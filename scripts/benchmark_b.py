"""
benchmark_b.py - The 6 knock-out cases, end to end, with simulation.

Configuration of benchmark/cases.json (kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr).
For each case:
  1. knock-out run (every strict equivalent of the hidden reaction removed);
  2. tapir.generator candidates for the target (the hidden set never reaches the generator);
  3. flux pre-ranking on the knock-out run (tapir.flux), generic rate from capture_guess;
  4. simulate the top K (K=30) by flux, each added alone to the knock-out network;
  5. score = |log10(X_cand / X_base)| + penalty (lower is better), where
     penalty = sum over observed TMC-1 species other than the target (observations.observed) of
               max(0, |log10(X_o,cand / X_o,base)| - log10 3): only changes beyond 3x count;
  6. rank of the hidden reaction in the final ranking (strict and channel levels).
The objective is X_base of the original model, not the TMC-1 observation. Hits@1/5/10, MRR and total
wall time; simulations run on 4 processes (~75 s).
usage: python -m scripts.benchmark_b [--k 30] [--procs 4]
Output: results/benchmark_b.csv (one row per simulated candidate), summary on stdout.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, json, math, multiprocessing as mp, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, T_REF, run, X, label
from tapir.observations import observed
from tapir.generator import Generator, load_cases, rank
from tapir.flux import FluxScorer, knockout_exclude
from tapir.paths import relative, result

OUT = result("benchmark_b.csv")
SCREEN = result("knockout_screen.csv")
KS = (1, 5, 10)
LOG3 = math.log10(3)
FLOOR = 1e-40


def _simulate(task):
    """(hidden key or None, extra row or None, species) -> ({species: X at t_ref}, runtime, error)"""
    hidden_key, row, species = task
    kw = {}
    if hidden_key is not None:
        kw["exclude"] = knockout_exclude(hidden_key)
    if row is not None:
        kw["extra_reactions"] = [row]
    try:
        net, t, y = run(**kw)
    except RuntimeError as e:
        return None, None, str(e)
    return {s: X(net, t, y, s, T_REF) for s in species if s in net.idx}, net.last_runtime, ""


def _knockout(task):
    """hidden key -> (FluxScorer, {species: X_ko})"""
    hidden_key, species = task
    net, t, y = run(exclude=knockout_exclude(hidden_key))
    return FluxScorer(net, t, y), {s: X(net, t, y, s, T_REF) for s in species if s in net.idx}


def objective(xs, xbase, target, obs):
    """(score, |log10 X_target / X_base|, penalty, number of observed species beyond 3x)"""
    err = abs(math.log10(max(xs[target], FLOOR) / max(xbase[target], FLOOR)))
    excess = [abs(math.log10(max(xs[o], FLOOR) / max(xbase[o], FLOOR))) - LOG3
              for o in obs if o != target and o in xs]
    pen = sum(e for e in excess if e > 0)
    return err + pen, err, pen, sum(e > 0 for e in excess)


def summary(ranks):
    n = len(ranks)
    hits = " ".join(f"H@{k}={sum(r is not None and r <= k for r in ranks)}/{n}" for k in KS)
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return f"{hits} MRR={mrr:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    cases = load_cases()
    targets = [n for n, _ in cases]
    keys = {n: chem.canonical_key(h) for n, h in cases}
    obs = sorted(observed())
    species = sorted(set(obs) | set(targets))
    recovery_screen = {r["species"]: r["capture_recovery"] for r in csv.DictReader(open(SCREEN))}

    with mp.Pool(a.procs) as pool:
        base = pool.apply_async(_simulate, ((None, None, species),))
        kos = pool.map(_knockout, [(keys[n], species) for n in targets])
        xbase, _, err = base.get()
        if xbase is None:
            raise RuntimeError(f"base run failed: {err}")
        obs = [o for o in obs if o in xbase]

        plan, tasks = {}, []
        for (name, h), (scorer, xko) in zip(cases, kos):
            cands = Generator(rows24, raw14, hidden=[h]).candidates(name)
            ranked, no_rate = scorer.score(cands, name)
            top = ranked[:a.k]
            plan[name] = dict(h=h, n_gen=len(cands), no_rate=no_rate, top=top, xko=xko,
                              rank_prior=rank(cands, h), rank_flux=rank(ranked, h, key="dX"))
            tasks += [(name, i, (keys[name], c["kinetic"], species)) for i, c in enumerate(top)]
        t_sim = time.time()
        results = pool.map(_simulate, [t for _, _, t in tasks])
        t_sim = time.time() - t_sim

    for (name, i, _), (xs, rt, err) in zip(tasks, results):
        c = plan[name]["top"][i]
        c["flux_rank"], c["runtime"], c["error"] = i + 1, rt, err
        c["prior"] = c["score"]                      # generator prior; "score" becomes the objective
        if xs is None:
            c["score"], c["log_err"], c["penalty"], c["n_3x"], c["X_cand"] = math.inf, None, None, None, None
        else:
            c["score"], c["log_err"], c["penalty"], c["n_3x"] = objective(xs, xbase, name, obs)
            c["X_cand"] = xs[name]
        c["neg_score"] = -c["score"]

    rows_out, ranks_s, ranks_c = [], [], []
    print(f"BENCHMARK B  K={a.k}  procs={a.procs}  t_ref={T_REF:.0e} yr")
    for name, _ in cases:
        p = plan[name]; h = p["h"]; top = p["top"]
        top.sort(key=lambda c: (c["score"], chem.canonical_key(c)))
        rs, rc = rank(top, h, key="neg_score"), rank(top, h, "channel", name, key="neg_score")
        ranks_s.append(rs); ranks_c.append(rc)
        ko_score = objective(p["xko"], xbase, name, obs)[0]
        hid = next((c for c in top if chem.equivalent(c, h)), None)
        rec = "" if hid is None or hid["X_cand"] is None else f"{hid['X_cand'] / xbase[name]:.2g}"
        print(f" {name:8s} N={p['n_gen']:4d} (no rate {p['no_rate']:2d})  prior rank={p['rank_prior']}  "
              f"flux rank={p['rank_flux']}  final rank={rs} (channel {rc})  "
              f"recovery={rec or '-'} (screen {recovery_screen.get(name) or '-'})  knock-out score={ko_score:.2f}")
        print(f"          top1: {label(top[0])}  score={top[0]['score']:.3f}  [{top[0]['family']}]")
        for j, c in enumerate(top):
            rows_out.append(dict(case=name, final_rank=j + 1, flux_rank=c["flux_rank"], reaction=label(c),
                                 family=c["family"], prior_score=f"{c['prior']:.3f}",
                                 k=f"{c['k']:.2e}", dX_pred=f"{c['dX']:.2e}",
                                 X_cand="" if c["X_cand"] is None else f"{c['X_cand']:.3e}",
                                 X_base=f"{xbase[name]:.3e}",
                                 log_err="" if c["log_err"] is None else f"{c['log_err']:.3f}",
                                 penalty="" if c["penalty"] is None else f"{c['penalty']:.3f}",
                                 n_obs_3x="" if c["n_3x"] is None else c["n_3x"], score=f"{c['score']:.3f}",
                                 hidden_strict=chem.equivalent(c, h),
                                 hidden_channel=chem.equivalent(c, h, "channel", name), error=c["error"]))
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_out[0])); wr.writeheader(); wr.writerows(rows_out)
    n_sims = 1 + len(cases) + len(tasks)
    rts = [c["runtime"] for p in plan.values() for c in p["top"] if c["runtime"]]
    print(f" strict : {summary(ranks_s)}")
    print(f" channel: {summary(ranks_c)}")
    print(f" {n_sims} simulations ({len(tasks)} candidates), {sum(1 for _, _, e in results if e)} failed, "
          f"mean {sum(rts) / len(rts):.1f} s/run, candidate stage {t_sim:.0f} s, "
          f"total {time.time() - t0:.0f} s  -> {relative(OUT)}")
