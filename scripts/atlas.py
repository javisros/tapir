"""
atlas.py - Discovery over every TMC-1 species the model underestimates, with a supply diagnostic.

Species: those of tapir.observations with an observed X, X_mod / X_obs < UNDER (1/3) at t_ref in the
reference configuration (CH2CCH included as control). Overestimated ones (> 1 / UNDER) are listed as
"not analysed (requires destruction mode)"; the rest are within a factor 3.
Per species, the pipeline of scripts/discover.py with the analogy generator only (no LLM, no Bond):
candidates -> flags (unpaired, ion, IE) and rates (rates.RateEstimator) -> flux pre-ranking -> top K
simulated -> k_needed by bisection (at most MAX_SIMS_ATLAS simulations per candidate).
  n_reachable        capture_feasible: k_needed <= rates.CAPTURE_MAX of the class
  n_reachable_clean  reachable and without any flag (ion_flag, unpaired_flag, ie_flag): the only ones
                     that would count as possible findings
Supply diagnostic (generalises the CH2CCH bottleneck): the main formation reaction at t_ref
(knockout_demo.budget) times FACTOR (1e3):
  "rate_limited"    X_target reaches X_obs
  "supply_limited"  it does not, and some reactant falls below SUPPLY_FLOOR (10 %) of its base value
  "other"           neither
Simulations run on --procs processes with OMP_NUM_THREADS=1.
usage: python -m scripts.atlas [--k 20] [--procs 4] [--species CH2CCH,NH3]
Output: results/discovery/atlas.json and atlas.csv (written to a temporary file, then renamed); the
per-species discovery JSONs are not touched.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, datetime, json, math, multiprocessing as mp, time
from tapir.darkcloud import parse_kida, load_rows, W24_GAS
from tapir import chem
from tapir.discovery import (DISCLAIMER, observed_set, simulate, simulate_scaled, hypothesis, rank_key, sig,
                             bisect_log_k, with_rate, k_bounds, capture_feasible)
from tapir.flux import FluxScorer
from tapir.generator import Generator
from tapir.knockout import NET, EL, T_REF, run, X, label
from tapir.observations import observed
from tapir.rates import RateEstimator, ion_flag, ie_flag, capture_max
from tapir.paths import RESULTS, relative
from scripts.knockout_demo import budget

UNDER = 1 / 3
K = 20
MAX_SIMS_ATLAS = 8
FACTOR = 1e3
SUPPLY_FLOOR = 0.1
FLAGS = ("ion_flag", "unpaired_flag", "ie_flag")
CSV_FIELDS = ("species", "X_obs", "x_ratio_base", "diagnosis", "main_formation", "limiting_reactant",
              "limiting_ratio", "x_ratio_scaled", "n_candidates", "n_simulated", "best_x_ratio", "n_reachable",
              "n_reachable_clean", "best_reaction", "best_flags")


def _simulate(task):
    return simulate(*task)


def _scaled(task):
    return simulate_scaled(*task)


def required_rate(task):
    """discovery.required_rate with at most MAX_SIMS_ATLAS simulations."""
    row, target, x_obs = task
    errors = []

    def f(k):
        s = simulate(with_rate(row, k), target, [target])
        if s["error"]:
            errors.append(s["error"])
            return None
        return s["xs"][target]
    res = bisect_log_k(f, x_obs, *k_bounds(row), max_sims=MAX_SIMS_ATLAS)
    res.pop("trace")
    res.update(error="; ".join(errors), capture_feasible=capture_feasible(row, res["k_needed"], res["k_status"]))
    return res


def species_lists(net, t, y):
    """(underestimated, within factor 3, overestimated): [(species, X_obs, x_ratio_base)], sorted by x_ratio."""
    out = ([], [], [])
    for s, (x, _) in observed().items():
        if x is None or s not in net.idx:
            continue
        r = X(net, t, y, s, T_REF) / x
        out[0 if r < UNDER else 2 if r > 1 / UNDER else 1].append((s, x, r))
    return tuple(sorted(o, key=lambda v: v[2]) for o in out)


def supply(row, x_obs, xbase, xscaled, target):
    """Supply diagnostic of the main formation reaction scaled by FACTOR."""
    reac = [s for s in row["reac"] if s in xbase and s in xscaled and xbase[s] > 0]
    ratios = {s: xscaled[s] / xbase[s] for s in reac}
    lim = min(ratios, key=ratios.get) if ratios else None
    reached = xscaled[target] >= x_obs
    diag = ("rate_limited" if reached else
            "supply_limited" if lim is not None and ratios[lim] < SUPPLY_FLOOR else "other")
    return dict(reaction=label(row), factor=FACTOR, diagnosis=diag, X_target_base=sig(xbase[target]),
                X_target_scaled=sig(xscaled[target]), x_ratio_scaled=sig(xscaled[target] / x_obs),
                target_gain=sig(xscaled[target] / xbase[target]),
                reactants={s: sig(v) for s, v in ratios.items()}, limiting_reactant=lim,
                limiting_ratio=sig(ratios[lim]) if lim else None)


def flags(h):
    return {f: h[f] for f in FLAGS if h.get(f)}


def analyse(target, x_obs, net, t, y, gen, est, scorer, pool, k):
    obs = observed_set(target)
    form, _ = budget(net, t, y, target, T_REF, 3)
    by_label = {label(r): r for r in net.rows}
    form = [(lab, f) for lab, f in form if lab in by_label]      # budget also names H2 formation on grains
    main = by_label[form[0][0]] if form else None
    species = sorted(set(obs) | (set(main["reac"]) if main else set()))
    xbase = {s: X(net, t, y, s, T_REF) for s in species if s in net.idx}
    bn = pool.apply_async(_scaled, ((main, FACTOR, target, species),)) if main else None

    cands = gen.candidates(target)
    for c in cands:
        c["source"] = ["analogy"]
        c["ion_flag"], c["ie_flag"] = ion_flag(c), ie_flag(c)
    ranked, no_rate = scorer.score(cands, target, estimator=est)
    live = [c for c in ranked if c["k"] > 0]
    top = live[:k]
    for i, c in enumerate(top):
        c["flux_rank"] = i + 1
    sims = pool.map(_simulate, [(c["kinetic"], target, species) for c in top])
    hyps = [hypothesis(c, s, xbase, obs, target) for c, s in zip(top, sims)]
    req = pool.map(required_rate, [(c["kinetic"], target, x_obs) for c in top])
    for h, r in zip(hyps, req):
        h.update(k_needed=sig(r["k_needed"]), k_status=r["k_status"], n_bisect=r["n_sims"],
                 capture_feasible=r["capture_feasible"])
    hyps.sort(key=rank_key)
    sup = None
    if bn is not None:
        s = bn.get()
        sup = dict(error=s["error"]) if s["error"] else supply(main, x_obs, xbase, s["xs"], target)
    reach = [h for h in hyps if h["capture_feasible"] is True]
    clean = [h for h in reach if not flags(h)]
    ok = [h for h in hyps if h["x_ratio"] is not None]
    best = hyps[0] if hyps else None
    keep = ("reaction", "class", "k", "k_source", "x_ratio", "delta_D", "k_needed", "k_status", "k_max_class",
            "capture_feasible") + FLAGS
    return dict(
        species=target, X_obs=x_obs, x_ratio_base=sig(xbase[target] / x_obs), main_formation_frac=
        round(float(form[0][1]), 4) if form else None,
        n_candidates=len(cands), n_no_rate=no_rate, n_k_zero=len(ranked) - len(live), n_simulated=len(top),
        n_simulations=len(top) + sum(r["n_sims"] for r in req) + (1 if main else 0),
        best_x_ratio=sig(max(h["x_ratio"] for h in ok)) if ok else None,
        n_reachable=len(reach), n_reachable_clean=len(clean),
        best=dict({k_: best.get(k_) for k_ in keep}, flags=flags(best)) if best else None,
        reachable_clean=[{k_: h.get(k_) for k_ in keep} for h in clean],
        reachable_flagged=[dict({k_: h.get(k_) for k_ in keep}, flags=flags(h)) for h in reach if flags(h)],
        supply=sup,
        candidates=[dict({k_: h.get(k_) for k_ in keep}, flags=flags(h)) for h in hyps])


def write_atomic(path, write):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--species", default="", help="comma-separated subset of the underestimated species")
    a = ap.parse_args(argv)
    t0 = time.time()
    net, t, y = run()
    under, within, over = species_lists(net, t, y)
    todo = [v for v in under if not a.species or v[0] in a.species.split(",")]
    rows24 = load_rows(NET)
    gen, est, scorer = Generator(rows24, parse_kida(chem.REACTIONS_2014)), RateEstimator(rows24), FluxScorer(net, t, y)
    results = []
    with mp.Pool(a.procs) as pool:
        for s, x_obs, r in todo:
            t1 = time.time()
            res = analyse(s, x_obs, net, t, y, gen, est, scorer, pool, a.k)
            res["runtime_s"] = round(time.time() - t1)
            results.append(res)
            sp = res["supply"] or {}
            print(f"{s:9s} x_ratio_base={r:.3g}  {sp.get('diagnosis')} ({sp.get('reaction')}, limiting "
                  f"{sp.get('limiting_reactant')} x{sp.get('limiting_ratio')}, target x{sp.get('target_gain')})  "
                  f"cands={res['n_candidates']} sim={res['n_simulated']} best_x_ratio={res['best_x_ratio']} "
                  f"reachable={res['n_reachable']} clean={res['n_reachable_clean']}  {res['runtime_s']} s", flush=True)
    out_dir = os.path.join(RESULTS, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    doc = dict(
        description="TAPIR over every TMC-1 species underestimated more than 3x at t_ref (analogy generator only)",
        disclaimer=DISCLAIMER,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        config=dict(network="kida.uva.2024", C_O=round(EL["C+"] / EL["O"], 3), S=EL["S+"], t_ref=T_REF,
                    physics=W24_GAS),
        settings=dict(generators=["analogy"], under=UNDER, K=a.k, max_sims=MAX_SIMS_ATLAS, factor=FACTOR,
                      supply_floor=SUPPLY_FLOOR, flags=list(FLAGS), procs=a.procs),
        species=results,
        within_factor_3=[dict(species=s, X_obs=x, x_ratio_base=sig(r)) for s, x, r in within],
        not_analysed=[dict(species=s, X_obs=x, x_ratio_base=sig(r), reason="overestimated: requires destruction mode")
                      for s, x, r in over],
        runtime_s=round(time.time() - t0))
    path = os.path.join(out_dir, "atlas.json")
    write_atomic(path, lambda fh: json.dump(doc, fh, indent=1))
    rows = []
    for r in results:
        sp, b = r["supply"] or {}, r["best"] or {}
        rows.append(dict(species=r["species"], X_obs=r["X_obs"], x_ratio_base=r["x_ratio_base"],
                         diagnosis=sp.get("diagnosis"), main_formation=sp.get("reaction"),
                         limiting_reactant=sp.get("limiting_reactant"), limiting_ratio=sp.get("limiting_ratio"),
                         x_ratio_scaled=sp.get("x_ratio_scaled"), n_candidates=r["n_candidates"],
                         n_simulated=r["n_simulated"], best_x_ratio=r["best_x_ratio"], n_reachable=r["n_reachable"],
                         n_reachable_clean=r["n_reachable_clean"], best_reaction=b.get("reaction"),
                         best_flags="+".join(f"{k}={v}" for k, v in (b.get("flags") or {}).items())))

    def write_csv(fh):
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    path_csv = os.path.join(out_dir, "atlas.csv")
    write_atomic(path_csv, write_csv)
    print(f"{len(results)} species, not analysed (overestimated) {len(over)}, within 3x {len(within)}; "
          f"total {doc['runtime_s']} s -> {relative(path)}, {relative(path_csv)}")


if __name__ == "__main__":
    main()
