"""
discover.py - Discovery mode against the TMC-1 observation (tapir.discovery).

Reference configuration (kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr):
  1. base run; the flux pre-ranking (tapir.flux) is computed on it;
  2. candidates producing the target from the chosen generators (nothing hidden), merged by canonical key;
  3. rate from tapir.rates.RateEstimator; candidates with k = 0 (unpaired_increase, ion_h_transfer)
     or without a rate are not simulated;
  4. simulate the top K by flux, each added alone to the base network;
  5. delta_D and x_ratio against the observation (tapir.discovery), ranking by delta_D;
  6. k_needed by bisection for the K simulated candidates (skipped with --no-bisect).
Simulations run on --procs processes with OMP_NUM_THREADS=1.
usage: python -m scripts.discover [CH2CCH] [--k 40] [--top 20] [--procs 4] [--generators analogy]
                                  [--no-bisect]
Output: results/discovery/<target>.json (top N with the target's curve) and <target>.csv (all K).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, datetime, json, multiprocessing as mp, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows, W24_GAS
from tapir.discovery import (DISCLAIMER, MAX_SIMS, K_MIN, K_MAX, K_MAX_DEFAULT, observation, observed_set,
                             distance, simulate, required_rate, hypothesis, rank_key, sig)
from tapir.flux import FluxScorer
from tapir.generator import Generator
from tapir.knockout import NET, EL, T_REF, run, X
from tapir.rates import RateEstimator, ion_flag
from tapir.paths import RESULTS, relative

GENERATORS = {"analogy": lambda rows24, raw14, target: Generator(rows24, raw14).candidates(target)}
CSV_FIELDS = ("rank", "reaction", "source", "family", "reaction_class", "k", "k_source", "unpaired_flag",
              "ion_flag", "flux_rank", "dX_pred", "X_target", "x_ratio", "D", "delta_D", "k_needed",
              "k_status", "k_needed_sims", "capture_max", "capture_feasible", "error")


def _simulate(task):
    return simulate(*task)


def candidates(names, rows24, raw14, target):
    """Union of the generators' candidates, merged by canonical key; 'source' joins the generators."""
    out = {}
    for g in names:
        for c in GENERATORS[g](rows24, raw14, target):
            k = chem.canonical_key(c)
            if k in out:
                out[k]["source"] += "+" + g
            else:
                out[k] = dict(c, source=g)
    return list(out.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="CH2CCH")
    ap.add_argument("--k", type=int, default=40, help="candidates simulated (top K by flux)")
    ap.add_argument("--top", type=int, default=20, help="hypotheses written to the JSON")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--generators", default="analogy", help=f"comma-separated: {', '.join(GENERATORS)}")
    ap.add_argument("--no-bisect", action="store_true", help="skip k_needed (k_needed = null)")
    a = ap.parse_args()
    gens = [g.strip() for g in a.generators.split(",") if g.strip()]
    unknown = [g for g in gens if g not in GENERATORS]
    if unknown or not gens:
        ap.error(f"unknown generators {unknown}; available: {', '.join(GENERATORS)}")
    t0 = time.time()
    target = a.target
    x_obs, obs_src = observation(target)
    obs = observed_set(target)
    species = sorted(obs)

    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    net, t, y = run()
    xbase = {s: X(net, t, y, s, T_REF) for s in species if s in net.idx}
    base = dict(X_target=sig(xbase[target]), x_ratio=sig(xbase[target] / x_obs),
                D=round(distance(xbase, obs), 4), curve=[sig(v) for v in 2 * y[net.idx[target]]])
    scorer = FluxScorer(net, t, y)

    cands = candidates(gens, rows24, raw14, target)
    for c in cands:
        c["ion_flag"] = ion_flag(c)
    ranked, no_rate = scorer.score(cands, target, estimator=RateEstimator(rows24))
    live = [c for c in ranked if c["k"] > 0]
    top = live[:a.k]
    for i, c in enumerate(top):
        c["flux_rank"] = i + 1
    print(f"DISCOVERY {target}  X_obs={x_obs:.2e} ({obs_src})  base X={xbase[target]:.2e} "
          f"(x_ratio {base['x_ratio']:.2g}, D={base['D']:.3f} on {len(obs)} species)")
    print(f" generators={','.join(gens)}: {len(cands)} candidates, {no_rate} without a rate, "
          f"{len(ranked) - len(live)} with k = 0, simulating {len(top)}")

    with mp.Pool(a.procs) as pool:
        sims = pool.map(_simulate, [(c["kinetic"], target, species) for c in top])
        hyps = [hypothesis(c, s, base, obs, target) for c, s in zip(top, sims)]
        rows = {h["reaction"]: c["kinetic"] for h, c in zip(hyps, top)}
        n_bisect = 0
        if not a.no_bisect:
            t_b = time.time()
            req = pool.map(required_rate, [(rows[h["reaction"]], target, x_obs) for h in hyps])
            for h, r in zip(hyps, req):
                h.update(k_needed=sig(r["k_needed"]), k_status=r["k_status"], k_needed_sims=r["n_sims"],
                         capture_feasible=r["capture_feasible"])
                if r["error"]:
                    h["error"] = "; ".join(e for e in (h["error"], "k_needed: " + r["error"]) if e)
            n_bisect = sum(r["n_sims"] for r in req)
            print(f" k_needed: {n_bisect} simulations, {time.time() - t_b:.0f} s")

    hyps.sort(key=rank_key)
    for i, h in enumerate(hyps):
        h["rank"] = i + 1
    n_sims = 1 + len(top) + n_bisect
    total = time.time() - t0
    out_dir = os.path.join(RESULTS, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    doc = dict(
        species=target, mode="observation", disclaimer=DISCLAIMER,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        config=dict(network="kida.uva.2024", C_O=round(EL["C+"] / EL["O"], 3), S=EL["S+"], t_ref_yr=T_REF, conditions=W24_GAS,
                    abundance_unit="X relative to H2", generators=gens, K=a.k, top=a.top,
                    k_needed=None if a.no_bisect else dict(k_min=K_MIN, k_max=K_MAX_DEFAULT,
                                                           k_max_by_class=K_MAX, max_sims=MAX_SIMS)),
        observation=dict(X_obs=x_obs, source=obs_src),
        D_species=sorted(s for s in obs if s in xbase),
        base=base, t_yr=[sig(v) for v in t],
        counts=dict(generated=len(cands), no_rate=no_rate, k_zero=len(ranked) - len(live),
                    simulated=len(top), failed=sum(1 for s in sims if s["error"]),
                    reachable=None if a.no_bisect else sum(h["capture_feasible"] is True for h in hyps)),
        runtime=dict(total_s=round(total), n_simulations=n_sims, procs=a.procs),
        hypotheses=[{k: h[k] for k in ("rank",) + tuple(k for k in h if k != "rank")} for h in hyps[:a.top]])
    path_json = os.path.join(out_dir, f"{target}.json")
    with open(path_json, "w") as fh:
        json.dump(doc, fh, indent=1)
    path_csv = os.path.join(out_dir, f"{target}.csv")
    with open(path_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({k: ("" if h[k] is None else h[k]) for k in CSV_FIELDS} for h in hyps)

    for h in hyps[:10]:
        print(f" {h['rank']:2d} dD={h['delta_D'] if h['delta_D'] is not None else 'nan':>7} "
              f"x_ratio={h['x_ratio'] or 0:8.2e} k={h['k']:.1e} [{h['k_source']}] "
              f"k_needed={h['k_needed'] or '-'} ({h['k_status'] or '-'}, feasible={h['capture_feasible']}) "
              f"{h['reaction']}  {h['unpaired_flag'] or ''} {h['ion_flag'] or ''}")
    print(f" {n_sims} simulations, total {total:.0f} s -> {relative(path_json)}, {relative(path_csv)}")
