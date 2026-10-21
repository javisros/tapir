"""
discover.py - Discovery mode against the TMC-1 observation (tapir.discovery).

Reference configuration (kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr):
  1. base run; the flux pre-ranking (tapir.flux) is computed on it;
  2. candidates producing the target from the chosen generators (nothing hidden), merged by canonical key;
  3. rate from tapir.rates.RateEstimator; candidates with k = 0 (unpaired_increase, ion_h_transfer)
     or without a rate are not simulated;
  4. simulate the top K by flux, each added alone to the base network;
  5. delta_D and x_ratio against the observation (tapir.discovery), ranking by delta_D;
  6. k_needed by bisection for the K simulated candidates (skipped with --no-bisect);
  7. "existing" block: the --existing main formation and destruction reactions of the target in the
     base network at t_ref (scripts.knockout_demo.budget), the factor on their k that reaches X_obs and
     whether it is within the KIDA uncertainty factor F (skipped with --existing 0).
Simulations run on --procs processes with OMP_NUM_THREADS=1.
usage: python -m scripts.discover [CH2CCH] [--k 40] [--top 20] [--procs 4] [--generators analogy]
                                  [--no-bisect] [--existing 5]
Output: results/discovery/<target>.json (top N with the target's curve) and <target>.csv (all K).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, datetime, json, multiprocessing as mp, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows, W24_GAS
from tapir.discovery import (DISCLAIMER, MAX_SIMS, K_MIN, K_MAX, K_MAX_DEFAULT, FACTOR_LIMIT, observation,
                             observed_set, distance, simulate, required_rate, hypothesis, rank_key, sig,
                             kida_uncertainty, row_key, rate_factor, existing_record)
from tapir.flux import FluxScorer
from tapir.generator import Generator
from tapir.knockout import NET, EL, T_REF, run, X, label
from tapir.rates import RateEstimator, ion_flag
from tapir.paths import RESULTS, relative
from scripts.knockout_demo import budget

GENERATORS = {"analogy": lambda rows24, raw14, target: Generator(rows24, raw14).candidates(target)}
CSV_FIELDS = ("rank", "reaction", "source", "family", "class", "k", "k_source", "unpaired_flag",
              "ion_flag", "flux_rank", "dX_flux", "X_cand", "x_ratio", "D", "delta_D", "k_needed",
              "k_status", "n_bisect", "k_max_class", "capture_feasible", "error")


def _simulate(task):
    return simulate(*task)


def candidates(names, rows24, raw14, target):
    """Union of the generators' candidates, merged by canonical key; 'source' lists the generators."""
    out = {}
    for g in names:
        for c in GENERATORS[g](rows24, raw14, target):
            k = chem.canonical_key(c)
            if k in out:
                out[k]["source"].append(g)
            else:
                out[k] = dict(c, source=[g])
    return list(out.values())


def existing_tasks(net, t, y, target, n):
    """budget() of the base run -> [(network row, role, frac)] for the n main formation and destruction
    reactions of the target at t_ref."""
    form, dest = budget(net, t, y, target, T_REF, n)
    by_label = {label(r): r for r in net.rows}
    return [(by_label[lab], role, frac) for role, part in (("formation", form), ("destruction", dest))
            for lab, frac in part if lab in by_label]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="CH2CCH")
    ap.add_argument("--k", type=int, default=40, help="candidates simulated (top K by flux)")
    ap.add_argument("--top", type=int, default=20, help="hypotheses written to the JSON")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--generators", default="analogy", help=f"comma-separated: {', '.join(GENERATORS)}")
    ap.add_argument("--no-bisect", action="store_true", help="skip k_needed (k_needed = null)")
    ap.add_argument("--existing", type=int, default=5,
                    help="main formation and destruction reactions of the existing network to scale (0: skip)")
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
    D_base = round(distance(xbase, obs), 4)
    scorer = FluxScorer(net, t, y)
    ex_tasks = existing_tasks(net, t, y, target, a.existing) if a.existing > 0 else []

    cands = candidates(gens, rows24, raw14, target)
    for c in cands:
        c["ion_flag"] = ion_flag(c)
    ranked, no_rate = scorer.score(cands, target, estimator=RateEstimator(rows24))
    live = [c for c in ranked if c["k"] > 0]
    top = live[:a.k]
    for i, c in enumerate(top):
        c["flux_rank"] = i + 1
    print(f"DISCOVERY {target}  X_obs={x_obs:.2e} ({obs_src})  base X={xbase[target]:.2e} "
          f"(x_ratio {xbase[target] / x_obs:.2g}, D={D_base:.3f} on {len(obs)} species)")
    print(f" generators={','.join(gens)}: {len(cands)} candidates, {no_rate} without a rate, "
          f"{len(ranked) - len(live)} with k = 0, simulating {len(top)}")

    with mp.Pool(a.procs) as pool:
        sims = pool.map(_simulate, [(c["kinetic"], target, species) for c in top])
        hyps = [hypothesis(c, s, D_base, obs, target) for c, s in zip(top, sims)]
        n_bisect = 0
        if not a.no_bisect:
            t_b = time.time()
            req = pool.map(required_rate, [(c["kinetic"], target, x_obs) for c in top])
            for h, r in zip(hyps, req):
                h.update(k_needed=sig(r["k_needed"]), k_status=r["k_status"], n_bisect=r["n_sims"],
                         capture_feasible=r["capture_feasible"])
                if r["error"]:
                    h["error"] = "; ".join(e for e in (h["error"], "k_needed: " + r["error"]) if e)
            n_bisect = sum(r["n_sims"] for r in req)
            print(f" k_needed: {n_bisect} simulations, {time.time() - t_b:.0f} s")
        existing, n_existing = [], 0
        if ex_tasks:
            t_e = time.time()
            unc = kida_uncertainty(NET)
            res = pool.map(rate_factor, [(row, role, target, species, x_obs) for row, role, _ in ex_tasks])
            existing = [existing_record(row, role, frac, unc.get(row_key(row)), r, xbase, obs, target)
                        for (row, role, frac), r in zip(ex_tasks, res)]
            n_existing = sum(r["n_bisect"] + 1 for r in res)
            print(f" existing reactions: {n_existing} simulations, {time.time() - t_e:.0f} s")

    hyps.sort(key=rank_key)
    for i, h in enumerate(hyps):
        h["rank"] = i + 1
    n_sims = 1 + len(top) + n_bisect + n_existing
    total = time.time() - t0
    out_dir = os.path.join(RESULTS, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    doc = dict(
        species=target, mode="formation", X_obs=x_obs, obs_source=obs_src, generators=gens,
        config=dict(network="kida.uva.2024", C_O=round(EL["C+"] / EL["O"], 3), S=EL["S+"], t_ref=T_REF,
                    physics=W24_GAS),
        config2=None,
        X_base=sig(xbase[target]), D_base=D_base, n_generated=len(cands), n_no_rate=no_rate,
        n_simulated=len(top), runtime_s=round(total),
        t=[sig(v) for v in t], curve_base=[sig(v) for v in 2 * y[net.idx[target]]],
        disclaimer=DISCLAIMER,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        settings=dict(abundance_unit="X relative to H2", K=a.k, top=a.top,
                      k_needed=None if a.no_bisect else dict(k_min=K_MIN, k_max=K_MAX_DEFAULT,
                                                             k_max_by_class=K_MAX, max_sims=MAX_SIMS),
                      existing=dict(n_per_role=a.existing, factor_limit=FACTOR_LIMIT, max_sims=MAX_SIMS)),
        x_ratio_base=sig(xbase[target] / x_obs),
        D_species=sorted(s for s in obs if s in xbase),
        n_k_zero=len(ranked) - len(live), n_failed=sum(1 for s in sims if s["error"]),
        n_reachable=None if a.no_bisect else sum(h["capture_feasible"] is True for h in hyps),
        n_simulations=n_sims, procs=a.procs,
        hypotheses=[{k: h[k] for k in ("rank",) + tuple(k for k in h if k != "rank")} for h in hyps[:a.top]],
        existing=existing)
    path_json = os.path.join(out_dir, f"{target}.json")
    with open(path_json, "w") as fh:
        json.dump(doc, fh, indent=1)
    path_csv = os.path.join(out_dir, f"{target}.csv")
    with open(path_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({k: ("" if h[k] is None else "+".join(h[k]) if k == "source" else h[k]) for k in CSV_FIELDS}
                     for h in hyps)

    for h in hyps[:10]:
        print(f" {h['rank']:2d} dD={h['delta_D'] if h['delta_D'] is not None else 'nan':>7} "
              f"x_ratio={h['x_ratio'] or 0:8.2e} k={h['k']:.1e} [{h['k_source']}] "
              f"k_needed={h['k_needed'] or '-'} ({h['k_status'] or '-'}, feasible={h['capture_feasible']}) "
              f"{h['reaction']}  {h['unpaired_flag'] or ''} {h['ion_flag'] or ''}")
    for e in existing:
        print(f" {e['role']:11s} {100 * e['frac']:5.1f}%  F={e['F']} g={e['g']}  factor={e['factor_needed']} "
              f"({e['factor_status']}, within F: {e['within_uncertainty']})  dD={e['delta_D']} "
              f"dD_gratier={e['delta_D_gratier']} n3x={e['n_gratier_3x']}  {e['reaction']}")
    print(f" {n_sims} simulations, total {total:.0f} s -> {relative(path_json)}, {relative(path_csv)}")
