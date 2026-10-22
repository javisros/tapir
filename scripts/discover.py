"""
discover.py - Discovery mode against the TMC-1 observation (tapir.discovery).

Reference configuration (kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr):
  1. base run; the flux pre-ranking (tapir.flux) is computed on it;
  2. candidates producing the target from the chosen generators (nothing hidden), merged by canonical key;
     llm (tapir.generator_llm): one LLM call per target, with the N_CONTEXT main formation and destruction
     reactions of the target in the base run at t_ref as context and needed_effect "increase"; one client for
     the whole run, so TAPIR_LLM_MAX_CALLS caps it; answers cached in results/llm_cache; a FormatError
     (raised by the client after its own retry) is logged as status "format_error" and that target keeps
     the other generators' candidates only; the reasoning effort is the configured one (no fallback chain);
  3. rate from tapir.rates.RateEstimator; candidates with k = 0 (unpaired_increase, ion_h_transfer)
     or without a rate are not simulated;
  4. simulate the top K by flux, each added alone to the base network;
  5. delta_D and x_ratio against the observation (tapir.discovery), ranking by delta_D;
  6. k_needed by bisection for the K simulated candidates (skipped with --no-bisect);
  7. "existing" block: the --existing main formation and destruction reactions of the target in the
     base network at t_ref (scripts.knockout_demo.budget), the factor on their k that reaches X_obs and
     whether it is within the KIDA uncertainty factor F (skipped with --existing 0 and in upstream mode);
  8. "bottleneck" block: reactants of the two main formation reactions (and the --upstream precursors)
     in the base run and with the main formation reaction times 1e3.
Upstream mode (--upstream C2H4,CH3): steps 2-4 use candidates producing each precursor, pre-ranked by
flux towards it, top K per precursor (field "via"); scoring and k_needed stay on the target.
Simulations run on --procs processes with OMP_NUM_THREADS=1.
--dry-run: base run and LLM prompts only; prints, per LLM call, whether it is cached and the estimated
tokens, then exits (no LLM call, no candidate simulation, nothing written).
usage: python -m scripts.discover [CH2CCH] [--k 40] [--top 20] [--procs 4] [--generators analogy,llm]
                                  [--no-bisect] [--existing 5] [--upstream C2H4,CH3] [--dry-run]
Output: results/discovery/<target>.json (top N with the target's curve) and <target>.csv (all K);
<target>_upstream.json / .csv in upstream mode.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, datetime, glob, hashlib, json, multiprocessing as mp, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows, W24_GAS
from tapir.discovery import (DISCLAIMER, MAX_SIMS, K_MIN, K_MAX, K_MAX_DEFAULT, FACTOR_LIMIT, observation,
                             observed_set, distance, simulate, required_rate, hypothesis, rank_key, sig,
                             kida_uncertainty, row_key, rate_factor, existing_record, simulate_scaled,
                             bottleneck)
from tapir.flux import FluxScorer
from tapir.generator import Generator
from tapir.generator_llm import LLMGenerator, N_CONTEXT, SCHEMA, CACHE_DIR
from tapir.llm_client import FormatError
from tapir.knockout import NET, EL, T_REF, run, X, label
from tapir.rates import RateEstimator, ion_flag, ie_flag
from tapir.paths import RESULTS, relative
from scripts.knockout_demo import budget

GENERATORS = ("analogy", "llm")
TOKENS_PER_CHAR = 0.62          # dry-run fallback: prompt tokens / characters of the first cached answer
COMPLETION_TOKENS = (2000, 8000)  # dry-run range per call (~2200 measured at the default reasoning effort)
CSV_FIELDS = ("rank", "reaction", "via", "source", "family", "class", "k", "k_source", "unpaired_flag",
              "ion_flag", "ie_flag", "flux_rank", "dX_flux", "X_cand", "x_ratio", "D", "delta_D", "n_worse", "k_needed",
              "k_status", "n_bisect", "k_max_class", "capture_feasible", "error")


def _simulate(task):
    return simulate(*task)


def llm_context(net, t, y, target):
    """(formation, destruction) labels of the N_CONTEXT main reactions of the target in the base run at t_ref."""
    form, dest = budget(net, t, y, target, T_REF, N_CONTEXT)
    return [lab for lab, _ in form], [lab for lab, _ in dest]


def llm_record(llm, target, before, error=None):
    """Log entry of one use of the LLM generator. usage: the client's counters for this use (None if
    cached); token counts None when a failed request reported no usage (never estimated)."""
    c = llm.client
    usage = {k: v - before.get(k, 0) for k, v in getattr(c, "stats", {}).items()}
    if error is not None and not usage.get("total_tokens"):
        usage.update(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    rec = dict(target=target, status="ok" if error is None else "format_error", model=c.model,
               reasoning_effort=getattr(c, "reasoning_effort", None), cached=False, proposed=None, kept=0,
               discarded=None, usage=usage, error=error)
    if error is None:
        last = llm.last
        rec.update(cached=last["cached"], proposed=last["returned_raw"], kept=last["considered"],
                   discarded={k: v for k, v in last["discarded"].items() if v},
                   usage=None if last["cached"] else usage)
    return rec


def make_generators(names, rows24, raw14, net, t, y, client=None):
    """{name: f(target) -> candidates}. The llm one keeps one LLMGenerator (and client) for the run and
    appends a llm_record of each use to f.log; on a FormatError it returns no candidates (graceful
    degradation to the other generators)."""
    gens = {}
    if "analogy" in names:
        gens["analogy"] = Generator(rows24, raw14).candidates
    if "llm" in names:
        llm = LLMGenerator(rows24, raw14, client=client, allow_fallback_context=False)

        def f(target):
            before = dict(getattr(llm.client, "stats", {}))
            try:
                cands = llm.candidates(target, llm_context(net, t, y, target), "increase")
            except FormatError as e:
                f.log.append(llm_record(llm, target, before, error=f"{type(e).__name__}: {e}"))
                return []
            f.log.append(llm_record(llm, target, before))
            return cands
        f.log, f.llm = [], llm
        gens["llm"] = f
    return gens


def dry_run(targets, rows24, raw14, net, t, y):
    """Per target: prompt of the llm generator, cached or not, estimated tokens; no LLM call."""
    from tapir import llm_client
    stub = type("DryRun", (), dict(model=llm_client.MODEL, reasoning_effort=llm_client.REASONING_EFFORT))
    llm = LLMGenerator(rows24, raw14, client=stub, allow_fallback_context=False)
    ratios = [d["usage"]["prompt_tokens"] / len(d["prompt"]) for d in map(lambda p: json.load(open(p)),
              sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))) if (d.get("usage") or {}).get("calls") == 1]
    ratio = sum(ratios) / len(ratios) if ratios else TOKENS_PER_CHAR
    print(f"DRY RUN llm: model={stub.model} reasoning_effort={stub.reasoning_effort} "
          f"max_calls={llm_client.MAX_CALLS} (one retry per call on a format error)")
    new, pt = 0, 0.0
    for target in targets:
        prompt = llm.prompt(target, llm_context(net, t, y, target), "increase")
        h = hashlib.sha256(json.dumps([stub.model, stub.reasoning_effort, SCHEMA, prompt]).encode()).hexdigest()
        cached = os.path.exists(os.path.join(CACHE_DIR, h + ".json"))
        print(f" {target}: {'cached' if cached else 'NEW CALL'}  prompt {len(prompt)} chars ~ {ratio * len(prompt):.0f} tokens")
        if not cached:
            new, pt = new + 1, pt + ratio * len(prompt)
    lo, hi = (round(pt + new * c) for c in COMPLETION_TOKENS)
    print(f" {new} new call(s): ~{lo}-{hi} tokens (x2 if every call is retried); "
          f"prompt tokens/char {ratio:.2f} from {len(ratios)} cached answer(s)")
    return new, lo, hi


def candidates(gens, target):
    """Union of the generators' candidates, merged by canonical key; 'source' lists the generators."""
    out = {}
    for g, gen in gens.items():
        for c in gen(target):
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


def upstream_candidates(gens, precursors, scorer, estimator, k):
    """Top k by flux towards each precursor -> (candidates with 'via' and 'flux_rank', counts).
    A candidate found for several precursors keeps its first flux rank; 'via' joins them with '+'."""
    top, n = {}, dict(generated=0, no_rate=0, k_zero=0)
    for p in precursors:
        cands = candidates(gens, p)
        for c in cands:
            c["ion_flag"], c["ie_flag"] = ion_flag(c), ie_flag(c)
        ranked, no_rate = scorer.score(cands, p, estimator=estimator)
        live = [c for c in ranked if c["k"] > 0]
        n["generated"] += len(cands); n["no_rate"] += no_rate; n["k_zero"] += len(ranked) - len(live)
        for i, c in enumerate(live[:k]):
            key = chem.canonical_key(c)
            if key in top:
                top[key]["via"] += "+" + p
            else:
                top[key] = dict(c, via=p, flux_rank=i + 1)
    return list(top.values()), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="CH2CCH")
    ap.add_argument("--k", type=int, default=40, help="candidates simulated (top K by flux)")
    ap.add_argument("--top", type=int, default=20, help="hypotheses written to the JSON")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--generators", default="analogy", help=f"comma-separated: {', '.join(GENERATORS)}")
    ap.add_argument("--dry-run", action="store_true", help="LLM prompts and token estimate only, then exit")
    ap.add_argument("--no-bisect", action="store_true", help="skip k_needed (k_needed = null)")
    ap.add_argument("--existing", type=int, default=5,
                    help="main formation and destruction reactions of the existing network to scale (0: skip)")
    ap.add_argument("--upstream", default="", help="comma-separated precursors: upstream mode (e.g. C2H4,CH3)")
    a = ap.parse_args()
    names = [g.strip() for g in a.generators.split(",") if g.strip()]
    unknown = [g for g in names if g not in GENERATORS]
    if unknown or not names:
        ap.error(f"unknown generators {unknown}; available: {', '.join(GENERATORS)}")
    precursors = [p.strip() for p in a.upstream.split(",") if p.strip()]
    t0 = time.time()
    target = a.target
    x_obs, obs_src = observation(target)
    obs = observed_set(target)

    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    net, t, y = run()
    missing = [p for p in precursors if p not in net.idx]
    if missing:
        ap.error(f"precursors not in the network: {missing}")
    if a.dry_run:
        if "llm" not in names:
            ap.error("--dry-run estimates the llm generator: add it to --generators")
        dry_run(precursors or [target], rows24, raw14, net, t, y)
        raise SystemExit(0)
    gens = make_generators(names, rows24, raw14, net, t, y)
    form = existing_tasks(net, t, y, target, 2)[:2]
    supply = [s for row, role, _ in form if role == "formation" for s in row["reac"]]
    species = sorted(set(obs) | set(precursors) | set(supply))
    xbase = {s: X(net, t, y, s, T_REF) for s in species if s in net.idx}
    D_base = round(distance(xbase, obs), 4)
    scorer = FluxScorer(net, t, y)
    ex_tasks = existing_tasks(net, t, y, target, a.existing) if a.existing > 0 and not precursors else []

    if precursors:
        top, n = upstream_candidates(gens, precursors, scorer, RateEstimator(rows24), a.k)
        n_gen, no_rate, n_k_zero = n["generated"], n["no_rate"], n["k_zero"]
    else:
        cands = candidates(gens, target)
        for c in cands:
            c["ion_flag"], c["ie_flag"] = ion_flag(c), ie_flag(c)
        ranked, no_rate = scorer.score(cands, target, estimator=RateEstimator(rows24))
        live = [c for c in ranked if c["k"] > 0]
        top = live[:a.k]
        for i, c in enumerate(top):
            c["flux_rank"] = i + 1
        n_gen, n_k_zero = len(cands), len(ranked) - len(live)
    print(f"DISCOVERY {target}  X_obs={x_obs:.2e} ({obs_src})  base X={xbase[target]:.2e} "
          f"(x_ratio {xbase[target] / x_obs:.2g}, D={D_base:.3f} on {len(obs)} species)"
          + (f"  upstream via {','.join(precursors)}" if precursors else ""))
    print(f" generators={','.join(names)}: {n_gen} candidates, {no_rate} without a rate, "
          f"{n_k_zero} with k = 0, simulating {len(top)}")
    llm_doc = None
    if "llm" in gens:
        f = gens["llm"]
        llm_doc = dict(model=f.llm.client.model, reasoning_effort=getattr(f.llm.client, "reasoning_effort", None),
                       usage=dict(getattr(f.llm.client, "stats", {})), calls=f.log)
        for e in f.log:
            if e["status"] == "ok":
                print(f" llm {e['target']}: ok, {'cached' if e['cached'] else 'called'}, {e['proposed']} proposed, "
                      f"{e['kept']} kept, discarded {e['discarded']}, usage {e['usage']}")
            else:
                print(f" llm {e['target']}: {e['status']} ({e['error']}), analogy only, usage {e['usage']}")
        print(f" llm usage: {llm_doc['usage']}")

    with mp.Pool(a.procs) as pool:
        bn_row = form[0][0]
        bn_sim = pool.apply_async(simulate_scaled, (bn_row, FACTOR_LIMIT["formation"], target, species))
        sims = pool.map(_simulate, [(c["kinetic"], target, species) for c in top])
        hyps = [hypothesis(c, s, xbase, obs, target) for c, s in zip(top, sims)]
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
        bn_sim = bn_sim.get()
    bn = None if bn_sim["error"] else bottleneck(bn_row, FACTOR_LIMIT["formation"], xbase, bn_sim["xs"],
                                                  sorted(set(supply) | set(precursors)), target)

    hyps.sort(key=rank_key)
    for i, h in enumerate(hyps):
        h["rank"] = i + 1
    n_sims = 2 + len(top) + n_bisect + n_existing
    total = time.time() - t0
    out_dir = os.path.join(RESULTS, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    doc = dict(
        species=target, mode="formation", search="upstream" if precursors else "direct", upstream=precursors,
        X_obs=x_obs, obs_source=obs_src, generators=names, llm=llm_doc,
        config=dict(network="kida.uva.2024", C_O=round(EL["C+"] / EL["O"], 3), S=EL["S+"], t_ref=T_REF,
                    physics=W24_GAS),
        config2=None,
        X_base=sig(xbase[target]), D_base=D_base, n_generated=n_gen, n_no_rate=no_rate,
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
        n_k_zero=n_k_zero, n_failed=sum(1 for s in sims if s["error"]),
        n_reachable=None if a.no_bisect else sum(h["capture_feasible"] is True for h in hyps),
        n_simulations=n_sims, procs=a.procs,
        hypotheses=[{k: h[k] for k in ("rank",) + tuple(k for k in h if k != "rank")} for h in hyps[:a.top]],
        existing=existing, bottleneck=bn)
    stem = f"{target}_upstream" if precursors else target
    path_json = os.path.join(out_dir, f"{stem}.json")
    with open(path_json, "w") as fh:
        json.dump(doc, fh, indent=1)
    path_csv = os.path.join(out_dir, f"{stem}.csv")
    with open(path_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({k: ("" if h[k] is None else "+".join(h[k]) if k == "source" else h[k]) for k in CSV_FIELDS}
                     for h in hyps)

    for h in hyps[:10]:
        print(f" {h['rank']:2d} dD={h['delta_D'] if h['delta_D'] is not None else 'nan':>7} "
              f"x_ratio={h['x_ratio'] or 0:8.2e} k={h['k']:.1e} [{h['k_source']}] "
              f"k_needed={h['k_needed'] or '-'} ({h['k_status'] or '-'}, feasible={h['capture_feasible']}) "
              f"worse={h['n_worse']} {h['reaction']}  {'via ' + h['via'] if h['via'] else ''} "
              f"{h['unpaired_flag'] or ''} {h['ion_flag'] or ''} {h['ie_flag'] or ''}")
    for e in existing:
        print(f" {e['role']:11s} {100 * e['frac']:5.1f}%  F={e['F']} g={e['g']}  factor={e['factor_needed']} "
              f"({e['factor_status']}, within F: {e['within_uncertainty']})  dD={e['delta_D']} "
              f"dD_gratier={e['delta_D_gratier']} n3x={e['n_gratier_3x']}  {e['reaction']}")
    if bn:
        print(f" bottleneck ({bn['reaction']} x{bn['factor']:g}): CH2CCH x{bn['X_target_scaled'] / bn['X_target_base']:.2f}, "
              + ", ".join(f"{s} x{v['ratio']:.2g}" for s, v in bn["species"].items())
              + f"  supply_limited={bn['supply_limited']}")
    print(f" {n_sims} simulations, total {total:.0f} s -> {relative(path_json)}, {relative(path_csv)}")
