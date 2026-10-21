"""
bench_llm.py - T5: the LLM generator (tapir.generator_llm) against the analogy generator and their union.

Strategies: analogy (tapir.generator), llm (tapir.generator_llm) and union (both merged by canonical key,
source "analogy", "llm" or "analogy+llm"). The three go through the same pipeline; the generator's own
score (the LLM order for llm) never ranks anything.

Benchmark B (benchmark/cases.json, the 6 knock-out cases of scripts.benchmark_b), for each case:
  1. knock-out run (every strict equivalent of the hidden reaction removed, as in benchmark_b);
  2. context for the LLM: the N_CONTEXT main formation and destruction reactions of the target in the
     knock-out run at t_ref (knockout_demo.budget); the prompt is checked not to contain the hidden
     reaction (any order of reactants and products);
  3. needed_effect from the knock-out error: "increase" if X_ko < X_base of the target, "decrease" if above;
  4. candidates; both generators get the hidden set (leak-proofing), and the LLM runs with
     drop_hidden=False so that the hidden reaction, if proposed, survives and can be ranked;
  5. as benchmark_b: flux pre-ranking with RateEstimator(hidden=[h]), top K by flux, each simulated alone
     on the knock-out network (one simulation per distinct candidate of the case), objective, ranking;
  6. metrics per strategy: Hits@1/5/10, MRR (strict and channel), rank, n, rank/n, recall of the hidden
     reaction in the final ranking and MRR conditional on it, and two random baselines:
       random_mrr_end_to_end   mean over ALL cases of H_n / n if the hidden reaction is in the final
                               ranking, else 0 (shuffling cannot recover a reaction that is not there);
                               the baseline for MRR (generator + ranking)
       random_mrr_conditional  mean of H_n / n over the recovered cases only; the baseline for
                               conditional_mrr (ranking alone)
     random_MRR (H_n / n over all cases, present or not) is kept for compatibility only.
  The analogy strategy reproduces benchmark_b; its positions are compared with results/benchmark_b.csv.
Benchmark A (recall, no simulation): the first --benchmark-a-cases of 50 reactions drawn with seed 0 by
  scripts.benchmark_a.sample, target from benchmark_a.pick_target; one knock-out run per reaction gives
  the LLM context and needed_effect. Recall (strict / channel) per strategy, split by whether the
  reaction is already in KIDA 2014. Runs only when --benchmark-a-cases is given.
Canaries: benchmark/canaries.json ({"canaries": [{"reac", "prod", "target", "rationale", "reviewed"}]}),
  invented reactions outside KIDA; run on the base network, only if reviewed and valid. They can show
  hallucination or generalisation, never the absence of memorisation (see NOTES).
LLM failures: FormatError (after the client's own retry) or BudgetExceeded leave the LLM without
  candidates for that case (union = analogy), are recorded and the benchmark continues. The reasoning
  effort is the configured one (TAPIR_LLM_REASONING_EFFORT); answers cached in results/llm_cache.
--dry-run: knock-out runs and prompts only; LLM uses, cached / new, estimated tokens; no LLM call, nothing
  written. Without --benchmark-a-cases it estimates the full 50 reactions of benchmark A.
usage: python -m scripts.bench_llm [--dry-run] [--cases 6] [--benchmark-b-only] [--benchmark-a-cases N]
                                   [--strategies analogy,llm,union] [--k 30] [--procs 4] [--max-llm-calls N]
Output: results/bench_llm.csv (one row per case and strategy) and results/bench_llm_summary.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, datetime, glob, hashlib, itertools, json, multiprocessing as mp, re, statistics, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, EL, T_REF, run, X, label, reaction_class
from tapir.observations import observed
from tapir.generator import Generator, load_cases, rank
from tapir.generator_llm import LLMGenerator, SCHEMA, CACHE_DIR, N_CONTEXT
from tapir.llm_client import FormatError, BudgetExceeded
from tapir.flux import FluxScorer, knockout_exclude
from tapir.rates import RateEstimator
from tapir.paths import RESULTS, BENCHMARK, relative
from scripts.benchmark_a import sample, pick_target
from scripts.benchmark_b import _simulate, objective, KS
from scripts.discover import llm_record, TOKENS_PER_CHAR, COMPLETION_TOKENS
from scripts.knockout_demo import budget

STRATEGIES = ("analogy", "llm", "union")
A_N, A_SEED = 50, 0
CANARIES = os.path.join(BENCHMARK, "canaries.json")
FIELDS = ("benchmark", "case", "target", "hidden_reaction", "strategy", "status", "needed_effect", "cached",
          "proposed", "valid", "n_no_rate", "simulated", "n", "rank_strict", "rank_channel", "rr_strict",
          "rr_channel", "rank_over_n", "random_rr", "random_rr_end_to_end", "hidden_generated", "hidden_prior_rank", "hidden_flux_rank",
          "hidden_source", "top1", "top1_source", "n_failed_sims", "reaction_class", "in_kida2014",
          "recall_strict", "recall_channel", "prompt_tokens", "completion_tokens", "total_tokens", "error")
NOTES = dict(
    ranking="The generator score (the LLM order for llm) is never used to rank: benchmark B ranks by flux "
            "pre-ranking and the simulated objective of scripts.benchmark_b, identical for the three strategies.",
    contamination="The hidden reactions of benchmarks A and B are KIDA reactions, public for years; the LLM may "
                  "have seen them (or the papers behind them) in training, so its recall on them is an upper "
                  "bound of what it would reach on chemistry it has not seen.",
    canaries="Canaries are invented reactions outside KIDA. An LLM proposing one may indicate hallucination or "
             "generalisation (it cannot have memorised it). Not proposing one does NOT demonstrate the absence of "
             "memorisation, and canaries cannot show that the model did not see KIDA during training.",
    kida2014="Benchmark A splits recall by whether the hidden reaction is already in KIDA 2014; a gap between the "
             "two is consistent with, but does not prove, memorisation.")


def _knockout_context(task):
    """(hidden key or None, target, species) -> (FluxScorer, {species: X at t_ref}, (formation, destruction)
    labels of the target at t_ref from budget(), error)."""
    key, target, species = task
    try:
        net, t, y = run(**({} if key is None else {"exclude": knockout_exclude(key)}))
        form, dest = budget(net, t, y, target, T_REF, N_CONTEXT)
    except (RuntimeError, KeyError) as e:
        return None, None, None, f"{type(e).__name__}: {e}"
    return (FluxScorer(net, t, y), {s: X(net, t, y, s, T_REF) for s in species if s in net.idx},
            ([lab for lab, _ in form], [lab for lab, _ in dest]), "")


def needed_effect(x_ko, x_base):
    """Direction that undoes the knock-out error on the target (None if there is no error)."""
    if x_ko < x_base * (1 - 1e-6):
        return "increase"
    if x_ko > x_base * (1 + 1e-6):
        return "decrease"
    return None


def assert_no_leak(prompt, h):
    """AssertionError if the prompt contains the reaction h as a reaction line, in any order."""
    for r, p in itertools.product(itertools.permutations(h["reac"]), itertools.permutations(h["prod"])):
        text = " + ".join(r) + " -> " + " + ".join(p)
        if re.search(rf"(^|\s){re.escape(text)}\s*$", prompt, re.M):
            raise AssertionError(f"hidden reaction in the prompt: {text}")


def union(analogy, llm):
    """Candidates of both strategies merged by canonical key, source 'analogy', 'llm' or 'analogy+llm'."""
    out = {}
    for src, cands in (("analogy", analogy), ("llm", llm)):
        for c in cands:
            k = chem.canonical_key(c)
            if k in out:
                out[k]["source"] = "analogy+llm"
            else:
                out[k] = dict(c, source=src)
    return list(out.values())


def make_llm(rows24, raw14, hidden, client, cache_dir):
    return LLMGenerator(rows24, raw14, hidden=hidden, client=client, cache_dir=cache_dir, drop_hidden=False,
                        allow_fallback_context=False)


def llm_candidates(llm, target, h, context, effect):
    """(candidates, record) of one use of the LLM generator; the prompt is checked against h first.
    No context, FormatError or BudgetExceeded -> no candidates, recorded with its status."""
    before = dict(getattr(llm.client, "stats", {}))
    if context is None:
        return [], dict(llm_record(llm, target, before, error="no context (knock-out run failed)"),
                        status="no_context")
    assert_no_leak(llm.prompt(target, context, effect), h)
    try:
        cands = llm.candidates(target, context, effect)
    except (FormatError, BudgetExceeded) as e:
        rec = llm_record(llm, target, before, error=f"{type(e).__name__}: {e}")
        if isinstance(e, BudgetExceeded):
            rec["status"] = "budget_exceeded"
        return [], rec
    return cands, llm_record(llm, target, before)


def generate(h, target, context, effect, rows24, raw14, use_llm, client, cache_dir, hide=True):
    """({strategy: candidates}, LLM record or None) for one case; hide=False for canaries (not in the network)."""
    hidden = [h] if hide else []
    ana = [dict(c, source="analogy") for c in Generator(rows24, raw14, hidden=hidden).candidates(target)]
    llm_c, rec = [], None
    if use_llm:
        llm_c, rec = llm_candidates(make_llm(rows24, raw14, hidden, client, cache_dir), target, h, context, effect)
    return dict(analogy=ana, llm=llm_c, union=union(ana, llm_c)), rec


def harmonic(n):
    return sum(1 / i for i in range(1, n + 1))


def b_metrics(recs):
    """Benchmark B metrics of one strategy from its per-case records. Recovered: the hidden reaction is in the
    final ranking (rank_strict not None)."""
    m = len(recs)
    rec = [r for r in recs if r["rank_strict"] is not None]
    h_n = lambda r: harmonic(r["n"]) / r["n"]
    out = {f"hits@{k}": sum(r["rank_strict"] is not None and r["rank_strict"] <= k for r in recs) for k in KS}
    out.update({f"hits@{k}_channel": sum(r["rank_channel"] is not None and r["rank_channel"] <= k for r in recs)
                for k in KS})
    out.update(n_cases=m, MRR=sum(r["rr_strict"] for r in recs) / m, MRR_channel=sum(r["rr_channel"] for r in recs) / m,
               recovered={r["case"]: r["rank_strict"] is not None for r in recs}, n_recovered=len(rec),
               recall=len(rec) / m,
               conditional_mrr=sum(r["rr_strict"] for r in rec) / len(rec) if rec else None,
               random_mrr_end_to_end=sum(h_n(r) for r in rec) / m,
               random_mrr_conditional=sum(h_n(r) for r in rec) / len(rec) if rec else None,
               random_baselines_definition="H_n / n is the expected 1/rank of the hidden reaction among n shuffled "
                                           "candidates, n = final ranking. random_mrr_end_to_end: mean over all "
                                           "cases, 0 when the hidden reaction is not in the ranking (compare with "
                                           "MRR). random_mrr_conditional: mean over the recovered cases only "
                                           "(compare with conditional_mrr); null when none is recovered",
               random_MRR=sum(r["random_rr"] for r in recs) / m,
               random_MRR_definition="kept for compatibility, use random_mrr_end_to_end or random_mrr_conditional: "
                                     "mean over all cases of H_n / n even when the hidden reaction is not in the "
                                     "ranking (n = final ranking; 0 when n = 0), so neither an end-to-end nor a "
                                     "conditional baseline",
               ranks={r["case"]: r["rank_strict"] for r in recs}, n={r["case"]: r["n"] for r in recs},
               rank_over_n={r["case"]: r["rank_over_n"] for r in recs},
               llm_failures=sum(r["status"] not in ("ok",) for r in recs))
    return out


def a_metrics(recs):
    """Benchmark A recall of one strategy, overall and split by in_kida2014."""
    def rec(sel):
        return None if not sel else dict(n=len(sel), recall_strict=sum(r["recall_strict"] for r in sel) / len(sel),
                                          recall_channel=sum(r["recall_channel"] for r in sel) / len(sel))
    return dict(all=rec(recs), in_kida2014=rec([r for r in recs if r["in_kida2014"]]),
                added_after_2014=rec([r for r in recs if not r["in_kida2014"]]),
                N_median=statistics.median([r["valid"] for r in recs]) if recs else None,
                llm_failures=sum(r["status"] not in ("ok",) for r in recs))


def tokens(rec):
    u = (rec or {}).get("usage") or {}
    return {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def llm_fields(s, rec):
    """status / cached / proposed / valid / tokens of a strategy row from the LLM record."""
    if s == "analogy" or rec is None:
        return dict(status="ok", cached=None)
    return dict(status=rec["status"], cached=rec["cached"], error=rec["error"], **tokens(rec),
                **({"proposed": rec["proposed"]} if s == "llm" else {}))


# --- canaries -------------------------------------------------------------------------------------

def load_canaries(path=CANARIES):
    """List of canaries, or None if the file does not exist."""
    return json.load(open(path))["canaries"] if os.path.exists(path) else None


def canary_problems(c, rows24, raw14):
    """Reasons that keep a canary out of the run (empty: usable)."""
    row = dict(reac=list(c["reac"]), prod=list(c["prod"]))
    species = {s for r in rows24 for s in r["reac"] + r["prod"]}
    out = [] if c.get("reviewed") else ["not reviewed by the chemist"]
    if not chem.conserves(row):
        out.append("not conserved")
    unknown = sorted(set(row["reac"] + row["prod"]) - species)
    if unknown:
        out.append(f"species not in the network: {unknown}")
    if any(chem.equivalent(r, row) for r in rows24 + raw14):
        out.append("already in KIDA 2014 or 2024")
    if c.get("target") not in row["prod"]:
        out.append("target not among the products")
    return out


# --- dry run --------------------------------------------------------------------------------------

def cache_path(llm, prompt):
    """Same key as LLMGenerator._ask."""
    c = llm.client
    h = hashlib.sha256(json.dumps([c.model, getattr(c, "reasoning_effort", None), SCHEMA, prompt]).encode()).hexdigest()
    return os.path.join(llm.cache_dir, h + ".json")


def cached_usage(cache_dir, model, effort):
    """(prompt tokens per character over all cached single-call answers, completion tokens of those at
    this model and effort)."""
    ds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(cache_dir, "*.json")))]
    one = [d for d in ds if (d.get("usage") or {}).get("calls") == 1]
    ratios = [d["usage"]["prompt_tokens"] / len(d["prompt"]) for d in one]
    comp = [d["usage"]["completion_tokens"] for d in one
            if d.get("model") == model and d.get("reasoning_effort") == effort]
    return (sum(ratios) / len(ratios) if ratios else TOKENS_PER_CHAR), comp


def estimate(uses, ratio, comp):
    """uses: [(name, prompt, cached)] -> dict with calls, cached, new and token range of the new ones."""
    new = [p for _, p, cached in uses if not cached]
    pt = sum(ratio * len(p) for p in new)
    lo, hi = (min(comp), max(comp)) if comp else COMPLETION_TOKENS
    mid = statistics.mean(comp) if comp else sum(COMPLETION_TOKENS) / 2
    return dict(calls=len(uses), cached=len(uses) - len(new), new=len(new), prompt_tokens=round(pt),
                tokens_expected=round(pt + len(new) * mid), tokens_range=[round(pt + len(new) * lo), round(pt + len(new) * hi)],
                tokens_worst_with_retries=round(2 * (pt + len(new) * hi)))


# --- main -----------------------------------------------------------------------------------------

def parse(argv):
    ap = argparse.ArgumentParser(description="T5: analogy vs LLM vs union on benchmarks B and A, and canaries")
    ap.add_argument("--dry-run", action="store_true", help="estimate LLM calls and tokens, no call, nothing written")
    ap.add_argument("--cases", type=int, default=None, help="first N cases of benchmark B (default: all 6)")
    ap.add_argument("--benchmark-b-only", action="store_true", help="only benchmark B (no A, no canaries)")
    ap.add_argument("--benchmark-a-cases", type=int, default=None,
                    help=f"first N of the {A_N} benchmark A reactions (seed {A_SEED}); A runs only when given "
                         f"(the dry run estimates all {A_N} by default)")
    ap.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated subset of analogy,llm,union")
    ap.add_argument("--k", type=int, default=30, help="candidates simulated per strategy and case (top K by flux)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-llm-calls", type=int, default=None,
                    help="cap on LLM requests (default: 2 per planned use, i.e. one retry each)")
    a = ap.parse_args(argv)
    a.strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    if not a.strategies or set(a.strategies) - set(STRATEGIES):
        ap.error(f"--strategies: subset of {','.join(STRATEGIES)}")
    return a


def main(argv=None, client=None, out_dir=RESULTS, cache_dir=CACHE_DIR):
    """client: LLMClient-like (tests); by default llm_client.LLMClient(max_calls=--max-llm-calls)."""
    a = parse(argv)
    t0 = time.time()
    use_llm = bool({"llm", "union"} & set(a.strategies))
    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    keys14 = {chem.canonical_key(r) for r in raw14}
    cases = load_cases()[:a.cases]
    n_a = 0 if a.benchmark_b_only else (a.benchmark_a_cases if a.benchmark_a_cases is not None
                                         else (A_N if a.dry_run else 0))
    a_hidden = sample(rows24, A_N, A_SEED)[:n_a]
    canaries = None if a.benchmark_b_only else load_canaries()
    canaries_ok = [c for c in canaries or [] if not canary_problems(c, rows24, raw14)]
    obs = sorted(observed())
    species = sorted(set(obs) | {n for n, _ in cases} | {pick_target(h) for h in a_hidden}
                     | {c["target"] for c in canaries_ok})
    key = chem.canonical_key

    with mp.Pool(a.procs) as pool:
        base = pool.apply_async(_simulate, ((None, None, species),))
        ko_b = pool.map_async(_knockout_context, [(key(h), n, species) for n, h in cases])
        ko_a = pool.map_async(_knockout_context, [(key(h), pick_target(h), [pick_target(h)]) for h in a_hidden])
        ko_c = pool.map_async(_knockout_context, [(None, c["target"], [c["target"]]) for c in canaries_ok])
        xbase, _, err = base.get()
        if xbase is None:
            raise RuntimeError(f"base run failed: {err}")
        ko_b, ko_a, ko_c = ko_b.get(), ko_a.get(), ko_c.get()
        obs = [o for o in obs if o in xbase]
        effect_of = lambda ko, tgt: (needed_effect(ko[1][tgt], xbase[tgt]) if not ko[3] and tgt in ko[1] else None)

        if a.dry_run:
            return dry_run(a, cases, ko_b, a_hidden, ko_a, canaries, canaries_ok, ko_c, effect_of, rows24, raw14,
                           use_llm, cache_dir)
        n_uses = use_llm * (len(cases) + len(a_hidden) + len(canaries_ok))
        if use_llm and client is None:
            from tapir.llm_client import LLMClient
            client = LLMClient(max_calls=a.max_llm_calls or max(1, 2 * n_uses))

        # benchmark B
        plan, tasks = [], {}
        for (name, h), ko in zip(cases, ko_b):
            scorer, xko, ctx, ko_err = ko
            if ko_err:
                raise RuntimeError(f"knock-out run of {name} failed: {ko_err}")
            effect = effect_of(ko, name)
            gens, rec = generate(h, name, ctx, effect, rows24, raw14, use_llm, client, cache_dir)
            est = RateEstimator(rows24, hidden=[h])
            for s in a.strategies:
                cands = [dict(c) for c in gens[s]]
                ranked, no_rate = scorer.score(cands, name, estimator=est)
                top = ranked[:a.k]
                for i, c in enumerate(top):
                    c["flux_rank"], c["prior"] = i + 1, c.get("score")
                    tasks.setdefault((name, key(c)), (key(h), c["kinetic"], species))
                plan.append(dict(name=name, h=h, s=s, effect=effect, rec=rec, gen=gens[s], ranked=ranked,
                                 top=top, no_rate=no_rate))
        t_sim = time.time()
        order = list(tasks)
        sims = dict(zip(order, pool.map(_simulate, [tasks[t] for t in order])))
        t_sim = time.time() - t_sim

    b_rows = []
    for p in plan:
        name, h, top = p["name"], p["h"], p["top"]
        n_failed = 0
        for c in top:
            xs, _, err = sims[(name, key(c))]
            c["error"] = err
            if xs is None:
                c["score"], n_failed = float("inf"), n_failed + 1
            else:
                c["score"] = objective(xs, xbase, name, obs)[0]
            c["neg_score"] = -c["score"]
        top.sort(key=lambda c: (c["score"], key(c)))
        n = len(top)
        rs, rc = rank(top, h, key="neg_score"), rank(top, h, "channel", name, key="neg_score")
        hid = next((c for c in p["gen"] if chem.equivalent(c, h)), None)
        b_rows.append(dict(
            benchmark="B", case=name, target=name, hidden_reaction=label(h), strategy=p["s"], needed_effect=p["effect"],
            proposed=len(p["gen"]), valid=len(p["gen"]), n_no_rate=p["no_rate"], simulated=n, n=n,
            rank_strict=rs, rank_channel=rc, rr_strict=1 / rs if rs else 0.0, rr_channel=1 / rc if rc else 0.0,
            rank_over_n=rs / n if rs else None, random_rr=harmonic(n) / n if n else 0.0,
            random_rr_end_to_end=harmonic(n) / n if rs else 0.0,
            hidden_generated=hid is not None,
            hidden_prior_rank=rank(p["gen"], h) if p["s"] != "union" else None,
            hidden_flux_rank=rank(p["ranked"], h, key="dX"), hidden_source=hid["source"] if hid else None,
            top1=label(top[0]) if top else None, top1_source=top[0]["source"] if top else None,
            n_failed_sims=n_failed,
            position_strict=next((i + 1 for i, c in enumerate(top) if chem.equivalent(c, h)), None),
            llm=p["rec"] if p["s"] != "analogy" else None) | llm_fields(p["s"], p["rec"]))

    # benchmark A (recall, no simulation)
    a_rows = []
    for h, ko in zip(a_hidden, ko_a):
        tgt = pick_target(h)
        effect = effect_of(ko, tgt)
        gens, rec = generate(h, tgt, ko[2], effect, rows24, raw14, use_llm, client, cache_dir)
        for s in a.strategies:
            cs = gens[s]
            a_rows.append(dict(
                benchmark="A", case=h["id"], target=tgt, hidden_reaction=label(h), strategy=s, needed_effect=effect,
                reaction_class=reaction_class(h), in_kida2014=key(h) in keys14, proposed=len(cs), valid=len(cs),
                recall_strict=rank(cs, h) is not None, recall_channel=rank(cs, h, "channel", tgt) is not None,
                hidden_prior_rank=rank(cs, h) if s != "union" else None,
                hidden_source=next((c["source"] for c in cs if chem.equivalent(c, h)), None),
                llm=rec if s != "analogy" else None) | llm_fields(s, rec))

    # canaries (base network, nothing hidden)
    c_rows = []
    for c, ko in zip(canaries_ok, ko_c):
        h, tgt = dict(reac=list(c["reac"]), prod=list(c["prod"])), c["target"]
        gens, rec = generate(h, tgt, ko[2], "increase", rows24, raw14, use_llm, client, cache_dir, hide=False)
        for s in a.strategies:
            cs = gens[s]
            c_rows.append(dict(
                benchmark="canary", case=label(h), target=tgt, hidden_reaction=label(h), strategy=s,
                needed_effect="increase", proposed=len(cs), valid=len(cs),
                recall_strict=any(chem.equivalent(x, h) for x in cs),
                recall_channel=any(chem.equivalent(x, h, "channel", tgt) for x in cs),
                llm=rec if s != "analogy" else None) | llm_fields(s, rec))

    doc = summary(a, cases, b_rows, a_rows, c_rows, canaries, canaries_ok, rows24, raw14, client, use_llm,
                  out_dir, time.time() - t0, t_sim, len(tasks))
    write(out_dir, b_rows + a_rows + c_rows, doc)
    report(doc, b_rows)
    return doc


def benchmark_b_positions(path):
    """{case: final position of the hidden reaction (strict)} from results/benchmark_b.csv, or None."""
    if not os.path.exists(path):
        return None
    return {r["case"]: int(r["final_rank"]) for r in csv.DictReader(open(path)) if r["hidden_strict"] == "True"}


def summary(a, cases, b_rows, a_rows, c_rows, canaries, canaries_ok, rows24, raw14, client, use_llm, out_dir,
            runtime, t_sim, n_sims):
    by = lambda rows, s: [r for r in rows if r["strategy"] == s]
    ref = benchmark_b_positions(os.path.join(out_dir, "benchmark_b.csv"))
    ana = {r["case"]: r["position_strict"] for r in by(b_rows, "analogy")}
    llm_recs = [r["llm"] for r in by(b_rows + a_rows + c_rows, "llm" if "llm" in a.strategies else "union")]
    if a.benchmark_b_only:
        can = dict(status="skipped (--benchmark-b-only)")
    elif canaries is None:
        can = dict(status="not_defined", path=relative(CANARIES),
                   todo="the chemist must define 5 plausible invented reactions (reac, prod, target, rationale, "
                        "reviewed: true); none are invented by this script")
    else:
        can = dict(status="run" if canaries_ok else "pending_review", path=relative(CANARIES),
                   items=[dict(reaction=label(c), target=c.get("target"), problems=canary_problems(c, rows24, raw14))
                          for c in canaries],
                   results={s: [dict(reaction=r["case"], recall_strict=r["recall_strict"], recall_channel=r["recall_channel"],
                                     status=r["status"]) for r in by(c_rows, s)] for s in a.strategies})
    stats = dict(getattr(client, "stats", {})) if use_llm and client is not None else {}
    return dict(
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        config=dict(network="kida.uva.2024", C_O=round(EL["C+"] / EL["O"], 3), S=EL["S+"], t_ref=T_REF, K=a.k,
                    strategies=a.strategies, benchmark_b_cases=[n for n, _ in cases],
                    benchmark_a=dict(n=len({r["case"] for r in a_rows}), of=A_N, seed=A_SEED), n_context=N_CONTEXT,
                    llm_drop_hidden=False, procs=a.procs),
        model=getattr(client, "model", None) if use_llm else None,
        reasoning_effort=getattr(client, "reasoning_effort", None) if use_llm else None,
        benchmark_b=dict(metrics={s: b_metrics(by(b_rows, s)) for s in a.strategies},
                         cases=[{k: v for k, v in r.items()} for r in b_rows],
                         analogy_vs_benchmark_b_csv=None if ref is None or "analogy" not in a.strategies else
                         {c: dict(bench_llm=ana.get(c), benchmark_b=ref.get(c), same=ana.get(c) == ref.get(c)) for c in ana},
                         n_simulations=n_sims, simulation_s=round(t_sim)) if b_rows else None,
        benchmark_a=dict(metrics={s: a_metrics(by(a_rows, s)) for s in a.strategies}, cases=a_rows)
        if a_rows else dict(status="not_run"),
        canaries=can,
        llm_usage=dict(known=stats, n_uses=len(llm_recs),
                       format_errors=sum(r is not None and r["status"] == "format_error" for r in llm_recs),
                       budget_exceeded=sum(r is not None and r["status"] == "budget_exceeded" for r in llm_recs),
                       note="tokens of requests that failed without usage are unknown (null), never estimated"),
        notes=NOTES, runtime_s=round(runtime))


def write(out_dir, rows, doc):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bench_llm.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({k: "" if r.get(k) is None else r[k] for k in FIELDS} for r in rows)
    with open(os.path.join(out_dir, "bench_llm_summary.json"), "w") as fh:
        json.dump(doc, fh, indent=1)


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def report(doc, b_rows):
    print(f"BENCH LLM  model={doc['model']} effort={doc['reasoning_effort']}  K={doc['config']['K']}")
    for r in b_rows:
        print(f" B {r['case']:8s} {r['strategy']:7s} {r['status']:15s} rank={r['rank_strict']} (channel "
              f"{r['rank_channel']}) n={r['n']} H_n/n={r['random_rr']:.3f} (end-to-end {r['random_rr_end_to_end']:.3f}) "
              f"source={r['hidden_source']}")
    if doc["benchmark_b"]:
        for s, m in doc["benchmark_b"]["metrics"].items():
            print(f" B {s:7s}: H@1={m['hits@1']}/{m['n_cases']} H@5={m['hits@5']}/{m['n_cases']} "
                  f"MRR={m['MRR']:.3f} (channel {m['MRR_channel']:.3f}) random end-to-end="
                  f"{m['random_mrr_end_to_end']:.3f}  recovered={m['n_recovered']}/{m['n_cases']} "
                  f"conditional MRR={fmt(m['conditional_mrr'])} random conditional={fmt(m['random_mrr_conditional'])} "
                  f"llm failures={m['llm_failures']}")
        chk = doc["benchmark_b"]["analogy_vs_benchmark_b_csv"]
        if chk:
            print(f" analogy = benchmark_b.csv positions: {all(v['same'] for v in chk.values())} {chk}")
    if doc["benchmark_a"].get("metrics"):
        for s, m in doc["benchmark_a"]["metrics"].items():
            print(f" A {s:7s}: {m['all']}  (in KIDA 2014: {m['in_kida2014']}; added after: {m['added_after_2014']})")
    print(f" canaries: {doc['canaries']['status']}")
    print(f" llm usage: {doc['llm_usage']}  total {doc['runtime_s']} s -> results/bench_llm.csv, "
          f"results/bench_llm_summary.json")


def dry_run(a, cases, ko_b, a_hidden, ko_a, canaries, canaries_ok, ko_c, effect_of, rows24, raw14, use_llm, cache_dir):
    from tapir import llm_client
    stub = type("DryRun", (), dict(model=llm_client.MODEL, reasoning_effort=llm_client.REASONING_EFFORT))
    ratio, comp = cached_usage(cache_dir, stub.model, stub.reasoning_effort)
    print(f"DRY RUN bench_llm  model={stub.model} reasoning_effort={stub.reasoning_effort} "
          f"strategies={','.join(a.strategies)} K={a.k}  (no LLM call, nothing written)")
    print(f" prompt tokens/char {ratio:.2f}; completion tokens from {len(comp)} cached answer(s) at this "
          f"model and effort: {comp or 'none, using ' + str(COMPLETION_TOKENS)}")
    out = {}
    groups = [("benchmark B", [(n, h, ko, True) for (n, h), ko in zip(cases, ko_b)]),
              (f"benchmark A ({len(a_hidden)} of {A_N}, seed {A_SEED})",
               [(pick_target(h), h, ko, True) for h, ko in zip(a_hidden, ko_a)]),
              ("canaries", [(c["target"], dict(reac=c["reac"], prod=c["prod"]), ko, False)
                            for c, ko in zip(canaries_ok, ko_c)])]
    for title, items in groups:
        if not items or not use_llm:
            continue
        uses = []
        for tgt, h, ko, hide in items:
            if ko[3]:
                print(f"  {tgt}: knock-out run failed ({ko[3]}): no LLM call")
                continue
            llm = make_llm(rows24, raw14, [h] if hide else [], stub, cache_dir)
            effect = effect_of(ko, tgt) if hide else "increase"
            prompt = llm.prompt(tgt, ko[2], effect)
            assert_no_leak(prompt, h)
            uses.append((tgt, prompt, os.path.exists(cache_path(llm, prompt))))
            if title == "benchmark B":
                print(f"  {tgt:8s} hidden {label(h):28s} needed_effect={effect}  "
                      f"{'cached' if uses[-1][2] else 'NEW CALL'}  prompt {len(prompt)} chars ~ {ratio * len(prompt):.0f} tokens")
        e = out[title] = estimate(uses, ratio, comp)
        print(f" {title}: {e['calls']} LLM uses, {e['cached']} cached, {e['new']} new -> ~{e['tokens_expected']} tokens "
              f"(range {e['tokens_range'][0]}-{e['tokens_range'][1]}; worst with every retry {e['tokens_worst_with_retries']})")
    if canaries is None and not a.benchmark_b_only:
        print(f" canaries: {relative(CANARIES)} not defined (needs the chemist): none run")
    elif canaries and len(canaries_ok) < len(canaries):
        print(f" canaries: {len(canaries) - len(canaries_ok)} of {len(canaries)} not usable (unreviewed or invalid)")
    if not use_llm:
        print(" no LLM strategy selected: 0 calls")
    print(f" cap on LLM requests for a real run: --max-llm-calls (default 2 per planned use)")
    return out


if __name__ == "__main__":
    main()
