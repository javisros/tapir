"""Tests for scripts.bench_llm with fake LLM clients (no call to LiteLLM).
Usage: pytest  (or, without pytest: python -m tests.test_bench_llm)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv, json, multiprocessing as mp, subprocess, sys, tempfile
from tests._compat import pytest, main
from tapir import chem
from tapir.darkcloud import load_rows, parse_kida
from tapir.generator import load_cases
from tapir.knockout import NET, label
from tapir.llm_client import FormatError, BudgetExceeded
from tapir.paths import ROOT
from scripts.bench_llm import (FIELDS, assert_no_leak, union, make_llm, llm_candidates, generate, b_metrics, harmonic,
                               _knockout_context, needed_effect, main as bench_main)

ROWS24, RAW14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
CASES = load_cases()


class FakeClient:
    """LLMClient interface: `proposals` for every target, FormatError (two requests, no usage) for those in
    `fail`, BudgetExceeded for those in `budget`."""
    model, reasoning_effort = "fake", "low"

    def __init__(self, proposals=(), fail=(), budget=()):
        self.proposals, self.fail, self.budget, self.prompts = list(proposals), set(fail), set(budget), []
        self.stats = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0, format_failures=0)

    def ask(self, prompt, schema, validate):
        self.prompts.append(prompt)
        target = prompt.split("Target species: ")[1].split("\n")[0]
        if target in self.budget:
            raise BudgetExceeded("TAPIR_LLM_MAX_CALLS=0 reached")
        if target in self.fail:
            self.stats["calls"] += 2; self.stats["format_failures"] += 2
            raise FormatError("invalid answer after one retry: json_validate_failed")
        self.stats["calls"] += 1
        for k, v in (("prompt_tokens", 60), ("completion_tokens", 40), ("total_tokens", 100)):
            self.stats[k] += v
        return validate(dict(candidates=self.proposals))


def prop(r):
    return dict(reactants=list(r["reac"]), products=list(r["prod"]), reaction_class="neutral-neutral",
                rationale="test")


# --- leak-proofing --------------------------------------------------------------------------------

def test_assert_no_leak():
    h = dict(reac=["S", "CH3"], prod=["H", "H2CS"])
    for line in ("  S + CH3 -> H + H2CS", "  CH3 + S -> H2CS + H"):
        with pytest.raises(AssertionError):
            assert_no_leak(f"Main formation reactions in the network:\n{line}\nMain destruction", h)
    assert_no_leak("  S + CH3 -> H + H2CS + X\n  O + CH3 -> H + H2CO", h)      # other reactions pass


def test_hidden_reaction_not_in_any_prompt():
    """Real prompts of the 6 cases (knock-out context, needed effect from the knock-out error)."""
    species = sorted({n for n, _ in CASES})
    with mp.Pool(2) as pool:
        kos = pool.map(_knockout_context, [(chem.canonical_key(h), n, species) for n, h in CASES])
        base = _knockout_context((None, CASES[0][0], species))
    for (name, h), (_, xko, ctx, err) in zip(CASES, kos):
        assert not err
        effect = needed_effect(xko[name], base[1][name])
        assert effect == "increase"                                   # every hidden reaction forms its target
        prompt = make_llm(ROWS24, RAW14, [h], FakeClient(), tempfile.mkdtemp()).prompt(name, ctx, effect)
        assert label(h) not in prompt
        assert_no_leak(prompt, h)
        assert all(chem.canonical_key(dict(reac=l.split(" -> ")[0].split(" + "), prod=l.split(" -> ")[1].split(" + ")))
                   != chem.canonical_key(h) for l in ctx[0] + ctx[1] if " -> " in l)


def test_needed_effect():
    assert needed_effect(1e-10, 1e-9) == "increase" and needed_effect(1e-8, 1e-9) == "decrease"
    assert needed_effect(1e-9, 1e-9) is None


# --- generators -----------------------------------------------------------------------------------

def test_drop_hidden_false_keeps_hidden():
    name, h = next(c for c in CASES if c[0] == "H2CS")
    kept = make_llm(ROWS24, RAW14, [h], FakeClient([prop(h)]), tempfile.mkdtemp()).candidates(name, ([], []), "increase")
    assert len(kept) == 1 and chem.equivalent(kept[0], h)
    from tapir.generator_llm import LLMGenerator
    g = LLMGenerator(ROWS24, RAW14, hidden=[h], client=FakeClient([prop(h)]), cache_dir=tempfile.mkdtemp())
    assert g.candidates(name, ([], []), "increase") == [] and g.last["discarded"]["hidden"] == 1


def test_union_deduplicates_with_source():
    a = [dict(reac=["A", "B"], prod=["C", "D"]), dict(reac=["E", "F"], prod=["G", "H"])]
    l = [dict(reac=["B", "A"], prod=["D", "C"]), dict(reac=["X", "Y"], prod=["Z", "W"])]
    u = union(a, l)
    assert len(u) == 3
    src = {chem.canonical_key(c): c["source"] for c in u}
    assert src[chem.canonical_key(a[0])] == "analogy+llm"
    assert src[chem.canonical_key(a[1])] == "analogy" and src[chem.canonical_key(l[1])] == "llm"


def test_llm_failure_does_not_abort():
    """FormatError on one target and BudgetExceeded on another: no LLM candidates, recorded, union = analogy;
    the next target still gets its LLM candidate."""
    (n1, h1), (n2, h2), (n3, h3) = CASES[3], CASES[4], CASES[0]          # HCCCHO, H2CS, CH2CHCN
    client = FakeClient([prop(h3)], fail=[n1], budget=[n2])
    ctx = ([], [])
    for name, h, status in ((n1, h1, "format_error"), (n2, h2, "budget_exceeded")):
        gens, rec = generate(h, name, ctx, "increase", ROWS24, RAW14, True, client, tempfile.mkdtemp())
        assert gens["llm"] == [] and rec["status"] == status and rec["kept"] == 0 and rec["error"]
        assert [chem.canonical_key(c) for c in gens["union"]] == [chem.canonical_key(c) for c in gens["analogy"]]
    assert rec["usage"]["total_tokens"] is None                       # failed without usage: not invented
    gens, rec = generate(h3, n3, ctx, "increase", ROWS24, RAW14, True, client, tempfile.mkdtemp())
    assert rec["status"] == "ok" and len(gens["llm"]) == 1 and rec["usage"]["total_tokens"] == 100
    hid = next(c for c in gens["union"] if chem.equivalent(c, h3))
    assert hid["source"] == "analogy+llm"
    _, rec = llm_candidates(make_llm(ROWS24, RAW14, [h3], client, tempfile.mkdtemp()), n3, h3, None, "increase")
    assert rec["status"] == "no_context"


# --- metrics --------------------------------------------------------------------------------------

def test_metrics():
    assert harmonic(1) == 1 and abs(harmonic(3) - 11 / 6) < 1e-12
    recs = [dict(case="a", rank_strict=1, rank_channel=1, rr_strict=1.0, rr_channel=1.0, n=1, rank_over_n=1.0,
                 random_rr=1.0, status="ok"),
            dict(case="b", rank_strict=4, rank_channel=2, rr_strict=0.25, rr_channel=0.5, n=4, rank_over_n=1.0,
                 random_rr=harmonic(4) / 4, status="ok"),
            dict(case="c", rank_strict=None, rank_channel=None, rr_strict=0.0, rr_channel=0.0, n=0,
                 rank_over_n=None, random_rr=0.0, status="format_error")]
    m = b_metrics(recs)
    assert (m["hits@1"], m["hits@5"], m["hits@10"], m["n_cases"]) == (1, 2, 2, 3)
    assert abs(m["MRR"] - 1.25 / 3) < 1e-12 and abs(m["MRR_channel"] - 0.5) < 1e-12
    assert abs(m["random_MRR"] - (1 + harmonic(4) / 4) / 3) < 1e-12 and m["llm_failures"] == 1
    assert m["recovered"] == {"a": True, "b": True, "c": False} and m["n_recovered"] == 2
    assert abs(m["recall"] - 2 / 3) < 1e-12 and abs(m["conditional_mrr"] - 1.25 / 2) < 1e-12
    assert abs(m["random_mrr_end_to_end"] - (1 + harmonic(4) / 4) / 3) < 1e-12
    assert abs(m["random_mrr_conditional"] - (1 + harmonic(4) / 4) / 2) < 1e-12


def b_rec(case, rank, n):
    """Per-case record as main() builds it, for the metric tests."""
    return dict(case=case, rank_strict=rank, rank_channel=rank, rr_strict=1 / rank if rank else 0.0,
                rr_channel=1 / rank if rank else 0.0, n=n, rank_over_n=rank / n if rank else None,
                random_rr=harmonic(n) / n if n else 0.0, status="ok")


def test_random_baseline_hidden_absent():
    """rank=None with n=1: the old H_n/n gives 1, but a shuffle can never recover an absent reaction."""
    m = b_metrics([b_rec("x", None, 1)])
    assert m["random_MRR"] == 1.0
    assert m["random_mrr_end_to_end"] == 0.0 and m["random_mrr_conditional"] is None
    assert m["n_recovered"] == 0 and m["recall"] == 0.0 and m["conditional_mrr"] is None and m["MRR"] == 0.0


def test_random_baselines_bench_b_llm():
    """The 6 benchmark B cases of the LLM strategy (ranks and n of results/bench_llm.csv, 2026-10-21)."""
    recs = [b_rec("CH2CHCN", None, 1), b_rec("CH3CN", 2, 5), b_rec("HNCCC", None, 2),
            b_rec("HCCCHO", None, 1), b_rec("H2CS", None, 5), b_rec("N2H+", 1, 4)]
    m = b_metrics(recs)
    assert (m["n_recovered"], m["n_cases"]) == (2, 6) and abs(m["recall"] - 1 / 3) < 1e-12
    assert abs(m["MRR"] - 0.25) < 1e-12 and abs(m["conditional_mrr"] - 0.75) < 1e-12
    both = harmonic(5) / 5 + harmonic(4) / 4
    assert abs(m["random_mrr_end_to_end"] - both / 6) < 1e-12 and round(m["random_mrr_end_to_end"], 3) == 0.163
    assert abs(m["random_mrr_conditional"] - both / 2) < 1e-12 and round(m["random_mrr_conditional"], 3) == 0.489
    # all recovered (analogy, union): both baselines equal the old random_MRR
    full = b_metrics([b_rec("a", 1, 2), b_rec("b", 4, 30)])
    assert full["random_mrr_end_to_end"] == full["random_mrr_conditional"] == full["random_MRR"]


# --- command line ---------------------------------------------------------------------------------

def test_dry_run_makes_no_call():
    """--dry-run --cases 1 with an unreachable proxy: one LLM use estimated, nothing written."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TAPIR_RESULTS=tmp, OMP_NUM_THREADS="1", LITELLM_BASE_URL="http://127.0.0.1:9/v1")
        out = subprocess.run([sys.executable, "-m", "scripts.bench_llm", "--dry-run", "--cases", "1",
                              "--benchmark-b-only", "--procs", "2"],
                             cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout
        assert os.listdir(tmp) == []
    assert "benchmark B: 1 LLM uses, 0 cached, 1 new" in out and "CH2CHCN" in out and "CH3CN" not in out
    assert "benchmark A" not in out


def test_cases_1_end_to_end():
    """--cases 1 with a fake LLM proposing the hidden reaction: one case x three strategies, valid CSV/JSON."""
    name, h = CASES[0]
    client = FakeClient([prop(h)])
    with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as cache:
        doc = bench_main(["--cases", "1", "--benchmark-b-only", "--k", "3", "--procs", "2"], client=client,
                         out_dir=out, cache_dir=cache)
        rows = list(csv.DictReader(open(os.path.join(out, "bench_llm.csv"))))
        assert json.load(open(os.path.join(out, "bench_llm_summary.json"))) == json.loads(json.dumps(doc))
    assert tuple(rows[0]) == FIELDS and client.stats["calls"] == 1
    assert [(r["case"], r["strategy"]) for r in rows] == [(name, s) for s in ("analogy", "llm", "union")]
    by = {r["strategy"]: r for r in rows}
    assert by["llm"]["status"] == "ok" and by["llm"]["n"] == "1" and by["llm"]["rank_strict"] == "1.0"
    assert by["llm"]["proposed"] == "1" and by["llm"]["total_tokens"] == "100"
    assert by["union"]["hidden_source"] == "analogy+llm" and by["analogy"]["hidden_source"] == "analogy"
    assert int(by["union"]["n"]) <= 3 and by["analogy"]["status"] == "ok"
    m = doc["benchmark_b"]["metrics"]
    assert set(m) == {"analogy", "llm", "union"} and m["llm"]["MRR"] == 1.0 and m["llm"]["random_MRR"] == 1.0
    assert m["llm"]["random_mrr_end_to_end"] == m["llm"]["random_mrr_conditional"] == 1.0
    assert by["llm"]["random_rr_end_to_end"] == "1.0"
    assert doc["benchmark_a"] == {"status": "not_run"} and doc["canaries"]["status"].startswith("skipped")
    assert doc["model"] == "fake" and doc["reasoning_effort"] == "low" and "canaries" in doc["notes"]


if __name__ == "__main__":
    main(__file__, globals())
