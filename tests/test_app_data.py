"""Tests for the Marimo app data (app/data/, written by scripts/export_app_data.py) and for the app's
independence from tapir. They only read files: no simulation, no networks, no LiteLLM.
Usage: pytest  (or, without pytest: python -m tests.test_app_data)"""
import ast, json, math, os
from tests._compat import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app", "tapir_app.py")
DATA = os.path.join(ROOT, "app", "data")
VALIDATION = os.path.join(DATA, "validation_curves.json")
HINT = "run python scripts/export_app_data.py"
CURVES = ("base", "knockout", "recovered")


def json_files():
    return [os.path.join(d, f) for d, _, fs in os.walk(DATA) for f in fs if f.endswith(".json")]


def validation():
    return json.load(open(VALIDATION))


def test_data_dir_exists():
    assert os.path.isdir(DATA), f"app/data/ missing: {HINT}"
    assert os.path.isfile(VALIDATION), f"app/data/validation_curves.json missing: {HINT}"


def test_all_json_load():
    files = json_files()
    assert files
    for p in files:
        json.load(open(p))


def test_size_below_10mb():
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DATA) for f in fs)
    assert total < 10 * 1024 * 1024, f"app/data/ is {total / 2**20:.1f} MB"


def test_validation_structure():
    v = validation()
    for key in ("configuration", "units", "metrics", "cases"):
        assert key in v, key
    assert v["configuration"]["t_ref_yr"] > 0
    m = v["metrics"]
    assert 0 < m["mrr"] <= 1 and 0 < m["random_mrr"] <= 1
    assert m["n_cases"] == len(v["cases"]) > 0
    assert all(c["ok"] for c in v["checks"]), "export checks against benchmark_b.csv failed"


def test_cases_metrics():
    v = validation()
    for c in v["cases"]:
        for key in ("case", "target", "hidden_reaction", "rank", "n", "rank_over_n", "random_reciprocal_rank"):
            assert key in c, (c.get("case"), key)
        n = c["n"]
        assert n >= 1
        assert math.isclose(c["random_reciprocal_rank"], sum(1 / i for i in range(1, n + 1)) / n)
        if c["rank"] is not None:
            assert 1 <= c["rank"] <= n and math.isclose(c["rank_over_n"], c["rank"] / n)
    rr = [0.0 if c["rank"] is None else 1 / c["rank"] for c in v["cases"]]
    assert math.isclose(v["metrics"]["mrr"], sum(rr) / len(rr))


def test_cases_curves():
    for c in validation()["cases"]:
        t = c["time"]
        assert len(t) > 1 and all(a < b for a, b in zip(t, t[1:])), c["case"]
        for k in CURVES:
            x = c["abundance"][k]
            assert len(x) == len(t), (c["case"], k)
            assert all(isinstance(v, (int, float)) and v >= 0 for v in x), (c["case"], k)


MANIFEST = os.path.join(DATA, "manifest.json")
DISCOVERY = "discovery/CH2CCH.json"
UPSTREAM = "discovery/CH2CCH_upstream.json"
ATLAS = "discovery/atlas.json"
CHEM_REVIEW = "discovery/chem_review.json"
K_STATUS = {"ok", "not_reachable", "already_above", "error", None}
LIT_STATUS = {"measured", "proposed", "computed", "no_evidence_found", "unchecked"}


def exported(name):
    """The Discovery tab's JSON if the manifest lists it as exported, else None (absence is valid)."""
    if name not in json.load(open(MANIFEST))["exported"]:
        return None
    return json.load(open(os.path.join(DATA, name)))


def check_hypotheses(d):
    hyps = d["hypotheses"]
    assert hyps and [h["rank"] for h in hyps] == list(range(1, len(hyps) + 1))
    for h in hyps:
        assert " -> " in h["reaction"], h["reaction"]
        assert set(h["source"]) <= {"analogy", "llm"} and h["source"], h["source"]
        assert h["k_status"] in K_STATUS, h["k_status"]
        assert h["capture_feasible"] in (True, False, None)
        assert h["k_status"] != "ok" or h["k_needed"] > 0
        assert h.get("literature") is None or h["literature"]["status"] in LIT_STATUS
        assert h.get("curve") is None or len(h["curve"]) == len(d["t"])
    assert len(d["curve_base"]) == len(d["t"])


def test_discovery_contract():
    d = exported(DISCOVERY)
    if d is None:
        return
    assert d["search"] == "direct" and d["species"] == "CH2CCH"
    assert d["X_obs"] > d["X_base"] > 0 and d["x_ratio_base"] > 0
    assert d["disclaimer"]
    check_hypotheses(d)
    for e in d["existing"]:
        assert {"reaction", "role", "frac", "factor_status"} <= set(e)
    assert d["config2"]["n_reachable"] >= 0 and d["bottleneck_config2"]["supply_limited"] in (True, False)
    assert all(isinstance(h["robust"], bool) and h["config2"] for h in d["hypotheses"])


def test_upstream_contract():
    d = exported(UPSTREAM)
    if d is None:
        return
    assert d["search"] == "upstream" and d["upstream"]
    check_hypotheses(d)
    assert all(h["via"] in d["upstream"] for h in d["hypotheses"])
    bn = d["bottleneck"]
    reac = bn["reaction"].split(" -> ")[0].split(" + ")
    assert any(s in bn["species"] for s in reac)
    assert all(v["ratio"] >= 0 for v in bn["species"].values())
    assert isinstance(bn["supply_limited"], bool) and bn["factor"] > 0


def test_atlas_contract():
    d = exported(ATLAS)
    if d is None:
        return
    assert d["species"] and d["settings"]["under"] > 0
    for s in d["species"]:
        assert s["x_ratio_base"] < d["settings"]["under"]
        assert 0 <= s["n_reachable_clean"] <= s["n_reachable"] <= s["n_simulated"]
        supply = s["supply"]
        assert supply["diagnosis"] in {"supply_limited", "rate_limited", "other"}
        assert supply["reaction"] and supply["factor"] > 0


def test_chemical_review_contract():
    d = exported(CHEM_REVIEW)
    if d is None:
        return
    assert d["reviewer"] and d["summary"] and d["reactions"]
    discovery_reactions = {
        name.removeprefix("discovery/").removesuffix(".json"): {h["reaction"] for h in exported(name)["hypotheses"]}
        for name in (DISCOVERY, UPSTREAM)
    }
    for r in d["reactions"]:
        assert r["verdict"] in {"implausible", "plausible", "uncertain"}
        assert r["reaction"] in discovery_reactions[r["file"]]


def test_bench_llm_contract():
    """Fields of bench_llm_summary.json read by the Generator tab; baselines recomputed from ranks and n."""
    d = exported("bench_llm_summary.json")
    if d is None:
        return
    b = d["benchmark_b"]
    for s, m in b["metrics"].items():
        n = m["n_cases"]
        assert s in ("analogy", "llm", "union") and n > 0
        assert 0 <= m["hits@1"] <= m["hits@5"] <= n and 0 <= m["n_recovered"] <= n
        assert set(m["ranks"]) == set(m["n"]) and len(m["ranks"]) == n
        rec = [c for c, r in m["ranks"].items() if r is not None]
        assert m["n_recovered"] == len(rec) and math.isclose(m["recall"], len(rec) / n)
        h_n = {c: sum(1 / i for i in range(1, m["n"][c] + 1)) / m["n"][c] for c in rec}
        assert math.isclose(m["random_mrr_end_to_end"], sum(h_n.values()) / n)
        assert math.isclose(m["MRR"], sum(1 / m["ranks"][c] for c in rec) / n)
        if rec:
            assert math.isclose(m["random_mrr_conditional"], sum(h_n.values()) / len(rec))
            assert math.isclose(m["conditional_mrr"], sum(1 / m["ranks"][c] for c in rec) / len(rec))
        else:
            assert m["random_mrr_conditional"] is None and m["conditional_mrr"] is None
    for c in b["cases"]:
        assert {"case", "strategy", "hidden_reaction", "hidden_generated"} <= set(c)


def test_app_reads_only_app_data():
    src = open(APP).read()
    assert "results/" not in src and "networks/" not in src
    tree = ast.parse(src)
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not mods & {"urllib", "requests", "httpx", "socket", "http", "openai"}, mods


def test_app_does_not_import_tapir():
    tree = ast.parse(open(APP).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(n == "tapir" or n.startswith("tapir.") for n in names), names
    assert "import tapir" not in open(APP).read() and "from tapir" not in open(APP).read()


if __name__ == "__main__":
    main(__file__, globals())
