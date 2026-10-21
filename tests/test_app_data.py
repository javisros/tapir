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
