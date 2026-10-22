"""Tests for the T4 fields written by scripts/robustness.py in results/discovery/CH2CCH.json.
Usage: pytest  (or, without pytest: python -m tests.test_robustness)"""
import json, os
from tests._compat import pytest, main
from tapir.paths import RESULTS

PATH = os.path.join(RESULTS, "discovery", "CH2CCH.json")
HYP_FIELDS = ("X_cand", "x_ratio", "delta_D", "k", "k_max", "X_at_k_max", "x_ratio_at_k_max", "reachable", "error")


def doc():
    return json.load(open(PATH))


def test_config2_block():
    c = doc()["config2"]
    assert (c["network"], c["C_O"], c["S"], c["t_ref"]) == ("kida.uva.2024", 1.2, 1.5e-5, 4.5e5)
    for k in ("X_base", "x_ratio_base", "D_base", "main_formation", "n_robust", "n_reachable"):
        assert k in c, k
    assert c["x_ratio_base"] > 0 and c["main_formation"]


def test_bottleneck_config2_block():
    d = doc()
    b = d["bottleneck_config2"]
    assert b["reaction"] == d["bottleneck"]["reaction"] and b["factor"] == d["bottleneck"]["factor"]
    assert {"X_target_base", "X_target_scaled", "species", "supply_limited"} <= set(b)
    assert set(b["species"]) == set(d["bottleneck"]["species"])


def test_every_hypothesis_has_config2_and_robust():
    d = doc()
    for h in d["hypotheses"]:
        assert set(HYP_FIELDS) <= set(h["config2"]), h["reaction"]
        assert isinstance(h["robust"], bool)
        assert h["config2"]["error"] or h["config2"]["x_ratio"] is not None
    assert d["config2"]["n_robust"] == sum(h["robust"] for h in d["hypotheses"])


if __name__ == "__main__":
    main(__file__, globals())
