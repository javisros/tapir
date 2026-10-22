"""Tests for results/discovery/atlas.json (scripts/atlas.py) and its supply diagnostic.
Usage: pytest  (or, without pytest: python -m tests.test_atlas)"""
import json, os
from tests._compat import pytest, main
from tapir.paths import RESULTS
from scripts.atlas import supply, UNDER, FLAGS

PATH = os.path.join(RESULTS, "discovery", "atlas.json")
SPECIES_FIELDS = ("species", "X_obs", "x_ratio_base", "n_candidates", "n_simulated", "best_x_ratio", "n_reachable",
                  "n_reachable_clean", "best", "supply", "reachable_clean", "candidates")
SUPPLY_FIELDS = ("reaction", "diagnosis", "limiting_reactant", "limiting_ratio", "target_gain", "x_ratio_scaled")


def atlas():
    return json.load(open(PATH))


def test_fields():
    d = atlas()
    assert d["species"] and "not_analysed" in d and "within_factor_3" in d
    for s in d["species"]:
        assert all(k in s for k in SPECIES_FIELDS), s["species"]
        assert s["x_ratio_base"] < UNDER
        assert all(k in s["supply"] for k in SUPPLY_FIELDS), s["species"]
        assert s["supply"]["diagnosis"] in ("supply_limited", "rate_limited", "other")
        assert 0 <= s["n_reachable_clean"] <= s["n_reachable"] <= s["n_simulated"]
        assert all(not any(c.get(f) for f in FLAGS) for c in s["reachable_clean"])
    for s in d["not_analysed"]:
        assert s["x_ratio_base"] > 1 / UNDER


def test_ch2cch_supply_limited_by_c2h4():
    """Coherent with the CH2CCH discovery: C + C2H4 is the main route and C2H4 runs out."""
    s = next(s for s in atlas()["species"] if s["species"] == "CH2CCH")
    sup = s["supply"]
    assert sup["reaction"] == "C + C2H4 -> H + CH2CCH" and sup["diagnosis"] == "supply_limited"
    assert sup["limiting_reactant"] == "C2H4" and sup["limiting_ratio"] < 0.1
    assert s["n_reachable_clean"] == 0
    stored = json.load(open(os.path.join(RESULTS, "discovery", "CH2CCH.json")))["bottleneck"]
    assert abs(sup["limiting_ratio"] / stored["species"]["C2H4"]["ratio"] - 1) < 0.01


def test_supply_classification():
    row = dict(reac=["A", "B"], prod=["T", "H"], id=1)
    base = dict(A=1.0, B=1.0, T=1.0)
    assert supply(row, 10.0, base, dict(A=0.05, B=1.0, T=2.0), "T")["diagnosis"] == "supply_limited"
    assert supply(row, 10.0, base, dict(A=0.05, B=1.0, T=20.0), "T")["diagnosis"] == "rate_limited"
    assert supply(row, 10.0, base, dict(A=0.5, B=1.0, T=2.0), "T")["diagnosis"] == "other"
    assert supply(row, 10.0, base, dict(A=0.05, B=1.0, T=2.0), "T")["limiting_reactant"] == "A"


if __name__ == "__main__":
    main(__file__, globals())
