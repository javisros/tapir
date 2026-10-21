"""Tests for tapir.discovery and scripts.discover, and for the ion H-transfer pattern of tapir.rates.
Usage: pytest  (or, without pytest: python -m tests.test_discovery)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, subprocess, sys, tempfile
import numpy as np
from tests._compat import pytest, main
from tapir.darkcloud import load_rows
from tapir.discovery import (bisect_log_k, distance, observed_set, capture_feasible, MAX_SIMS, DISCLAIMER)
from tapir.flux import kinetic_row
from tapir.knockout import NET, T_REF, run, X
from tapir.paths import ROOT
from tapir.rates import (RateEstimator, ion_h_transfer, ion_flag, ion_h_transfer_validation, H_TRANSFER,
                         ION_H_TRANSFER_MAX, ION_H_TRANSFER_EXCLUDE)
from scripts.best_time_w24 import D_of_t

ROWS24 = load_rows(NET)


def rx(lab):
    """'A + B -> C + D' -> candidate dict"""
    reac, prod = lab.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


# --- ion H-transfer pattern (tapir.rates) ------------------------------------------------------

@pytest.mark.parametrize("lab, match", (("c-C3H2 + H3+ -> CH2CCH + H2+", True),
                                        ("c-C3H2 + HCO+ -> CH2CCH + CO+", True),
                                        ("N2 + H3+ -> H2 + N2H+", False),      # hidden reaction of benchmark B
                                        ("c-C3H2 + CH3 -> CH2CCH + CH2", False),
                                        ("CH3CCH + H3+ -> H2 + C3H5+", False)))
def test_ion_h_transfer_pattern(lab, match):
    assert ion_h_transfer(rx(lab)) == match
    assert ion_flag(rx(lab)) == (H_TRANSFER if match else None)


def test_ion_h_transfer_validation():
    """The decision in tapir.rates: the pattern is not rare among fast KIDA ion-neutral reactions,
    so it is flagged, not excluded."""
    n, matches = ion_h_transfer_validation(ROWS24)
    assert n > 3000
    assert len(matches) / n >= ION_H_TRANSFER_MAX
    assert ION_H_TRANSFER_EXCLUDE is False


def test_ion_h_transfer_flag_only():
    c = rx("c-C3H2 + H3+ -> CH2CCH + H2+")
    assert RateEstimator(ROWS24).estimate(c) == (kinetic_row(c), "capture_guess")


# --- distance D --------------------------------------------------------------------------------

def test_distance_reproduces_D_of_t():
    """D at t_ref on the base network = D_of_t of scripts/best_time_w24 at the same output time."""
    net, t, y = run()
    obs = observed_set("CH2CCH")
    xs = {s: X(net, t, y, s, T_REF) for s in obs if s in net.idx}
    d_t, n = D_of_t(net, y, obs)
    assert n == len(xs) == len(obs)
    assert math.isclose(distance(xs, obs), d_t[np.argmin(abs(t - T_REF))], rel_tol=1e-12)


# --- bisection ---------------------------------------------------------------------------------

def synthetic(x0=1e-10, power=0.5):
    """Monotone X(k) = x0 + 1e-9 (k / 1e-10)^power, counting calls."""
    calls = []

    def f(k):
        calls.append(k)
        return x0 + 1e-9 * (k / 1e-10) ** power
    return f, calls


def exact_k(x_obs, x0=1e-10, power=0.5):
    return 1e-10 * ((x_obs - x0) / 1e-9) ** (1 / power)


@pytest.mark.parametrize("x_obs", (3e-10, 8.7e-9, 2e-8))
def test_bisection_converges(x_obs):
    f, calls = synthetic()
    r = bisect_log_k(f, x_obs, 1e-14, 1e-7)
    assert r["k_status"] == "found" and r["n_sims"] == len(calls) <= MAX_SIMS
    assert abs(math.log10(r["k_needed"] / exact_k(x_obs))) < 0.05
    assert abs(math.log10(f(r["k_needed"]) / x_obs)) < 0.02


def test_bisection_extremes():
    f, calls = synthetic()
    assert bisect_log_k(f, 1e-6, 1e-14, 1e-7)["k_status"] == "not_reachable" and len(calls) == 1
    f, calls = synthetic(x0=1e-9)
    r = bisect_log_k(f, 5e-10, 1e-14, 1e-7)
    assert r["k_status"] == "already_above" and r["k_needed"] is None and len(calls) == 2


def test_bisection_error():
    r = bisect_log_k(lambda k: None if k < 1e-8 else 1.0, 1e-3, 1e-14, 1e-7)
    assert r["k_status"] == "error" and r["k_needed"] is None


def test_capture_feasible():
    c = rx("c-C3H2 + C5H -> CH2CCH + C5")                      # neutral-neutral, limit 1e-9
    assert capture_feasible(c, 5e-10, "found") is True
    assert capture_feasible(c, 5e-9, "found") is False
    assert capture_feasible(c, None, "not_reachable") is False
    assert capture_feasible(c, None, "already_above") is True
    assert capture_feasible(c, None, "error") is None


# --- output of scripts.discover ----------------------------------------------------------------

TOP_FIELDS = ("species", "mode", "disclaimer", "config", "observation", "base", "t_yr", "counts", "runtime",
              "hypotheses")
HYP_FIELDS = ("rank", "reaction", "reac", "prod", "source", "rationale", "family", "reaction_class", "k",
              "k_source", "unpaired_flag", "ion_flag", "flux_rank", "dX_pred", "X_target", "x_ratio", "D",
              "delta_D", "k_needed", "k_status", "k_needed_sims", "capture_max", "capture_feasible", "robust",
              "literature", "k_lit", "error", "curve")


def test_discover_json_fields():
    """End to end with K = 1 (at most 12 simulations), written to a temporary results directory."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TAPIR_RESULTS=tmp, OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, "-m", "scripts.discover", "CH2CCH", "--k", "1", "--top", "1",
                        "--procs", "1"], cwd=ROOT, env=env, check=True, capture_output=True)
        doc = json.load(open(os.path.join(tmp, "discovery", "CH2CCH.json")))
        assert os.path.exists(os.path.join(tmp, "discovery", "CH2CCH.csv"))
    assert all(f in doc for f in TOP_FIELDS)
    assert doc["mode"] == "observation" and doc["disclaimer"] == DISCLAIMER
    assert doc["observation"]["X_obs"] == 8.7e-9 and doc["config"]["t_ref_yr"] == T_REF
    assert len(doc["base"]["curve"]) == len(doc["t_yr"])
    [h] = doc["hypotheses"]
    assert all(f in h for f in HYP_FIELDS)
    assert h["source"] == "analogy" and len(h["curve"]) == len(doc["t_yr"])
    assert h["k_status"] in ("found", "not_reachable", "already_above", "error")
    assert h["k_needed_sims"] <= MAX_SIMS
    assert math.isclose(h["x_ratio"], h["X_target"] / doc["observation"]["X_obs"], rel_tol=1e-3)
    assert math.isclose(h["delta_D"], h["D"] - doc["base"]["D"], abs_tol=2e-4)


if __name__ == "__main__":
    main(__file__, globals())
