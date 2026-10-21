"""Tests for tapir.discovery and scripts.discover, and for the ion H-transfer pattern of tapir.rates.
Usage: pytest  (or, without pytest: python -m tests.test_discovery)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, subprocess, sys, tempfile
import numpy as np
from tests._compat import pytest, main
from tapir.darkcloud import load_rows
from tapir.discovery import (bisect_log_k, distance, observed_set, capture_feasible, kida_uncertainty, row_key,
                             simulate, simulate_scaled, MAX_SIMS, DISCLAIMER, OK)
from tapir.flux import kinetic_row
from tapir.knockout import NET, T_REF, run, X, label
from tapir.paths import ROOT
from tapir.llm_client import FormatError
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
    assert r["k_status"] == OK and r["n_sims"] == len(calls) <= MAX_SIMS
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
    assert capture_feasible(c, 5e-10, OK) is True
    assert capture_feasible(c, 5e-9, OK) is False
    assert capture_feasible(c, None, "not_reachable") is False
    assert capture_feasible(c, None, "already_above") is True
    assert capture_feasible(c, None, "error") is None


# --- existing reactions -----------------------------------------------------------------------

def test_kida_uncertainty():
    """F and g are read from the columns after alpha, beta, gamma (first rows of kida.uva.2024)."""
    unc = kida_uncertainty(NET)
    assert unc[(1, ("N2", "CR"), ("N", "N"), -9999.0, 9999.0)] == (1.25, 0.0)
    assert unc[(2, ("H", "CR"), ("H+", "e-"), -9999.0, 9999.0)] == (2.0, 0.0)
    assert all(row_key(r) in unc for r in ROWS24)


def test_factor_one_is_base():
    """Replacing a reaction by itself times 1 gives the base network."""
    row = next(r for r in ROWS24 if label(r) == "C + C2H4 -> H + CH2CCH")
    species = sorted(observed_set("CH2CCH"))
    base, same = simulate(None, "CH2CCH", species), simulate_scaled(row, 1.0, "CH2CCH", species)
    assert not base["error"] and not same["error"]
    assert all(math.isclose(same["xs"][s], base["xs"][s], rel_tol=1e-6, abs_tol=1e-30) for s in base["xs"])
    lower = simulate_scaled(row, 0.1, "CH2CCH", ["CH2CCH"])
    assert lower["xs"]["CH2CCH"] < 0.9 * base["xs"]["CH2CCH"]


# --- output of scripts.discover ----------------------------------------------------------------

TOP_FIELDS = ("species", "mode", "X_obs", "obs_source", "generators", "config", "config2", "X_base", "D_base",
              "n_generated", "n_no_rate", "n_simulated", "runtime_s", "t", "curve_base", "hypotheses",
              "existing", "disclaimer")
CONFIG_FIELDS = ("network", "C_O", "S", "t_ref", "physics")
HYP_FIELDS = ("rank", "reaction", "reac", "prod", "class", "source", "family", "rationale", "k_source", "k",
              "dX_flux", "X_cand", "x_ratio", "D", "delta_D", "k_needed", "k_status", "k_max_class",
              "capture_feasible", "n_bisect", "curve", "robust", "config2", "literature", "lit_run",
              "unpaired_flag", "ion_flag")
EXISTING_FIELDS = ("reaction", "role", "frac", "k_kida", "F", "factor_needed", "factor_status",
                   "within_uncertainty", "delta_D", "delta_D_gratier", "curve")


def test_discover_json_fields():
    """End to end with K = 1 and one existing reaction per role, in a temporary results directory."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TAPIR_RESULTS=tmp, OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, "-m", "scripts.discover", "CH2CCH", "--k", "1", "--top", "1",
                        "--procs", "2", "--existing", "1"], cwd=ROOT, env=env, check=True, capture_output=True)
        doc = json.load(open(os.path.join(tmp, "discovery", "CH2CCH.json")))
        assert os.path.exists(os.path.join(tmp, "discovery", "CH2CCH.csv"))
    assert all(f in doc for f in TOP_FIELDS) and all(f in doc["config"] for f in CONFIG_FIELDS)
    assert doc["mode"] == "formation" and doc["disclaimer"] == DISCLAIMER
    assert doc["X_obs"] == 8.7e-9 and doc["config"]["t_ref"] == T_REF
    assert len(doc["curve_base"]) == len(doc["t"])
    [h] = doc["hypotheses"]
    assert all(f in h for f in HYP_FIELDS)
    assert h["source"] == ["analogy"] and len(h["curve"]) == len(doc["t"])
    assert h["k_status"] in (OK, "not_reachable", "already_above")
    assert h["n_bisect"] <= MAX_SIMS
    assert math.isclose(h["x_ratio"], h["X_cand"] / doc["X_obs"], rel_tol=1e-3)
    assert math.isclose(h["delta_D"], h["D"] - doc["D_base"], abs_tol=2e-4)
    assert [e["role"] for e in doc["existing"]] == ["formation", "destruction"]
    for e in doc["existing"]:
        assert all(f in e for f in EXISTING_FIELDS)
        assert e["factor_status"] in (OK, "not_reachable", "already_above") and e["n_bisect"] <= MAX_SIMS
        assert 0 < e["frac"] <= 1 and e["F"] >= 1 and len(e["curve"]) == len(doc["t"])
    bn = doc["bottleneck"]
    assert bn["reaction"] == "C + C2H4 -> H + CH2CCH" and {"C", "C2H4", "CH3"} <= set(bn["species"])
    assert isinstance(bn["supply_limited"], bool)


def test_upstream_mode():
    """--upstream: candidates produce the precursor, the score is on the target."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TAPIR_RESULTS=tmp, OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, "-m", "scripts.discover", "CH2CCH", "--k", "2", "--top", "4",
                        "--procs", "2", "--no-bisect", "--upstream", "C2H4,CH3"],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        doc = json.load(open(os.path.join(tmp, "discovery", "CH2CCH_upstream.json")))
    assert all(f in doc for f in TOP_FIELDS) and doc["search"] == "upstream" and doc["upstream"] == ["C2H4", "CH3"]
    assert doc["species"] == "CH2CCH" and doc["existing"] == []
    hs = doc["hypotheses"]
    assert {h["via"] for h in hs} == {"C2H4", "CH3"}
    i_ref = min(range(len(doc["t"])), key=lambda i: abs(doc["t"][i] - T_REF))
    for h in hs:
        assert all(f in h for f in HYP_FIELDS + ("via", "n_worse"))
        assert all(p in h["prod"] for p in h["via"].split("+"))
        assert "CH2CCH" not in h["prod"]
        assert math.isclose(h["curve"][i_ref], h["X_cand"], rel_tol=1e-3)          # X_cand is CH2CCH
        assert math.isclose(h["x_ratio"], h["X_cand"] / doc["X_obs"], rel_tol=1e-3)


# --- llm generator in scripts.discover (fake client, no call to LiteLLM) -----------------------

class FakeClient:
    """LLMClient interface; raises FormatError (as the real client after its retry: two requests, no
    usage) for the targets in `fail`, otherwise returns `proposals` with 100 tokens per request."""
    model, reasoning_effort = "fake", "low"

    def __init__(self, proposals, fail=()):
        self.proposals, self.fail, self.prompts = proposals, set(fail), []
        self.stats = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0, format_failures=0)

    def ask(self, prompt, schema, validate):
        self.prompts.append(prompt)
        if any(f"Target species: {t}\n" in prompt for t in self.fail):
            self.stats["calls"] += 2; self.stats["format_failures"] += 2
            raise FormatError("invalid answer after one retry: json_validate_failed")
        self.stats["calls"] += 1
        for k, v in (("prompt_tokens", 60), ("completion_tokens", 40), ("total_tokens", 100)):
            self.stats[k] += v
        return validate(dict(candidates=self.proposals))


def prop(reac, prod):
    return dict(reactants=reac, products=prod, reaction_class="neutral-neutral", rationale="test")


def base_and_generators(client):
    from scripts.discover import make_generators
    from tapir.darkcloud import parse_kida
    from tapir import chem
    net, t, y = run()
    raw14 = parse_kida(chem.REACTIONS_2014)
    gens = make_generators(["analogy", "llm"], ROWS24, raw14, net, t, y, client=client)
    gens["llm"].llm.cache_dir = tempfile.mkdtemp()
    return net, t, y, gens


def best_analogy(target):
    """Best analogy candidate forming `target` (a valid novel proposal for the fake LLM)."""
    from tapir.generator import Generator
    from tapir.darkcloud import parse_kida
    from tapir import chem
    return next(c for c in Generator(ROWS24, parse_kida(chem.REACTIONS_2014)).candidates(target)
                if target in c["prod"] and target not in c["reac"])


def test_llm_generator_merged_with_analogy():
    """An LLM proposal equal to an analogy candidate is merged (source lists both); the context is the
    target's budget in the base run and the direction is 'increase'."""
    from scripts.discover import candidates, llm_context
    from tapir import chem
    best = best_analogy("CH2CCH")
    client = FakeClient([prop(best["reac"], best["prod"]), prop(best["prod"], best["reac"])])   # + its reverse
    net, t, y, gens = base_and_generators(client)
    cands = candidates(gens, "CH2CCH")
    merged = next(c for c in cands if chem.canonical_key(c) == chem.canonical_key(best))
    assert merged["source"] == ["analogy", "llm"]
    assert all(c["source"] != ["llm"] for c in cands)             # the reverse is wrong_direction
    [e] = gens["llm"].log
    assert e["target"] == "CH2CCH" and e["status"] == "ok" and e["cached"] is False and e["error"] is None
    assert (e["model"], e["reasoning_effort"], e["proposed"], e["kept"]) == ("fake", "low", 2, 1)
    assert e["discarded"] == {"wrong_direction": 1} and client.stats["calls"] == 1
    assert e["usage"]["total_tokens"] == 100 and e["usage"]["calls"] == 1
    form, dest = llm_context(net, t, y, "CH2CCH")
    assert all(lab in client.prompts[0] for lab in form + dest) and "net formation" in client.prompts[0]
    candidates(gens, "CH2CCH")                                     # second time: from the cache
    assert client.stats["calls"] == 1 and gens["llm"].log[-1]["cached"] is True
    assert gens["llm"].log[-1]["usage"] is None


def test_llm_format_error_degrades_to_analogy():
    """FormatError on C2H4: logged as format_error without invented tokens, C2H4 keeps the analogy
    candidates, and CH3 is still asked and contributes its LLM candidate."""
    from scripts.discover import candidates, upstream_candidates
    from tapir.flux import FluxScorer
    from tapir import chem
    best = best_analogy("CH3")
    client = FakeClient([prop(best["reac"], best["prod"])], fail=["C2H4"])
    net, t, y, gens = base_and_generators(client)
    analogy_c2h4 = {chem.canonical_key(c) for c in gens["analogy"]("C2H4")}
    assert {chem.canonical_key(c) for c in candidates(gens, "C2H4")} == analogy_c2h4
    gens["llm"].log.clear()
    top, n = upstream_candidates(gens, ["C2H4", "CH3"], FluxScorer(net, t, y), RateEstimator(ROWS24), 5)
    fail, ok = gens["llm"].log
    assert (fail["target"], fail["status"], ok["target"], ok["status"]) == ("C2H4", "format_error", "CH3", "ok")
    assert "json_validate_failed" in fail["error"] and fail["kept"] == 0 and fail["proposed"] is None
    assert fail["usage"]["total_tokens"] is None and fail["usage"]["calls"] == 2       # tokens not invented
    assert ok["kept"] == 1 and ok["usage"]["total_tokens"] == 100
    assert {h["via"] for h in top} == {"C2H4", "CH3"} and n["generated"] > 0


def test_dry_run_makes_no_call():
    """--dry-run with an unreachable proxy: prompts and estimate only, nothing written."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TAPIR_RESULTS=tmp, OMP_NUM_THREADS="1", LITELLM_BASE_URL="http://127.0.0.1:9/v1")
        out = subprocess.run([sys.executable, "-m", "scripts.discover", "CH2CCH", "--generators", "analogy,llm",
                              "--upstream", "C2H4,CH3", "--dry-run"],
                             cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout
        assert not os.path.exists(os.path.join(tmp, "discovery")) and not os.path.exists(os.path.join(tmp, "llm_cache"))
    assert "C2H4: NEW CALL" in out and "CH3: NEW CALL" in out and "2 new call(s)" in out


if __name__ == "__main__":
    main(__file__, globals())
