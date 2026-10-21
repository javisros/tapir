"""
discovery.py - Discovery mode: candidate reactions ranked against the TMC-1 observation of a target.

Reference configuration (tapir.knockout): kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr.
The objective is the observed abundance (tapir.observations), not the base model.

Score of a network (base, or base + one candidate) at t_ref:
  D = mean over s in GRATIER + target of |log10 X_s - log10 X_obs,s|     (X relative to H2, floor 1e-30)
the D(t) of scripts/best_time_w24.D_of_t (Wakelam+2024 Eq. 3) with the target's observation added.
  delta_D = D_cand - D_base   (negative: the candidate brings the model closer to TMC-1)
  x_ratio = X_target / X_obs

Required rate (k_needed): bisection in log10(k) at 10 K, between K_MIN and K_MAX of the candidate's
class, until X_target(t_ref) = X_obs. The extremes are evaluated first:
  k_status = "not_reachable"  X_target < X_obs even at K_MAX
             "already_above"  X_target > X_obs already at K_MIN
             "found"          k_needed interpolated in log-log inside the final bracket
             "error"          an integration failed
at most MAX_SIMS simulations per candidate. Assumes X_target grows with k (it does for a formation
reaction unless the candidate also consumes a precursor of the target). capture_feasible =
k_needed <= rates.CAPTURE_MAX of the class (True if already_above, False if not_reachable).
"""
import math
import numpy as np
from .darkcloud import W24_GAS
from .knockout import EL, T_REF, reaction_class, run, X, label
from .observations import GRATIER, observed
from .rates import capture_max

FLOOR = 1e-30                                   # same floor as scripts/best_time_w24.D_of_t
K_MIN = 1e-14
K_MAX = {"dissociative recombination": 1e-5}    # other classes: K_MAX_DEFAULT
K_MAX_DEFAULT = 1e-7
MAX_SIMS = 10
LOG_TOL = 0.01                                  # |log10 X / X_obs| that stops the bisection early
DISCLAIMER = "Ranked hypotheses, not confirmed reactions."
FAMILY_NAMES = dict(PT="proton transfer", DR="dissociative recombination (H loss)",
                    AE="addition-H elimination", OS="O/S isovalent substitution",
                    CT="charge transfer", HA="H-atom abstraction")


def observation(target):
    """(X_obs relative to H2, source) of the target; ValueError if it has no quoted value."""
    x, src = observed().get(target, (None, None))
    if x is None:
        raise ValueError(f"no observed abundance for {target!r}")
    return x, src


def observed_set(target):
    """GRATIER plus the target's observation: the species D is computed on."""
    obs = dict(GRATIER)
    obs[target] = observation(target)[0]
    return obs


def distance(xs, obs):
    """D: mean |log10 X - log10 X_obs| over the species of obs present in xs (X relative to H2)."""
    sp = [s for s in obs if s in xs]
    return float(np.mean([abs(math.log10(max(xs[s], FLOOR)) - math.log10(obs[s])) for s in sp]))


def with_rate(row, k):
    """Copy of a candidate row whose rate is k (cm3 s-1) at the model temperature."""
    return dict(row, A=k, B=0.0, C=0.0, formula=3, itype=4, Teff=W24_GAS["T"])


def k_bounds(r):
    return K_MIN, K_MAX.get(reaction_class(r), K_MAX_DEFAULT)


def simulate(row, target, species, el=EL, t_ref=T_REF):
    """Reference network (+ row if not None) -> dict(xs={species: X at t_ref}, t, curve of the target,
    runtime, error). Integration errors are returned, not raised."""
    try:
        net, t, y = run(el, extra_reactions=[row] if row is not None else ())
    except Exception as e:                                  # solve_ivp failure or bad row
        return dict(xs=None, t=None, curve=None, runtime=None, error=f"{type(e).__name__}: {e}")
    return dict(xs={s: X(net, t, y, s, t_ref) for s in species if s in net.idx}, t=t.tolist(),
                curve=(2 * y[net.idx[target]]).tolist(), runtime=net.last_runtime, error="")


def bisect_log_k(f, x_obs, lo, hi, max_sims=MAX_SIMS, tol=LOG_TOL):
    """Bisection in log10(k) for f(k) = x_obs, f increasing; f returns None on failure.
    -> dict(k_needed, k_status, n_sims, trace=[(k, f(k))])."""
    trace = []

    def ev(lk):
        v = f(10 ** lk)
        trace.append((10 ** lk, v))
        return v

    def out(status, k=None):
        return dict(k_needed=k, k_status=status, n_sims=len(trace), trace=trace)

    llo, lhi = math.log10(lo), math.log10(hi)
    xhi = ev(lhi)
    if xhi is None:
        return out("error")
    if xhi < x_obs:
        return out("not_reachable")
    xlo = ev(llo)
    if xlo is None:
        return out("error")
    if xlo > x_obs:
        return out("already_above")
    target = math.log10(x_obs)
    while len(trace) < max_sims:
        mid = (llo + lhi) / 2
        xm = ev(mid)
        if xm is None:
            return out("error")
        if abs(math.log10(max(xm, FLOOR)) - target) <= tol:
            return out("found", 10 ** mid)
        if xm < x_obs:
            llo, xlo = mid, xm
        else:
            lhi, xhi = mid, xm
    a, b = math.log10(max(xlo, FLOOR)), math.log10(max(xhi, FLOOR))
    lk = llo if b == a else llo + (target - a) / (b - a) * (lhi - llo)
    return out("found", 10 ** lk)


def capture_feasible(r, k_needed, k_status):
    """k_needed <= capture limit of the class; None if undetermined."""
    if k_status == "already_above":
        return True
    if k_status == "not_reachable":
        return False
    cap = capture_max(r)
    if k_status != "found" or cap is None:
        return None
    return bool(k_needed <= cap)


def required_rate(task):
    """(candidate row, target, X_obs) -> bisect_log_k result plus capture_max and capture_feasible.
    One process per candidate (multiprocessing)."""
    row, target, x_obs = task
    errors = []

    def f(k):
        s = simulate(with_rate(row, k), target, [target])
        if s["error"]:
            errors.append(s["error"])
            return None
        return s["xs"][target]

    lo, hi = k_bounds(row)
    res = bisect_log_k(f, x_obs, lo, hi)
    res.update(capture_max=capture_max(row), error="; ".join(errors),
               capture_feasible=capture_feasible(row, res["k_needed"], res["k_status"]))
    return res


def rationale(c):
    """One line from the generator's best derivation (analogy) or the candidate's own rationale."""
    if c.get("rationale"):
        return c["rationale"]
    fam, template, support, local = max(c["derivations"],
                                        key=lambda d: math.log10(1 + d[2]) + math.log10(1 + d[3]))
    return (f"analogy, {FAMILY_NAMES.get(fam, fam)}: template {template} "
            f"({support} instances in KIDA, local evidence {local})")


def sig(v, n=4):
    """Float rounded to n significant digits (None and 0 unchanged), to keep the JSON small."""
    return v if not v else float(f"{v:.{n - 1}e}")


def hypothesis(c, sim, base, obs, target):
    """Record of one simulated candidate (fields of the discovery JSON, without rank and k_needed)."""
    x_obs = obs[target]
    rec = dict(reaction=label(c), reac=list(c["reac"]), prod=list(c["prod"]),
               source=c.get("source", "analogy"), rationale=rationale(c), family=c.get("family"),
               reaction_class=reaction_class(c), prior_score=c.get("score"),
               k=sig(c["k"]), k_source=c["k_source"], unpaired_flag=c.get("unpaired_flag"),
               ion_flag=c.get("ion_flag"), flux_rank=c["flux_rank"], dX_pred=sig(c["dX"]),
               X_target=None, x_ratio=None, D=None, delta_D=None,
               k_needed=None, k_status=None, k_needed_sims=None, capture_max=capture_max(c),
               capture_feasible=None, robust=None, literature=None, k_lit=None,
               error=sim["error"], curve=None)
    if sim["xs"] is not None:
        d = distance(sim["xs"], obs)
        rec.update(X_target=sig(sim["xs"][target]), x_ratio=sig(sim["xs"][target] / x_obs),
                   D=round(d, 4), delta_D=round(d - base["D"], 4),
                   curve=[sig(v) for v in sim["curve"]])
    return rec


def rank_key(h):
    """Ranking: delta_D ascending (failed runs last), then the reaction label."""
    return (math.inf if h["delta_D"] is None else h["delta_D"], h["reaction"])
