"""
flux.py - Flux pre-ranking of candidate reactions on the knock-out network (no new simulation).

For a candidate A + B -> target + ... with generic rate k (knockout.capture_guess, never KIDA):
  dX ~= m * k * n_H * x_A * x_B / L_X        at t_ref, on the knock-out network,
where L_X (s-1) is the destruction frequency of the target at t_ref, m its multiplicity among the
products and X = 2x (relative to H2, as in knockout.X). Candidates whose class has no generic
rate (capture_guess -> None: radiative association, ion-ion, ...) get dX = None and are not ranked.
"""
import numpy as np
from . import chem
from .darkcloud import W24_GAS, rate
from .knockout import capture_guess, T_REF, run

PHYSICAL = set(chem.PLACEHOLDERS)


def knockout_exclude(hidden_key):
    """exclude callback for Network: drop every row strictly equivalent to the hidden reaction."""
    return lambda r: chem.canonical_key(r) == hidden_key


def kinetic_row(cand):
    """Candidate -> row in parse_kida format with the generic rate, or None if there is none.
    itype=4 (bimolecular) so darkcloud.rate uses the Kooij formula set by capture_guess."""
    g = capture_guess(dict(reac=list(cand["reac"]), prod=list(cand["prod"]), itype=4))
    if g is None:
        return None
    return dict(reac=g["reac"], prod=g["prod"], A=g["A"], B=g["B"], C=g["C"], itype=4,
                tmin=g["tmin"], tmax=g["tmax"], formula=g["formula"], id=-1, Teff=g["Teff"])


class FluxScorer:
    def __init__(self, net, t, y, t_ref=T_REF):
        """net, t, y: a knock-out run (knockout.run(exclude=...)). Keeps only x, L and the
        species index, so the scorer can be sent between processes."""
        self.idx = dict(net.idx)
        self.x = y[:, np.argmin(abs(t - t_ref))]
        v = net.rates(self.x)
        S = net.S.tocsr()
        destruction = -(S.multiply(S < 0) @ v)                  # per species, x units s-1
        with np.errstate(divide="ignore", invalid="ignore"):
            self.L = np.where(self.x > 0, destruction / self.x, 0.0)

    def k(self, row):
        return rate(row, row["Teff"], W24_GAS["zeta"], W24_GAS["Av"], 1.0)

    def delta_X(self, row, target):
        reac = [s for s in row["reac"] if s not in PHYSICAL]
        flux = self.k(row) * np.prod([self.x[self.idx[s]] for s in reac])
        if len(reac) == 2:
            flux *= W24_GAS["nH"]
        L = self.L[self.idx[target]]
        return 2 * row["prod"].count(target) * flux / L if L > 0 else float("inf")

    def score(self, cands, target):
        """Adds 'kinetic' (row or None), 'k' and 'dX' to each candidate; returns the ranked ones
        (dX descending) and the number without a generic rate."""
        ranked, no_rate = [], 0
        for c in cands:
            row = kinetic_row(c)
            c["kinetic"] = row
            if row is None:
                c["k"] = c["dX"] = None; no_rate += 1
                continue
            c["k"], c["dX"] = self.k(row), self.delta_X(row, target)
            ranked.append(c)
        ranked.sort(key=lambda c: (-c["dX"], chem.canonical_key(c)))
        return ranked, no_rate


def knockout_scorer(hidden_row):
    net, t, y = run(exclude=knockout_exclude(chem.canonical_key(hidden_row)))
    return FluxScorer(net, t, y), (net, t, y)
