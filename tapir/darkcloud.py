"""
darkcloud.py - 0D dark-cloud model, gas phase only, for kida.uva networks (2014 and 2024).

MODEL REFERENCE
  Wakelam, V., Gratier, P., Loison, J.-C., Hickson, K. M., Penguen, J., & Mechineau, A. 2024,
  "The 2024 KIDA network for interstellar chemistry", A&A, 689, A63.
  doi:10.1051/0004-6361/202450606 ; arXiv:2407.15958 ; bibcode 2024A&A...689A..63W
  kida.uva.2014 network: Wakelam, V., et al. 2015, ApJS, 217, 20.
  Rate formulas: Wakelam, V., et al. 2012, ApJS, 199, 21 (Eqs. 1, 5-7).

Taken from Wakelam et al. (2024):
  - Physical conditions of the gas-phase-only test (Sect. 2.3): T = 10 K, n_H = 2e4 cm-3,
    zeta = 1e-17 s-1, A_V = 15; 2014 vs 2024 comparison at 1e5 yr (Table 2) -> scripts/validate_w24.py.
  - Formulas by type (Sect. 2.2): CR/CRP k = alpha*zeta; photons k = alpha*exp(-gamma*A_V)*G0;
    Kooij; ionpol1 (Eq. 1) and ionpol2 (Eq. 2) for ion-dipole reactions.
  - Types 10/11 = forced H2 formation on grains (Wakelam+2012 Sect. 3.2); replaced here by the
    standard rate R = 3e-17 cm3 s-1.
  - Disagreement distance D(t) (Eq. 3) for the "best time", with the observed set of
    Gratier et al. (2016, ApJS 225, 25), the same one used in Wakelam+2024 Sect. 3.2.

Known differences from Nautilus/Wakelam+2024 (documented, not hidden):
  - Gas phase only. The TMC-1 comparison of Wakelam+2024 (Sect. 3.2) is gas-grain with
    n_H = 3e4 cm-3; ours is not, so magnitudes cannot be compared one to one.
  - Temperature ranges (t_extrap option): Nautilus 2.0 does not extrapolate, it uses the rate at the
    edge of the range ("nautilus"). By default here ("barrier") reactions with a barrier (Kooij,
    gamma > 0) are evaluated at the actual T (same spirit as the N + H2 correction of Wakelam+2024
    Sect. 2.1.2). Affects only 6 reactions in 2014 and 5 in 2024 and does not change the validation.
  - Duplicates (dedup option): if the same reaction (same reactants and products) appears under two
    numbers, the higher number (the most recent) is kept, following Wakelam+2024 Sect. 2.1.1.
    Without this the rate is counted twice (7 cases in 2014, 21 in 2024).

Conventions:
  itype 1, 2 (CR, CRP)    : k = alpha * zeta
  itype 3 (UV photons)    : k = alpha * exp(-gamma * Av) * G0
  formula 3 (Kooij)       : k = alpha (T/300)^beta exp(-gamma/T)
  formula 4 (ionpol1)     : k = alpha beta (0.62 + 0.4767 gamma sqrt(300/T))
  formula 5 (ionpol2)     : k = alpha beta (1 + 0.0967 gamma sqrt(300/T) + gamma^2/10.526 * 300/T)
  itype 0 (grains)        : k = alpha (T/300)^beta
  itype 10/11 (forced H2) : replaced by standard H2 formation, R = 3e-17 cm3 s-1
Fractional abundances x_i = n_i / n_H. Integrates dx/dt with BDF + sparse analytic Jacobian.
"""
import re, time
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

PLACEHOLDERS = {"CR", "CRP", "Photon"}
YR = 3.15576e7
NUM = re.compile(r"^[-+]?\d*\.?\d+[eE][-+]?\d+$")


def parse_kida(path):
    rows = []
    for line in open(path):
        if line.startswith("!") or not line.strip():
            continue
        reac = line[0:34].split()
        prod = line[34:91].split()
        t = line[91:].split()
        A, B, C = float(t[0]), float(t[1]), float(t[2])
        itype, tmin, tmax, formula, rid = int(t[6]), float(t[7]), float(t[8]), int(t[9]), int(t[10])
        rows.append(dict(reac=reac, prod=prod, A=A, B=B, C=C, itype=itype,
                         tmin=tmin, tmax=tmax, formula=formula, id=rid))
    return rows


def rate(r, T, zeta, Av, G0):
    A, B, C, it, f = r["A"], r["B"], r["C"], r["itype"], r["formula"]
    if it in (1, 2):
        return A * zeta
    if it == 3:
        return A * np.exp(-C * Av) * G0
    if it == 0:
        return A * (T / 300.0) ** B
    if f == 3:
        return A * (T / 300.0) ** B * np.exp(-C / T)
    if f == 4:
        return A * B * (0.62 + 0.4767 * C * np.sqrt(300.0 / T))
    if f == 5:
        return A * B * (1 + 0.0967 * C * np.sqrt(300.0 / T) + C**2 / 10.526 * 300.0 / T)
    raise ValueError(f"unsupported formula {f}: {r}")


# Conditions of the gas-phase-only test of Wakelam et al. (2024), Sect. 2.3
W24_GAS = dict(T=10.0, nH=2e4, zeta=1e-17, Av=15.0)


def select_entries(rows, T, t_extrap="barrier"):
    """One entry per reaction (same id) according to its temperature range.
    t_extrap: "barrier" (default) or "nautilus" (rate at the edge of the range, as Nautilus 2.0)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["id"], tuple(r["reac"]), tuple(r["prod"])), []).append(r)
    chosen = []
    for g in groups.values():
        inside = [r for r in g if r["tmin"] <= T <= r["tmax"]]
        if inside:
            r = dict(inside[0]); r["Teff"] = T
        else:
            r = dict(min(g, key=lambda r: min(abs(T - r["tmin"]), abs(T - r["tmax"]))))
            barrier = r["formula"] == 3 and r["C"] > 0 and t_extrap == "barrier"
            r["Teff"] = T if barrier else min(max(T, r["tmin"]), r["tmax"])
        chosen.append(r)
    return chosen


def dedup_latest(rows):
    """Same reaction (reactants and products) under several numbers -> keep the most recent."""
    best = {}
    for r in rows:
        key = (tuple(sorted(r["reac"])), tuple(sorted(r["prod"])))
        if key not in best or r["id"] > best[key]["id"]:
            best[key] = r
    keep = {id(r) for r in best.values()}
    return [r for r in rows if id(r) in keep]


_CACHE = {}


def load_rows(path, T=10.0, t_extrap="barrier", dedup=True):
    """Read and prepare the network once (cached): useful for hundreds of knock-outs."""
    key = (path, T, t_extrap, dedup)
    if key not in _CACHE:
        rows = select_entries(parse_kida(path), T, t_extrap)
        rows = [r for r in rows if r["itype"] not in (10, 11)]  # forced H2 -> replaced
        if dedup:
            rows = dedup_latest(rows)
        _CACHE[key] = rows
    return [dict(r) for r in _CACHE[key]]


class Network:
    def __init__(self, path, T=10.0, nH=2e4, zeta=1e-17, Av=15.0, G0=1.0, Rh2=3e-17,
                 n_grain=1.8e-12, extra_reactions=(), exclude=None, t_extrap="barrier", dedup=True):
        """extra_reactions: candidate reactions (dicts in the parse_kida format).
        exclude: function(row) -> True to remove the reaction (knock-out).
        t_extrap, dedup: see the module docstring."""
        self.path, self.T, self.nH = path, T, nH
        rows = load_rows(path, T, t_extrap, dedup)
        if exclude is not None:
            rows = [r for r in rows if not exclude(r)]
        rows += list(extra_reactions)
        names = sorted({s for r in rows for s in r["reac"] + r["prod"]} - PLACEHOLDERS)
        self.species = names
        self.idx = {s: i for i, s in enumerate(names)}
        self.n_grain = n_grain
        ks, R1, R2, S_rows, S_cols, S_vals = [], [], [], [], [], []
        for j, r in enumerate(rows):
            k = rate(r, r.get("Teff", T), zeta, Av, G0)
            reac = [s for s in r["reac"] if s not in PLACEHOLDERS]
            if len(reac) == 2:
                k *= nH                       # bimolecular -> times n_H (fractional abundances)
            elif len(reac) != 1:
                raise ValueError(f"reaction with {len(reac)} reactants: {r}")
            ks.append(k)
            R1.append(self.idx[reac[0]])
            R2.append(self.idx[reac[1]] if len(reac) == 2 else -1)
            for s in reac:
                S_rows.append(self.idx[s]); S_cols.append(j); S_vals.append(-1.0)
            for s in r["prod"]:
                if s in PLACEHOLDERS:
                    continue
                S_rows.append(self.idx[s]); S_cols.append(j); S_vals.append(1.0)
        # H2 formation on grains: H -> 1/2 H2, rate per H2 formed = Rh2 n_H x_H
        jH = len(ks)
        ks.append(Rh2 * nH); R1.append(self.idx["H"]); R2.append(-1)
        S_rows += [self.idx["H"], self.idx["H2"]]; S_cols += [jH, jH]; S_vals += [-2.0, 1.0]
        self.rows = rows
        self.k = np.array(ks)
        self.r1 = np.array(R1); self.r2 = np.array(R2)
        nr, ns = len(ks), len(names)
        self.S = sp.csr_matrix((S_vals, (S_rows, S_cols)), shape=(ns, nr))
        # sparsity pattern of the rate Jacobian D[j, i] = d rate_j / d x_i
        uni = self.r2 < 0
        bi = ~uni
        self._jr = np.concatenate([np.where(uni)[0], np.where(bi)[0], np.where(bi)[0]])
        self._jc = np.concatenate([self.r1[uni], self.r1[bi], self.r2[bi]])
        self._uni, self._bi = uni, bi
        self.nr, self.ns = nr, ns

    def rates(self, x):
        v = self.k * x[self.r1]
        v[self._bi] *= x[self.r2[self._bi]]
        return v

    def f(self, t, x):
        return self.S @ self.rates(x)

    def jac(self, t, x):
        k = self.k
        d = np.concatenate([k[self._uni],
                            k[self._bi] * x[self.r2[self._bi]],
                            k[self._bi] * x[self.r1[self._bi]]])
        D = sp.csr_matrix((d, (self._jr, self._jc)), shape=(self.nr, self.ns))
        return (self.S @ D).tocsc()

    def initial(self, elements):
        x = np.zeros(self.ns)
        for s, v in elements.items():
            x[self.idx[s]] = v
        x[self.idx["GRAIN0"]] = self.n_grain
        charge = sum(v for s, v in elements.items() if s.endswith("+"))
        x[self.idx["e-"]] = charge
        return x

    def run(self, elements, t_end_yr=1e7, n_out=161, rtol=1e-6, atol=1e-28):
        x0 = self.initial(elements)
        t_eval = np.logspace(2, np.log10(t_end_yr), n_out) * YR
        t0 = time.perf_counter()
        sol = solve_ivp(self.f, (0, t_eval[-1]), x0, method="BDF", jac=self.jac,
                        t_eval=t_eval, rtol=rtol, atol=atol)
        self.last_runtime = time.perf_counter() - t0
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t / YR, np.clip(sol.y, 0, None)


# Standard "low-metal" elemental abundances (relative to n_H), all H in H2.
def elements(C_over_O=0.7, S=8e-8):
    C = 1.7e-4
    return {"H2": 0.5, "He": 0.09, "N": 6.2e-5, "O": C / C_over_O, "C+": C, "S+": S,
            "Si+": 8e-9, "Fe+": 3e-9, "Na+": 2e-9, "Mg+": 7e-9, "P+": 2e-10,
            "Cl+": 1e-9, "F": 6.68e-9}
