"""
knockout.py - Reference configuration and shared helpers for knock-out experiments.

Reference configuration: kida.uva.2024, C/O = 1.2, S = 8e-8, conditions of Wakelam+2024 Sect. 2.3.
Reference time: minimum of D(t) against Gratier+2016 (scripts/best_time_w24.py) -> 6e6 yr.
Abundances X are relative to H2 (X = 2 x, with x = n / n_H).
"""
import numpy as np
from .darkcloud import Network, elements, W24_GAS
from .paths import NET_2024

NET = NET_2024
EL = elements(C_over_O=1.2, S=8e-8)
T_REF, T_EARLY = 6e6, 1e5
PHYS = {"CR", "CRP", "Photon"}

PHYSICAL = "physical (CR/CRP/photon)"
REACTION_CLASSES = (PHYSICAL, "radiative electron attachment", "dissociative recombination",
                    "radiative association", "grain", "ion-neutral", "ion-ion", "neutral-neutral")


def label(r):
    return " + ".join(r["reac"]) + " -> " + " + ".join(r["prod"])


def reaction_class(r):
    reac, prod = r["reac"], r["prod"]
    if PHYS & set(reac): return PHYSICAL
    if "e-" in reac: return "radiative electron attachment" if "Photon" in prod else "dissociative recombination"
    if "Photon" in prod: return "radiative association"
    if any(s.startswith("GRAIN") for s in reac): return "grain"
    ion = [s for s in reac if s.endswith(("+", "-"))]
    return "ion-neutral" if len(ion) == 1 else ("ion-ion" if len(ion) == 2 else "neutral-neutral")


def capture_guess(r):
    """Generic rate that never looks at KIDA: what the pipeline gives a new candidate."""
    c = reaction_class(r)
    k = {"ion-neutral": (2e-9, 0.0), "neutral-neutral": (1e-10, 0.0),
         "dissociative recombination": (1e-7, -0.5)}.get(c)
    if k is None: return None
    return dict(r, A=k[0], B=k[1], C=0.0, formula=3, tmin=10.0, tmax=300.0, id=-1, Teff=W24_GAS["T"])


def is_row(target):
    key = (target["id"], tuple(target["reac"]), tuple(target["prod"]))
    return lambda r: (r["id"], tuple(r["reac"]), tuple(r["prod"])) == key


def run(el=EL, **kw):
    """One run of the reference network; kw goes to Network (exclude=, extra_reactions=)."""
    net = Network(NET, **W24_GAS, **kw); t, y = net.run(el)
    return net, t, y


def X(net, t, y, s, tt):
    return 2 * y[net.idx[s], np.argmin(abs(t - tt))]


def peak(net, t, y, s):
    w = (t >= 1e4) & (t <= 1e7); return 2 * y[net.idx[s], w].max()
