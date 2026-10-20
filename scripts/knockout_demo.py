"""
knockout_demo.py - Walk-through of the TAPIR loop with CH2CCH (kida.uva.2024, C/O=1.2, S=8e-8):
1) formation/destruction budget, 2) knock-out of the dominant reaction,
3) re-injection as a candidate with a capture-type rate (1e-10 cm3 s-1).
usage: python -m scripts.knockout_demo
"""
import time
import numpy as np
from tapir.darkcloud import Network, parse_kida
from tapir.knockout import NET, EL
from tapir.observations import TARGETS

SP, T_REF = "CH2CCH", 6e6
OBS = TARGETS[SP][0]


def budget(net, t, y, sp, tt, n=5):
    k = np.argmin(abs(t - tt)); v = net.rates(y[:, k]); col = net.S[net.idx[sp]].toarray().ravel() * v
    name = lambda j: (" + ".join(net.rows[j]["reac"]) + " -> " + " + ".join(net.rows[j]["prod"])
                      if j < len(net.rows) else "H2 formation on grains")
    form = [(name(j), col[j] / col[col > 0].sum()) for j in np.argsort(col)[::-1][:n] if col[j] > 0]
    dest = [(name(j), col[j] / col[col < 0].sum()) for j in np.argsort(col)[:n] if col[j] < 0]
    return form, dest


def run(**kw):
    t0 = time.perf_counter(); net = Network(NET, **kw); t, y = net.run(EL)
    x = 2 * y[net.idx[SP]]; w = (t >= 1e5) & (t <= 1e7)
    return net, t, y, x[w].max(), x[np.argmin(abs(t - T_REF))], time.perf_counter() - t0


if __name__ == "__main__":
    net, t, y, pk, at, dt = run()
    form, dest = budget(net, t, y, SP, T_REF)
    print(f"Budget of {SP} at t={T_REF:.0e} yr:")
    for r, p in form: print(f"  formation   {100*p:5.1f}%  {r}")
    for r, p in dest: print(f"  destruction {100*p:5.1f}%  {r}")

    top = form[0][0]
    is_top = lambda r: " + ".join(r["reac"]) + " -> " + " + ".join(r["prod"]) == top
    orig = [r for r in parse_kida(NET) if is_top(r)][0]
    cand = dict(orig, A=1e-10, B=0.0, C=0.0, formula=3, itype=4, tmin=10, tmax=300, id=-1)
    print(f"\nKnock-out of: {top}")
    for lab, kw in [("base", {}), ("knock-out", dict(exclude=is_top)),
                    ("candidate 1e-10", dict(exclude=is_top, extra_reactions=[cand]))]:
        *_, pk, at, dt = run(**kw)
        print(f"  {lab:16s} peak={pk:.1e} ({pk/OBS:.2f} x obs)  at {T_REF:.0e} yr={at:.1e}  [{dt:.1f} s]")
