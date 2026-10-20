"""
plot_targets.py - Time evolution of the mystery molecules from results/timeseries.csv.
Output: results/targets.png
usage: python -m scripts.plot_targets
"""
import csv
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tapir.observations import TARGETS
from tapir.paths import result

OBS = {s: x for s, (x, _) in TARGETS.items()}
LAB = {"CH2CCH": "CH$_2$CCH", "CCS": "CCS", "CH3CHCH2": "CH$_3$CHCH$_2$",
       "HC3O+": "HC$_3$O$^+$", "C5O": "C$_5$O", "CH3CO+": "CH$_3$CO$^+$"}



def series(rows, net, co, S, sp):
    sel = [r for r in rows if r["network"] == str(net) and float(r["C_O"]) == co and np.isclose(float(r["S"]), S)]
    return np.array([float(r["t_yr"]) for r in sel]), np.array([float(r[sp]) for r in sel])


if __name__ == "__main__":
    ts = list(csv.DictReader(open(result("timeseries.csv"))))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    style = {(2014, 0.7): ("#c0504d", "--"), (2014, 1.2): ("#c0504d", "-"),
             (2024, 0.7): ("#1f5fa8", "--"), (2024, 1.2): ("#1f5fa8", "-")}
    for ax, sp in zip(axs.ravel(), OBS):
        ax.axhspan(OBS[sp] / 3, OBS[sp] * 3, color="0.85", zorder=0)
        ax.axhline(OBS[sp], color="k", lw=1)
        for (net, co), (c, ls) in style.items():
            for S, a in ([(8e-8, 1.0)] if sp != "CCS" else [(8e-8, 1.0), (1.5e-5, 0.45)]):
                t, x = series(ts, net, co, S, sp)
                lab = f"KIDA {net}, C/O={co}" + (f", S={S:.1e}" if sp == "CCS" else "")
                ax.plot(t, np.clip(x, 1e-20, None), color=c, ls=ls, alpha=a, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e3, 1e7)
        ax.set_ylim(OBS[sp] * 1e-5, OBS[sp] * 1e3); ax.set_title(LAB[sp])
        ax.grid(alpha=0.3, which="major")
    for ax in axs[1]: ax.set_xlabel("time (yr)")
    for ax in axs[:, 0]: ax.set_ylabel("abundance relative to H$_2$")
    axs[0, 1].legend(fontsize=7, loc="lower right")
    axs[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("0D gas-phase model (T=10 K, n$_H$=2e4 cm$^{-3}$, $\\zeta$=1e-17 s$^{-1}$, A$_V$=15). "
                 "Black line: observed in TMC-1; grey band: x3. S=8e-8 unless stated.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(result("targets.png"), dpi=150)
