"""
run_targets.py - Does the base model reproduce the published TMC-1 discrepancies?
Networks: kida.uva.2014 and kida.uva.2024. Sweep in C/O and elemental sulphur (12 runs, ~1 min).
Outputs: results/summary.csv, results/timeseries.csv (plot them with scripts/plot_targets.py)
usage: python -m scripts.run_targets
"""
import itertools, csv
import numpy as np
from tapir.darkcloud import Network, elements
from tapir.observations import TARGETS, CONTROL
from tapir.paths import NETWORKS, result

CO_LIST = [0.7, 1.2]
S_LIST = [8e-8, 1.5e-6, 1.5e-5]
TWIN = (1e4, 1e7)


def to_H2(x): return 2.0 * x   # x relative to n_H -> relative to H2


if __name__ == "__main__":
    summary, series = [], []
    nets = {k: Network(v) for k, v in NETWORKS.items()}
    for (net_name, net), co, S in itertools.product(nets.items(), CO_LIST, S_LIST):
        t, y = net.run(elements(C_over_O=co, S=S))
        w = (t >= TWIN[0]) & (t <= TWIN[1])
        X = {s: to_H2(y[net.idx[s]]) for s in list(TARGETS) + list(CONTROL) if s in net.idx}
        # best time = minimum disagreement distance with the control species
        D = np.mean([np.abs(np.log10(np.maximum(X[s], 1e-30)) - np.log10(v))
                     for s, v in CONTROL.items()], axis=0)
        D[~w] = np.inf
        kb = int(np.argmin(D))
        for s, (xo, ref) in TARGETS.items():
            xs = X[s]
            kp = np.argmax(np.where(w, xs, -1))
            summary.append(dict(network=net_name, C_O=co, S=S, species=s, X_obs=xo,
                                t_best_yr=f"{t[kb]:.2e}", D_control=round(float(D[kb]), 2),
                                X_mod_best=f"{xs[kb]:.2e}", ratio_best=f"{xs[kb]/xo:.2e}",
                                X_mod_peak=f"{xs[kp]:.2e}", t_peak_yr=f"{t[kp]:.2e}",
                                ratio_peak=f"{xs[kp]/xo:.2e}", ref=ref))
        for k in range(len(t)):
            series.append(dict(network=net_name, C_O=co, S=S, t_yr=f"{t[k]:.4e}",
                               **{s: f"{X[s][k]:.4e}" for s in X}))
        print(f"network {net_name} C/O={co} S={S:.0e}: {net.last_runtime:.1f} s, "
              f"best t={t[kb]:.1e} yr (D={D[kb]:.2f})")

    for name, rows in [("summary", summary), ("timeseries", series)]:
        with open(result(f"{name}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
