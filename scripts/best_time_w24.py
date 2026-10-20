"""
best_time_w24.py - "Best time" with the disagreement distance D(t) of Wakelam+2024 (Eq. 3)
against the Gratier+2016 set (31 species with a value), in the 12 configurations.
Compared with the earlier criterion (the 8 control species of scripts/run_targets.py).
Output: results/best_time_w24.csv  (~1 min)
usage: python -m scripts.best_time_w24
"""
import itertools, csv, numpy as np
from tapir.darkcloud import Network, elements
from tapir.observations import GRATIER, CONTROL
from tapir.paths import NETWORKS, result


def D_of_t(net, y, obs):
    sp = [s for s in obs if s in net.idx]
    return np.mean([abs(np.log10(np.maximum(2 * y[net.idx[s]], 1e-30)) - np.log10(obs[s])) for s in sp], axis=0), len(sp)


if __name__ == "__main__":
    rows = []
    for (nn, path), co, S in itertools.product(NETWORKS.items(), [0.7, 1.2], [8e-8, 1.5e-6, 1.5e-5]):
        net = Network(path); t, y = net.run(elements(co, S))
        w = (t >= 1e4) & (t <= 1e7)
        Dg, ng = D_of_t(net, y, GRATIER); Dc, _ = D_of_t(net, y, CONTROL)
        Dg[~w] = np.inf; Dc[~w] = np.inf
        kg, kc = np.argmin(Dg), np.argmin(Dc)
        f10 = np.mean([abs(np.log10(max(2 * y[net.idx[s], kg], 1e-30) / GRATIER[s])) <= 1 for s in GRATIER if s in net.idx])
        rows.append(dict(network=nn, C_O=co, S=S, n_species=ng, t_best_gratier=f"{t[kg]:.1e}",
                         D_gratier=round(float(Dg[kg]), 2), frac_within_10x=round(float(f10), 2),
                         t_best_control=f"{t[kc]:.1e}", D_control=round(float(Dc[kc]), 2)))
        print(rows[-1])
    with open(result("best_time_w24.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
