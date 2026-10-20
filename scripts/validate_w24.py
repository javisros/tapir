"""
validate_w24.py - Validation of the model against Wakelam et al. (2024, A&A 689, A63), Table 2:
log10(X_2024 / X_2014) at 1e5 yr in the gas-phase-only test (T=10 K, n_H=2e4, zeta=1e-17, Av=15),
for species with X_2024 > 1e-12 (rel. to n_H) and |log ratio| > 0.3.
The elemental abundances of that test are not given in Sect. 2.3: ours are tried.
Output: results/validate_w24.csv  (~30 s)
usage: python -m scripts.validate_w24
"""
import csv, numpy as np
from tapir.darkcloud import Network, elements, W24_GAS
from tapir.paths import NET_2014 as N14, NET_2024 as N24, result

TABLE2 = {  # Wakelam+2024 Table 2 (KIDA names)
 "HCl":-0.5,"PN":-0.8,"SiC":0.5,"SO":-0.9,"CCS":-0.5,"CCO":0.7,"HNSi":0.7,"C3O":-0.8,
 "C3S":-2.3,"H2CO":-2.0,"NH3":-2.3,"HCCSi":-0.9,"l-C3H":-0.7,"C6H6":-0.6,"H2CCO":-2.7,
 "c-C3H2":-0.5,"HCNCC":-0.5,"HCCNC":-0.4,"HNCCC":-0.5,"l-C3H2":-2.0,"NC4N":-0.5,"C4H2":-0.4,
 "C5H":-0.5,"C5N":-0.6,"C6":-0.5,"C5O":-0.6,"C5H2":-0.6,"C6H":-0.7,"C7":-0.6,"HC5N":-0.6,
 "C6N":-0.6,"NC6N":-0.5,"C6H2":-0.8,"C7H":-0.7,"C7N":-0.8,"C8":-0.6,"C7O":-0.8,"C7H2":-0.8,
 "C8H":-0.7,"C9":-0.6,"HC7N":-1.0,"C8N":-0.8,"C8H2":-0.8,"C9H":-0.7,"C10":-0.7,"C9N":-0.8,
 "C10H":-0.7,"C11":-0.7,"C+":-0.3,"Na+":0.3,"S+":0.6,"Si+":1.5,"CF+":-0.4,"SO+":-0.5,
 "CN-":-0.6,"CHSi+":5.6,"HSiO+":-0.7,"H2CCO+":-1.1,"H2COH+":-1.8,"HC3O+":0.8,"HC3S+":-0.5,
 "l-C3H2+":-0.3,"C5-":-0.4,"CH3CO+":-0.7,"HC3NH+":-0.9,"C4H2+":-0.6,"C5H+":-0.4,
 "H2C3O+":3.1,"HC4O+":-0.6,"C5H-":-0.5,"C6-":-0.5,"C4H3+":-0.9,"C5H2+":-0.7,"C7-":-0.6,
 "C7H-":-0.7,"C8-":-0.6,"C9-":-0.6}
T_CMP = 1e5


def at(net, t, y, s): return y[net.idx[s], np.argmin(abs(t - T_CMP))] if s in net.idx else np.nan


if __name__ == "__main__":
    n14, n24 = Network(N14, **W24_GAS), Network(N24, **W24_GAS)
    rows = []
    for co, S in [(0.7, 8e-8), (0.7, 1.5e-5), (1.2, 8e-8)]:
        el = elements(co, S)
        t14, y14 = n14.run(el); t24, y24 = n24.run(el)
        common = sorted(set(n14.species) & set(n24.species))
        lr = {s: np.log10(max(at(n24, t24, y24, s), 1e-40) / max(at(n14, t14, y14, s), 1e-40)) for s in common}
        abund = {s: at(n24, t24, y24, s) for s in common}
        tab = [s for s in TABLE2 if s in lr]
        a = np.array([TABLE2[s] for s in tab]); b = np.array([lr[s] for s in tab])
        sign = np.mean(np.sign(a) == np.sign(b)); within = np.mean(abs(a - b) <= 0.5)
        r = np.corrcoef(a, b)[0, 1]
        # abundant species NOT in the table: they should have |lr| < 0.3
        others = [s for s in common if abund[s] > 1e-12 and s not in TABLE2 and s not in ("GRAIN0", "GRAIN-")]
        fp = [s for s in others if abs(lr[s]) > 0.3]
        print(f"C/O={co} S={S:.0e}: {len(tab)} species of Table 2 | same sign {sign:.0%} | "
              f"|diff|<=0.5 dex {within:.0%} | r={r:.2f} | off-table with |lr|>0.3: {len(fp)}/{len(others)}")
        for s in tab:
            rows.append(dict(C_O=co, S=S, species=s, w24=TABLE2[s], model=round(lr[s], 2)))
        if (co, S) == (0.7, 8e-8):
            worst = sorted(tab, key=lambda s: -abs(TABLE2[s] - lr[s]))[:8]
            print("   worst:", ", ".join(f"{s} ({TABLE2[s]:+.1f} vs {lr[s]:+.1f})" for s in worst))
            print("   off-table with largest |lr|:", ", ".join(f"{s} {lr[s]:+.1f}" for s in sorted(fp, key=lambda s: -abs(lr[s]))[:8]))
    with open(result("validate_w24.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
