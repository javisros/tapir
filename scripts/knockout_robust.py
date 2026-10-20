"""
knockout_robust.py - Repeats the knock-out of the species that pass the screen (drop >= 6x) in a
second configuration (the one with the lowest D against Gratier+2016: kida.uva.2024, C/O=1.2,
S=1.5e-5, t=4.5e5 yr) and at 1e6 yr in the reference one (~2 min).
Output: results/knockout_robust.csv
usage: python -m scripts.knockout_robust
"""
import csv
from tapir.darkcloud import Network, elements, W24_GAS
from tapir.knockout import NET, label, X
from tapir.paths import result

CONFIGS = {"ref_1e6yr": (elements(1.2, 8e-8), 1e6), "S1.5e-5_4.5e5yr": (elements(1.2, 1.5e-5), 4.5e5)}

if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(result("knockout_screen.csv"))) if float(r["drop_tref"]) >= 6]
    base = {}
    for name, (el, tt) in CONFIGS.items():
        n = Network(NET, **W24_GAS); t, y = n.run(el); base[name] = (n, t, y)
    out = []
    for r in rows:
        s = r["species"]; rid = int(r["kida_id"])
        excl = lambda q, rid=rid, lab=r["hidden_reaction"]: q["id"] == rid and label(q) == lab
        res = dict(species=s, reaction=r["hidden_reaction"], drop_ref_6e6yr=r["drop_tref"])
        for name, (el, tt) in CONFIGS.items():
            n, t, y = base[name]
            nk = Network(NET, **W24_GAS, exclude=excl); tk, yk = nk.run(el)
            res[f"drop_{name}"] = f"{X(n, t, y, s, tt) / max(X(nk, tk, yk, s, tt), 1e-40):.3g}"
        out.append(res); print(res, flush=True)
    with open(result("knockout_robust.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
