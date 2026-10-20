"""
knockout_top2.py - Borderline cases (drop 3-10x): knock-out of the TWO main formation reactions at
once, in the reference configuration. Prints to stdout (~1 min).
usage: python -m scripts.knockout_top2
"""
import csv, numpy as np
from tapir.knockout import T_REF, PHYSICAL, label, reaction_class, run, X
from tapir.paths import result

if __name__ == "__main__":
    net, t, y = run(); k = np.argmin(abs(t - T_REF)); v = net.rates(y[:, k])
    for r in csv.DictReader(open(result("knockout_screen.csv"))):
        if not 3 <= float(r["drop_tref"]) < 10:
            continue
        s = r["species"]; col = net.S[net.idx[s]].toarray().ravel() * v
        form = [j for j in np.argsort(col)[::-1]
                if col[j] > 0 and j < len(net.rows) and reaction_class(net.rows[j]) != PHYSICAL][:2]
        keys = {(net.rows[j]["id"], label(net.rows[j])) for j in form}
        nk, tk, yk = run(exclude=lambda q: (q["id"], label(q)) in keys)
        print(f"{s:8s} top2={col[form].sum()/col[col>0].sum():.0%} "
              f"drop={X(net, t, y, s, T_REF)/max(X(nk, tk, yk, s, T_REF), 1e-40):.3g}  ",
              " | ".join(label(net.rows[j]) for j in form), flush=True)
