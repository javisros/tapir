"""
knockout_screen.py - Sensitivity screen for the knock-out benchmark (~6 min, resumable).

Reference configuration (tapir.knockout): kida.uva.2024, C/O = 1.2, S = 8e-8, t_ref = 6e6 yr.
For the 50 TMC-1 observed species (tapir.observations) that are most abundant in the model:
  1. formation budget at t_ref -> dominant formation reaction (CR/CRP/photon reactions are skipped:
     a generator would never propose them);
  2. knock-out of that reaction -> drop = X_base / X_ko at t_ref, at 1e5 yr and at the peak;
  3. collateral damage: how many other observed species change by more than 3x;
  4. if the drop is >= 10x: re-injection with a generic capture-type rate (not KIDA's) -> recovery.
Output: results/knockout_screen.csv (written row by row and resumed if interrupted)
usage: python -m scripts.knockout_screen [max_species_per_call]
"""
import csv, os, sys, time, numpy as np
from tapir.knockout import NET, EL, T_REF, T_EARLY, PHYSICAL, W24_GAS, PHYS, \
    label, reaction_class, capture_guess, is_row, run, X, peak
from tapir.observations import observed
from tapir.paths import result

OUT = result("knockout_screen.csv")
N_TOP, DROP_MIN = 50, 10.0

if __name__ == "__main__":
    t0 = time.time()
    net, t, y = run()
    obs = {s: v for s, v in observed().items() if s in net.idx}
    xb = {s: X(net, t, y, s, T_REF) for s in obs}
    top = sorted(obs, key=lambda s: -xb[s])[:N_TOP]
    k = np.argmin(abs(t - T_REF)); v = net.rates(y[:, k])
    done = set()
    if os.path.exists(OUT):
        done = {r["species"] for r in csv.DictReader(open(OUT))}
    budget_n = int(sys.argv[1]) if len(sys.argv) > 1 else len(top)
    todo = [s for s in top if s not in done][:budget_n]
    for i, s in enumerate(todo):
        col = net.S[net.idx[s]].toarray().ravel() * v
        form = [j for j in np.argsort(col)[::-1] if col[j] > 0 and j < len(net.rows)]
        tot = col[col > 0].sum()
        j = next(j for j in form if reaction_class(net.rows[j]) != PHYSICAL)
        r = net.rows[j]; frac = col[j] / tot
        try:
            nk, tk, yk = run(exclude=is_row(r))
            xko = X(nk, tk, yk, s, T_REF)
            drop = xb[s] / max(xko, 1e-40)
            drop_e = X(net, t, y, s, T_EARLY) / max(X(nk, tk, yk, s, T_EARLY), 1e-40)
            drop_p = peak(net, t, y, s) / max(peak(nk, tk, yk, s), 1e-40)
            coll = sum(abs(np.log10(max(X(nk, tk, yk, o, T_REF), 1e-40) / max(xb[o], 1e-40))) > np.log10(3)
                       for o in obs if o != s)
        except RuntimeError as e:
            print(s, "failed:", e); continue
        rec = ""
        cand = capture_guess(r)
        if drop >= DROP_MIN and cand is not None:
            nr, tr, yr = run(exclude=is_row(r), extra_reactions=[cand])
            rec = X(nr, tr, yr, s, T_REF) / xb[s]
        xo, ref = obs[s]
        n_reac = len(r["reac"]) - len(PHYS & set(r["reac"]))
        row = dict(species=s, X_model=f"{xb[s]:.2e}", X_obs="" if xo is None else f"{xo:.1e}",
                   model_over_obs="" if xo is None else f"{xb[s]/xo:.2g}", obs_source=ref,
                   hidden_reaction=label(r), reaction_class=reaction_class(r), kida_id=r["id"],
                   k_kida_10K=f"{net.k[j] / (W24_GAS['nH'] if n_reac == 2 else 1):.2e}",
                   formation_fraction=f"{frac:.2f}", drop_tref=f"{drop:.3g}", drop_1e5yr=f"{drop_e:.3g}",
                   drop_peak=f"{drop_p:.3g}", collateral_3x=coll,
                   capture_recovery="" if rec == "" else f"{rec:.2g}")
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
        print(f"[{len(done)+i+1:2d}/{len(top)}] {s:10s} X={xb[s]:.1e} {frac:4.0%} {label(r):40s} drop={drop:9.3g} "
              f"(1e5:{drop_e:.3g}, peak:{drop_p:.3g}) collat={coll} rec={rec if rec == '' else f'{rec:.2g}'}  [{time.time()-t0:.0f}s]", flush=True)
    print("pending:", len([s for s in top if s not in done]) - len(todo))
