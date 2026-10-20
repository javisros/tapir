"""
build_cases.py - The 6 knock-out benchmark cases -> benchmark/cases.json

Configuration: kida.uva.2024, C/O = 1.2, S = 8e-8, W24_GAS, t_ref = 6e6 yr (tapir.knockout).
For each case: hidden reaction (from results/knockout_screen.csv), KIDA id, canonical key,
equivalents in kida.uva.2014 (strict and channel levels) and in the raw 2024 file (strict).
One base run and 6 knock-outs; X_base and X_ko are checked against knockout_screen.csv (1 % tol.).
The objective of each case is X_base of the original model, not the TMC-1 observation.
usage: python -m scripts.build_cases  (~20 s)
"""
import csv, json, os, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows, W24_GAS
from tapir.knockout import NET, EL, T_REF, run, X, is_row, label, reaction_class
from tapir.paths import CASES as OUT, relative, result

CASES = ("CH2CHCN", "CH3CN", "HNCCC", "HCCCHO", "H2CS", "N2H+")
SCREEN = result("knockout_screen.csv")
TOL = 0.01


def brief(r):
    return dict(id=r["id"], reaction=label(r))


def hidden_row(rows, rid, lab):
    hits = [r for r in rows if r["id"] == rid and label(r) == lab]
    if len(hits) != 1:
        raise RuntimeError(f"{len(hits)} rows for id {rid} '{lab}'")
    return hits[0]


if __name__ == "__main__":
    t0 = time.time()
    screen = {r["species"]: r for r in csv.DictReader(open(SCREEN))}
    rows24 = load_rows(NET)                       # network the model simulates (dedup, one T range)
    raw24, raw14 = parse_kida(NET), parse_kida(chem.REACTIONS_2014)

    net, t, y = run()
    cases, all_ok = [], True
    for s in CASES:
        sc = screen[s]
        r = hidden_row(rows24, int(sc["kida_id"]), sc["hidden_reaction"])
        assert chem.conserves(r)
        strict14 = chem.find_equivalents(r, raw14, "strict")
        channel14 = [q for q in chem.find_equivalents(r, raw14, "channel", s) if q not in strict14]
        strict24 = chem.find_equivalents(r, raw24, "strict")
        in_model = chem.find_equivalents(r, rows24, "strict")
        if len(in_model) != 1:
            raise RuntimeError(f"{s}: {len(in_model)} strict equivalents in the simulated network")

        nk, tk, yk = run(exclude=is_row(r))
        xb, xko = X(net, t, y, s, T_REF), X(nk, tk, yk, s, T_REF)
        xb_csv, drop_csv = float(sc["X_model"]), float(sc["drop_tref"])
        err_b, err_d = abs(xb / xb_csv - 1), abs((xb / xko) / drop_csv - 1)
        ok = bool(err_b <= TOL and err_d <= TOL)
        all_ok &= ok

        cases.append(dict(
            species=s,
            hidden_reaction=dict(reaction=label(r), kida_id=r["id"], reac=r["reac"], prod=r["prod"],
                                 reaction_class=reaction_class(r), A=r["A"], B=r["B"], C=r["C"],
                                 itype=r["itype"], formula=r["formula"]),
            canonical_key=[list(k) for k in chem.canonical_key(r)],
            equivalents_2014=dict(strict=[brief(q) for q in strict14],
                                  channel=[brief(q) for q in channel14]),
            equivalents_2024_file=dict(strict=sorted({q["id"] for q in strict24})),
            X_base=float(xb), X_ko=float(xko), drop=float(xb / xko),
            check=dict(X_model_csv=xb_csv, drop_csv=drop_csv, rel_err_X_base=float(err_b),
                       rel_err_drop=float(err_d), ok=ok)))
        print(f"{s:8s} {label(r):30s} id={r['id']:5d} 2014:{len(strict14)}s/{len(channel14)}c  "
              f"Xb={xb:.3e} (csv {xb_csv:.2e}) drop={xb/xko:.3g} (csv {drop_csv:.3g})  "
              f"{'OK' if ok else 'MISMATCH'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(
        configuration=dict(network=relative(NET), C_O=1.2, S=8e-8, elements=EL, physics=W24_GAS, t_ref_yr=T_REF,
                           X="2 * x_i = n_i / n(H2) at t_ref (as in tapir.knockout.X)",
                           objective="X_base of the original model, not the TMC-1 observation",
                           knockout="only the hidden row (id, reactants, products) is removed from load_rows",
                           check_tolerance=TOL),
        cases=cases), open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"-> {relative(OUT)}  all within {TOL:.0%}: {all_ok}  [{time.time()-t0:.0f}s]")
