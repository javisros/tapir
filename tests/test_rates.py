"""Tests for tapir.rates: class medians, leak-proofing, reactive species and consistency with KIDA.
Usage: pytest  (or, without pytest: python -m tests.test_rates)"""
from tests._compat import pytest, main
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, label
from tapir.generator import Generator, load_cases
from tapir.flux import kinetic_row
from tapir.rates import (RateEstimator, REACTIVE_EVEN_ELECTRON, CAPTURE_MAX, NEUTRAL_H2, CLOSED_SHELL,
                         KIDA_MEDIAN, UNPAIRED_INCREASE, INCREASE, UNPAIRED_FLAG_ONLY, EVIDENCE_K,
                         electrons, k10, unpaired, unpaired_flag, unpaired_validation)

ROWS24 = load_rows(NET)
RAW14 = parse_kida(chem.REACTIONS_2014)
EST = RateEstimator(ROWS24)


def row(lab):
    """The kida.uva.2024 row with this label (reactants and products in file order)."""
    return next(r for r in ROWS24 if label(r) == lab)


def rx(lab):
    """'A + B -> C + D' -> candidate dict"""
    reac, prod = lab.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


def keys(est, c):
    return {k for k, _ in est.members[c]}


def evidence(est):
    return {s: set(k) for s, k in est.evidence.items()}


def flipped(r):
    """Same reaction with reactants and products listed in another order: a strict equivalent."""
    return dict(reac=r["reac"][::-1], prod=r["prod"][::-1])


# --- class medians -----------------------------------------------------------------------------

def test_median_neutral_h2():
    assert len(EST.members[NEUTRAL_H2]) >= 10
    assert EST.median[NEUTRAL_H2] < 1e-12


def test_medians_below_capture():
    assert all(m <= CAPTURE_MAX["neutral-neutral"] for m in EST.median.values())


@pytest.mark.parametrize("c", (NEUTRAL_H2, CLOSED_SHELL))
def test_estimate_uses_median(c):
    r = next(r for r in ROWS24 if EST.kclass(r) == c and not unpaired_flag(r))
    est_row, src = EST.estimate(dict(reac=r["reac"], prod=r["prod"]))
    assert src == KIDA_MEDIAN
    assert k10(est_row) == EST.median[c]


# --- leak-proofing -----------------------------------------------------------------------------

@pytest.mark.parametrize("lab, c", (("CN + H2 -> H + HCN", NEUTRAL_H2),
                                    ("CO + HNO -> NH + CO2", CLOSED_SHELL)))
def test_hiding_class_member(lab, c):
    """Hiding a member of a class removes exactly its equivalents from the median set."""
    h = row(lab)
    est = RateEstimator(ROWS24, hidden=[flipped(h)])
    assert chem.canonical_key(h) in keys(EST, c)
    assert keys(est, c) == keys(EST, c) - {chem.canonical_key(h)}
    other = NEUTRAL_H2 if c == CLOSED_SHELL else CLOSED_SHELL
    assert keys(est, other) == keys(EST, other)
    assert evidence(est) == evidence(EST)


def test_hiding_evidence():
    """Hiding an evidence reaction (iii) removes exactly its equivalents from the evidence."""
    k = sorted(EST.evidence["S"])[0]
    h = flipped(next(r for r in ROWS24 if chem.canonical_key(r) == k))
    est = RateEstimator(ROWS24, hidden=[h])
    expected = {s: v - {k} for s, v in evidence(EST).items()}
    assert evidence(est) == {s: v for s, v in expected.items() if v}
    assert all(keys(est, c) == keys(EST, c) for c in (NEUTRAL_H2, CLOSED_SHELL))


def test_hiding_evidence_threshold():
    """A species with exactly the minimum evidence loses it when one of its reactions is hidden."""
    s = next(s for s in sorted(EST.by_evidence) if len(EST.evidence[s]) == 2)
    k = sorted(EST.evidence[s])[0]
    est = RateEstimator(ROWS24, hidden=[next(r for r in ROWS24 if chem.canonical_key(r) == k)])
    assert s not in est.by_evidence


def test_hiding_absent_reaction_changes_nothing():
    est = RateEstimator(ROWS24, hidden=[dict(reac=["Mg", "HCl"], prod=["Mg", "H", "Cl"])])
    assert est.median == EST.median and evidence(est) == evidence(EST)


# --- reactive species --------------------------------------------------------------------------

@pytest.mark.parametrize("s", ("C", "O", "S"))
def test_atoms_reactive(s):
    assert EST.reactive(s)


def test_c_plus_c2h4_not_closed_shell():
    c = dict(reac=["C", "C2H4"], prod=["H", "CH2CCH"])
    assert EST.kclass(c) == "neutral-neutral"
    assert EST.estimate(c) == (kinetic_row(c), "capture_guess")


def test_table_species():
    """Table entries are even-electron neutrals; only NCN and HCCN are absent from kida.uva.2024."""
    absent = {s for s in REACTIVE_EVEN_ELECTRON if s not in chem.COMPOSITION}
    assert absent == {"NCN", "HCCN"}
    for s in set(REACTIVE_EVEN_ELECTRON) - absent:
        assert chem.charge(s) == 0 and electrons(s) % 2 == 0, s


# --- other classes unchanged -------------------------------------------------------------------

@pytest.mark.parametrize("target", ("CH2CCH", "HNCCC", "N2H+", "H2CS"))
def test_other_classes_use_capture_guess(target):
    cands = Generator(ROWS24, RAW14).candidates(target)
    others = [c for c in cands if EST.kclass(c) not in EST.median
              and (not unpaired_flag(c) or EST.kclass(c) in UNPAIRED_FLAG_ONLY)]
    assert others
    for c in others:
        r = kinetic_row(c)
        assert EST.estimate(c) == (r, None if r is None else "capture_guess")


# --- unpaired electrons ------------------------------------------------------------------------

def test_unpaired_counts():
    assert [unpaired(s) for s in ("H2", "CO", "H3+", "e-", "H", "CH3", "C", "CH2", "N", "P")] == \
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("lab, flagged", (("c-C3H2 + CH3 -> CH2CCH + CH2", True),
                                          ("c-C3H2 + H3+ -> CH2CCH + H2+", True),
                                          ("CN + C2H4 -> H + CH2CHCN", False),
                                          ("O + CH2CCH2 -> OH + CH2CCH", False),
                                          ("S + CH3 -> H + H2CS", False)))
def test_unpaired_flag(lab, flagged):
    assert (unpaired_flag(rx(lab)) == INCREASE) == flagged


@pytest.mark.parametrize("name", [n for n, _ in load_cases()])
def test_hidden_reactions_not_flagged(name):
    assert unpaired_flag(dict(load_cases())[name]) is None


def test_unpaired_exclusion_neutral_only():
    """Flagged neutral-neutral -> k = 0; flagged ion-neutral keeps its rate (flag only)."""
    row, src = EST.estimate(rx("c-C3H2 + CH3 -> CH2CCH + CH2"))
    assert src == UNPAIRED_INCREASE and k10(row) == 0
    c = rx("c-C3H2 + H3+ -> CH2CCH + H2+")
    assert EST.estimate(c) == (kinetic_row(c), "capture_guess")


def test_unpaired_validation():
    """The decision in tapir.rates: exceptions concentrate in ionic classes; the rest stays below 2 %."""
    v = unpaired_validation(ROWS24)
    n_rest = sum(n for c, (n, _) in v.items() if c not in UNPAIRED_FLAG_ONLY)
    e_rest = sum(len(e) for c, (_, e) in v.items() if c not in UNPAIRED_FLAG_ONLY)
    e_ion = sum(len(e) for c, (_, e) in v.items() if c in UNPAIRED_FLAG_ONLY)
    assert n_rest > 500 and e_rest / n_rest < 0.02
    assert e_ion / (e_ion + e_rest) > 0.95
    assert (e_ion + e_rest) / sum(n for n, _ in v.values()) > 0.02    # the filter could not apply to all


# --- consistency with KIDA ---------------------------------------------------------------------

def test_closed_shell_consistency():
    """No kida.uva.2024 reaction classified closed-shell is fast at 10 K."""
    fast = [f"{label(r)} k={k10(r):.1e}" for r in ROWS24
            if EST.kclass(r) == CLOSED_SHELL and k10(r) > EVIDENCE_K]
    assert fast == []


if __name__ == "__main__":
    main(__file__, globals())
