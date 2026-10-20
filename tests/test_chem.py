"""Tests for tapir.chem. Usage: pytest  (or, without pytest: python -m tests.test_chem)"""
from tests._compat import pytest, main
from tapir import chem
from tapir.darkcloud import parse_kida

NETS = {"2014": chem.REACTIONS_2014, "2024": chem.REACTIONS_2024}
_ROWS = {}


def rows(year):
    if year not in _ROWS:
        _ROWS[year] = parse_kida(NETS[year])
    return _ROWS[year]


def R(reac, prod, id=-1):
    return dict(reac=list(reac), prod=list(prod), id=id)


# --- composition ------------------------------------------------------------------------

def test_column_order():
    assert chem.atoms("H2CS") == {"H": 2, "C": 1, "S": 1}
    assert chem.atoms("CH3CN") == {"H": 3, "C": 2, "N": 1}
    assert chem.atoms("SiO") == {"O": 1, "Si": 1}
    assert chem.atoms("NaH") == {"H": 1, "Na": 1}
    assert chem.atoms("PN") == {"N": 1, "P": 1}
    assert chem.atoms("HF") == {"H": 1, "F": 1}
    for el in ("He", "Fe", "Mg", "Cl"):
        assert chem.atoms(el) == {el: 1}
    assert chem.charge("N2H+") == 1 and chem.atoms("N2H+") == {"H": 1, "N": 2}
    assert chem.charge("e-") == -1 and chem.atoms("e-") == {}
    assert chem.charge("GRAIN-") == -1


def test_missing_in_2024():
    s24 = chem.read_species(chem.SPECIES_2024)
    s14 = chem.read_species(chem.SPECIES_2014)
    assert sorted(set(s14) - set(s24)) == sorted(chem.MISSING_IN_2024)
    assert all(s in chem.COMPOSITION for s in chem.MISSING_IN_2024)


@pytest.mark.parametrize("year", NETS)
def test_all_species_known(year):
    used = {s for r in rows(year) for s in r["reac"] + r["prod"]}
    assert sorted(used - set(chem.COMPOSITION)) == []


# --- conservation -----------------------------------------------------------------------

@pytest.mark.parametrize("year", NETS)
def test_conservation_all_rows(year):
    bad = [(r["id"], " + ".join(r["reac"]) + " -> " + " + ".join(r["prod"]), chem.imbalance(r))
           for r in rows(year) if not chem.conserves(r)]
    assert bad == [], f"{len(bad)} rows do not conserve in {year}: {bad[:20]}"


def test_conserves_detects_violation():
    assert chem.conserves(R(["N2", "H3+"], ["H2", "N2H+"]))
    assert not chem.conserves(R(["N2", "H3+"], ["H2", "N2H"]))           # charge
    assert not chem.conserves(R(["CN", "C2H4"], ["CH2CHCN"]))            # missing H
    assert not chem.conserves(R(["CN", "C2H4"], ["H", "NOSUCH"]))        # unknown species
    assert chem.imbalance(R(["CN", "C2H4"], ["CH2CHCN"])) == {"H": -1}


# --- canonical key and equivalence --------------------------------------------------------------

def test_key_permutation_invariant():
    a = R(["CN", "C2H4"], ["H", "CH2CHCN"])
    b = R(["C2H4", "CN"], ["CH2CHCN", "H"])
    assert chem.canonical_key(a) == chem.canonical_key(b)
    assert chem.equivalent(a, b, "strict")


def test_key_keeps_multiplicity():
    a = R(["HCNH+", "e-"], ["H", "H", "CN"])
    b = R(["HCNH+", "e-"], ["H", "CN"])
    assert chem.canonical_key(a) != chem.canonical_key(b)


def test_photon_distinguishes_channels():
    rad = R(["H", "CH2CCH"], ["CH3CCH", "Photon"])
    bare = R(["H", "CH2CCH"], ["CH3CCH"])
    assert chem.canonical_key(rad) != chem.canonical_key(bare)
    assert not chem.equivalent(rad, bare, "strict")
    assert chem.equivalent(rad, bare, "channel", target="CH3CCH")


def test_channel_level():
    a = R(["CN", "C2H4"], ["H", "CH2CHCN"])
    b = R(["CN", "C2H4"], ["H2", "CH2CCN"])
    c = R(["CN", "C2H2"], ["H", "HC3N"])
    assert not chem.equivalent(a, b, "strict")
    assert not chem.equivalent(a, b, "channel", target="CH2CHCN")   # b does not yield the target
    assert chem.equivalent(a, b, "channel", target="CN") is False  # CN is not a product
    assert not chem.equivalent(a, c, "channel", target="H")         # different reactants
    d = R(["C2H4", "CN"], ["CH2CHCN", "H", "Photon"])
    assert chem.equivalent(a, d, "channel", target="CH2CHCN")
    with pytest.raises(ValueError):
        chem.equivalent(a, b, "channel")


def test_duplicates_synthetic():
    rs = [R(["A", "B"], ["C", "D"], 1), R(["B", "A"], ["D", "C"], 2),   # true duplicate
          R(["E", "F"], ["G"], 3), R(["E", "F"], ["G"], 3),             # same id: T ranges
          R(["E", "F"], ["G", "Photon"], 4)]                            # another channel
    assert chem.duplicates(rs) == {(("A", "B"), ("C", "D")): [1, 2]}


@pytest.mark.parametrize("year,n", [("2014", 7), ("2024", 21)])
def test_duplicates_in_networks(year, n):
    # README: 7 duplicated reactions in 2014 and 21 in 2024 (same criterion as dedup_latest)
    assert len(chem.duplicates(rows(year))) == n


if __name__ == "__main__":
    main(__file__, globals())
