"""Tests for tapir.generator: leak-proofing, canary and positive control.
Usage: pytest  (or, without pytest: python -m tests.test_generator)"""
from tests._compat import pytest, main
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, label
from tapir.generator import (Generator, rank, load_cases, open_or_unsaturated, NONSPECIES,
                                 HP, H, E, RADICALS, vec, add, sub)

ROWS24 = load_rows(NET)
RAW14 = parse_kida(chem.REACTIONS_2014)
CASES = load_cases()
NAMES = [c[0] for c in CASES]
# Canary: an invented reaction that conserves and uses network species, but that no family can
# derive: every family yields two-product channels except OS, and it has no O or S to swap.
CANARY = dict(reac=["Mg", "HCl"], prod=["Mg", "H", "Cl"], A=1e-10, B=0.0, C=0.0, itype=4,
              tmin=10.0, tmax=300.0, formula=3, id=-999, Teff=10.0)


def case(name):
    return dict(CASES)[name]


def snapshot(cands):
    return [(chem.canonical_key(c), c["score"]) for c in cands]


# --- leak-proofing ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_hidden_absent_from_sources(name):
    h = case(name)
    g = Generator(ROWS24, RAW14, hidden=[h])
    assert not any(chem.equivalent(r, h) for r in g.source)
    assert not any(chem.equivalent(r, h) for r in g.net)
    assert chem.canonical_key(h) not in g.net_keys
    # and it was in the sources before hiding (otherwise the test proves nothing)
    assert any(chem.equivalent(r, h) for r in ROWS24)


@pytest.mark.parametrize("name", NAMES)
def test_hiding_equals_removing(name):
    """Hiding == deleting every strict equivalent from 2014 and 2024 by hand: nothing else leaks."""
    h = case(name)
    k = chem.canonical_key(h)
    g1 = Generator(ROWS24, RAW14, hidden=[h])
    g2 = Generator([r for r in ROWS24 if chem.canonical_key(r) != k],
                   [r for r in RAW14 if chem.canonical_key(r) != k])
    assert snapshot(g1.candidates(name)) == snapshot(g2.candidates(name))


@pytest.mark.parametrize("name", NAMES)
def test_candidates_valid(name):
    g = Generator(ROWS24, RAW14, hidden=[case(name)])
    cs = g.candidates(name)
    assert cs
    for c in cs:
        assert chem.conserves(c)
        assert name in c["prod"]
        assert all(s in g.species or s in NONSPECIES for s in c["reac"] + c["prod"])  # closed world
        assert chem.canonical_key(c) not in g.net_keys                                # novelty
        assert not set(c["reac"]) & set(chem.PLACEHOLDERS)                            # no CR/photons


# --- canary ------------------------------------------------------------------------------------

def test_canary_is_valid():
    g = Generator(ROWS24)
    assert chem.conserves(CANARY)
    assert all(s in g.species for s in CANARY["reac"] + CANARY["prod"])
    assert not any(chem.equivalent(r, CANARY) for r in ROWS24 + RAW14)


def test_canary_not_recoverable():
    net = ROWS24 + [CANARY]
    for g in (Generator(net, RAW14, hidden=[CANARY]),     # hidden, as in the benchmark
              Generator(net, RAW14, novelty=False)):      # visible and unfiltered: no copying
        for tgt in ("Mg", "H", "Cl"):
            assert rank(g.candidates(tgt), CANARY) is None


# --- positive control ----------------------------------------------------------------------------

def test_positive_control_cases():
    """Nothing hidden and no novelty filter: the 6 hidden reactions are generated."""
    g = Generator(ROWS24, RAW14, novelty=False)
    missing = [n for n, h in CASES if rank(g.candidates(n), h) is None]
    assert missing == []


def test_leave_one_out_cases():
    """With each one hidden, all 6 are still generated (the analogy comes from other reactions)."""
    missing = [n for n, h in CASES if rank(Generator(ROWS24, RAW14, hidden=[h]).candidates(n), h) is None]
    assert missing == []


def _heavier(a, b):
    return a if sum(chem.atoms(a).values()) >= sum(chem.atoms(b).values()) else b


def _instance(r):
    """Classifier independent of the generator: PT, DR, AE, CT or HA by composition -> (family, target)."""
    reac, prod = r["reac"], r["prod"]
    if len(reac) != 2 or len(prod) != 2 or set(reac) & set(chem.PLACEHOLDERS):
        return None
    if any(s.startswith("GRAIN") for s in reac + prod):
        return None
    if "e-" in reac:
        ion = [s for s in reac if s != "e-"][0]; X = [s for s in prod if s != "H"] or ["H"]
        if chem.charge(ion) != 1 or "H" not in prod:
            return None
        return ("DR", X[0]) if vec(X[0]) == sub(add(vec(ion), E), H) else None
    for X, D in (reac, reac[::-1]):
        for XH, A in (prod, prod[::-1]):
            if chem.charge(D) == 1 and chem.charge(X) == 0 and vec(XH) == add(vec(X), HP) \
                    and vec(A) == sub(vec(D), HP):
                return "PT", _heavier(XH, A)
            if chem.charge(D) == 1 and chem.charge(X) == 0 and vec(XH) == add(vec(D), E) \
                    and vec(A) == sub(vec(X), E):
                return "CT", _heavier(XH, A)
            if vec(XH) == add(vec(X), H) and vec(A) == sub(vec(D), H):
                return "HA", _heavier(XH, A)
    if "H" in prod:
        P = [s for s in prod if s != "H"] or ["H"]
        for R, M in (reac, reac[::-1]):
            if R in RADICALS and chem.charge(M) == 0 and vec(P[0]) == sub(add(vec(R), vec(M)), H):
                return ("AE", P[0]) if open_or_unsaturated(M) else None
    return None


def test_positive_control_all_instances():
    """Every network reaction that is a PT, DR, AE, CT or HA instance is regenerated with nothing hidden."""
    g = Generator(ROWS24, RAW14, novelty=False)
    inst = [(r, _instance(r)) for r in ROWS24]
    inst = [(r, i) for r, i in inst if i is not None]
    assert len(inst) > 1000
    cache, missing = {}, []
    for r, (fam, tgt) in inst:
        if tgt not in cache:
            cache[tgt] = g.candidates(tgt)
        if rank(cache[tgt], r) is None:
            missing.append((fam, label(r)))
    assert missing == [], f"{len(missing)} of {len(inst)} not recovered: {missing[:10]}"


if __name__ == "__main__":
    main(__file__, globals())
