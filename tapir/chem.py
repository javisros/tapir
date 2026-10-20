"""
chem.py - Elemental composition, canonical key and reaction equivalence for kida.uva networks.

Composition: gas_species_kida.uva.2024.in (format a10 + 14 integers, see kida.uva.2024_readme.txt):
  charge, H, He, C, N, O, Si, S, Fe, Na, Mg, Cl, P, F
Column order checked against C, H, He, Fe, Cl, Mg, Na, P, Si, F, H2CS, HF, NaH, PN, SiO, H3+ and e-.
The 14 kida.uva.2014 species missing from the 2024 file (MISSING_IN_2024) are taken from the 2014
gas_species.dat, which has the same format; species present in both files agree.

Reaction equivalence (two levels):
  "strict"  : same multiset of reactants and of products, Photon included.
  "channel" : same reactants and the target species among the products of both.
"""
from collections import Counter
from .paths import SPECIES_2014, SPECIES_2024, NET_2014 as REACTIONS_2014, NET_2024 as REACTIONS_2024

ELEMENTS = ("H", "He", "C", "N", "O", "Si", "S", "Fe", "Na", "Mg", "Cl", "P", "F")
PLACEHOLDERS = ("CR", "CRP", "Photon")   # no atoms, no charge

# kida.uva.2014 species absent from the 2024 species file (checked against read_species).
MISSING_IN_2024 = ("CH2CHC2H", "H2C10N+", "H2C5N+", "H2C6N+", "H2C8N+", "H2CSiCH", "H3C3O+",
                   "H3CS+", "PNH3+", "SiC2CH3", "SiC3H5", "SiC4H", "SiC6H", "SiC8H")


def read_species(path):
    """name -> (charge, H, He, C, N, O, Si, S, Fe, Na, Mg, Cl, P, F)"""
    out = {}
    for line in open(path):
        t = line.split()
        if not t:
            continue
        vals = tuple(int(v) for v in t[1:])
        if len(vals) != 1 + len(ELEMENTS):
            raise ValueError(f"{path}: {len(vals)} columns for {t[0]!r}")
        out[t[0]] = vals
    return out


def _load_composition():
    s24, s14 = read_species(SPECIES_2024), read_species(SPECIES_2014)
    missing = tuple(sorted(set(s14) - set(s24)))
    if missing != tuple(sorted(MISSING_IN_2024)):
        raise RuntimeError(f"MISSING_IN_2024 is out of date: {missing}")
    clash = [s for s in set(s14) & set(s24) if s14[s] != s24[s]]
    if clash:
        raise RuntimeError(f"composition differs between 2014 and 2024: {clash}")
    comp = dict(s24)
    comp.update({s: s14[s] for s in missing})
    comp.update({p: (0,) * (1 + len(ELEMENTS)) for p in PLACEHOLDERS})
    return comp


COMPOSITION = _load_composition()


def charge(species):
    return COMPOSITION[species][0]


def atoms(species):
    """dict element -> number of atoms (non-zero only)"""
    return {e: n for e, n in zip(ELEMENTS, COMPOSITION[species][1:]) if n}


def imbalance(row):
    """Products minus reactants for charge and each element, non-zero entries only.
    KeyError if a species has no known composition."""
    tot = [0] * (1 + len(ELEMENTS))
    for sign, side in ((-1, row["reac"]), (1, row["prod"])):
        for s in side:
            for i, n in enumerate(COMPOSITION[s]):
                tot[i] += sign * n
    names = ("charge",) + ELEMENTS
    return {k: v for k, v in zip(names, tot) if v}


def conserves(row):
    """True if the reaction conserves atoms and charge; False otherwise or if a species is unknown."""
    try:
        return not imbalance(row)
    except KeyError:
        return False


def canonical_key(row):
    """Sorted multisets of reactants and products (Photon, CR and CRP included).
    Same criterion as darkcloud.dedup_latest."""
    return (tuple(sorted(row["reac"])), tuple(sorted(row["prod"])))


def equivalent(r1, r2, level="strict", target=None):
    """level="strict": same canonical key.
    level="channel": same reactants and `target` among the products of both (target required)."""
    if level == "strict":
        return canonical_key(r1) == canonical_key(r2)
    if level == "channel":
        if target is None:
            raise ValueError("level='channel' needs a target")
        return (Counter(r1["reac"]) == Counter(r2["reac"])
                and target in r1["prod"] and target in r2["prod"])
    raise ValueError(f"unknown level: {level!r}")


def find_equivalents(row, rows, level="strict", target=None):
    """Rows of `rows` equivalent to `row` at the given level."""
    return [r for r in rows if equivalent(row, r, level, target)]


def duplicates(rows):
    """Reactions listed under more than one KIDA number (same canonical key, different ids).
    Repeated rows with the same id (several temperature ranges) do not count.
    Returns {key: [sorted ids]}."""
    ids = {}
    for r in rows:
        ids.setdefault(canonical_key(r), set()).add(r["id"])
    return {k: sorted(v) for k, v in ids.items() if len(v) > 1}
