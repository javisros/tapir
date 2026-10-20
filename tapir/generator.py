"""
generator.py - Analogy-based generator of candidate reactions, directed at a target species.

Template sources: the base network (kida.uva.2024 as simulated by darkcloud.load_rows) + kida.uva.2014,
merged by canonical key, MINUS the hidden set (the hidden reaction and every strict equivalent, in
2014 or 2024). Leak-proof by construction: the hidden set is removed before anything is learned.

Closed world: reactants and products of a candidate are species of the base network after hiding.
Novelty: candidates already in the base network after hiding (strict level) are discarded.
Atom and charge conservation: chem.conserves. Reactions driven by CR/CRP/photons are never proposed.

Template families (instances are recognised by composition, not by name):
  PT  proton transfer           X + AH+ -> XH+ + A         template = proton donor AH+ (H3+, HCO+...)
  DR  dissociative recomb.      XH+ + e- -> X + H          template = H loss
  AE  addition-H elimination    R + M -> P + H, R in {CN, CCH, O}, M neutral unsaturated or radical
                                                           template = radical R
  OS  isovalent substitution    O->S or S->O image of a source reaction
                                                           template = (direction, class of the image)
  CT  charge transfer           X+ + Y -> X + Y+           template = ion X+
  HA  H-atom abstraction        A + BH -> AH + B           template = abstractor A
PT, DR, AE and OS are the agreed families (BASE_FAMILIES); CT and HA extend coverage to the two
largest classes they miss (scripts/benchmark_a.py compares both sets).

Prior score = template support: log10(1 + number of instances in the sources), plus log10(1 + n) of
the local evidence for that use of the template (PT: times X <-> XH+ appears as a protonation pair;
DR: known recombination channels of the parent ion; AE: times M is an AE substrate; OS: verified O/S
pairs in the sources with the same substituted reactants; CT: times Y is ionised by charge
transfer; HA: times BH donates an H atom). A candidate produced by several derivations keeps the
best one (max, not sum: an OS image of a PT is the same analogy counted twice).
"""
import itertools, math, re
from collections import Counter, defaultdict
from . import chem
from .chem import COMPOSITION, ELEMENTS, canonical_key, conserves
from .knockout import reaction_class, label
from .paths import CASES

RADICALS = ("CN", "CCH", "O")
PHYSICAL = set(chem.PLACEHOLDERS)
NONSPECIES = PHYSICAL | {"e-"}
VALENCE = dict(H=1, He=2, C=4, N=5, O=6, Si=4, S=6, Fe=8, Na=1, Mg=2, Cl=7, P=5, F=7)
BASE_FAMILIES = ("PT", "DR", "AE", "OS")
FAMILIES = BASE_FAMILIES + ("CT", "HA")
_IO, _IS = 1 + ELEMENTS.index("O"), 1 + ELEMENTS.index("S")

def vec(s): return COMPOSITION[s]
def add(*vs): return tuple(map(sum, zip(*vs)))
def sub(a, b): return tuple(x - y for x, y in zip(a, b))


H, E = vec("H"), vec("e-")
HP = sub(H, E)                                    # proton: one H and charge +1


def other(seq, s):
    """seq without one occurrence of s, or None if s is absent"""
    rest = list(seq)
    if s not in rest:
        return None
    rest.remove(s)
    return rest


def open_or_unsaturated(s):
    """Neutral and either odd valence-electron count (radical) or degree of unsaturation > 0."""
    if chem.charge(s) != 0:
        return False
    a = chem.atoms(s)
    if sum(VALENCE[e] * n for e, n in a.items()) % 2:
        return True
    dou = (a.get("C", 0) + a.get("Si", 0) + 1 + (a.get("N", 0) + a.get("P", 0)) / 2
           - (a.get("H", 0) + a.get("F", 0) + a.get("Cl", 0) + a.get("Na", 0)) / 2)
    return dou > 0


def w(n): return math.log10(1 + n)


def orderings(pair):
    """Both orderings of a 2-tuple, once if they coincide."""
    return list(dict.fromkeys(((pair[0], pair[1]), (pair[1], pair[0]))))


class Generator:
    def __init__(self, network_rows, extra_sources=(), hidden=(), novelty=True, families=FAMILIES):
        """network_rows: base network (e.g. darkcloud.load_rows); extra_sources: other networks (2014).
        hidden: rows to hide; every strict equivalent is removed from the network and the sources.
        novelty=False only for the positive control (keeps candidates already in the network)."""
        self.hidden_keys = {canonical_key(h) for h in hidden}
        self.net = [r for r in network_rows if canonical_key(r) not in self.hidden_keys]
        src = {}
        for r in itertools.chain(self.net, extra_sources):
            k = canonical_key(r)
            if k not in self.hidden_keys and k not in src and r.get("itype") not in (10, 11):
                src[k] = r
        self.source = list(src.values())
        self.net_keys = {canonical_key(r) for r in self.net}
        self.species = {s for r in self.net for s in r["reac"] + r["prod"]} - NONSPECIES
        self.by_comp = defaultdict(list)
        for s in sorted(self.species):
            if not s.startswith("GRAIN"):
                self.by_comp[vec(s)].append(s)
        self.novelty, self.families = novelty, families
        self._swap_cache = {}
        self._learn(set(src))

    # --- template learning ---------------------------------------------------------------------
    def _learn(self, src_keys):
        self.pt_donor, self.pt_pair = Counter(), Counter()
        self.dr_h, self.dr_ion = 0, Counter()
        self.ae, self.ae_sub = Counter(), Counter()
        self.ct_ion, self.ct_neutral, self.ct_pair = Counter(), defaultdict(Counter), Counter()
        self.ha_abs, self.ha_prod, self.ha_donor = Counter(), defaultdict(Counter), Counter()
        self.ha_pair = Counter()
        self.os_pairs, self.os_local, self.os_images = Counter(), Counter(), defaultdict(list)
        for r in self.source:
            reac, prod = r["reac"], r["prod"]
            if len(reac) != 2 or set(reac) & PHYSICAL:
                continue
            ion = other(reac, "e-")
            if ion is not None:
                if chem.charge(ion[0]) == 1:
                    self.dr_ion[ion[0]] += 1
                    rest = other(prod, "H")
                    if rest is not None and len(rest) == 1 and vec(rest[0]) == sub(add(vec(ion[0]), E), H):
                        self.dr_h += 1
                continue
            if len(prod) != 2:
                continue
            pairs = list(itertools.product(orderings(reac), orderings(prod)))
            for (X, D), (XH, A) in pairs:
                if (chem.charge(D) == 1 and chem.charge(X) == 0 and vec(XH) == add(vec(X), HP)
                        and vec(A) == sub(vec(D), HP)):
                    self.pt_donor[D] += 1; self.pt_pair[(X, XH)] += 1
                    break
            for (Xp, Y), (X, Yp) in pairs:
                if (chem.charge(Xp) == 1 and chem.charge(Y) == 0 and vec(X) == add(vec(Xp), E)
                        and vec(Yp) == sub(vec(Y), E)):
                    self.ct_ion[Xp] += 1; self.ct_neutral[Xp][X] += 1; self.ct_pair[(Y, Yp)] += 1
                    break
            for (A, BH), (AH, B) in pairs:
                if vec(AH) == add(vec(A), H) and vec(B) == sub(vec(BH), H):
                    self.ha_abs[A] += 1; self.ha_prod[A][AH] += 1
                    self.ha_donor[BH] += 1; self.ha_pair[(BH, B)] += 1
                    break
            rest = other(prod, "H")
            if rest is not None:
                for R, M in orderings(reac):
                    if (R in RADICALS and chem.charge(M) == 0
                            and vec(rest[0]) == sub(add(vec(R), vec(M)), H)):
                        self.ae[R] += 1; self.ae_sub[M] += 1
        for d in ("O->S", "S->O"):
            for r in self.source:
                if set(r["reac"]) & PHYSICAL:
                    continue
                img = self._swap_row(r, d)
                if img is None or canonical_key(img) == canonical_key(r):
                    continue
                cls = reaction_class(img)
                swapped = tuple(sorted(s for s in r["reac"] if self._swap(s, d) != s))
                if canonical_key(img) in src_keys:
                    self.os_pairs[(d, cls)] += 1
                    self.os_local[(d, swapped)] += 1
                for p in set(img["prod"]):
                    self.os_images[p].append((d, cls, swapped, img))

    def _swap(self, s, d):
        if s in NONSPECIES or s.startswith("GRAIN"):
            return s
        if (s, d) not in self._swap_cache:
            v = list(vec(s)); a, b = (_IO, _IS) if d == "O->S" else (_IS, _IO)
            out = s
            if v[a]:
                v[b] += v[a]; v[a] = 0; v = tuple(v)
                name = s.replace("O", "S") if d == "O->S" else re.sub(r"S(?!i)", "O", s)
                iso = self.by_comp.get(v, [])
                out = name if name in iso else (iso[0] if len(iso) == 1 else None)
            self._swap_cache[(s, d)] = out
        return self._swap_cache[(s, d)]

    def _swap_row(self, r, d):
        reac = [self._swap(s, d) for s in r["reac"]]
        prod = [self._swap(s, d) for s in r["prod"]]
        if None in reac or None in prod:
            return None
        return dict(reac=reac, prod=prod)

    # --- directed generation -------------------------------------------------------------------
    def candidates(self, target):
        """Candidates producing `target`, sorted by prior score (descending)."""
        out = {}
        T, vT = target, vec(target)

        def put(reac, prod, family, template, support, local=0):
            if family not in self.families or support <= 0:
                return
            row = dict(reac=list(reac), prod=list(prod))
            if not all(s in self.species or s in NONSPECIES for s in row["reac"] + row["prod"]):
                return
            if not conserves(row):
                return
            k = canonical_key(row)
            if self.novelty and k in self.net_keys:
                return
            c = out.setdefault(k, dict(row, id=-1, derivations=[], fam={}))
            c["derivations"].append((family, template, support, local))
            c["fam"][family] = max(c["fam"].get(family, 0.0), w(support) + w(local))

        if chem.charge(T) == 1:
            Xs = [x for x in self.by_comp.get(sub(vT, HP), []) if chem.charge(x) == 0]
            for D, n in self.pt_donor.items():                   # PT: T = XH+
                for A in self.by_comp.get(sub(vec(D), HP), []):
                    for X in Xs:
                        put([X, D], [T, A], "PT", D, n, self.pt_pair[(X, T)])
            Ys = [y for y in self.by_comp.get(add(vT, E), []) if chem.charge(y) == 0]
            for Xp, n in self.ct_ion.items():                    # CT: T = Y+
                for X in self.ct_neutral[Xp]:
                    for Y in Ys:
                        put([Xp, Y], [X, T], "CT", Xp, n, self.ct_pair[(Y, T)])
        if chem.charge(T) == 0:
            for D in self.by_comp.get(add(vT, HP), []):          # PT: T = A, conjugate of the donor
                for (X, XH), m in self.pt_pair.items():
                    put([X, D], [XH, T], "PT", D, self.pt_donor[D], m)
            for Xp in self.by_comp.get(sub(vT, E), []):          # CT: T = X, neutralised ion
                for (Y, Yp), m in self.ct_pair.items():
                    put([Xp, Y], [T, Yp], "CT", Xp, self.ct_ion[Xp], m)
            for P in self.by_comp.get(sub(add(vT, H), E), []):   # DR
                put([P, "e-"], [T, "H"], "DR", "H-loss", self.dr_h, self.dr_ion[P])
            for R in RADICALS:                                   # AE
                for M in self.by_comp.get(sub(add(vT, H), vec(R)), []):
                    if open_or_unsaturated(M):
                        put([R, M], [T, "H"], "AE", R, self.ae[R], self.ae_sub[M])
        for A in self.by_comp.get(sub(vT, H), []):               # HA: T = AH
            for (BH, B), m in self.ha_pair.items():
                put([A, BH], [T, B], "HA", A, self.ha_abs[A], m)
        for BH in self.by_comp.get(add(vT, H), []):              # HA: T = B
            for A, n in self.ha_abs.items():
                for AH in self.ha_prod[A]:
                    put([A, BH], [AH, T], "HA", A, n, self.ha_donor[BH])
        for d, cls, swapped, img in self.os_images.get(T, []):   # OS
            put(img["reac"], img["prod"], "OS", f"{d} {cls}", self.os_pairs[(d, cls)],
                self.os_local[(d, swapped)])

        cands = list(out.values())
        for c in cands:
            c["score"] = round(max(c["fam"].values()), 9)
            c["family"] = "+".join(sorted(c["fam"]))
        cands.sort(key=lambda c: (-c["score"], canonical_key(c)))
        return cands


def rank(cands, hidden, level="strict", target=None, key="score"):
    """'Realistic' rank (mean of the tie group) of the best candidate equivalent to `hidden`
    under `key` (higher is better); None if absent."""
    hits = [c[key] for c in cands if chem.equivalent(c, hidden, level, target)]
    if not hits:
        return None
    s = max(hits)
    higher = sum(c[key] > s for c in cands)
    ties = sum(c[key] == s for c in cands)
    return higher + (ties + 1) / 2


def load_cases(path=CASES):
    """[(target species, hidden row)] from benchmark/cases.json"""
    import json
    return [(c["species"], dict(reac=c["hidden_reaction"]["reac"], prod=c["hidden_reaction"]["prod"]))
            for c in json.load(open(path))["cases"]]

