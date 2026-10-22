"""
rates.py - Rate estimator for candidate reactions: two KIDA class medians, then capture_guess.

Neutral-neutral candidates fall into two classes whose generic capture rate (1e-10) is wrong at 10 K:
  "neutral + H2"                 any neutral + H2 (abstractions from H2 have barriers at 10 K);
  "closed-shell neutral-neutral" no reactant is barrierless-reactive (see below).
Their k is the median of k(10 K) over the reactions of that class in kida.uva.2024, WITHOUT the hidden
reactions and their equivalents (same canonical-key criterion as Generator.hidden_keys), with
k_source = "kida_class_median". Every other candidate is delegated unchanged to knockout.capture_guess
(k_source = "capture_guess"); classes without a generic rate give (None, None).

Unpaired-electron filter (all classes except dissociative recombination). unpaired(s): 0 closed shell,
1 odd valence-electron count (e- included), 2 REACTIVE_EVEN_ELECTRON, 3 atomic N and P. A candidate
whose products carry more unpaired electrons than its reactants gets unpaired_flag = "increase": a SIGN
that bonds break without new ones forming (often endothermic), not proven endothermicity nor a spin rule.
Validation on kida.uva.2024 (unpaired_validation: no barrier, k(10 K) > 1e-11, no CR, photons,
dissociative recombination or grains; 5411 reactions): 26.6 % increase it, but 1426 of the 1437 exceptions have an
ionic reactant (ion-ion 68 %, ion-neutral 15 %: charge transfer, whose ionisation energy breaks bonds,
and triplet ions such as OH+, HS+, NH2+ or N+ that the count takes as closed shell). Neutral-neutral:
11 of 587 (1.9 %), mostly carbon-chain growth into a triplet product (CH + C3 -> H + C4). Hence ion-ion
and ion-neutral are flagged only, and a flagged candidate of any other class with a rate is excluded:
k = 0, k_source = "unpaired_increase".

Ion H-transfer pattern (ion_h_transfer): ion-neutral, both reactants with an even electron count and
outside REACTIVE_EVEN_ELECTRON, two products with an odd count (c-C3H2 + H3+ -> CH2CCH + H2+): an H
atom where real chemistry moves a proton. Validation on kida.uva.2024 (ion_h_transfer_validation:
ion-neutral, gamma = 0 for Kooij rows, k(10 K) > 1e-11; 3511 reactions): 189 match (5.4 %), above the
2 % threshold (ION_H_TRANSFER_MAX), so matching candidates are flagged only (ion_flag = "h_transfer")
and keep their rate. The matches are charge transfer (118: H+ + X -> H + X+, N+, Mg, Fe), other
rearrangements (46, mostly C-chain growth with H loss: C2H2 + C4H+ -> H + C6H2+) and H abstraction
by the ion (24: H2 + CN+ -> H + HCN+). Only one (CO + H- -> HCO + e-) moves the H from the ion to the
neutral, as the candidates do; the count is by electron parity, not by mechanism.

Wrong charge partner (ie_flag): ion-neutral with a single cationic product D+ and neutral products
N: if IE(D) - min IE(N) > IE_MARGIN (1 eV), the charge would sit on the other product (c-C3H2 + HCO+ ->
CH2CCH + CO+: IE(CO) 14.0 vs IE(CH2CCH) 8.7 eV) -> ie_flag = "wrong_charge_partner". IE from
data/ionization_energies.csv (NIST Chemistry WebBook only, see its header); applied only when IE(D) and
at least one IE(N) are in the table (a missing IE(N) could only lower the minimum, so it cannot clear
a flag). Validation on kida.uva.2024 (ie_validation: ion-neutral, gamma = 0 for Kooij rows,
k(10 K) > 1e-11; 3511 reactions, 1518 with both IE): 103 break the rule (6.8 %), above the 2 % threshold
(IE_WRONG_PARTNER_MAX), so candidates are flagged only and keep their rate. Exceptions: 44 dissociative
charge transfer from He+ (24.6 eV, three products: SiO + He+ -> He + Si + O+) and 59 of 1178 two-product
reactions (5.0 %), mostly heavy-atom rearrangements and C-chain growth (CO2 + N+ -> NO + CO+,
C4 + C+ -> C + C4+, CH3OH + C+ -> HCO + CH3+).

A neutral species is barrierless-reactive if ANY of:
  (i)   odd number of valence electrons (radical);
  (ii)  it is in REACTIVE_EVEN_ELECTRON (triplet ground state or reactive singlet);
  (iii) network evidence: reactant in >= 2 neutral-neutral reactions of kida.uva.2024 with gamma = 0
        and k(10 K) > 1e-11 against a closed-shell partner, i.e. an even-electron species that is not
        in REACTIVE_EVEN_ELECTRON. A partner that is itself reactive (C + C2H2, O + OCS) says nothing
        about the other reactant, so it does not count. Computed without the hidden reactions.
Medians and evidence are both computed on the rows passed in (kida.uva.2024 as darkcloud.load_rows
prepares it: entry valid at 10 K, duplicates removed), at the temperature the model uses (row Teff).
"""
import csv
import numpy as np
from . import chem
from .chem import canonical_key
from .darkcloud import W24_GAS, rate
from .flux import estimate_row
from .generator import VALENCE
from .knockout import reaction_class, PHYSICAL
from .paths import IONIZATION_ENERGIES

NEUTRAL_H2 = "neutral + H2"
CLOSED_SHELL = "closed-shell neutral-neutral"
KIDA_MEDIAN, UNPAIRED_INCREASE, ION_H_TRANSFER = "kida_class_median", "unpaired_increase", "ion_h_transfer"
IE_WRONG_PARTNER = "ie_wrong_partner"
INCREASE, H_TRANSFER, WRONG_CHARGE_PARTNER = "increase", "h_transfer", "wrong_charge_partner"

# Even-electron neutrals that react without a barrier. Editable.
REACTIVE_EVEN_ELECTRON = {
    "C": "triplet ground state (3P)",
    "O": "triplet ground state (3P)",
    "S": "triplet ground state (3P)",
    "Si": "triplet ground state (3P)",
    "O2": "triplet ground state (3Sigma-)",                                   # UNCERTAIN: triplet, but slow with closed-shell partners
    "CH2": "triplet ground state (3B1), methylene",
    "NH": "triplet ground state (3Sigma-)",
    "SO": "triplet ground state (3Sigma-), isovalent with O2",                # UNCERTAIN: same caveat as O2
    "S2": "triplet ground state (3Sigma-), isovalent with O2",                # UNCERTAIN: same caveat as O2
    "CCO": "triplet ground state (3Sigma-), ketenylidene",
    "C2": "reactive singlet (low-lying triplet), barrierless with C2H2 in KIDA",
    "C4": "triplet ground state (3Sigma-) of the linear chain",              # UNCERTAIN: rhombic singlet nearly degenerate
    "NCN": "triplet ground state (3Sigma-); not a kida.uva.2024 species",
    "PH": "triplet ground state (3Sigma-), isovalent with NH",
    "HCCN": "triplet ground state, cyanocarbene; not a kida.uva.2024 species",
    # Not in the initial list; same criterion (triplet ground state), present in kida.uva.2024.
    "CCS": "triplet ground state (3Sigma-), isovalent with CCO",
    "C4S": "triplet ground state (3Sigma-), even CnS chain like CCS",
    "SiC": "triplet ground state (3Pi)",
    "C6": "triplet ground state (3Sigma-) of the linear chain, like C4",
    "C8": "triplet ground state (3Sigma-) of the linear chain, like C4",
    "C10": "triplet ground state (3Sigma-) of the linear chain, like C4",    # UNCERTAIN: the C10 ring (singlet) is lower
    "t-C3H2": "triplet propargylene HCCCH",                                   # UNCERTAIN: assumes KIDA's t- means triplet
}

# Upper limit of a rate coefficient by class (cm3 s-1 at 10 K): what is physically reachable. Editable.
CAPTURE_MAX = {"ion-neutral": 1e-8, "neutral-neutral": 1e-9, "dissociative recombination": 3e-6}

EVIDENCE_K, EVIDENCE_N = 1e-11, 2

# Atoms with a quartet ground state (4S): three unpaired electrons.
UNPAIRED_ATOMS = {"N": 3, "P": 3}
# Classes where an increase of unpaired electrons is flagged but does not change k (see the docstring).
UNPAIRED_FLAG_ONLY = ("ion-neutral", "ion-ion")
UNPAIRED_EXEMPT = ("dissociative recombination",)
# Ion H-transfer: excluded (k = 0) only if the pattern is rarer than this among fast KIDA ion-neutral
# reactions. Measured 5.4 % (see the docstring) -> flag only.
ION_H_TRANSFER_MAX = 0.02
ION_H_TRANSFER_EXCLUDE = False
# Wrong charge partner: IE(D) above the lowest IE of the neutral products by more than IE_MARGIN (eV).
# Excluded only if rarer than IE_WRONG_PARTNER_MAX among fast KIDA ion-neutral reactions. Measured 6.8 %
# (see the docstring) -> flag only.
IE_MARGIN = 1.0
IE_WRONG_PARTNER_MAX = 0.02
IE_WRONG_PARTNER_EXCLUDE = False
_IE = {}


def electrons(s):
    """Valence electrons of a species (charge included)."""
    return sum(VALENCE[e] * n for e, n in chem.atoms(s).items()) - chem.charge(s)


def k10(r):
    """Rate coefficient of a network row at the temperature the model uses for it."""
    return rate(r, r.get("Teff", W24_GAS["T"]), W24_GAS["zeta"], W24_GAS["Av"], 1.0)


def unpaired(s):
    """Unpaired electrons: 3 atomic N and P, 2 REACTIVE_EVEN_ELECTRON, 1 odd electron count, 0 otherwise."""
    if s in UNPAIRED_ATOMS:
        return UNPAIRED_ATOMS[s]
    if s in REACTIVE_EVEN_ELECTRON:
        return 2
    return electrons(s) % 2


def unpaired_flag(r):
    """INCREASE if the products carry more unpaired electrons than the reactants, else None.
    Never for dissociative recombination."""
    if reaction_class(r) in UNPAIRED_EXEMPT:
        return None
    if sum(map(unpaired, r["prod"])) > sum(map(unpaired, r["reac"])):
        return INCREASE
    return None


def unpaired_validation(rows):
    """Reactions of `rows` without barrier and with k(10 K) > EVIDENCE_K, excluding CR, photons,
    dissociative recombination and grains (never proposed by the generator)
    -> {class: (number of reactions, [those flagged INCREASE])}.
    gamma is only defined for the Kooij formula (3); ionpol rows (4, 5) are capture rates, kept."""
    out = {}
    for r in rows:
        c = reaction_class(r)
        if (c in (PHYSICAL, "grain") or c in UNPAIRED_EXEMPT or "Photon" in r["reac"] + r["prod"]
                or (r["formula"] == 3 and r["C"] != 0) or k10(r) <= EVIDENCE_K):
            continue
        n, exc = out.get(c, (0, []))
        out[c] = (n + 1, exc + [r] if unpaired_flag(r) else exc)
    return out


def ion_h_transfer(r):
    """Ion-neutral with both reactants even-electron and outside REACTIVE_EVEN_ELECTRON, and two
    odd-electron products (see the docstring)."""
    if reaction_class(r) != "ion-neutral" or len(r["prod"]) != 2:
        return False
    return (all(electrons(s) % 2 == 0 and s not in REACTIVE_EVEN_ELECTRON for s in r["reac"])
            and all(electrons(s) % 2 == 1 for s in r["prod"]))


def ion_flag(r):
    """H_TRANSFER if the reaction matches ion_h_transfer, else None."""
    return H_TRANSFER if ion_h_transfer(r) else None


def ion_h_transfer_validation(rows):
    """Ion-neutral reactions of `rows` without barrier (gamma = 0 for Kooij rows; ionpol rows kept)
    and with k(10 K) > EVIDENCE_K -> (number of reactions, [those matching ion_h_transfer])."""
    fast = [r for r in rows if reaction_class(r) == "ion-neutral"
            and not (r["formula"] == 3 and r["C"] != 0) and k10(r) > EVIDENCE_K]
    return len(fast), [r for r in fast if ion_h_transfer(r)]


def ionization_energies(path=IONIZATION_ENERGIES):
    """{neutral species: IE in eV} from the NIST table (comment lines start with #), read once."""
    if path not in _IE:
        with open(path) as fh:
            _IE[path] = {r["species"]: float(r["ie_eV"]) for r in csv.DictReader(l for l in fh if not l.startswith("#"))}
    return _IE[path]


def charge_partner(r, ie=None):
    """(D, IE(D), N, IE(N)) for an ion-neutral reaction with one cationic product D+ and neutral
    products, N the neutral product of lowest known IE; None if the rule does not apply."""
    ie = ionization_energies() if ie is None else ie
    if reaction_class(r) != "ion-neutral":
        return None
    cations = [s for s in r["prod"] if chem.charge(s) > 0]
    neutrals = [s for s in r["prod"] if chem.charge(s) == 0 and s not in chem.PLACEHOLDERS]
    if len(cations) != 1 or chem.charge(cations[0]) != 1 or not neutrals:
        return None
    d, known = cations[0][:-1], [(ie[s], s) for s in neutrals if s in ie]
    if d not in ie or not known:
        return None
    ie_n, n = min(known)
    return d, ie[d], n, ie_n


def ie_flag(r, ie=None):
    """WRONG_CHARGE_PARTNER if IE(D) - min IE(N) > IE_MARGIN (see charge_partner), else None."""
    p = charge_partner(r, ie)
    return WRONG_CHARGE_PARTNER if p and p[1] - p[3] > IE_MARGIN else None


def ie_validation(rows, ie=None):
    """Ion-neutral reactions of `rows` without barrier (gamma = 0 for Kooij rows; ionpol rows kept),
    with k(10 K) > EVIDENCE_K and the rule applicable -> (number of reactions, [those flagged])."""
    fast = [r for r in rows if reaction_class(r) == "ion-neutral"
            and not (r["formula"] == 3 and r["C"] != 0) and k10(r) > EVIDENCE_K and charge_partner(r, ie)]
    return len(fast), [r for r in fast if ie_flag(r, ie)]


def capture_max(r):
    """Capture limit of the reaction's class, or None if the class has none."""
    return CAPTURE_MAX.get(reaction_class(r))


class RateEstimator:
    def __init__(self, rows, hidden=()):
        """rows: kida.uva.2024 (darkcloud.load_rows). hidden: rows whose strict equivalents never enter
        the medians nor the evidence (iii)."""
        self.hidden_keys = {canonical_key(h) for h in hidden}
        rows = [r for r in rows if canonical_key(r) not in self.hidden_keys
                and reaction_class(r) == "neutral-neutral"]
        self.evidence = {}                       # species -> canonical keys of its evidence (iii)
        for r in rows:
            if r["C"] != 0 or k10(r) <= EVIDENCE_K:
                continue
            for x, partner in (r["reac"], r["reac"][::-1]):
                if electrons(partner) % 2 == 0 and partner not in REACTIVE_EVEN_ELECTRON:
                    self.evidence.setdefault(x, set()).add(canonical_key(r))
        self.by_evidence = {s for s, keys in self.evidence.items() if len(keys) >= EVIDENCE_N}
        self.members = {NEUTRAL_H2: [], CLOSED_SHELL: []}   # class -> [(canonical key, k(10 K))]
        for r in rows:
            c = self.kclass(r)
            if c in self.members:
                self.members[c].append((canonical_key(r), k10(r)))
        self.median = {}
        for c, m in self.members.items():
            if not m:
                raise ValueError(f"no kida.uva.2024 reaction left in class {c!r}")
            self.median[c] = float(np.median([k for _, k in m]))

    def reactive(self, s):
        """Barrierless-reactive neutral: radical, REACTIVE_EVEN_ELECTRON or network evidence."""
        return s in REACTIVE_EVEN_ELECTRON or electrons(s) % 2 == 1 or s in self.by_evidence

    def kclass(self, r):
        """NEUTRAL_H2, CLOSED_SHELL or knockout.reaction_class(r)."""
        c = reaction_class(r)
        if c != "neutral-neutral":
            return c
        if "H2" in r["reac"]:
            return NEUTRAL_H2
        if not any(self.reactive(s) for s in r["reac"]):
            return CLOSED_SHELL
        return c

    def flag(self, cand):
        """unpaired_flag of the candidate (stored by FluxScorer.score)."""
        return unpaired_flag(cand)

    def estimate(self, cand):
        """Candidate -> (row in parse_kida format or None, k_source)."""
        c = self.kclass(cand)
        if c in self.median:
            row, src = dict(reac=list(cand["reac"]), prod=list(cand["prod"]), A=self.median[c], B=0.0,
                            C=0.0, itype=4, tmin=10.0, tmax=300.0, formula=3, id=-1,
                            Teff=W24_GAS["T"]), KIDA_MEDIAN
        else:
            row, src = estimate_row(cand)
        if row is not None and reaction_class(cand) not in UNPAIRED_FLAG_ONLY and unpaired_flag(cand):
            row, src = dict(row, A=0.0), UNPAIRED_INCREASE
        if row is not None and ION_H_TRANSFER_EXCLUDE and ion_h_transfer(cand):
            row, src = dict(row, A=0.0), ION_H_TRANSFER
        if row is not None and IE_WRONG_PARTNER_EXCLUDE and ie_flag(cand):
            row, src = dict(row, A=0.0), IE_WRONG_PARTNER
        return row, src
