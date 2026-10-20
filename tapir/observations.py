"""
observations.py - Observed abundances in TMC-1(CP), relative to H2, with their source.

1) GRATIER: Gratier et al. (2016, ApJS 225, 25), Table 2, median log N with N(H2) = 1e22 cm-2.
   This is the set Wakelam et al. (2024) Sect. 3.2 use for the "best time".
   Excluded: limits (C2O <, OCS <, CS >, SO >) and C4H (interval only, depends on the dipole).
   C4H2 = ortho + para.
2) QUIJOTE: values from the mystery-molecule papers (Agundez+2021, Cernicharo+2020/2021,
   Marcelino+2007), only for species not in GRATIER.
3) DETECTED: species firmly detected in TMC-1 whose abundance has NOT been checked here
   (X = None). They count as "observed" for the knock-out screen; before using one as a mystery
   molecule its abundance must be set (e.g. from Agundez & Wakelam 2013, Chem. Rev. 113, 8710).

TARGETS are the published mystery molecules and CONTROL the 8 non-mystery species with a quoted
value that scripts/run_targets.py uses to pick the "best time".
"""
NH2 = 1e22  # N(H2) in TMC-1 (Cernicharo & Guelin 1987), to convert N -> X(H2)

_G_LOGN = {
    "NH3": 14.70, "CH3OH": 13.16, "c-C3H": 13.48, "l-C3H": 12.75, "l-C3H2": 11.77,
    "c-C3H2": 13.27, "CH3CCH": 14.06, "H2CCN": 13.58, "CH3CN": 12.61, "H2CCO": 12.68,
    "HNCO": 13.03, "CH3CHO": 12.43, "HCS+": 12.76, "H2CS": 13.62, "C3N": 13.55,
    "HC3N": 14.37, "HNCCC": 11.68, "HCCNC": 12.93, "C3O": 11.92, "HC3NH+": 11.87,
    "CH2CHCN": 12.81, "HCCCHO": 11.26, "CCS": 14.01, "C5H": 12.27, "CH3C4H": 13.17,
    "CH3C3N": 11.99, "C3S": 13.14, "C6H": 12.74, "HC5N": 13.77, "HC7N": 13.66, "HC9N": 13.02,
}
GRATIER = {s: 10 ** (v - 22) for s, v in _G_LOGN.items()}
GRATIER["C4H2"] = (10 ** 13.28 + 10 ** 12.42) / 1e22

QUIJOTE = {  # species: (X/H2, source)
    "CH2CCH": (8.7e-9, "Agundez+2021"), "CH3CHCH2": (4.0e-9, "Marcelino+2007"),
    "HC3O+": (1.7e-11, "Cernicharo+2020"), "C5O": (1.5e-12, "Cernicharo+2021"),
    "CH3CO+": (3.2e-11, "Cernicharo+2021"), "HCO": (1.1e-10, "Cernicharo+2021"),
    "CCO": (7.5e-11, "Cernicharo+2021"),
}

DETECTED = ["CO", "HCO+", "N2H+", "HCN", "HNC", "CN", "CCH", "OH", "H2CO", "CS", "SO",
            "OCS", "C4H", "C4H-", "C6H-", "HCNH+", "NO", "NS", "H2S", "HC3S+"]

# Published mystery molecules: species -> (X_obs/H2, source)
TARGETS = {
    "CH2CCH":   (8.7e-9,  "Agundez+2021 A&A 647 L10"),
    "CCS":      (5.5e-9,  "Cernicharo+2021a A&A 648 L3"),
    "CH3CHCH2": (4.0e-9,  "Marcelino+2007 ApJ 665 L127"),
    "HC3O+":    (1.7e-11, "Cernicharo+2020 A&A 642 L17 (C3O/7)"),
    "C5O":      (1.5e-12, "Cernicharo+2021 A&A 656 L21"),
    "CH3CO+":   (3.2e-11, "Cernicharo+2021 A&A 646 L7"),
}

# Control species (not mysteries) with a quoted value, to pick the "best time".
CONTROL = {
    "CH3CCH": 1.2e-8,   # Gratier+2016 / Cabezas+2021
    "c-C3H2": 5.9e-9,   # Cernicharo+2021 (rings)
    "c-C3H":  6.2e-10,  # idem
    "C3O":    1.2e-10,  # Cernicharo+2021 (HCCCO/C5O)
    "CCO":    7.5e-11,  # idem
    "HCO":    1.1e-10,  # idem
    "H2CCO":  1.4e-9,   # 44 x CH3CO+ (Cernicharo+2021)
    "C3S":    1.3e-9,   # 65 x HC3S+ (Cernicharo+2021)
}

UNVERIFIED = "detected; X not verified"


def observed():
    """dict species -> (X_obs or None, source)."""
    out = {s: (x, "Gratier+2016") for s, x in GRATIER.items()}
    for s, (x, ref) in QUIJOTE.items():
        out.setdefault(s, (x, ref))
    for s in DETECTED:
        out.setdefault(s, (None, UNVERIFIED))
    return out
