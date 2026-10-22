# Chemical networks

Files as distributed by KIDA, the KInetic Database for Astrochemistry
(https://kida.astrochem-tools.org). Check KIDA's terms of use before redistributing them, and cite the
network papers when using them.

| Directory | Files | Reference |
|---|---|---|
| `kida.uva.2014/` | `kida.uva.2014.dat` (reactions), `gas_species.dat`, `kida.uva.2014.bib` | Wakelam, V., et al. 2015, ApJS, 217, 20 |
| `kida.uva.2024/` | `gas_reactions_kida.uva.2024.in`, `gas_species_kida.uva.2024.in`, `kida.uva.2024.bib`, `kida.uva.2024_readme.txt` | Wakelam, V., et al. 2024, A&A, 689, A63 |

Formats are described in `kida.uva.2024/kida.uva.2024_readme.txt`. Species files: name, charge and the
number of atoms of H, He, C, N, O, Si, S, Fe, Na, Mg, Cl, P, F (order checked in `tests/test_chem.py`).
The 14 species of the 2014 network missing from the 2024 species file are listed in
`tapir/chem.py` (`MISSING_IN_2024`). Every row of both reaction files conserves atoms and charge.
The `.bib` files hold the references of the individual rate coefficients.
