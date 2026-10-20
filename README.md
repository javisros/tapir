# TAPIR — Tracking Absent Pathways in Interstellar Reactions

Software that proposes reactions missing from the KIDA gas-phase networks, filters them, gives them a
rate, simulates them in a 0D dark-cloud model and ranks them by how well they explain a target
abundance. Built for the OSI Hackathon (21–22 October 2026).

The product is the pipeline. The knock-out benchmark only **validates** it: we hide a reaction that the
network needs and check whether the pipeline rediscovers it without having seen it.

**Status.** This is the pre-hackathon baseline: 0D model, analogy generator, flux pre-ranking and the two
validation benchmarks. The LLM generator, the discovery mode against the TMC-1 observation, the
literature check and the Marimo app are being built during the event (see [Roadmap](#roadmap)).

```
target species ─► generator (analogy templates) ─► conservation / closed world / novelty filters
             ─► generic rate (capture_guess) ─► flux pre-ranking ─► 0D simulation of the top K ─► ranking
```

## Quick start

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib (pytest optional); or: pip install -e .
python -m tests.test_chem              # 30 tests in total; or simply: pytest
python -m scripts.generate CH3CN       # candidate reactions for a species, no simulation
python -m scripts.benchmark_b          # end-to-end knock-out benchmark, ~75 s on 4 processes
```

All scripts are run from the repository root as `python -m scripts.<name>`. Tested with Python 3.12,
numpy 2.4 and scipy 1.18.

| Script | What it does | Time |
|---|---|---|
| `run_targets` | 2 networks × 2 C/O × 3 S: does the base model reproduce the published discrepancies? → `summary.csv`, `timeseries.csv` | ~1 min |
| `plot_targets` | `results/targets.png` from `timeseries.csv` | s |
| `knockout_demo` | CH₂CCH budget + knock-out + re-injection as a candidate | ~10 s |
| `validate_w24` | Validation against Table 2 of Wakelam et al. (2024) | ~30 s |
| `best_time_w24` | "Best time" with D(t) against Gratier et al. (2016) | ~1 min |
| `knockout_screen` | Knock-out sensitivity screen (50 species), resumable | ~6 min |
| `knockout_robust` | Robustness of the screen in another configuration and time | ~2 min |
| `knockout_top2` | Borderline cases: knock-out of the two main reactions | ~1 min |
| `build_cases` | The 6 benchmark cases → `benchmark/cases.json`, checked against the screen | ~20 s |
| `generate` | Candidates for any target (`--flux` re-ranks them, `--cases` shows the 6 cases) | s |
| `benchmark_a` | Leave-one-out of the generator on 200 reactions, no simulation | ~20 s |
| `benchmark_b` | The 6 cases end to end, with simulation | ~75 s |

## Repository layout

```
tapir/                library
  darkcloud.py          0D gas-phase model (KIDA rate formulas, BDF + sparse Jacobian)
  chem.py               elemental composition, conservation, canonical key, reaction equivalence
  generator.py          analogy-based candidate generator (leak-proof by construction)
  flux.py               flux pre-ranking dX ≈ k·n_H·x_A·x_B / L_X on the knock-out network
  knockout.py           reference configuration, reaction classes, generic rates, run helpers
  observations.py       TMC-1 abundances (Gratier+2016, QUIJOTE), mystery and control species
  paths.py              repository paths
scripts/              command-line entry points (table above)
tests/                pytest suite (runs without pytest too: python -m tests.test_generator)
networks/             kida.uva.2014 and kida.uva.2024 as downloaded from KIDA (see networks/README.md)
benchmark/cases.json  the 6 knock-out cases
results/              outputs of the scripts (CSV, PNG)
```

## The pipeline

**Generator** (`tapir/generator.py`). Templates are learned from the base network (kida.uva.2024)
plus kida.uva.2014, merged by canonical key. Instances are recognised by elemental composition, not by
name. Generation is directed: only candidates that produce the target species.

| Family | Pattern | Template support counted per |
|---|---|---|
| PT proton transfer | X + AH⁺ → XH⁺ + A | proton donor AH⁺ (H₃⁺, HCO⁺, …) |
| DR dissociative recombination | XH⁺ + e⁻ → X + H | H loss |
| AE addition–H elimination | R + M → P + H, R ∈ {CN, CCH, O}, M unsaturated or radical | radical R |
| OS isovalent substitution | O→S or S→O image of a known reaction | direction × reaction class |
| CT charge transfer | X⁺ + Y → X + Y⁺ | ion X⁺ |
| HA H-atom abstraction | A + BH → AH + B | abstractor A |

PT, DR, AE and OS were the families planned first; CT and HA were added because they cover the two
largest classes the first four miss. Prior score = log₁₀(1 + template support) + log₁₀(1 + local evidence),
best derivation per candidate (see the module docstring).

**Filters.** Atom and charge conservation; closed world (all species already exist in the base
network); novelty (not already in the network); never CR/CRP/photon-driven reactions.

**Rate.** A generic rate that never looks at KIDA (`knockout.capture_guess`): 2×10⁻⁹ cm³ s⁻¹ ion–neutral,
10⁻¹⁰ neutral–neutral, 10⁻⁷ (T/300)⁻⁰·⁵ dissociative recombination. Other classes (radiative
association, ion–ion) have no estimator yet and are not simulated.

**Flux pre-ranking** (`tapir/flux.py`) on the knock-out run at t_ref, then **simulation** of the top
K = 30, each candidate alone. Score = |log₁₀(X_cand / X_base)| + Σ over the other observed TMC-1 species
of max(0, |log₁₀ change| − log₁₀ 3). Lower is better.

**Reaction equivalence**, two levels (`tapir/chem.py`): *strict* = same multiset of reactants and of
products, Photon included; *channel* = same reactants and the target among the products.

**Leak-proofing.** The hidden reaction and every strict equivalent in KIDA 2014 or 2024 are removed
before anything is learned. KIDA numbers change between versions (CN + C₂H₄ is #907 in 2014 and #961 in
2024), so equivalence is always by canonical key. `tests/test_generator.py` checks that hiding a reaction
gives exactly the same candidates as deleting it from the sources by hand, and includes a canary (an
invented reaction no template can derive) and a positive control (all 1000+ template instances of the
network are regenerated when nothing is hidden).

## Validation: knock-out benchmark

Reference configuration: kida.uva.2024, C/O = 1.2, S = 8×10⁻⁸, conditions of Wakelam et al. (2024)
Sect. 2.3, t_ref = 6×10⁶ yr. The objective of each case is **X_base of the original model, not the TMC-1
observation** (see the screen below for why).

**Benchmark B – the 6 cases, end to end** (`scripts/benchmark_b.py`, `results/benchmark_b.csv`)

| Case | Hidden reaction | Candidates | Prior rank | Flux rank | **Final rank** | Recovery |
|---|---|---|---|---|---|---|
| CH₂CHCN | CN + C₂H₄ → H + CH₂CHCN | 2 | 2 | 1 | **1** | 0.85 |
| CH₃CN | CH₃CNH⁺ + e⁻ → H + CH₃CN | 1 | 1 | 1 | **1** | 0.79 |
| HNCCC | HC₃NH⁺ + e⁻ → H + HNCCC | 124 | 1 | 5 | **4** | 0.44 |
| HCCCHO | O + CH₂CCH → HCCCHO + H | 4 | 3.5 | 1 | **1** | 0.65 |
| H₂CS | S + CH₃ → H + H₂CS | 376 | 5 | 1 | **1** | 0.74 |
| N₂H⁺ | N₂ + H₃⁺ → H₂ + N₂H⁺ | 172 | 5 | 3 | **1** | 1.2 |

Hits@1 = 5/6, Hits@5 = 6/6, MRR = 0.875 (same at both equivalence levels), against MRR = 0.39 and
Hits@1 ≈ 1.8/6 for a random ordering of the generated candidates (the baseline is high because two
cases have only one or two candidates). 104 simulations, 2.5 s per run, 75 s in total on 4 processes. Recovery = X(hidden reaction at the generic rate) / X_base; it matches
the independent re-injection of the screen in all six cases. The one miss, HNCCC, loses to
CN + C₂H₂ → HNCCC + H: the real CN + C₂H₂ → HC₃N + H chemistry with the wrong isomer, because the
generator works by composition.

**Benchmark A – generator leave-one-out, no simulation** (`scripts/benchmark_a.py`): 200 neutral–neutral
and ion–neutral reactions of kida.uva.2024 (seed 0), target = heaviest product.

| Families | recall@N | Hits@1 | Hits@5 | Hits@10 | MRR | median N |
|---|---|---|---|---|---|---|
| PT, DR, AE, OS | 0.28 | 0.03 | 0.16 | 0.20 | 0.089 | 60 |
| + CT, HA (default) | 0.45 | 0.04 | 0.20 | 0.27 | 0.119 | 376 |
| positive control, all families (nothing hidden) | 0.47 | 0.01 | 0.14 | 0.17 | 0.074 | 392 |

The ceiling is template coverage, not hiding: the control recall is only 0.47. Of the 110 reactions not
recovered, 45 are heavy-atom exchanges, 40 are two-product channels releasing H or H₂ that no template
covers (insertions, H₂ elimination, addition of radicals other than CN/CCH/O) and 25 have three or more
products. Prior ranks are poor (many ties among proton donors); the flux pre-ranking and the simulation
are what separate them.

## Base model

0D dark cloud, gas phase only, reading **kida.uva.2014** and **kida.uva.2024** directly. It reproduces the
**gas-phase-only** test of:

> Wakelam, V., Gratier, P., Loison, J.-C., Hickson, K. M., Penguen, J., & Mechineau, A. 2024, *The 2024
> KIDA network for interstellar chemistry*, A&A, 689, A63.
> [doi:10.1051/0004-6361/202450606](https://doi.org/10.1051/0004-6361/202450606) · arXiv:2407.15958

Taken from that paper: the physical conditions of Sect. 2.3 (T = 10 K, n_H = 2×10⁴ cm⁻³, ζ = 10⁻¹⁷ s⁻¹,
A_V = 15), the rate formulas by type of Sect. 2.2, the duplicate criterion of Sect. 2.1.1 (keep the most
recent entry) and the disagreement distance D(t) of Eq. 3 with the observed set of Gratier et al. (2016,
ApJS 225, 25), the same as their Sect. 3.2. Other references: kida.uva.2014 is Wakelam et al. (2015, ApJS
217, 20) and the formulas are from Wakelam et al. (2012, ApJS 199, 21).

What is **not** the same: their TMC-1 comparison (Sect. 3.2) is gas-grain with Nautilus and
n_H = 3×10⁴ cm⁻³; ours is gas only. And outside the temperature range Nautilus uses the rate at the edge,
whereas here reactions with a barrier are evaluated at 10 K (`t_extrap` option; affects 6 reactions in
2014 and 5 in 2024, with no measurable effect).

- **Elemental abundances.** Standard "low-metal" set, hydrogen initially all in H₂ and C⁺ = 1.7×10⁻⁴.
  Sweeps over C/O ∈ {0.7, 1.2} and S ∈ {8×10⁻⁸, 1.5×10⁻⁶, 1.5×10⁻⁵}.
- **Rate coefficients.** KIDA formulas: cosmic rays, Draine, Kooij, ionpol1 and ionpol2. Reactions with
  several temperature ranges use the range containing 10 K. Types 10/11 (forced H₂) are replaced by
  standard H₂ formation on grains, R = 3×10⁻¹⁷ cm³ s⁻¹.
- **Duplicates.** 7 reactions in 2014 and 21 in 2024 appear twice under different numbers; only the
  most recent is kept (`dedup=True`; effect on CH₂CCH below 1 %).
- **Integration.** scipy BDF with a sparse analytic Jacobian; 2.5–6 s per run on one core. The network
  is read once per process (`load_rows` cache).
- **Checks.** Every row of both files (7505 in 2014, 8275 in 2024) conserves atoms and charge
  (`tests/test_chem.py`); elemental drift over 10⁷ yr is below 10⁻¹³.
- **External validation** (`validate_w24`). log(X₂₀₂₄/X₂₀₁₄) at 10⁵ yr against Table 2 of Wakelam et al.
  (2024). With C/O = 0.7 and S = 8×10⁻⁸: **77 of 77 species with the same sign, all within 0.5 dex,
  r = 0.99**, and only 2 of 97 abundant species outside the table exceed the threshold (Mg⁺ and Fe⁺,
  +0.4). With S = 1.5×10⁻⁵ the agreement degrades (92 % within 0.5 dex), which suggests their test used
  low sulphur.
- **Best time** (`best_time_w24`). With the 31 species of Gratier et al. (2016), in the reference
  configuration the best time is **6×10⁶ yr** (D = 1.06; 66 % of species within 10×), the same as
  Wakelam et al. (2024) obtain with that set. Note: S = 1.5×10⁻⁵ gives a slightly better D (0.98 at
  4.5×10⁵ yr), so the reference is defensible but not unique.

### Does the base model reproduce each published discrepancy?

Model/observed ratio at the best time, range over the 12 configurations (`run_targets`):

| Molecule | Observed (X/H₂) | Published discrepancy | This model (range) | Verdict |
|---|---|---|---|---|
| CH₂CCH | 8.7×10⁻⁹ | ~100× too low | 5×10⁻⁶ – 0.3 | **Yes, in all 12.** The size depends strongly on C/O. |
| C₅O | 1.5×10⁻¹² | ~10⁴× too high | 6 – 260× too high | **Yes in sign, in all 12.** Smaller than published. |
| CH₃CHCH₂ | 4.0×10⁻⁹ | ≥100× too low | 0 (2014) – 0.02 (2024) | **Yes.** No gas-phase source in 2014. |
| CH₃CO⁺ | 3.2×10⁻¹¹ | cation/neutral ratio 5–10× | absolute 0.01 – 0.6; ratio 2–8× low | Weak: the factor is small. |
| HC₃O⁺ | 1.7×10⁻¹¹ | HC₃O⁺/C₃O ratio ~70× | absolute 6×10⁻⁴ – 2; ratio 10–100× low | Ambiguous: the ratio fails because C₃O is **overproduced**. |
| CCS | 5.5×10⁻⁹ | 100–1000× too low | 2×10⁻³ – 60 | **Not robust:** the sign changes with S and C/O. |

What it means for the project: CH₂CCH is the main mystery molecule (the only large one that fails in
every configuration; its "factor 60" ranges from ~10× at C/O = 1.2 to over 1000× at C/O = 0.7, so a
reference configuration has to be fixed). C₅O is the reserve, the mystery in reverse: there the
TAPIR should look for missing **destructions**. Propene is a demo case only (its likely fix involves
grain chemistry). CCS (depends on the free sulphur parameter) and HC₃O⁺ (the problem is C₃O) are out.
From 2014 to 2024 the dominant destruction of CH₂CCH changes: H + CH₂CCH → H₂ + C₃H₂ in 2014, replaced by
the radiative associations H + CH₂CCH → C₃H₄ + photon in 2024.

### Knock-out sensitivity screen

Reference configuration, t = 6×10⁶ yr (`knockout_screen`, `results/knockout_screen.csv` and
`results/knockout_robust.csv`). For the 50 TMC-1 species (`tapir/observations.py`) most abundant in the
model, the dominant formation reaction (excluding cosmic rays and photons) is removed and the drop is
measured. Removing the dominant formation of CH₂CCH itself (C + C₂H₄, 60 %) only lowers it 3×, because
the network is redundant, hence the screen.

**12 of 50 drop ≥10×.** "Robust" means it also drops ≥10× at 10⁶ yr and in the S = 1.5×10⁻⁵
configuration at 4.5×10⁵ yr. "Recovery" is X(re-injected with a generic rate)/X(base): 1 is perfect.

| Species | Hidden reaction | Class | Drop | Robust | Recovery |
|---|---|---|---|---|---|
| CH₂CHCN | CN + C₂H₄ → H + CH₂CHCN | neutral–neutral | 3500× | yes | 0.85 |
| CH₃CN | CH₃CNH⁺ + e⁻ → H + CH₃CN | recombination | 3200× | yes | 0.79 |
| C₆H⁻ | C₆H + e⁻ → C₆H⁻ + hν | radiative e⁻ attachment | 920× | yes | no estimator |
| C₄H⁻ | C₄H + e⁻ → C₄H⁻ + hν | radiative e⁻ attachment | 650× | yes | no estimator |
| HNCCC | HC₃NH⁺ + e⁻ → H + HNCCC | recombination | 320× | yes | 0.44 |
| HCCCHO | O + CH₂CCH → HCCCHO + H | neutral–neutral | 220× | yes | 0.65 |
| H₂CS | S + CH₃ → H + H₂CS | neutral–neutral | 120× | yes | 0.74 |
| HC₉N | CN + C₈H₂ → H + HC₉N | neutral–neutral | 50× | **no** (1.1×) | 0.52 |
| N₂H⁺ | N₂ + H₃⁺ → H₂ + N₂H⁺ | ion–neutral | 35× | yes | 1.2 |
| HC₇N | CN + C₆H₂ → H + HC₇N | neutral–neutral | 18× | **no** (1.05×) | 0.62 |
| HC₅N | CN + C₄H₂ → H + HC₅N | neutral–neutral | 14× | **no** (1.1×) | 0.65 |
| C₅O | O + C₅H → H + C₅O | neutral–neutral | only source | yes | 9.8 |

Borderline cases (3–10×): OCS, CH₃CCH, H₂CO, C₃O, HCCNC, H₂CCN, NH₃, l-C₃H₂, HNCO and CCO. Knocking out the
**two** main reactions (`knockout_top2`) makes HNCO (5000×), CCO (44×) and HCCNC (12×) pass.

How the 6 benchmark cases were chosen:

1. The robust, clean set is 8 species: CH₂CHCN, CH₃CN, C₆H⁻, C₄H⁻, HNCCC, HCCCHO, H₂CS and N₂H⁺.
2. C₆H⁻ and C₄H⁻ need a radiative electron-attachment rate estimator, which capture theory does not give;
   they are left out until one exists → **6 cases**.
3. The three cyanopolyynes only pass at late times with low sulphur and are one family (one analogy,
   CN + C₂H₂ → HC₃N, finds all three): they would count as one case, not three.
4. C₅O is not a knock-out case: it is the reserve mystery molecule (overproduced 260×), the hidden
   reaction is its only source and the generic rate overproduces it 10× again.
5. The knock-out "observation" must be the base model, not TMC-1: HC₉N, HNCCC and HCCCHO already come out
   12–32× above the observation, so knocking out their reaction would *improve* the fit to TMC-1 and
   there would be nothing to rediscover.
6. Leakage: HCCCHO forms from CH₂CCH, the mystery molecule. Fine for the knock-out benchmark, but a model
   trained without that reaction cannot be used to evaluate CH₂CCH.

## Limitations

- **Gas phase only.** The published discrepancies come from gas-grain models (e.g. NAUTILUS with the
  GOTHAM network); magnitudes are not comparable one to one.
- **Photodissociation.** With A_V = 15 external UV is negligible; only cosmic-ray-induced photons act.
- **Control species.** Their observed values come from the QUIJOTE papers cited in
  `tapir/observations.py`; the "best time" depends on that choice.
- **Isomers.** The generator works by composition and cannot tell HNCCC from HC₃N (the one miss of
  benchmark B).
- **Generic rates.** No barriers: neutral–neutral candidates with H₂ get absurd flux estimates
  (e.g. C₃N + H₂ → HNCCC + H, dX ≈ 2×10⁻³). Simulation discards them, but a small K would not.
- **Coverage.** No templates for heavy-atom exchange, insertion, radiative association or channels with
  three or more products (55 % of benchmark A is not generated at all); no estimator for radiative
  electron attachment (C₄H⁻, C₆H⁻).

## Roadmap

Planned for the hackathon (21–22 October 2026); none of this is in the repository yet.

1. **Discovery mode.** Target the TMC-1 observation instead of the base model, starting with CH₂CCH.
   For each hypothesis, the rate it would need to explain the observation and whether that rate is
   physically reachable (below the capture limit of its class); robustness in a second configuration
   (C/O = 1.2, S = 1.5×10⁻⁵, t = 4.5×10⁵ yr).
2. **LLM generator.** A second generator that proposes candidates with a one-line chemical rationale.
   Its output goes through exactly the same filters, ranking and simulation as the analogy generator, so
   the two can be compared on the same benchmarks; memorisation is measured with canaries.
3. **Literature check** with Valency for the top hypotheses: was the reaction measured, proposed or
   computed? A published rate replaces the generic one and is simulated again. Discovery mode only,
   never in a benchmark (it would know the hidden reactions).
4. **Marimo app** with a Validation tab (benchmark cases) and a Discovery tab (hypotheses, required rate,
   robustness, literature), published on molab from precomputed results.
5. **Rates for neutral + H₂** from the KIDA class median, so barrier reactions stop dominating the flux
   pre-ranking.
6. Optional: **C₅O in destruction mode** (the model overproduces it, so we look for missing destructions).

The output is a short list of **ranked hypotheses, not confirmed reactions**.

## Data and references

`networks/` holds kida.uva.2014 and kida.uva.2024 as distributed by KIDA
(https://kida.astrochem-tools.org), with their bibliographies; see `networks/README.md` for citations.
Observed abundances: Gratier et al. (2016, ApJS 225, 25) and the QUIJOTE papers listed in
`tapir/observations.py`.

## License

Code under the MIT License (`LICENSE`). The KIDA network files keep their own terms of use; cite the
network papers listed in `networks/README.md` when using them.
