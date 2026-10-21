"""
generate.py - Candidate reactions for a target species, without simulation.

  python -m scripts.generate CH3CN               top candidates producing CH3CN (nothing hidden)
  python -m scripts.generate CH3CN --flux        ... re-ranked by flux on the reference network (1 run),
                                                 rates from tapir.rates.RateEstimator
  python -m scripts.generate --cases             rank of each benchmark case's hidden reaction
"""
import argparse
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, run, label
from tapir.generator import Generator, load_cases, rank, BASE_FAMILIES, FAMILIES
from tapir.flux import FluxScorer
from tapir.rates import RateEstimator

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--families", choices=("base", "all"), default="all")
    ap.add_argument("--flux", action="store_true", help="re-rank by flux on the reference network")
    ap.add_argument("--cases", action="store_true", help="leave-one-out ranks of the 6 benchmark cases")
    a = ap.parse_args()
    fams = BASE_FAMILIES if a.families == "base" else FAMILIES
    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    if a.cases:
        for name, h in load_cases():
            cs = Generator(rows24, raw14, hidden=[h], families=fams).candidates(name)
            fam = next((c["family"] for c in cs if chem.equivalent(c, h)), "-")
            print(f"{name:8s} N={len(cs):4d}  strict rank={rank(cs, h)}  "
                  f"channel rank={rank(cs, h, 'channel', name)}  family={fam}")
    elif a.target:
        cs = Generator(rows24, raw14, families=fams).candidates(a.target)
        print(f"{len(cs)} candidates for {a.target}")
        if a.flux:
            cs, no_rate = FluxScorer(*run()).score(cs, a.target, estimator=RateEstimator(rows24))
            print(f"ranked by flux ({no_rate} without a rate)")
        for c in cs[:a.top]:
            extra = (f"  dX={c['dX']:.1e}  k={c['k']:.1e} [{c['k_source']}]"
                     f"{'  unpaired: ' + c['unpaired_flag'] if c['unpaired_flag'] else ''}") if a.flux else ""
            print(f"  {c['score']:6.3f}  {c['family']:8s} {label(c)}{extra}")
    else:
        ap.print_help()
