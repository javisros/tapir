"""
benchmark_a.py - Leave-one-out WITHOUT simulation for tapir.generator (~20 s).

Sample: neutral-neutral and ion-neutral reactions (tapir.knockout.reaction_class) of kida.uva.2024 as the
model simulates it (load_rows), drawn with a fixed seed.
For each one: hide it (and its strict equivalents in 2014/2024), take as target the product with the
most atoms (ties -> alphabetical; never e- or Photon, and H only if nothing else), generate
candidates and measure the rank of the hidden reaction at both equivalence levels.
Realistic rank: mean of the tie group. Metrics: recall@N (among the N generated), Hits@1/5/10 and
MRR (1/rank, 0 if absent).
Positive control (--control): nothing hidden and no novelty filter -> upper bound of the template
language.
usage: python -m scripts.benchmark_a [--n 200] [--seed 0] [--families base|all] [--control]
Output: results/benchmark_a_<families>[_control].csv (one row per reaction), summary on stdout.
"""
import argparse, csv, random, statistics, time
from tapir import chem
from tapir.darkcloud import parse_kida, load_rows
from tapir.knockout import NET, reaction_class, label
from tapir.generator import Generator, rank, NONSPECIES, BASE_FAMILIES, FAMILIES
from tapir.paths import relative, result

CLASSES = ("neutral-neutral", "ion-neutral")
KS = (1, 5, 10)


def pick_target(r):
    prods = [s for s in r["prod"] if s not in NONSPECIES]
    heavy = [s for s in prods if s != "H"] or prods
    return min(heavy, key=lambda s: (-sum(chem.atoms(s).values()), s))


def sample(rows, n, seed):
    pool = sorted((r for r in rows if reaction_class(r) in CLASSES), key=lambda r: r["id"])
    return random.Random(seed).sample(pool, n)


def metrics(ranks, n_cands):
    m = dict(n=len(ranks), recall_N=sum(r is not None for r in ranks) / len(ranks))
    for k in KS:
        m[f"hits@{k}"] = sum(r is not None and r <= k for r in ranks) / len(ranks)
    m["MRR"] = sum(1 / r for r in ranks if r is not None) / len(ranks)
    m["N_median"] = statistics.median(n_cands)
    return m


def fmt(m):
    return (f"n={m['n']:3d} recall@N={m['recall_N']:.2f} " + " ".join(f"H@{k}={m[f'hits@{k}']:.2f}" for k in KS)
            + f" MRR={m['MRR']:.3f} N_med={m['N_median']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", choices=("base", "all"), default="all")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    fams = BASE_FAMILIES if a.families == "base" else FAMILIES
    t0 = time.time()
    rows24, raw14 = load_rows(NET), parse_kida(chem.REACTIONS_2014)
    hidden = sample(rows24, a.n, a.seed)
    gen_all = Generator(rows24, raw14, novelty=False, families=fams) if a.control else None
    out = []
    for h in hidden:
        tgt = pick_target(h)
        g = gen_all or Generator(rows24, raw14, hidden=[h], families=fams)
        if not a.control:
            assert not any(chem.equivalent(r, h) for r in g.source)       # leak check
        cs = g.candidates(tgt)
        rs, rc = rank(cs, h), rank(cs, h, "channel", tgt)
        fam = next((c["family"] for c in cs if chem.equivalent(c, h)), "")
        out.append(dict(id=h["id"], reaction=label(h), reaction_class=reaction_class(h), target=tgt,
                        n_cand=len(cs), rank_strict="" if rs is None else rs,
                        rank_channel="" if rc is None else rc, family=fam))
    path = result(f"benchmark_a_{a.families}{'_control' if a.control else ''}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)

    def R(o, key): return None if o[key] == "" else o[key]
    print(f"{'POSITIVE CONTROL' if a.control else 'LEAVE-ONE-OUT'}  families={'+'.join(fams)}  "
          f"n={a.n} seed={a.seed}  -> {relative(path)}")
    for lev, key in (("strict", "rank_strict"), ("channel", "rank_channel")):
        print(f" {lev:7s} all            ", fmt(metrics([R(o, key) for o in out], [o["n_cand"] for o in out])))
        for c in CLASSES:
            sel = [o for o in out if o["reaction_class"] == c]
            if sel:
                print(f" {lev:7s} {c:15s}", fmt(metrics([R(o, key) for o in sel], [o["n_cand"] for o in sel])))
    fam_count = {}
    for o in out:
        if o["family"]:
            fam_count[o["family"]] = fam_count.get(o["family"], 0) + 1
    print(" recovered by family:", dict(sorted(fam_count.items(), key=lambda x: -x[1])),
          f" [{time.time() - t0:.0f}s]")
