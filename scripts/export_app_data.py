"""
export_app_data.py - Self-contained data for the Marimo app (app/tapir_app.py), which never imports tapir.

Writes app/data/:
  validation_curves.json  the 6 benchmark B cases: X(t) of the target for the base network, the knock-out
                          network and the knock-out network + the hidden reaction re-injected with the
                          generic rate the pipeline gives it (knockout.capture_guess, never KIDA); ranks
                          read back from results/benchmark_b.csv, MRR and the random baseline mean(H_n / n)
                          with n = candidates in the final ranking (the README's 0.39 takes n = all
                          generated candidates, before the flux pre-ranking keeps the top K).
  discovery/*.json        copies of results/discovery/*.json, when present
  bench_llm*.json         results/bench_llm* (JSON copied, CSV converted to a list of rows), when present
  manifest.json           what was exported and what was missing
The 13 runs (1 base + 6 knock-outs + 6 re-injections) take ~20 s; X(t_ref) of each run is checked against
benchmark/cases.json and results/benchmark_b.csv.
usage: python -m scripts.export_app_data   (or python scripts/export_app_data.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv, glob, json, shutil
from tapir import chem
from tapir.knockout import T_REF, run, X
from tapir.flux import knockout_exclude, kinetic_row
from tapir.generator import rank
from tapir.paths import ROOT, CASES, RESULTS, relative

APP_DATA = os.path.join(ROOT, "app", "data")
BENCH_B = os.path.join(RESULTS, "benchmark_b.csv")
MAX_BYTES = 10 * 1024 * 1024
TOL = 0.01                     # relative tolerance of the X(t_ref) checks (as cases.json check_tolerance)


def sig(v, n=5):
    return float(f"{v:.{n}g}")


def harmonic(n):
    return sum(1 / i for i in range(1, n + 1))


def parse_label(s):
    reac, prod = s.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


def rel_err(a, b):
    return abs(a - b) / abs(b) if b else None


def ranking(case_rows, hidden, target):
    """Final ranking of benchmark B (lower score is better) with tapir.generator.rank on its CSV rows."""
    cands = [dict(parse_label(r["reaction"]), neg_score=-float(r["score"])) for r in case_rows]
    strict = rank(cands, hidden, key="neg_score")
    channel = rank(cands, hidden, "channel", target, key="neg_score")
    flags = [r["final_rank"] for r in case_rows if r["hidden_strict"] == "True"]
    assert (strict is None) == (not flags), f"{target}: CSV hidden_strict disagrees with chem.equivalent"
    return strict, channel


def curve(net, t, y, s):
    return [sig(2 * v) for v in y[net.idx[s]]]


def validation():
    cfg = json.load(open(CASES))
    rows = list(csv.DictReader(open(BENCH_B)))
    print("base run ...", flush=True)
    net0, t0, y0 = run()
    out, checks = [], []
    for c in cfg["cases"]:
        name, h = c["species"], c["hidden_reaction"]
        hidden = dict(reac=h["reac"], prod=h["prod"])
        case_rows = sorted((r for r in rows if r["case"] == name), key=lambda r: int(r["final_rank"]))
        n = len(case_rows)
        r_strict, r_channel = ranking(case_rows, hidden, name)
        hid = next((r for r in case_rows if r["hidden_strict"] == "True"), None)

        print(f"{name}: knock-out and re-injection runs ...", flush=True)
        excl = knockout_exclude(chem.canonical_key(hidden))
        net_k, t_k, y_k = run(exclude=excl)
        row = kinetic_row(hidden)
        net_r, t_r, y_r = run(exclude=excl, extra_reactions=[row])
        assert list(t_k) == list(t0) and list(t_r) == list(t0)

        x_base, x_ko, x_rec = (X(*r, name, T_REF) for r in ((net0, t0, y0), (net_k, t_k, y_k), (net_r, t_r, y_r)))
        chk = dict(case=name, X_base_rel_err=rel_err(x_base, c["X_base"]), X_ko_rel_err=rel_err(x_ko, c["X_ko"]),
                   X_recovered_rel_err=None if hid is None or not hid["X_cand"] else rel_err(x_rec, float(hid["X_cand"])))
        chk["ok"] = all(v is None or v < TOL for k, v in chk.items() if k.endswith("rel_err"))
        checks.append(chk)

        out.append(dict(
            case=name, target=name,
            hidden_reaction=h["reaction"], hidden_kida_id=h.get("kida_id"),
            hidden_reaction_class=h.get("reaction_class"),
            reinjected_rate=dict(A=row["A"], B=row["B"], C=row["C"],
                                 k_10K=float(hid["k"]) if hid else None,
                                 note="generic rate from knockout.capture_guess, not the KIDA value"),
            rank=r_strict, rank_channel=r_channel, n=n,
            rank_over_n=None if r_strict is None else r_strict / n,
            reciprocal_rank=0.0 if r_strict is None else 1 / r_strict,
            random_reciprocal_rank=harmonic(n) / n,
            X_at_t_ref=dict(base=x_base, knockout=x_ko, recovered=x_rec),
            drop_at_t_ref=c.get("drop"),
            top_candidates=[dict(rank=int(r["final_rank"]), reaction=r["reaction"], family=r["family"],
                                 score=float(r["score"]), hidden=r["hidden_strict"] == "True")
                            for r in case_rows],
            time=[sig(v) for v in t0],
            abundance=dict(base=curve(net0, t0, y0, name), knockout=curve(net_k, t_k, y_k, name),
                           recovered=curve(net_r, t_r, y_r, name)),
        ))

    rr = [c["reciprocal_rank"] for c in out]
    rr_c = [0.0 if c["rank_channel"] is None else 1 / c["rank_channel"] for c in out]
    conf = cfg["configuration"]
    return dict(
        schema_version=1,
        source=dict(cases=relative(CASES), ranks=relative(BENCH_B),
                    curves="tapir.knockout.run (base / exclude= / exclude= + extra_reactions=)"),
        configuration=dict(network=conf["network"], C_O=conf["C_O"], S=conf["S"], physics=conf["physics"],
                           t_ref_yr=T_REF),
        units=dict(time="yr", abundance="n_i / n(H2) (dimensionless)", rate="cm3 s-1"),
        ranking=dict(level="strict", n="candidates in the final ranking of benchmark B (top K by flux, "
                     "simulated)", score="|log10 X_cand / X_base| + penalty on observed species (lower is better)"),
        metrics=dict(mrr=sum(rr) / len(rr), mrr_channel=sum(rr_c) / len(rr_c),
                     random_mrr=sum(c["random_reciprocal_rank"] for c in out) / len(out),
                     random_mrr_definition="mean over cases of H_n / n, the expected 1/rank of one item "
                                           "among n shuffled candidates",
                     random_mrr_note="n = final ranking (top K by flux, simulated). The README's 0.39 uses "
                                     "the same formula with n = all generated candidates (2, 1, 124, 4, 376, "
                                     "172), which are not stored in benchmark_b.csv",
                     hits={str(k): sum(c["rank"] is not None and c["rank"] <= k for c in out) for k in (1, 5, 10)},
                     n_cases=len(out)),
        checks=checks,
        cases=out,
    )


def csv_rows(path):
    return list(csv.DictReader(open(path)))


def optional_results():
    """Copies whatever discovery / bench_llm results exist; their absence is normal."""
    exported, missing = [], []
    src = sorted(glob.glob(os.path.join(RESULTS, "discovery", "*.json")))
    if src:
        os.makedirs(os.path.join(APP_DATA, "discovery"), exist_ok=True)
        for p in src:
            json.load(open(p))
            shutil.copy(p, os.path.join(APP_DATA, "discovery", os.path.basename(p)))
            exported.append(f"discovery/{os.path.basename(p)}")
    else:
        missing.append("results/discovery/*.json")
    src = sorted(glob.glob(os.path.join(RESULTS, "bench_llm*.json")) + glob.glob(os.path.join(RESULTS, "bench_llm*.csv")))
    for p in src:
        base, ext = os.path.splitext(os.path.basename(p))
        data = json.load(open(p)) if ext == ".json" else csv_rows(p)
        dest = base + (".json" if ext == ".json" else ".csv.json")
        json.dump(data, open(os.path.join(APP_DATA, dest), "w"), indent=1)
        exported.append(dest)
    if not src:
        missing.append("results/bench_llm*.json|csv")
    return exported, missing


def size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


if __name__ == "__main__":
    os.makedirs(APP_DATA, exist_ok=True)
    val = validation()
    with open(os.path.join(APP_DATA, "validation_curves.json"), "w") as fh:
        json.dump(val, fh, separators=(",", ":"))
    exported, missing = optional_results()
    manifest = dict(exported=["validation_curves.json"] + exported, missing=missing,
                    note="missing results are expected until the discovery / LLM pipelines have run")
    json.dump(manifest, open(os.path.join(APP_DATA, "manifest.json"), "w"), indent=1)

    m = val["metrics"]
    for c in val["cases"]:
        print(f" {c['case']:8s} rank={c['rank']} / n={c['n']}  rank/n={c['rank_over_n']:.3f}  "
              f"H_n/n={c['random_reciprocal_rank']:.3f}")
    bad = [c["case"] for c in val["checks"] if not c["ok"]]
    print(f" MRR={m['mrr']:.3f} (channel {m['mrr_channel']:.3f})  random={m['random_mrr']:.3f}")
    print(f" checks against cases.json / benchmark_b.csv: {'all ok' if not bad else 'FAILED ' + ', '.join(bad)}")
    print(f" missing (not exported): {', '.join(missing) or '-'}")
    total = size(APP_DATA)
    print(f" {relative(APP_DATA)}: {total / 1024:.1f} KiB")
    if total > MAX_BYTES or bad:
        sys.exit(1)
