"""
check_literature.py - Literature check of a discovery JSON with Valency MCP Bond (tapir.literature).

Three uses, chosen with --uses (default all):
  hypotheses  top --top hypotheses -> their "literature" field
  existing    key existing reactions (KEY_EXISTING, from the "existing" block) -> their "literature"
              field, plus kida_comparison (tapir.literature.kida_comparison): compatible, incompatible
              or not_comparable, with KIDA's F(T) = F0 * exp(|g * (1/T - 1/300)|) read from the network
  context     "literature_context" block: up to 5 articles per question of CONTEXT_QUESTIONS[species],
              each with ref, finding and stance (SUPPORTS / CONTRADICTS / NEUTRAL) towards the
              grain-surface hypothesis
Discovery mode only (tapir.literature.guard). Results cached in results/literature_cache.json.
usage: python -m scripts.check_literature results/discovery/CH2CCH.json [--uses hypotheses,existing,context]
                                          [--top 10] [--refresh] [--out PATH] [--cache PATH] [--max-llm-calls 40]
Output: the input JSON updated in place (or --out).
"""
import argparse, json
from tapir.chem import canonical_key
from tapir.literature import Literature, CACHE, GRAIN_HYPOTHESIS, guard, kida_comparison

USES = ("hypotheses", "existing", "context")
KEY_EXISTING = {"CH2CCH": ("C + C2H4 -> H + CH2CCH", "H + CH2CCH -> CH3CCH + Photon",
                           "H + CH2CCH -> CH2CCH2 + Photon")}
CONTEXT_QUESTIONS = {"CH2CCH": (
    "Which formation routes of the propargyl radical (CH2CCH) in the dark cloud TMC-1 have been proposed?",
    "Has it been proposed that ethylene (C2H4) or the propargyl radical (CH2CCH) form on the surface of "
    "dust grains in dark clouds?",
    "Do chemical models underestimate the abundance of the propargyl radical (CH2CCH) in TMC-1, and to "
    "what do they attribute it?")}


def rx(lab):
    reac, prod = lab.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


def check_hypotheses(lit, data, top):
    hyps = sorted(data["hypotheses"], key=lambda h: h["rank"])[:top]
    for h in hyps:
        h["literature"] = lit.check_reaction(h, data)
    return hyps


def check_existing(lit, data):
    by_key = {canonical_key(e): e for e in data.get("existing", [])}
    out = []
    for lab in KEY_EXISTING.get(data["species"], ()):
        e = by_key.get(canonical_key(rx(lab)))
        if e is None:
            print(f"  {lab}: not in the existing block, skipped")
            continue
        entry = dict(lit.check_reaction(e, data))
        entry["kida_comparison"] = kida_comparison(entry, e["kida_id"])
        e["literature"] = entry
        out.append(e)
    return out


def check_context(lit, data):
    qs = CONTEXT_QUESTIONS.get(data["species"], ())
    data["literature_context"] = dict(hypothesis=GRAIN_HYPOTHESIS, backend="valency-bond",
                                      questions=[lit.check_question(q, data) for q in qs])
    return data["literature_context"]


def report(name, entries):
    print(f"\n{name}:")
    for e in entries:
        L = e["literature"]
        extra = f" k_lit={L['k_lit']:.3g} at {L['k_lit_T']} K" if L.get("k_lit") else ""
        if "kida_comparison" in L:
            c = L["kida_comparison"]
            extra += f" KIDA: {c['result']}" + (f" ({c['reason']})" if c["reason"] else f" (F(T)={c['F_T']:.3g})")
        refs = [f"{r['ref']}{'' if r['same_channel'] else ' [other channel]'}" for r in L["refs"]]
        print(f"  {e['reaction']}: {L['status']}{extra} refs={refs}" + (f" ({L['error']})" if L.get("error") else ""))


def main(argv=None, literature=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("discovery_json")
    ap.add_argument("--uses", default=",".join(USES), help=f"comma-separated: {', '.join(USES)}")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--out", help="output path (default: the input, updated in place)")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--max-llm-calls", type=int, default=40, help="cap on LLM requests (TAPIR_LLM_MAX_CALLS)")
    a = ap.parse_args(argv)
    uses = [u.strip() for u in a.uses.split(",") if u.strip()]
    if set(uses) - set(USES):
        ap.error(f"unknown uses {sorted(set(uses) - set(USES))}; available: {', '.join(USES)}")
    data = json.load(open(a.discovery_json))
    guard(data)                                    # before any call to Bond or the LLM
    lit = literature or Literature(cache_path=a.cache, refresh=a.refresh, max_calls=a.max_llm_calls)
    try:
        if "hypotheses" in uses:
            report(f"hypotheses (top {a.top})", check_hypotheses(lit, data, a.top))
        if "existing" in uses:
            report("existing reactions", check_existing(lit, data))
        if "context" in uses:
            print("\nliterature context:")
            for q in check_context(lit, data)["questions"]:
                print(f"  {q['question']}\n    {q['status']}" + (f" ({q['error']})" if q.get("error") else ""))
                for x in q["articles"]:
                    print(f"    [{x['stance']}] {x['ref']}: {x['finding']}")
    finally:
        lit.save()
    with open(a.out or a.discovery_json, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"\n{lit.stats}\nwritten {a.out or a.discovery_json}; cache {a.cache}")
    return data


if __name__ == "__main__":
    main()
