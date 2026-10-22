"""
robustness.py - T4: the stored hypotheses of a discovery JSON in a second configuration of the model.

Second configuration: kida.uva.2024, C/O = 1.2, S = 1.5e-5, t_ref = 4.5e5 yr (W24_GAS physics).
No generator, LLM or Bond call: the candidates are those already in the JSON, each simulated with its
stored k (with_rate) and, to test reachability, with the capture limit of its class (rates.capture_max).
  config2 (top level)    base run: X_base, x_ratio_base, D_base, main formation reactions at t_ref2
  bottleneck_config2     main formation reaction of the reference configuration (bottleneck.reaction)
                         times FACTOR_LIMIT["formation"] (discovery.bottleneck)
  hypothesis.config2     X_cand, x_ratio, delta_D with its k; X_at_k_max, x_ratio_at_k_max and
                         reachable (x_ratio_at_k_max >= 1) with the capture limit
  hypothesis.robust      in both configurations: delta_D < 0 and X_target closer to X_obs than the base
                         (|log10 x_ratio| < |log10 x_ratio_base|)
Simulations run on --procs processes with OMP_NUM_THREADS=1.
usage: python -m scripts.robustness [results/discovery/CH2CCH.json] [--procs 4]
Output: the JSON updated in place; every other field is left as it was.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, math, multiprocessing as mp, time
from tapir.darkcloud import elements, W24_GAS
from tapir.discovery import (FACTOR_LIMIT, observed_set, distance, simulate, with_rate, scaled, bottleneck, sig)
from tapir.knockout import run, X, label, is_row
from tapir.paths import RESULTS
from tapir.rates import capture_max
from scripts.knockout_demo import budget

C_O2, S2, T_REF2 = 1.2, 1.5e-5, 4.5e5
EL2 = elements(C_over_O=C_O2, S=S2)


def row(h, k):
    """Candidate of the JSON with rate k (cm3 s-1) at the model temperature."""
    return with_rate(dict(reac=list(h["reac"]), prod=list(h["prod"]), id=-1, tmin=10.0, tmax=300.0), k)


def _simulate(task):
    """(row, target, species, network row to replace or None); is_row is built here (not picklable)."""
    r, target, species, replaced = task
    return simulate(r, target, species, el=EL2, t_ref=T_REF2, exclude=is_row(replaced) if replaced else None)


def closer(x_ratio, x_ratio_base):
    return x_ratio is not None and abs(math.log10(max(x_ratio, 1e-30))) < abs(math.log10(x_ratio_base))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("discovery_json", nargs="?", default=os.path.join(RESULTS, "discovery", "CH2CCH.json"))
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    doc = json.load(open(a.discovery_json))
    target, x_obs, hyps = doc["species"], doc["X_obs"], doc["hypotheses"]
    obs = observed_set(target)

    net, t, y = run(EL2)
    bn_lab = doc["bottleneck"]["reaction"]
    bn_row = next(r for r in net.rows if label(r) == bn_lab)
    supply = sorted(doc["bottleneck"]["species"])
    species = sorted(set(obs) | set(supply))
    xbase = {s: X(net, t, y, s, T_REF2) for s in species if s in net.idx}
    D_base = distance(xbase, obs)
    form, _ = budget(net, t, y, target, T_REF2, 3)

    tasks = [(scaled(bn_row, FACTOR_LIMIT["formation"]), target, species, bn_row)]
    tasks += [(row(h, h["k"]), target, species, None) for h in hyps]
    tasks += [(row(h, capture_max(h) or h["k"]), target, [target], None) for h in hyps]
    with mp.Pool(a.procs) as pool:
        sims = pool.map(_simulate, tasks)
    bn_sim, at_k, at_max = sims[0], sims[1:1 + len(hyps)], sims[1 + len(hyps):]

    x_ratio_base2 = xbase[target] / x_obs
    for h, s, m in zip(hyps, at_k, at_max):
        c2 = dict(X_cand=None, x_ratio=None, delta_D=None, k=h["k"], k_max=capture_max(h),
                  X_at_k_max=None, x_ratio_at_k_max=None, reachable=None, error=s["error"] or m["error"] or None)
        if not s["error"]:
            c2.update(X_cand=sig(s["xs"][target]), x_ratio=sig(s["xs"][target] / x_obs),
                      delta_D=round(distance(s["xs"], obs) - D_base, 4))
        if not m["error"]:
            c2.update(X_at_k_max=sig(m["xs"][target]), x_ratio_at_k_max=sig(m["xs"][target] / x_obs),
                      reachable=bool(m["xs"][target] >= x_obs))
        h["config2"] = c2
        ok1 = h["delta_D"] is not None and h["delta_D"] < 0 and closer(h["x_ratio"], doc["x_ratio_base"])
        ok2 = c2["delta_D"] is not None and c2["delta_D"] < 0 and closer(c2["x_ratio"], x_ratio_base2)
        h["robust"] = bool(ok1 and ok2)

    doc["config2"] = dict(network="kida.uva.2024", C_O=C_O2, S=S2, t_ref=T_REF2, physics=W24_GAS,
                          X_base=sig(xbase[target]), x_ratio_base=sig(x_ratio_base2), D_base=round(D_base, 4),
                          main_formation=[dict(reaction=lab, frac=round(float(f), 4)) for lab, f in form],
                          n_robust=sum(h["robust"] for h in hyps),
                          n_reachable=sum(h["config2"]["reachable"] is True for h in hyps),
                          runtime_s=round(time.time() - t0))
    doc["bottleneck_config2"] = (None if bn_sim["error"] else
                                 bottleneck(bn_row, FACTOR_LIMIT["formation"], xbase, bn_sim["xs"], supply, target))
    tmp = a.discovery_json + ".tmp"                # never leave a half-written JSON
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, a.discovery_json)

    c2, bn = doc["config2"], doc["bottleneck_config2"]
    print(f"config2 (C/O={C_O2}, S={S2:g}, t_ref={T_REF2:g} yr): X_base={c2['X_base']:.3g} "
          f"x_ratio_base={c2['x_ratio_base']:.3g} (config1 {doc['x_ratio_base']:.3g}), D_base={c2['D_base']}")
    print(" main formation: " + "; ".join(f"{m['reaction']} {100 * m['frac']:.0f}%" for m in c2["main_formation"]))
    if bn:
        print(f" bottleneck ({bn['reaction']} x{bn['factor']:g}): {target} x{bn['X_target_scaled'] / bn['X_target_base']:.2f}, "
              + ", ".join(f"{s} x{v['ratio']:.3g}" for s, v in bn["species"].items())
              + f"  supply_limited={bn['supply_limited']}")
    for h in hyps:
        d = h["config2"]
        print(f" {h['rank']:2d} x_ratio={d['x_ratio']} dD={d['delta_D']} at k_max: x_ratio={d['x_ratio_at_k_max']} "
              f"reachable={d['reachable']} robust={h['robust']}  {h['reaction']}")
    print(f" robust {c2['n_robust']}/{len(hyps)}, reachable in config2 {c2['n_reachable']}/{len(hyps)}, "
          f"{len(tasks) + 1} simulations, {c2['runtime_s']} s -> {a.discovery_json}")


if __name__ == "__main__":
    main()
