# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "matplotlib"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TAPIR")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    return Path, json, mo, plt


@app.cell
def _(Path, json, mo):
    import sys

    # Self-contained: only precomputed files in app/ (python scripts/export_app_data.py), never tapir.
    APP_DIR = mo.notebook_dir() or Path(__file__).resolve().parent
    # In the browser (molab, WebAssembly) a file missing next to the notebook is fetched from GitHub instead.
    IN_BROWSER = sys.platform == "emscripten"
    RAW_URL = "https://raw.githubusercontent.com/javisros/tapir/main/app"
    EXPORT_HINT = "Run `python scripts/export_app_data.py` from the repository root, then reload this app."

    def read_text(rel):
        path = APP_DIR / rel
        if path.exists() or not IN_BROWSER:
            return path.read_text()
        from pyodide.http import open_url
        return open_url(f"{RAW_URL}/{rel}").read()

    def load_json(name):
        """(data, error message): a missing or unreadable file gives a message, never a traceback."""
        try:
            return json.loads(read_text(f"data/{name}")), None
        except FileNotFoundError:
            return None, f"`app/data/{name}` was not found. {EXPORT_HINT}"
        except Exception as e:      # OSError, ValueError, or a browser fetch error (JsException)
            return None, f"`app/data/{name}` could not be read ({e}). {EXPORT_HINT}"

    validation, validation_error = load_json("validation_curves.json")
    discovery, discovery_error = load_json("discovery/CH2CCH.json")
    upstream, upstream_error = load_json("discovery/CH2CCH_upstream.json")
    atlas, atlas_error = load_json("discovery/atlas.json")
    chem_review, chem_review_error = load_json("discovery/chem_review.json")
    # Optional until the LLM benchmark has been exported: absence is a placeholder, not an error.
    bench_summary, bench_error = (load_json("bench_llm_summary.json")
                                  if IN_BROWSER or (APP_DIR / "data" / "bench_llm_summary.json").exists()
                                  else (None, None))
    return (APP_DIR, IN_BROWSER, RAW_URL, atlas, atlas_error, bench_error, bench_summary, chem_review,
            chem_review_error, discovery, discovery_error, upstream, upstream_error, validation, validation_error)


@app.cell
def _(mo, validation):
    case_picker = None
    if validation is not None:
        _names = [c["case"] for c in validation["cases"]]
        case_picker = mo.ui.dropdown(options=_names, value=_names[0], label="Test case")
    return (case_picker,)




@app.cell
def _(mo, plt, validation):
    COLORS = dict(base="#16253d", knockout="#d8694f", recovered="#2f6fed")
    LABELS = dict(base="Full network", knockout="Missing reaction", recovered="TAPIR re-adds pathway (generic rate)")
    STYLES = dict(base=dict(lw=3.0, ls="-", alpha=0.45), knockout=dict(lw=2.0, ls="--"),
                  recovered=dict(lw=2.0, ls="-"))

    def fmt_rank(r):
        return "not ranked" if r is None else f"{r:g}"

    def arrow(reaction):
        return reaction.replace(" -> ", " → ")

    def plot_case(c):
        t_ref = validation["configuration"]["t_ref_yr"]
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=110)
        for key in ("base", "knockout", "recovered"):
            pts = [(t, x) for t, x in zip(c["time"], c["abundance"][key]) if x > 0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[key], label=LABELS[key], **STYLES[key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        top = max(max(v) for v in c["abundance"].values())
        ax.set_ylim(top * 1e-5, top * 3)
        ax.set_xlim(c["time"][0], c["time"][-1])
        ax.axvline(t_ref, color="#8a8984", lw=1, ls=":")
        ax.annotate(f"scored here ({t_ref / 1e6:g} Myr)", (t_ref, top * 2), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color="#52514e")
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(f"Abundance of {c['target']} relative to H$_2$")
        ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        fig.tight_layout()
        html = mo.as_html(fig)
        plt.close(fig)
        return html

    def candidates_table(c, k=5):
        rows = [r for r in c["top_candidates"] if r["rank"] <= k or r["hidden"]]
        lines = ["| Rank | Candidate reaction proposed by TAPIR | |", "|---:|:---|:---|"]
        for r in rows:
            mark = "**← the hidden reaction**" if r["hidden"] else ""
            lines.append(f"| {r['rank']} | `{arrow(r['reaction'])}` | {mark} |")
        return mo.md("\n".join(lines))

    def case_result(c):
        """One line next to the case selector."""
        n, rank = c["n"], c["rank"]
        if rank is None:
            return mo.md(f"Not in TAPIR's final ranking of {n} candidates")
        if n == 1:
            return mo.md("Rank **#1 of 1**: only one candidate, so this case is trivial for ranking")
        return mo.md(f"Recovered at rank **#{fmt_rank(rank)} of {n}**")

    def hidden_block(c):
        drop = c.get("drop_at_t_ref")
        t_ref = validation["configuration"]["t_ref_yr"]
        effect = (f"Removing it lowers {c['target']} by ~{float(f'{drop:.2g}'):,.0f}× at the scoring time "
                  f"({t_ref / 1e6:g} Myr)." if drop else "")
        return mo.callout(mo.md(f"**Hidden pathway** &nbsp; `{arrow(c['hidden_reaction'])}`  \n{effect}"),
                          kind="info")

    def case_view(c):
        n, rank = c["n"], c["rank"]
        stats = mo.hstack([
            mo.stat(fmt_rank(rank), label="Rank of the hidden reaction", bordered=True,
                    caption="1 = TAPIR's first choice"),
            mo.stat(str(n), label="Candidates ranked (n)", bordered=True,
                    caption="only one candidate: trivial for ranking" if n == 1 else None),
            mo.stat("-" if c["rank_over_n"] is None else f"{c['rank_over_n']:.3f}", label="rank / n",
                    bordered=True, caption="lower is better"),
            mo.stat(f"{c['random_reciprocal_rank']:.3f}", label="Random expectation H_n / n", bordered=True,
                    caption=f"expected 1/rank if shuffled (here 1/rank = {c['reciprocal_rank']:.3f})"),
        ], widths="equal", gap=0.5)
        return mo.vstack([
            hidden_block(c),
            plot_case(c),
            mo.md("<small>TAPIR's job is to find the missing pathway. It re-adds it with a generic rate, not "
                  "the original KIDA value, so the blue curve need not match the grey one exactly.</small>"),
            stats,
            mo.md("**TAPIR's ranking for this case**"),
            candidates_table(c),
        ], gap=0.75)

    def global_view():
        # The headline hits@1 lives in the app's summary row; here only the metric behind it.
        m = validation["metrics"]
        return mo.md(
            f"**{m['hits']['1']} of {m['n_cases']}** hidden reactions ranked #1 · "
            f"**MRR {m['mrr']:.2f}** vs **{m['random_mrr']:.2f}** random baseline  \n"
            "<small>Mean reciprocal rank: higher is better (1 = always first). The baseline is the MRR "
            "expected if TAPIR's final candidates were shuffled.</small>"
        )

    return case_result, case_view, global_view


@app.cell
def _(case_picker, case_result, case_view, global_view, mo, validation, validation_error):
    INTRO = mo.md(
        "<div style='font-size:1.3rem;font-weight:600'>Can TAPIR find a reaction we hid on purpose?</div>"
        "<div>We delete one known reaction from the chemistry model, then ask TAPIR to find it again.</div>"
    )
    if validation is None:
        validation_view = mo.vstack([INTRO, mo.callout(mo.md(validation_error), kind="danger",
                                                       title="Validation data not available")])
    else:
        _case = next(c for c in validation["cases"] if c["case"] == case_picker.value)
        validation_view = mo.vstack([
            INTRO,
            global_view(),
            mo.hstack([case_picker, case_result(_case)], justify="start", align="center", gap=1.5),
            case_view(_case),
        ], gap=1)
    return (validation_view,)


@app.cell
def _(chem_review, mo, plt):
    DISCLAIMER = "Ranked hypotheses — not confirmed reactions."
    MOTTO = "LLM proposes. Code verifies. Physics decides."
    NO_EVIDENCE = ("No supporting evidence found in the literature search; this does not prove that no "
                   "literature exists.")
    SOURCE_LABELS = {"analogy": "Analogy", "llm": "LLM", "analogy+llm": "Analogy + LLM"}
    LIT_LABELS = {"measured": "Measured in the laboratory", "computed": "Computed (theory)",
                  "proposed": "Proposed in the literature", "no_evidence_found": "No evidence found",
                  "unchecked": "Not checked"}
    K_STATUS_LABELS = {"ok": "found by bisection", "not_reachable": "not reachable in the tested range",
                       "already_above": "observation reached even at the lowest rate", "error": "simulation error"}
    K_SOURCE_LABELS = {"capture_guess": "generic estimate for its reaction class"}
    DROP_LABELS = {"not_conserved": "not balanced", "already_in_network": "already in the network",
                   "unknown_species": "unknown species", "no_target": "do not make the target"}
    REVIEW_LABELS = {"implausible": "Rejected", "plausible": "Plausible", "uncertain": "Uncertain"}
    LLM_COLOR = "#6f5bd3"
    SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    REVIEW_BY_REACTION = {r["reaction"]: r for r in (chem_review or {}).get("reactions") or []}

    def sci(v, digits=2):
        if v is None:
            return "–"
        m, e = f"{v:.{digits - 1}e}".split("e")
        return f"{m} × 10{str(int(e)).translate(SUP)}"

    def signed(v):
        return "–" if v is None else f"{v:+.2g}".replace("-", "−")

    def rxn(reaction):
        return reaction.replace(" -> ", " → ")

    def cell(text):
        return str(text).replace("|", "\\|")

    def source_key(h):
        return "+".join(sorted(h.get("source") or []))

    def source_badge(h):
        """LLM candidates get an outlined tag: identifiable, not highlighted as better."""
        key = source_key(h)
        label = SOURCE_LABELS.get(key, key or "–")
        if "llm" not in key:
            return label
        return (f"<span style='border:1px solid {LLM_COLOR};color:{LLM_COLOR};border-radius:999px;"
                f"padding:0 0.5em;font-size:0.85em;white-space:nowrap'>{label}</span>")

    def k_tested_max(d, h):
        s = (d.get("settings") or {}).get("k_needed") or {}
        return (s.get("k_max_by_class") or {}).get(h.get("class"), s.get("k_max"))

    def feasibility(d, h):
        """(short label for tables, sentence for the detail view), from k_status / k_needed / capture_feasible."""
        status, need, cap = h.get("k_status"), h.get("k_needed"), h.get("k_max_class")
        cls = h.get("class") or "this class of"
        if h.get("capture_feasible") is True:
            if status == "already_above":
                return "Plausible", "Reaches the observed abundance even at the lowest rate tested."
            return "Plausible", (f"Needs k ≈ {sci(need)} cm³ s⁻¹, within the capture limit of {cls} reactions "
                                 f"({sci(cap, 1)} cm³ s⁻¹).")
        if status == "not_reachable":
            km = k_tested_max(d, h)
            top = f" even at k = {sci(km, 1)} cm³ s⁻¹, the fastest rate tested" if km else ""
            return "Not reachable", f"The model stays below the observed abundance{top}."
        if status == "ok" and need and cap:
            return (f"Implausible: {need / cap:.2g}× the capture limit",
                    f"Needs k ≈ {sci(need)} cm³ s⁻¹, {need / cap:.2g}× the capture limit of {cls} reactions "
                    f"({sci(cap, 1)} cm³ s⁻¹): physically implausible.")
        return "Undetermined", "The required rate could not be determined."

    def lit_status(h):
        """(raw status, readable label). A missing literature block is 'unchecked', never 'no evidence'."""
        s = (h.get("literature") or {}).get("status") or "unchecked"
        return s, LIT_LABELS.get(s, s)

    def ref_link(r):
        ref = r.get("ref") or ""
        url = ref if ref.startswith("http") else f"https://doi.org/{ref}"
        channel = "" if r.get("same_channel") else " <small>(other products)</small>"
        return f"[{cell(r.get('title') or ref)}]({url}){channel}"

    def ratio_change(d, h):
        base, x = d.get("x_ratio_base"), h.get("x_ratio")
        return None if x is None else (f"{x:.2f}" if base is None else f"{base:.2f} → {x:.2f}")

    def review_label(h):
        r = REVIEW_BY_REACTION.get(h.get("reaction"))
        return "Not reviewed" if r is None else REVIEW_LABELS.get(r.get("verdict"), r.get("verdict", "–"))

    def hypotheses_table(d, hyps, upstream=False):
        sp, base = d.get("species", "target"), d.get("x_ratio_base")
        if upstream:
            lines = ["| Rank | Reaction | Feeds precursor | ΔD | Physical feasibility | Chemical review |",
                     "|---:|:---|:---|---:|:---|:---|"]
        else:
            ratio_head = f"{sp} model / observed" + (f" (base {base:.2f})" if base is not None else "")
            lines = [f"| Rank | Reaction | Source | {ratio_head} | ΔD | Physical feasibility | Chemical review |",
                     "|---:|:---|:---|---:|---:|:---|:---|"]
        for h in hyps:
            row = [str(h["rank"]), f"`{rxn(h['reaction'])}`"]
            if upstream:
                row += [f"`{h['via']}`" if h.get("via") else "–"]
            else:
                x = h.get("x_ratio")
                row += [source_badge(h), "–" if x is None else f"{x:.2f}"]
            row += [signed(h.get("delta_D")), feasibility(d, h)[0], review_label(h)]
            lines.append("| " + " | ".join(row) + " |")
        return mo.md("\n".join(lines))

    def table_caption(d):
        n_obs = len(d.get("D_species") or [])
        caps = sorted({(h["class"], h["k_max_class"]) for h in d.get("hypotheses", [])
                       if h.get("class") and h.get("k_max_class")})
        caps = ", ".join(f"{c} {sci(k, 1)}" for c, k in caps)
        return mo.md(
            f"<small>**Model / observed**: predicted abundance of {d.get('species')} with the hypothesis added, "
            "divided by the observed one (1 = deficit closed). "
            f"**ΔD**: change in the mean log-distance to the {n_obs} observed species (negative = closer to the "
            "observations). **Physical feasibility**: rate needed to reach the observed abundance, compared "
            f"with the capture limit of the reaction class (the fastest plausible rate: {caps} cm³ s⁻¹).</small>"
        )

    def verdict(d):
        hyps = d.get("hypotheses") or []
        better = [h for h in hyps if (h.get("delta_D") or 0) < 0]
        ranked = [h for h in hyps if h.get("x_ratio") is not None]
        n_ok = d.get("n_reachable")
        n_ok = sum(h.get("capture_feasible") is True for h in hyps) if n_ok is None else n_ok
        best = max(ranked, key=lambda h: h["x_ratio"]) if ranked else None
        best_txt = "" if best is None or d.get("x_ratio_base") is None else (
            f" The best one (#{best['rank']}) takes {d['species']} from {d['x_ratio_base']:.0%} to "
            f"{best['x_ratio']:.0%} of the observed abundance.")
        if n_ok == 0:
            n_better = f"All {len(hyps)}" if len(better) == len(hyps) else f"{len(better)} of the {len(hyps)}"
            return mo.callout(mo.md(
                f"**{n_better} top hypotheses improve the model, but none closes the deficit at a physically "
                f"plausible rate.**{best_txt}"), kind="warn")
        return mo.callout(mo.md(f"**{n_ok} of the top {len(hyps)} hypotheses reach the observed abundance at "
                                f"a physically plausible rate.**{best_txt}"), kind="success")

    def llm_funnel(d):
        """What the code did with the LLM's proposals for the target, from d['llm']['calls']."""
        llm = d.get("llm") or {}
        c = next((c for c in llm.get("calls") or [] if c.get("target") == d.get("species")), None)
        if not c or c.get("returned_raw") is None:
            return None
        dropped = {k: v for k, v in (c.get("discarded") or {}).items() if v}
        reasons = ", ".join(f"{v} {DROP_LABELS.get(k, k.replace('_', ' '))}" for k, v in dropped.items())
        hyps = d.get("hypotheses") or []
        n_llm = sum("llm" in source_key(h) for h in hyps)
        kept = c.get("considered")
        model = f" ({llm['model']})" if llm.get("model") else ""
        return (f"The LLM{model} returned {c['returned_raw']} reactions. Code checks discarded "
                f"{sum(dropped.values())}" + (f" ({reasons})" if reasons else "")
                + (f"; the remaining {kept} were simulated and ranked exactly like the analogy candidates"
                   if kept is not None else "")
                + f". {n_llm} of the top {len(hyps)} come from the LLM.")

    def llm_spotlight(d, h):
        s, label = lit_status(h)
        lit = h.get("literature") or {}
        x = ratio_change(d, h)
        bullets = []
        if x:
            bullets.append(f"- **Improves the fit a little**: model / observed {x} (ΔD {signed(h.get('delta_D'))}).")
        bullets.append(f"- **Physics**: {feasibility(d, h)[1]}")
        lit_line = f"- **Literature**: {label} (`{s}`)."
        if s == "no_evidence_found":
            lit_line += f" {NO_EVIDENCE}"
        bullets.append(lit_line)
        if lit.get("n_bond_refs"):
            bullets.append(f"  <small>{lit['n_bond_refs']} search results were reviewed.</small>")
        funnel = llm_funnel(d)
        return mo.callout(mo.vstack([
            mo.md(f"{source_badge(h)} &nbsp; **Proposed only by the LLM** · rank #{h['rank']} of "
                  f"{len(d.get('hypotheses') or [])}"),
            mo.md(f"<div style='font-size:1.25rem;font-weight:600'><code>{rxn(h['reaction'])}</code></div>"),
            mo.md("\n".join(bullets)),
            mo.md(f"<small style='opacity:0.75'>{funnel}</small>") if funnel else mo.md(""),
            mo.md(f"<div style='font-weight:600;letter-spacing:0.02em;color:{LLM_COLOR}'>{MOTTO}</div>"),
        ], gap=0.4), kind="neutral")

    def plot_hypothesis(d, h):
        t, base, curve, x_obs = d.get("t"), d.get("curve_base"), h.get("curve"), d.get("X_obs")
        if not (t and base and curve and x_obs):
            return None
        t_ref = (d.get("config") or {}).get("t_ref")
        fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=110)
        for ys, color, label, style in ((base, "#52514e", "Current model", dict(lw=3.0, alpha=0.45)),
                                        (curve, "#2a78d6", "Model + this hypothesis", dict(lw=2.0))):
            pts = [(a, b) for a, b in zip(t, ys) if b > 0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, label=label, **style)
        obs_label = "Observed" + (f" ({d['obs_source']})" if d.get("obs_source") else "")
        ax.axhline(x_obs, color="#eb6834", lw=1.6, ls="--", label=obs_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(x_obs * 1e-3, x_obs * 4)
        ax.set_xlim(t[0], t[-1])
        if t_ref:
            ax.axvline(t_ref, color="#8a8984", lw=1, ls=":")
            ax.annotate(f"scored here ({t_ref / 1e6:g} Myr)", (t_ref, x_obs * 2.6), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9, color="#52514e")
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(f"Abundance of {d.get('species')} relative to H$_2$")
        ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="lower right")
        fig.tight_layout()
        html = mo.as_html(fig)
        plt.close(fig)
        return html

    def detail_view(d, h):
        rows = []

        def add(label, value):
            if value not in (None, "", []):
                rows.append(f"| {label} | {value} |")

        add("Reaction", f"`{rxn(h['reaction'])}`")
        add("Source", f"{source_badge(h)} <small>(`{source_key(h)}`)</small>")
        add("Rationale", cell(h.get("rationale") or ""))
        add("Reaction class", h.get("class"))
        add("Feeds precursor", f"`{h['via']}`" if h.get("via") else None)
        x = ratio_change(d, h)
        add("Model / observed", f"{x} <small>(`x_ratio`; 1 = deficit closed)</small>" if x else None)
        add("ΔD", f"{signed(h['delta_D'])} <small>(`delta_D`; negative = better fit)</small>"
            if h.get("delta_D") is not None else None)
        if h.get("k") is not None:
            k_src = h.get("k_source")
            add("Rate used in the simulation", f"{sci(h['k'])} cm³ s⁻¹" + (
                f" <small>({K_SOURCE_LABELS.get(k_src, k_src)}, `{k_src}`)</small>" if k_src else ""))
        add("Rate needed to match the observation", f"{sci(h['k_needed'])} cm³ s⁻¹ <small>(`k_needed`)</small>"
            if h.get("k_needed") is not None else None)
        if h.get("k_status"):
            add("Rate search", f"{K_STATUS_LABELS.get(h['k_status'], h['k_status'])} <small>(`{h['k_status']}`)</small>")
        add("Physical feasibility", feasibility(d, h)[1] + (
            f" <small>(`capture_feasible = {h['capture_feasible']}`)</small>"
            if h.get("capture_feasible") is not None else ""))
        s, label = lit_status(h)
        lit = h.get("literature") or {}
        add("Literature", f"**{label}** <small>(`{s}`)</small>")
        if s == "no_evidence_found":
            add("", f"<small>{NO_EVIDENCE}</small>")
        if s != "unchecked":
            add("Literature note", cell(lit.get("note") or ""))
            add("References", "<br>".join(ref_link(r) for r in lit.get("refs") or []))
        review = REVIEW_BY_REACTION.get(h.get("reaction"))
        if review is not None:
            add("Chemical review", f"**{review_label(h)}** <small>(team chemist)</small>")
            add("Chemical reason", cell(review.get("reason") or ""))
            refs = review.get("refs") or []
            add("Chemical-review references", "<br>".join(f"[{cell(ref)}](https://doi.org/{ref})" for ref in refs))
        table = mo.md("| | |\n|:---|:---|\n" + "\n".join(rows))
        plot = plot_hypothesis(d, h)
        return mo.vstack([table] + ([plot] if plot is not None else []), gap=0.75)

    def robustness_section(d):
        """Comparison of the stored CH2CCH run with the second model configuration."""
        c2, b1, b2 = d.get("config2"), d.get("bottleneck"), d.get("bottleneck_config2")
        if not (c2 and b1 and b2):
            return None
        reaction = b1.get("reaction")
        share1 = next((e.get("frac") for e in d.get("existing") or [] if e.get("reaction") == reaction), None)
        share2 = next((e.get("frac") for e in c2.get("main_formation") or [] if e.get("reaction") == reaction), None)

        def row(label, cfg, ratio, share, bn, reachable):
            gain = bn["X_target_scaled"] / bn["X_target_base"]
            limiting = min(bn.get("species") or {}, key=lambda s: bn["species"][s]["ratio"], default=None)
            left = bn["species"][limiting]["ratio"] if limiting else None
            share_text = "–" if share is None else f"{share:.0%}"
            limiting_text = "–" if limiting is None else f"`{limiting}` at {left:.1%}"
            return (f"| {label} | {cfg.get('C_O', '–'):g} | {sci(cfg.get('S'), 2)} | "
                    f"{cfg.get('t_ref', 0) / 1e6:g} | {ratio:.3f} | "
                    f"{share_text} | {gain:.2f}× | {limiting_text} | {reachable} |")

        lines = [
            "| Model configuration | C/O | S | t_ref [Myr] | Model / observed | Main-route share | Target with route ×1000 | Limiting supply left | Reachable hypotheses |",
            "|:---|---:|---:|---:|---:|---:|---:|:---|---:|",
            row("Reference", d.get("config") or {}, d["x_ratio_base"], share1, b1, d.get("n_reachable", 0)),
            row("Second", c2, c2["x_ratio_base"], share2, b2, c2.get("n_reachable", 0)),
        ]
        robust = c2.get("n_robust", 0)
        total = len(d.get("hypotheses") or [])
        same = b1.get("supply_limited") and b2.get("supply_limited") and not d.get("n_reachable") and not c2.get("n_reachable")
        conclusion = ("**The diagnosis survives the second configuration:** the main route exhausts its limiting "
                      "precursor and no tested hypothesis reaches the observation below its capture limit."
                      if same else "The second configuration changes part of the diagnosis; inspect the comparison below.")
        return mo.vstack([
            mo.md("<div style='font-size:1.15rem;font-weight:600'>Does the conclusion depend on the model configuration?</div>"),
            mo.callout(mo.md(conclusion), kind="success" if same else "warn"),
            mo.md("\n".join(lines)),
            mo.md(f"<small>{robust}/{total} hypotheses move the target closer to the observation in both settings, "
                  "but this is not enough to make them physically reachable. The second setting uses more sulphur "
                  "and an earlier scoring time.</small>"),
        ], gap=0.6)

    def chemical_review_section():
        if chem_review is None:
            return None
        reactions = chem_review.get("reactions") or []
        rejected = sum(r.get("verdict") == "implausible" for r in reactions)
        direct = sum(r.get("file") == "CH2CCH" for r in reactions)
        upstream = len(reactions) - direct
        lines = ["| Candidate | Search | Verdict | Reason |", "|:---|:---|:---|:---|"]
        for r in reactions:
            where = "Direct" if r.get("file") == "CH2CCH" else "Upstream"
            lines.append(f"| `{rxn(r['reaction'])}` | {where} | {REVIEW_LABELS.get(r['verdict'], r['verdict'])} | "
                         f"{cell(r['reason'])} |")
        return mo.vstack([
            mo.md("<div style='font-size:1.15rem;font-weight:600'>Chemical review</div>"),
            mo.callout(mo.md(f"**{rejected} of {len(reactions)} reviewed candidates are chemically implausible** "
                             f"({direct} direct and {upstream} upstream). The automatic physics screen is necessary, "
                             "but a favorable model score alone is not chemical evidence."), kind="warn"),
            mo.accordion({f"Why the {len(reactions)} candidates were rejected": mo.md("\n".join(lines))}),
            mo.md(f"<small>{chem_review.get('reviewer', '')}. Energetic estimates are approximate.</small>"),
        ], gap=0.6)

    def upstream_section(up, d):
        sp, x_obs = d.get("species", "the target"), d.get("X_obs")
        precursors = up.get("upstream") or []
        hyps = up.get("hypotheses") or []
        bn = up.get("bottleneck") or d.get("bottleneck") or {}
        parts = [mo.md("<div style='font-size:1.15rem;font-weight:600'>The direct search is not enough</div>"),
                 mo.md(f"TAPIR also traces the precursors of {sp}"
                       + (f" ({', '.join(f'`{p}`' for p in precursors)})" if precursors else "")
                       + f": it looks for reactions that make more of them, still scored on {sp} itself.")]

        limiting, sp_info = None, bn.get("species") or {}
        if bn.get("reaction"):
            reac = bn["reaction"].split(" -> ")[0].split(" + ")
            limiting = min((s for s in reac if s in sp_info), key=lambda s: sp_info[s]["ratio"], default=None)
        if bn.get("supply_limited") and limiting:
            parts.append(mo.callout(mo.md(
                f"**Upstream analysis points to {limiting} supply as the bottleneck.**"), kind="info"))
            main = next((e for e in d.get("existing") or [] if e.get("reaction") == bn["reaction"]), {})
            share = f" ({main['frac']:.0%} of {sp} formation in the model)" if main.get("frac") is not None else ""
            others = [f"`{s}` {sp_info[s]['ratio']:.0%}" for s in reac if s in sp_info and s != limiting]
            reached = (f" ({bn['X_target_scaled'] / x_obs:.0%} of the observed abundance)"
                       if bn.get("X_target_scaled") and x_obs else "")
            evidence = [
                f"- Multiplying the main formation reaction `{rxn(bn['reaction'])}`{share} by {bn['factor']:g} "
                f"only raises {sp} from {sci(bn.get('X_target_base'))} to {sci(bn.get('X_target_scaled'))}"
                f"{reached}: `{limiting}` runs out, falling to {sp_info[limiting]['ratio']:.1%} of its base "
                "abundance" + (f" (the other reactant stays: {', '.join(others)})" if others else "") + "."]
            ex = d.get("existing") or []
            if ex and all(e.get("factor_status") == "not_reachable" for e in ex):
                evidence.append(f"- None of the {len(ex)} main known formation and destruction reactions of {sp} "
                                "reaches the observation by scaling its rate either.")
            via = [h for h in hyps[:5] if h.get("via") == limiting]
            best_up = max((h for h in hyps if h.get("x_ratio") is not None), key=lambda h: h["x_ratio"], default=None)
            best_dir = max((h["x_ratio"] for h in d.get("hypotheses") or [] if h.get("x_ratio") is not None),
                           default=None)
            if best_up and best_dir:
                evidence.append(
                    f"- {len(via)} of the top 5 upstream hypotheses produce `{limiting}`, and the best one reaches "
                    f"{best_up['x_ratio']:.0%} of the observed {sp} (best direct hypothesis: {best_dir:.0%}).")
            parts.append(mo.md("\n".join(evidence)))
        if hyps:
            n_ok = up.get("n_reachable")
            parts.append(mo.md(f"**Top 5 upstream hypotheses**" + (
                " · none reaches the observation at a physically plausible rate" if n_ok == 0 else "")))
            parts.append(hypotheses_table(up, hyps[:5], upstream=True))
            if all(lit_status(h)[0] == "unchecked" for h in hyps[:5]):
                parts.append(mo.md("<small>The literature search has not been run for upstream hypotheses yet, "
                                   "so no literature status is shown here.</small>"))
        return mo.vstack(parts, gap=0.75)

    return (DISCLAIMER, MOTTO, chemical_review_section, detail_view, feasibility, hypotheses_table, lit_status,
            llm_spotlight, robustness_section, rxn, sci, signed, source_key, table_caption, upstream_section,
            verdict)


@app.cell
def _(discovery, mo, rxn, source_key):
    hypothesis_picker = None
    if discovery is not None and discovery.get("hypotheses"):
        _opts = {f"#{h['rank']} · {rxn(h['reaction'])}" + (" · LLM" if "llm" in source_key(h) else ""): h["rank"]
                 for h in discovery["hypotheses"]}
        hypothesis_picker = mo.ui.dropdown(options=_opts, value=next(iter(_opts)), label="Hypothesis")
    return (hypothesis_picker,)


@app.cell
def _(DISCLAIMER, chem_review_error, chemical_review_section, detail_view, discovery, discovery_error,
      hypotheses_table, hypothesis_picker, llm_spotlight, mo, robustness_section, sci, source_key,
      table_caption, upstream, upstream_error, upstream_section, verdict):
    _banner = mo.md(f"<div style='padding:0.35rem 0.75rem;border-left:3px solid #8a8984;opacity:0.85'>"
                    f"<b>{DISCLAIMER}</b></div>")
    if discovery is None:
        discovery_view = mo.vstack([
            mo.md("<div style='font-size:1.3rem;font-weight:600'>Discovery</div>"), _banner,
            mo.callout(mo.md(discovery_error), kind="danger", title="Discovery data not available")])
    else:
        _d = discovery
        _sp, _hyps = _d.get("species", "the target"), _d.get("hypotheses") or []
        _intro = mo.md(
            "<div style='font-size:0.8rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;"
            "opacity:0.6'>Discovery</div>"
            f"<div style='font-size:1.3rem;font-weight:600'>Why does the model underproduce {_sp}?</div>"
        )
        _x_obs, _x_base = _d.get("X_obs"), _d.get("X_base")
        _stats = []
        if _x_obs:
            _stats.append(mo.stat(sci(_x_obs), label=f"Observed {_sp}", bordered=True,
                                  caption=_d.get("obs_source") and f"relative to H₂ · {_d['obs_source']}"))
        if _x_base:
            _stats.append(mo.stat(sci(_x_base), label="Model", bordered=True,
                                  caption=f"relative to H₂ · at {_d['config']['t_ref'] / 1e6:g} Myr"
                                  if (_d.get("config") or {}).get("t_ref") else "relative to H₂"))
        if _x_obs and _x_base:
            _stats.append(mo.stat(f"{_x_obs / _x_base:.1f}×", label="Deficit", bordered=True,
                                  caption="observed / model"))
        _gens = " and ".join(f"**{g}**" for g in _d.get("generators") or [])
        _search = mo.md(
            "TAPIR searches for the missing pathways: it generated "
            + (f"{_d['n_generated']} candidate reactions" if _d.get("n_generated") else "candidate reactions")
            + (f" by {_gens}" if _gens else "")
            + (f", simulated the {_d['n_simulated']} with the largest flux" if _d.get("n_simulated") else "")
            + f" and ranked them by how much closer they bring the model to the observations."
        )
        _with_rate = (_d.get("n_generated", 0) - _d.get("n_no_rate", 0) - _d.get("n_k_zero", 0))
        _funnel = mo.hstack([
            mo.stat(str(_d.get("n_generated", "–")), label="Generated", bordered=True,
                    caption="analogy + LLM"),
            mo.stat(str(_with_rate), label="With an estimated rate", bordered=True,
                    caption="after rate and radical checks"),
            mo.stat(str(_d.get("n_simulated", "–")), label="Simulated", bordered=True,
                    caption="largest estimated flux"),
            mo.stat(str(_d.get("n_reachable", "–")), label="Physically reachable", bordered=True,
                    caption="no single reaction closes the gap: see the upstream diagnosis"
                    if _d.get("n_reachable") == 0 else "at or below the capture limit"),
        ], widths="equal", gap=0.5)
        _llm_only = next((h for h in _hyps if source_key(h) == "llm"), None)
        _parts = [_intro, mo.hstack(_stats, widths="equal", gap=0.5), _search, _funnel,
                  mo.md(f"**Top 5 direct hypotheses** for {_sp}"), hypotheses_table(_d, _hyps[:5]),
                  table_caption(_d), verdict(_d)]
        if _llm_only is not None:
            _parts.append(llm_spotlight(_d, _llm_only))
        if hypothesis_picker is not None:
            _h = next(h for h in _hyps if h["rank"] == hypothesis_picker.value)
            _parts += [mo.md("**Explore a hypothesis**"), hypothesis_picker, detail_view(_d, _h)]
        if upstream is not None:
            _parts.append(upstream_section(upstream, _d))
        else:
            _parts.append(mo.callout(mo.md(upstream_error), kind="neutral", title="Upstream data not available"))
        _robustness = robustness_section(_d)
        if _robustness is not None:
            _parts.append(_robustness)
        _review = chemical_review_section()
        if _review is not None:
            _parts.append(_review)
        elif chem_review_error:
            _parts.append(mo.callout(mo.md(chem_review_error), kind="neutral",
                                     title="Chemical review not available"))
        _parts.append(_banner)
        discovery_view = mo.vstack(_parts, gap=1)
    return (discovery_view,)


@app.cell
def _(MOTTO, bench_error, bench_summary, discovery, feasibility, lit_status, mo, rxn, signed, source_key):
    STRATEGIES = (("analogy", "Analogy", "templates from analogous KIDA reactions"),
                  ("llm", "LLM", "reactions proposed by the language model"),
                  ("union", "Combined", "analogy and LLM candidates ranked together"))
    BASELINE_LABEL = "Expected MRR from random ordering of recoverable candidates, counting misses as zero."
    _intro = ("<div style='font-size:0.8rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;"
              "opacity:0.6'>Generator</div>"
              "<div style='font-size:1.3rem;font-weight:600'>Does an LLM improve hypothesis generation?</div>")

    def _num(v, digits=3):
        return "—" if v is None else f"{v:.{digits}f}"

    def _of(v, n):
        return "—" if v is None or not n else f"{v}/{n}"

    def _rank(r, n=None):
        if r is None:
            return "—"
        return f"#{r:g}" + (f" <small style='opacity:0.6'>/ {n}</small>" if n else "")

    def _card(title, desc, m):
        n = m.get("n_cases")
        rows = [("Hits@1", _of(m.get("hits@1"), n)), ("Hits@5", _of(m.get("hits@5"), n)),
                ("MRR", _num(m.get("MRR"))), ("Random ordering", _num(m.get("random_mrr_end_to_end")))]
        rows = "".join(f"<div style='display:flex;justify-content:space-between;padding:0.15rem 0;"
                       f"{'opacity:0.6;font-size:0.85rem' if k == 'Random ordering' else ''}'>"
                       f"<span>{k}</span><b>{v}</b></div>" for k, v in rows)
        return mo.md(
            "<div style='border:1px solid rgba(128,128,128,0.35);border-radius:10px;padding:0.9rem 1.1rem'>"
            f"<div style='font-weight:600'>{title}</div><div style='font-size:0.8rem;opacity:0.65'>{desc}</div>"
            f"<div style='font-size:2rem;font-weight:700;line-height:1.1;margin-top:0.6rem'>"
            f"{_of(m.get('n_recovered'), n)}</div>"
            "<div style='font-size:0.85rem;opacity:0.7;margin-bottom:0.5rem'>hidden reactions recovered</div>"
            f"{rows}</div>")

    def _conclusion(ms):
        a, l, u = (ms.get(k, {}).get("MRR") for k in ("analogy", "llm", "union"))
        if None not in (a, l, u) and a > l and a >= u:
            return ("**Not on this validation benchmark.** Analogy is the stronger generator for recovering known "
                    "chemistry, while the LLM adds diversity.")
        return "MRR on this benchmark: " + ", ".join(f"{t} {_num(ms[k].get('MRR'))}" for k, t, _ in STRATEGIES
                                                      if k in ms) + "."

    def _case_table(ms, cases, targets):
        hidden = {c["case"]: c.get("hidden_reaction") for c in cases}
        present = [(k, t) for k, t, _ in STRATEGIES if k in ms]
        lines = ["| Target | Hidden reaction | " + " | ".join(t for _, t in present) + " |",
                 "|:---|:---|" + "---:|" * len(present)]
        for c in targets:
            cells = [_rank((ms[k].get("ranks") or {}).get(c), (ms[k].get("n") or {}).get(c)) for k, _ in present]
            lines.append(f"| `{c}` | " + (f"`{rxn(hidden[c])}`" if hidden.get(c) else "–") + " | "
                         + " | ".join(cells) + " |")
        return mo.vstack([mo.md("\n".join(lines)), mo.md(
            "<small>Rank of the hidden reaction in each final ranking (/ n = candidates ranked); "
            "— = not recovered: the hidden reaction is not among the ranked candidates.</small>")], gap=0.25)

    def _examples(ms, cases, targets):
        """Two patterns read from the ranks: LLM candidates pushing the known pathway down, and agreement at #1."""
        if not all(k in ms for k in ("analogy", "llm", "union")):
            return None
        rk = {k: ms[k].get("ranks") or {} for k in ("analogy", "llm", "union")}
        top1 = {c["case"]: c for c in cases if c.get("strategy") == "union"}
        notes = []
        for c in targets:
            a, l, u = rk["analogy"].get(c), rk["llm"].get(c), rk["union"].get(c)
            if a == 1 and u is not None and u > a and l is not None and not any(n[0] == "push" for n in notes):
                first = top1.get(c, {})
                src = {"llm": "proposed by the LLM", "analogy": "from analogy",
                       "analogy+llm": "proposed by both"}.get(first.get("top1_source"))
                by = (f" `{rxn(first['top1'])}`" + (f" ({src})" if src else "") + " took #1."
                      if first.get("top1") else "")
                notes.append(("push", c, a, l, u, "Adding LLM candidates introduced competition and moved the known "
                                                  f"pathway from #{a:g} to #{u:g}.{by}"))
            elif a == l == u == 1 and not any(n[0] == "agree" for n in notes):
                notes.append(("agree", c, a, l, u, "Both generators independently recovered the known pathway at #1."))
        if not notes:
            return None
        return mo.hstack([mo.callout(mo.md(
            f"**`{c}`** <small style='opacity:0.7'>analogy {_rank(a)} · LLM {_rank(l)} · combined {_rank(u)}</small>"
            f"  \n{msg}"), kind="neutral") for _, c, a, l, u, msg in notes], widths="equal", gap=0.75)

    def _llm_lesson(ms, cases):
        m = ms.get("llm")
        if not m or m.get("n_recovered") is None:
            return None
        n, rec = m.get("n_cases"), m["n_recovered"]
        missing = sum(c.get("strategy") == "llm" and c.get("hidden_generated") is False for c in cases)
        cm, rc = m.get("conditional_mrr"), m.get("random_mrr_conditional")
        bullets = [f"- The LLM recovers **{rec}/{n}** hidden reactions (recall {_num(m.get('recall'))})."]
        if missing:
            bullets.append(f"- Its main limitation is generation: in {missing} of {n} cases it never proposed the hidden "
                           "reaction, so no ranking could recover it.")
        if cm is not None and rc is not None:
            bullets.append(f"- When the hidden reaction is among its candidates, it is ranked well: conditional MRR "
                           f"**{_num(cm)}** vs **{_num(rc)}** for random ordering of the same candidates.")
            if cm > rc and rec < n:
                bullets.append("- This suggests the main problem is not ranking the right candidate, but generating it.")
        return mo.md("\n".join(bullets))

    def _metrics_detail(ms, notes):
        present = [(k, t) for k, t, _ in STRATEGIES if k in ms]
        lines = ["| | Recovered | MRR (overall) | Random, end-to-end | Conditional MRR | Random, conditional |",
                 "|:---|---:|---:|---:|---:|---:|"]
        for k, t in present:
            m = ms[k]
            lines.append(f"| {t} | {_of(m.get('n_recovered'), m.get('n_cases'))} | {_num(m.get('MRR'))} | "
                         f"{_num(m.get('random_mrr_end_to_end'))} | {_num(m.get('conditional_mrr'))} | "
                         f"{_num(m.get('random_mrr_conditional'))} |")
        defs = [
            "- **MRR (overall)**: mean of 1/rank over all cases; a hidden reaction that is not recovered counts as 0.",
            f"- **Random, end-to-end**: {BASELINE_LABEL} The baseline for the overall MRR.",
            "- **Conditional MRR**: mean of 1/rank over the recovered cases only.",
            "- **Random, conditional**: expected MRR from random ordering, over the recovered cases only. The "
            "baseline for the conditional MRR.",
        ]
        tech = ["<small>The export also keeps `random_MRR`, only for compatibility: it averages H<sub>n</sub>/n even "
                "when the hidden reaction is absent, so it is neither baseline and is not used here.</small>"]
        if notes.get("contamination"):
            tech.append(f"<small>{notes['contamination']}</small>")
        return mo.accordion({"Overall vs conditional metrics": mo.vstack(
            [mo.md("\n".join(lines)), mo.md("\n".join(defs))] + [mo.md(t) for t in tech], gap=0.5)})

    def _discovery_link(ms):
        parts = [mo.md("<div style='font-size:1.15rem;font-weight:600'>Known chemistry vs open discovery</div>")]
        bullets = []
        a, l = ms.get("analogy", {}).get("MRR"), ms.get("llm", {}).get("MRR")
        if a is not None and l is not None and a > l:
            bullets.append(f"- On this benchmark of known chemistry, analogy is stronger (MRR {_num(a)} vs {_num(l)}).")
        hyps = (discovery or {}).get("hypotheses") or []
        h = next((h for h in hyps if source_key(h) == "llm"), None)
        if h is not None:
            sp = discovery.get("species", "the target")
            bullets.append(f"- In the real {sp} discovery run, the LLM proposed `{rxn(h['reaction'])}` (rank #{h['rank']}), "
                           "a reaction the analogy generator did not produce.")
            checks = []
            if h.get("delta_D") is not None:
                checks.append(f"simulation shows only a small improvement (ΔD {signed(h['delta_D'])})")
            if h.get("capture_feasible") is False:
                checks.append("the physical limit rules out the rate it needs" + (
                    f" ({feasibility(discovery, h)[0].split(': ', 1)[-1]})" if h.get("k_status") == "ok" else ""))
            s, label = lit_status(h)
            if s == "no_evidence_found":
                checks.append("the literature search found no supporting evidence")
            if checks:
                bullets.append("- TAPIR did not accept it automatically: " + "; ".join(checks) + ".")
        if bullets:
            parts.append(mo.md("\n".join(bullets)))
        parts.append(mo.callout(mo.md("**The LLM expands the hypothesis space; it does not replace chemistry or "
                                      "physics.**"), kind="info"))
        parts.append(mo.md(f"<div style='font-weight:600;letter-spacing:0.02em;color:#6f5bd3'>{MOTTO}</div>"))
        return mo.vstack(parts, gap=0.6)

    _b = (bench_summary or {}).get("benchmark_b") or {}
    _ms = _b.get("metrics") or {}
    if bench_summary is None:
        _body = [mo.callout(mo.md(bench_error), kind="neutral") if bench_error else mo.md(
            "<span style='opacity:0.7'>The LLM-versus-analogy benchmark has not been exported yet. "
            "Its results will appear here once `app/data/bench_llm_summary.json` exists.</span>")]
    elif not _ms:
        _body = [mo.md("<span style='opacity:0.7'>The exported LLM benchmark contains no benchmark B metrics.</span>")]
    else:
        _cases = _b.get("cases") or []
        _targets = ((bench_summary.get("config") or {}).get("benchmark_b_cases")
                    or list(next(iter(_ms.values())).get("ranks") or {}))
        _n = next(iter(_ms.values())).get("n_cases")
        _meta = [f"Benchmark B: {_n} known reactions, each removed from the network and searched for again"
                 if _n else "Benchmark B", bench_summary.get("model") and f"LLM {bench_summary['model']}"]
        _body = [
            mo.md(_conclusion(_ms)),
            mo.md("<small style='opacity:0.7'>" + " · ".join(x for x in _meta if x) + "</small>"),
            mo.hstack([_card(t, d, _ms[k]) for k, t, d in STRATEGIES if k in _ms], widths="equal", gap=0.75),
            mo.md(f"<small><b>Random ordering</b>: {BASELINE_LABEL}</small>"),
            mo.md("**Rank of the hidden reaction, case by case**"),
            _case_table(_ms, _cases, _targets),
            _examples(_ms, _cases, _targets),
            mo.md("<div style='font-size:1.15rem;font-weight:600'>What TAPIR learns from the LLM</div>"),
            _llm_lesson(_ms, _cases),
            _metrics_detail(_ms, bench_summary.get("notes") or {}),
            _discovery_link(_ms),
        ]
    generator_view = mo.vstack([mo.md(_intro)] + [x for x in _body if x is not None], gap=1)
    return (generator_view,)


@app.cell
def _(atlas, mo):
    atlas_picker = None
    if atlas is not None and atlas.get("species"):
        _options = {s["species"]: s["species"] for s in atlas["species"]}
        atlas_picker = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Species")
    return (atlas_picker,)


@app.cell
def _(atlas, atlas_error, atlas_picker, mo, plt, rxn, sci):
    DIAGNOSIS_LABELS = {"supply_limited": "Supply-limited", "rate_limited": "Rate-limited", "other": "Other"}
    DIAGNOSIS_COLORS = {"supply_limited": "#2f6fed", "rate_limited": "#d8694f", "other": "#7d8792"}

    def _ratio(v):
        return f"{v:.2g}" if v >= 1e-3 else f"{v:.1e}"

    def _atlas_plot(data):
        rows = sorted(data["species"], key=lambda s: s["x_ratio_base"])
        xmin = min(s["x_ratio_base"] for s in rows) / 3
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=110)
        for i, s in enumerate(rows):
            diagnosis = s["supply"]["diagnosis"]
            color = DIAGNOSIS_COLORS[diagnosis]
            ax.hlines(i, xmin, s["x_ratio_base"], color=color, lw=4, alpha=0.55)
            ax.scatter(s["x_ratio_base"], i, color=color, s=55, zorder=3,
                       label=DIAGNOSIS_LABELS[diagnosis])
            ax.annotate(_ratio(s["x_ratio_base"]), (s["x_ratio_base"], i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.axvline(1 / 3, color="#8a8984", ls="--", lw=1, label="3× below observation")
        ax.axvline(1, color="#52514e", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(xmin, 2)
        ax.set_yticks(range(len(rows)), [s["species"] for s in rows])
        ax.set_xlabel("Model / observed abundance (1 = observation matched)")
        ax.grid(True, axis="x", which="major", color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        html = mo.as_html(fig)
        plt.close(fig)
        return html

    def _atlas_table(data):
        lines = ["| Species | Model / observed | Deficit | Diagnosis | Main formation route | Runs short of | Clean reachable |",
                 "|:---|---:|---:|:---|:---|:---|---:|"]
        for s in sorted(data["species"], key=lambda x: x["x_ratio_base"]):
            sup, ratio = s["supply"], s["x_ratio_base"]
            limiting = f"`{sup['limiting_reactant']}`" if sup["diagnosis"] == "supply_limited" else "–"
            lines.append(f"| `{s['species']}` | {_ratio(ratio)} | {1 / ratio:,.0f}× | "
                         f"{DIAGNOSIS_LABELS[sup['diagnosis']]} | `{rxn(sup['reaction'])}` | {limiting} | "
                         f"{s['n_reachable_clean']} |")
        return mo.md("\n".join(lines))

    def _atlas_detail(s):
        sup = s["supply"]
        diagnosis = sup["diagnosis"]
        factor = sup.get("factor", 1)
        if diagnosis == "supply_limited":
            explanation = (f"Multiplying the main formation route by {factor:g} raises `{s['species']}` only "
                           f"{sup['target_gain']:.2g}× because `{sup['limiting_reactant']}` falls to "
                           f"{sup['limiting_ratio']:.1%} of its original abundance.")
            kind = "info"
        elif diagnosis == "rate_limited":
            explanation = (f"Multiplying the main route by {factor:g} raises the target above the observation "
                           f"(model / observed = {sup['x_ratio_scaled']:.2f}); its rate is the leading issue in "
                           "this gas-phase test.")
            kind = "success"
        else:
            explanation = (f"The faster main route reaches only {_ratio(sup['x_ratio_scaled'])} of the observation, "
                           "without the severe precursor exhaustion used to diagnose supply limitation. This test "
                           "does not isolate a single gas-phase cause.")
            kind = "warn"
        best = s.get("best")
        if best is None:
            best_text = "No candidate was available with an estimated gas-phase rate."
        else:
            reach = ("It is physically reachable below the capture limit."
                     if best.get("capture_feasible") is True else
                     "It is not physically reachable below the capture limit.")
            best_text = (f"Best simulated candidate: `{rxn(best['reaction'])}`, taking model / observed from "
                         f"{_ratio(s['x_ratio_base'])} to {_ratio(best['x_ratio'])}. {reach}")
        return mo.vstack([
            mo.hstack([
                mo.stat(_ratio(s["x_ratio_base"]), label="Model / observed", bordered=True,
                        caption=f"{1 / s['x_ratio_base']:,.0f}× below observation"),
                mo.stat(str(s["n_candidates"]), label="Candidates generated", bordered=True),
                mo.stat(str(s["n_simulated"]), label="Candidates simulated", bordered=True),
                mo.stat(str(s["n_reachable_clean"]), label="Reachable and unflagged", bordered=True),
            ], widths="equal", gap=0.5),
            mo.callout(mo.md(f"**{DIAGNOSIS_LABELS[diagnosis]}:** {explanation}"), kind=kind),
            mo.md(f"Main route: `{rxn(sup['reaction'])}` "
                  f"({s['main_formation_frac']:.0%} of modeled formation).  \n{best_text}"),
        ], gap=0.6)

    _intro = mo.md(
        "<div style='font-size:0.8rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;"
        "opacity:0.6'>Atlas</div>"
        "<div style='font-size:1.3rem;font-weight:600'>Is CH₂CCH an isolated discrepancy?</div>"
    )
    if atlas is None:
        atlas_view = mo.vstack([_intro, mo.callout(mo.md(atlas_error), kind="danger",
                                                   title="Atlas data not available")])
    else:
        _rows = atlas.get("species") or []
        _supply = sum(s["supply"]["diagnosis"] == "supply_limited" for s in _rows)
        _clean = sum(s.get("n_reachable_clean", 0) for s in _rows)
        _selected = next(s for s in _rows if s["species"] == atlas_picker.value)
        atlas_view = mo.vstack([
            _intro,
            mo.md("TAPIR applied the same analogy, simulation, capture-limit and chemical-flag pipeline to every "
                  "TMC-1 species that the reference model underproduces by more than 3×."),
            mo.hstack([
                mo.stat(str(len(_rows)), label="Underproduced species", bordered=True),
                mo.stat(f"{_supply}/{len(_rows)}", label="Supply-limited", bordered=True),
                mo.stat(str(_clean), label="Reachable unflagged candidates", bordered=True,
                        caption="no single missing reaction explains any deficit" if _clean == 0
                        else "across all atlas species"),
            ], widths="equal", gap=0.5),
            _atlas_plot(atlas),
            _atlas_table(atlas),
            mo.md("<small>Supply-limited means that making the main formation reaction 1000× faster exhausts a "
                  "reactant below 10% while the target remains underproduced. “Clean” excludes candidates flagged "
                  "for proton/charge transfer, unpaired-electron growth or charge transfer toward the higher-"
                  "ionization-energy product.</small>"),
            mo.md("**Inspect one species**"),
            atlas_picker,
            _atlas_detail(_selected),
        ], gap=1)
    return (atlas_view,)


@app.cell
def _(APP_DIR, IN_BROWSER, RAW_URL, atlas, atlas_view, discovery, discovery_view, generator_view, mo, validation,
      validation_view):
    import base64

    ACCENT, VIOLET = "#2f6fed", "#6f5bd3"
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    _metrics = (validation or {}).get("metrics") or {}
    _hits = _metrics.get("hits") or {}
    _mrr, _random_mrr = _metrics.get("mrr"), _metrics.get("random_mrr")

    _d = discovery or {}
    _sp = _d.get("species") or "the target"
    _deficit = _d["X_obs"] / _d["X_base"] if _d.get("X_obs") and _d.get("X_base") else None
    _bottleneck = _d.get("bottleneck") or {}
    _limiting = min((_bottleneck.get("species") or {}).items(), key=lambda kv: kv[1].get("ratio", 1),
                    default=(None, None))[0]

    _atlas_rows = (atlas or {}).get("species") or []
    _n_supply = sum(s.get("supply", {}).get("diagnosis") == "supply_limited" for s in _atlas_rows)

    _logo_path = APP_DIR / "assets" / "tapir-mark.png"
    try:
        _logo_src = "data:image/png;base64," + base64.b64encode(_logo_path.read_bytes()).decode()
    except OSError:
        _logo_src = f"{RAW_URL}/assets/tapir-mark.png" if IN_BROWSER else None
    _logo = (f"<img src='{_logo_src}' alt='TAPIR logo' style='width:108px;height:108px;object-fit:contain;flex:none'>"
             if _logo_src else "")

    # Colors inherit from the marimo theme so the header reads in light and dark mode; only the brand accents are fixed.
    HEADER = mo.Html(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem 2rem;flex-wrap:wrap;
                padding:0.6rem 1.4rem 0.6rem 0.8rem;border:1px solid rgba(127,127,127,.22);border-radius:18px;
                background:linear-gradient(120deg,rgba(47,111,237,.07),rgba(111,91,211,.07) 60%,transparent)">
      <div style="display:flex;align-items:center;gap:1rem">
        {_logo}
        <div>
          <div style="font-family:Montserrat,Poppins,'Segoe UI',system-ui,sans-serif;font-size:2.4rem;
                      font-weight:800;letter-spacing:.06em;line-height:1">TAPIR</div>
          <div style="font-size:1rem;opacity:.75;margin-top:.3rem">
            Tracking Absent Pathways in Interstellar Reactions</div>
          <div style="height:3px;width:100%;margin-top:.45rem;border-radius:2px;
                      background:linear-gradient(90deg,{ACCENT},{VIOLET})"></div>
        </div>
      </div>
      <div style="font-size:0.95rem;font-weight:700;letter-spacing:.03em;color:{VIOLET};line-height:1.6">
        LLM proposes.<br>Code verifies.<br>Physics decides.</div>
    </div>
    """)

    _mrr_caption = (f"MRR {_mrr:.2f} vs {_random_mrr:.2f} if shuffled"
                    if isinstance(_mrr, (int, float)) and isinstance(_random_mrr, (int, float)) else None)
    if _d.get("n_reachable") == 0 and _bottleneck.get("supply_limited"):
        _discovery_stat = mo.stat(
            "Supply-limited", bordered=True,
            label=f"Why {_sp.translate(SUB)} is {_deficit:.0f}× too low" if _deficit else f"Diagnosis for {_sp}",
            caption=f"none of {_d.get('n_generated', 0)} hypotheses closes it"
                    + (f"; {_limiting.translate(SUB)} runs out" if _limiting else ""))
    else:
        _discovery_stat = mo.stat(str(_d.get("n_reachable", "–")), bordered=True,
                                  label=f"Plausible hypotheses for {_sp.translate(SUB)}",
                                  caption=f"of {_d.get('n_generated', 0)} generated")
    SUMMARY = mo.hstack([
        mo.stat(f"{_hits.get('1', _hits.get(1, 0))} / {_metrics.get('n_cases') or 0}", bordered=True,
                label="Hidden reactions recovered at #1", caption=_mrr_caption),
        _discovery_stat,
        mo.stat(f"{_n_supply} / {len(_atlas_rows)}", bordered=True, label="Atlas deficits that are supply-limited",
                caption="TMC-1 species underproduced by >3×"),
    ], widths="equal", gap=0.6)

    CONTEXT = mo.md(
        f"<div style='border-left:3px solid {VIOLET};padding:0.3rem 0.8rem;font-size:0.92rem;max-width:62rem'>"
        "<b>Ranked hypotheses, not confirmed reactions.</b> <span style='opacity:.75'>"
        "TAPIR generates candidates, tests them in the network, compares the required rate with physical limits, "
        "and checks the literature before a scientist decides what to pursue.</span></div>"
    )

    TABS = mo.ui.tabs({"Validation": validation_view, "Discovery": discovery_view,
                       "Atlas": atlas_view, "Generator": generator_view})
    mo.vstack([HEADER, SUMMARY, CONTEXT, TABS], gap=0.9)
    return


if __name__ == "__main__":
    app.run()
