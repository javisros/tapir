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
    # Self-contained: only precomputed files in app/data/ (python scripts/export_app_data.py), never tapir.
    DATA_DIR = (mo.notebook_dir() or Path(__file__).resolve().parent) / "data"
    EXPORT_HINT = "Run `python scripts/export_app_data.py` from the repository root, then reload this app."

    def load_json(name):
        """(data, error message): a missing or unreadable file gives a message, never a traceback."""
        path = DATA_DIR / name
        try:
            return json.loads(path.read_text()), None
        except FileNotFoundError:
            return None, f"`app/data/{name}` was not found. {EXPORT_HINT}"
        except (OSError, ValueError) as e:
            return None, f"`app/data/{name}` could not be read ({e}). {EXPORT_HINT}"

    validation, validation_error = load_json("validation_curves.json")
    return validation, validation_error


@app.cell
def _(mo, validation):
    case_picker = None
    if validation is not None:
        _names = [c["case"] for c in validation["cases"]]
        case_picker = mo.ui.dropdown(options=_names, value=_names[0], label="Test case")
    return (case_picker,)




@app.cell
def _(mo, plt, validation):
    COLORS = dict(base="#52514e", knockout="#eb6834", recovered="#2a78d6")
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
        m = validation["metrics"]
        hero = mo.md(
            f"<div style='font-size:2.6rem;font-weight:700;line-height:1.1'>{m['hits']['1']} / {m['n_cases']}</div>"
            "<div>hidden reactions ranked <b>#1</b></div>"
        )
        secondary = mo.md(
            f"**MRR {m['mrr']:.2f}** &nbsp; vs &nbsp; **{m['random_mrr']:.2f}** random baseline  \n"
            "<small>Mean reciprocal rank: higher is better (1 = always first). The baseline is the MRR "
            "expected if TAPIR's final candidates were shuffled.</small>"
        )
        return mo.hstack([hero, secondary], widths=[1, 2.2], align="center", gap=2)

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
def _(mo, validation_view):
    HEADER = mo.md(
        "<span style='font-size:1.6rem;font-weight:700;letter-spacing:0.05em'>TAPIR</span> &nbsp; "
        "<span style='opacity:0.75'>Tracking Absent Pathways in Interstellar Reactions</span>  \n"
        "<small style='opacity:0.6'>Finding missing chemistry in interstellar reaction networks.</small>"
    )
    PENDING = mo.md("<span style='opacity:0.6'>Results will appear after the corresponding pipeline has finished.</span>")
    mo.vstack([HEADER, mo.ui.tabs({"Validation": validation_view, "Discovery": PENDING, "Generator": PENDING})],
              gap=0.5)
    return


if __name__ == "__main__":
    app.run()
