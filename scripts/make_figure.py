"""
make_figure.py - Three-panel summary figure from stored results (no simulation, no LLM, no Bond).

  A  CH2CCH funnel (results/discovery/CH2CCH.json, chem_review.json): candidates generated, with a
     rate, simulated, reachable below the capture limit and passing the chemical review
  B  CH2CCH vs time (results/discovery/CH2CCH.json): base model, best-ranked candidate (at its
     estimated rate), the main formation reaction C + C2H4 at 1000x its KIDA rate (existing block)
     and the TMC-1 observation, with t_ref marked
  C  atlas (results/discovery/atlas.json): model / observed abundance of the species underestimated
     more than 3x, hatched by supply diagnosis

Readable in black and white: line styles and hatches carry identity, colour only repeats it.
usage: python -m scripts.make_figure
Output: figure.png (300 dpi) and figure.pdf at the repository root.
"""
import json, math, os, re
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tapir.paths import ROOT, RESULTS, relative

DISCOVERY = os.path.join(RESULTS, "discovery", "CH2CCH.json")
ATLAS = os.path.join(RESULTS, "discovery", "atlas.json")
REVIEW = os.path.join(RESULTS, "discovery", "chem_review.json")
OUT = os.path.join(ROOT, "figure")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIAGNOSIS = {  # diagnosis: (label, face colour, hatch)
    "supply_limited": ("supply-limited", BLUE, "////"),
    "rate_limited": ("rate-limited", ORANGE, "...."),
    "other": ("other", "white", ""),
}
AQUA = "#1baf7a"
NOTES = {"CH3OH": "known grain-surface product"}  # atlas annotations, inside the bar
METHODS = ("Generators: analogy templates + LLM  ·  Checks: conservation, unpaired electrons, "
           "ionization energy (NIST)  ·  Literature: Valency MCP Bond")


def tex(s):
    """KIDA name -> mathtext: digits subscript, trailing charge superscript (CH3OH, HCS+, NH4+)."""
    charge = re.search(r"[+-]+$", s)
    body = s[:charge.start()] if charge else s
    body = re.sub(r"(\d+)", r"$_{\1}$", body)
    return body + (f"$^{{{charge.group()}}}$" if charge else "")


def ratio_label(x):
    if x < 1e-3:
        return f"$10^{{{round(math.log10(x))}}}$"
    return f"{x:.2g}"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7.5)


def funnel_stages(d, review):
    """[(count, label)] for CH2CCH, all from stored counts. With a rate = generated - no rate - k = 0
    (unpaired-electron filter); chemical review counts the CH2CCH candidates judged plausible."""
    mine = [r for r in review["reactions"] if r["file"] == "CH2CCH"]
    ok = sum(r["verdict"] != "implausible" for r in mine)
    return [(d["n_generated"], "generated\n(analogy + LLM)"),
            (d["n_generated"] - d["n_no_rate"] - d["n_k_zero"], "with a rate\n(after k = 0 filters)"),
            (d["n_simulated"], "simulated\n(top by flux)"),
            (d["n_reachable"], "reachable\n(k ≤ capture limit)"),
            (ok, f"pass chemical\nreview ({ok} of {len(mine)})")]


def panel_funnel(ax, stages):
    """Bars that narrow left to right; height ∝ sqrt(count) so the small stages stay visible."""
    n0 = stages[0][0]
    ax.set_xlim(0, len(stages)); ax.set_ylim(-0.62, 0.72); ax.axis("off")
    w, hmax, hmin = 0.46, 0.8, 0.025
    heights = [max(hmax * math.sqrt(n / n0), hmin) for n, _ in stages]
    for i, ((n, label), h) in enumerate(zip(stages, heights)):
        x = i + 0.5 - w / 2
        if i:
            xp, hp = i - 0.5 + w / 2, heights[i - 1]
            ax.fill([xp, x, x, xp], [-hp / 2, -h / 2, h / 2, hp / 2], fc=GRID, ec="none", zorder=1)
        ax.add_patch(plt.Rectangle((x, -h / 2), w, h, fc=BLUE if n else "white", ec=INK, lw=0.8,
                                   hatch="" if n else "////", zorder=2))
        ax.text(i + 0.5, 0.46, f"{n}", ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=INK)
        ax.text(i + 0.5, -0.46, label, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.2)
    ax.set_title(f"{n0} hypotheses, {stages[-1][0]} survive", fontsize=8.5, color=INK, loc="left")
    ax.text(0.5, -0.2, METHODS, transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
            color=MUTED)


def panel_ch2cch(ax, d):
    t, x_obs, t_ref = d["t"], d["X_obs"], d["config"]["t_ref"]
    best = d["hypotheses"][0]
    ax.plot(t, d["curve_base"], color=BLUE, ls="-", lw=1.6, label="base model")
    ax.plot(t, best["curve"], color=ORANGE, ls="--", lw=1.6,
            label=f"best-ranked candidate (fails chemical review)\n{rxn(best['reaction'])}")
    main = next(e for e in d["existing"] if e["reaction"] == d["bottleneck"]["reaction"])
    if main["curve"] and main["factor_evaluated"] == d["bottleneck"]["factor"]:
        ax.plot(t, main["curve"], color=AQUA, ls="-.", lw=1.6,
                label=f"{rxn(main['reaction']).split(' →')[0]} rate ×{main['factor_evaluated']:.0f}: still below")
        c2h4 = d["bottleneck"]["species"]["C2H4"]["ratio"]
        ax.annotate(f"C$_2$H$_4$ runs out\n(→ {100 * c2h4:.1f} %)", xy=(t_ref, main["X_cand"]),
                    xytext=(4e6, x_obs * 3e-3), fontsize=6.5, color=INK, ha="right",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=2, shrinkB=3))
    ax.axhline(x_obs, color=INK, ls=":", lw=1.4, label="TMC-1 observation")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e3, t[-1]); ax.set_ylim(x_obs * 1e-4, x_obs * 50)
    ax.axvline(t_ref, color=MUTED, lw=0.8)
    ax.text(t_ref, x_obs * 2e-4, "$t_{ref}$ ", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("time (yr)", fontsize=8, color=INK)
    ax.set_ylabel("abundance relative to H$_2$", fontsize=8, color=INK)
    ax.set_title("CH$_2$CCH: the bottleneck is C$_2$H$_4$ supply,\nnot a missing reaction",
                 fontsize=8.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.5, which="major"); ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=2.6)
    style(ax)


def rxn(label):
    reac, prod = label.split(" -> ")
    return " + ".join(map(tex, reac.split(" + "))) + " → " + " + ".join(map(tex, prod.split(" + ")))


def panel_atlas(ax, a):
    rows = sorted(a["species"], key=lambda s: s["x_ratio_base"], reverse=True)
    for i, s in enumerate(rows):
        x = s["x_ratio_base"]
        _, fc, hatch = DIAGNOSIS[s["supply"]["diagnosis"]]
        ax.barh(i, 1 - x, left=x, height=0.62, fc=fc, ec=INK, lw=0.8, hatch=hatch)
        ax.text(x / 1.4, i, ratio_label(x), ha="right", va="center", fontsize=7, color=INK)
        if s["species"] in NOTES:
            ax.text(x * 1.6, i, NOTES[s["species"]], ha="left", va="center", fontsize=6.5,
                    color=MUTED, style="italic")
    ax.axvline(1, color=INK, lw=1.0)
    ax.axvline(1 / 3, color=MUTED, ls="--", lw=0.8)
    ax.text(1 / 3, len(rows) - 0.45, "3× below ", color=MUTED, fontsize=6.5, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1e-8, 2)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_yticks(range(len(rows)), [tex(s["species"]) for s in rows], fontsize=7.5)
    ax.set_xlabel("model / observed abundance", fontsize=8, color=INK)
    n = sum(s["supply"]["diagnosis"] == "supply_limited" for s in rows)
    ax.set_title(f"Atlas: {n} of {len(rows)} underestimated species are supply-limited",
                 fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    present = [k for k in DIAGNOSIS if any(s["supply"]["diagnosis"] == k for s in rows)]
    ax.legend([Patch(fc=DIAGNOSIS[k][1], ec=INK, hatch=DIAGNOSIS[k][2], lw=0.8) for k in present],
              [DIAGNOSIS[k][0] for k in present], fontsize=6.5, frameon=False, loc="lower left")
    style(ax)


if __name__ == "__main__":
    d, a, review = (json.load(open(p)) for p in (DISCOVERY, ATLAS, REVIEW))
    plt.rcParams.update({"hatch.linewidth": 0.6, "font.size": 8})
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 2.4), width_ratios=(1.15, 1), hspace=0.42, wspace=0.32)
    axes = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    panel_funnel(axes[0], funnel_stages(d, review)); panel_ch2cch(axes[1], d); panel_atlas(axes[2], a)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.02 if letter == "A" else -0.16, 1.02, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")
    fig.text(0.99, 0.005, "Ranked hypotheses, not confirmed reactions.", ha="right", va="bottom",
             fontsize=6.5, color=MUTED, style="italic")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"-> {relative(OUT)}.png, {relative(OUT)}.pdf")
