#!/usr/bin/env python3
"""Fig 1: mechanism schematic for Proposition 1 (pure drawing, no data).

Two runs share one seed. Every training step consumes one draw from the run's
RNG stream (ribbon under each timeline). In Run B a mid-training probe taps
the SAME stream and consumes k draws. The +k account holds only on the
aligned prefix; later draw counts/control flow may change. An update-relevant
read can cause divergence, and later reconvergence is possible. A realized
printed gap may shift without implying marginal mean bias. Inset: under
Proposition 1's deterministic, identical-program and state-purity premises,
isolating all relevant streams preserves the training trajectory.

Visual language: a dashed vermillion line drawn on top of a solid blue line
means "the two runs are bitwise identical here".

Sized for AAAI full column width: exactly 7.0 x 3.05 in (504 x 219.6 pt),
all text >= 8 pt, all lines >= 0.7 pt.

Deterministic, CPU-only, no network, no data files.
Outputs figs/fig1_mechanism.{pdf,png}.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

OUTDIR = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"       # Run A
VERM = "#D55E00"       # Run B
VERM_DARK = "#A34700"
ORANGE = "#E69F00"     # probe
ORANGE_DARK = "#7A5200"
GREEN = "#009E73"      # fork_rng / identical
GREEN_DARK = "#007A5C"
INK = "0.15"
GRAY = "0.45"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

DASH = (0, (3.2, 2.2))

# AAAI visual floors for the camera-ready column width.
FIGSIZE = (7.0, 3.05)
FS_BODY = 8.0          # minimum for every visible string
FS_RUN_TITLE = 9.0     # the two run titles
FS_OUTCOME = 8.5       # the outcome-box title ("printed Δ")
LW_HAIR = 0.7          # minimum for every visible line


def ribbon(ax, x0, x1, y0, y1, cells, label_colors=None, highlight=None):
    """Draw a stream band with cell separators and draw-index labels.

    cells: list of (x_left, x_right, text) triples.
    highlight: optional (x_left, x_right, text) orange block (probe draws).
    """
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.94",
                           edgecolor="0.55", lw=LW_HAIR, zorder=1))
    yc = (y0 + y1) / 2.0
    for i, (cl, cr, txt) in enumerate(cells):
        if cl > x0:
            ax.plot([cl, cl], [y0, y1], color="0.78", lw=LW_HAIR, zorder=1.2)
        color = (label_colors or {}).get(i, "0.25")
        ax.text((cl + cr) / 2.0, yc, txt, ha="center", va="center",
                fontsize=FS_BODY, color=color, zorder=1.4)
    if highlight is not None:
        hl, hr, htxt = highlight
        ax.add_patch(Rectangle((hl, y0), hr - hl, y1 - y0,
                               facecolor=ORANGE, alpha=0.55,
                               edgecolor=ORANGE_DARK, lw=LW_HAIR, zorder=1.3))
        if htxt:
            ax.text((hl + hr) / 2.0, yc, htxt, ha="center", va="center",
                    fontsize=FS_BODY, color=ORANGE_DARK, zorder=1.5,
                    fontweight="bold")


def connectors(ax, xs, y_top, y_bot, color="0.55"):
    for x in xs:
        ax.plot([x, x], [y_top, y_bot], color=color, lw=LW_HAIR,
                ls=(0, (1, 1.4)), zorder=2)


def ticks(ax, xy_list, color, half=1.1):
    for x, y in xy_list:
        ax.plot([x, x], [y - half, y + half], color=color, lw=1.1, zorder=3.5)


def arrow(ax, p0, p1, color, lw=1.2, style="-|>", ms=9, ls="-", zorder=3,
          alpha=1.0, shrinkA=0.0, shrinkB=0.0, connectionstyle=None):
    ax.annotate(
        "", xy=p1, xytext=p0, zorder=zorder,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                        alpha=alpha, mutation_scale=ms, shrinkA=shrinkA,
                        shrinkB=shrinkB, connectionstyle=connectionstyle),
    )


def rounded_box(ax, x0, y0, x1, y1, edge, face, lw=1.0, ls="-", zorder=4):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=face, edgecolor=edge, lw=lw, ls=ls, zorder=zorder,
    ))


def bcurve(t):
    """Run B's diverged trajectory, t in [0, 1]."""
    return 61.0 + 49.0 * t, 38.0 + 7.5 * t * t


def build_figure() -> Figure:
    """Draw the mechanism schematic and return the (unsaved) figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(-4, 144.9)
    ax.set_ylim(3, 68.2)
    ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

    step_xs = [30, 40, 50, 60, 70, 80, 90, 100]

    # ---------------- seed icon (shared by both runs) --------------------
    rounded_box(ax, 2, 42, 16, 53, edge="0.35", face="0.97", lw=0.9)
    # a small die glyph
    rounded_box(ax, 6.6, 47.2, 11.4, 51.6, edge=INK, face="white", lw=0.8,
                zorder=4.2)
    for dx, dy in ((-1.15, 1.1), (0, 0), (1.15, -1.1)):
        ax.add_patch(mpl.patches.Circle((9 + dx, 49.4 + dy), 0.38,
                                        color=INK, zorder=4.4))
    ax.text(9, 44.6, "same seed", ha="center", va="center", fontsize=FS_BODY,
            color=INK, zorder=4.2)
    # elbow connectors seed -> each run, leaving the box at its top and bottom
    # centre so that neither vertical crosses the stream labels on the left
    ax.plot([9, 9, 19], [53.2, 62, 62], color=INK, lw=0.9, zorder=3,
            solid_capstyle="round")
    arrow(ax, (19, 62), (21.6, 62), INK, lw=0.9, ms=7)
    ax.plot([9, 9, 19], [41.8, 38, 38], color=INK, lw=0.9, zorder=3,
            solid_capstyle="round")
    arrow(ax, (19, 38), (21.6, 38), INK, lw=0.9, ms=7)

    # ---------------- Run A: baseline ------------------------------------
    yA = 62
    ax.text(22, 65.8, "Run A: baseline", fontsize=FS_RUN_TITLE, color=BLUE,
            fontweight="bold", va="center", zorder=4)
    ax.plot([22, 108.6], [yA, yA], color=BLUE, lw=1.6, zorder=3,
            solid_capstyle="round")
    arrow(ax, (104, yA), (110, yA), BLUE, lw=1.6, ms=10)
    ticks(ax, [(x, yA) for x in step_xs], BLUE)

    # ribbon A: draws #1..#8 consumed in order
    cellsA = [(25 + 10 * i, 35 + 10 * i, f"#{i + 1}") for i in range(8)]
    ribbon(ax, 25, 105, 54.5, 58.5, cellsA)
    ax.text(106.3, 56.5, "…", fontsize=FS_BODY, color="0.35", va="center",
            zorder=1.4)
    connectors(ax, step_xs, yA - 1.1, 58.5)
    ax.text(24, 56.5, "training\nRNG stream", ha="right", va="center",
            fontsize=FS_BODY, color=GRAY, style="italic", zorder=4,
            linespacing=1.05)

    # ---------------- Run B: same seed + mid-training probe --------------
    yB = 38
    # two lines, so the title ends left of the probe arrow at x = 58
    ax.text(22, 42.6, "Run B: same seed\n+ mid-training probe",
            fontsize=FS_RUN_TITLE, color=VERM, fontweight="bold",
            va="center", zorder=4, linespacing=1.05)
    # pre-probe segment: bitwise identical to A (blue + dashed vermillion)
    ax.plot([22, 61], [yB, yB], color=BLUE, lw=1.8, zorder=3,
            solid_capstyle="round")
    ax.plot([22, 61], [yB, yB], color=VERM, lw=1.0, ls=DASH, zorder=3.2)
    # post-probe: diverged trajectory (vermillion only)
    ts = [i / 60.0 for i in range(61)]
    bx = [bcurve(t)[0] for t in ts]
    by = [bcurve(t)[1] for t in ts]
    ax.plot(bx, by, color=VERM, lw=1.6, zorder=3, solid_capstyle="round")
    arrow(ax, bcurve(0.94), (111.5, 46.0), VERM, lw=1.6, ms=10)
    # ghost continuation = counterfactual: where B would have gone (= A)
    ax.plot([61, 108], [yB, yB], color=BLUE, lw=0.9, ls=DASH, alpha=0.55,
            zorder=2.5)
    # label sits on the extension of the dashed ghost line, clear of the
    # dotted step connectors below the diverged path
    ax.text(109.2, yB, "counterfactual A path", ha="left", va="center",
            fontsize=FS_BODY, color=GRAY, zorder=4)
    # divergence wedge
    ax.fill([61] + bx + [108, 61], [yB] + by + [yB, yB], color=VERM,
            alpha=0.15, lw=0, zorder=1.8)
    ax.text(94.5, 45.4, "possible divergence", fontsize=FS_BODY,
            color=VERM_DARK, style="italic", rotation=9, ha="center",
            va="center", zorder=4)

    # ticks: 3 steps before the probe, then steps on the diverged path
    pre = [30, 40, 50]
    post = [66, 76, 86, 96]
    ticks(ax, [(x, yB) for x in pre], VERM)
    ticks(ax, [(x, 38 + 7.5 * ((x - 61) / 49.0) ** 2) for x in post], VERM)

    # ribbon B: same stream; probe consumes k draws mid-stream
    cellsB = ([(25 + 10 * i, 35 + 10 * i, f"#{i + 1}") for i in range(3)]
              + [(61, 71, "#4+k"), (71, 81, "?"), (81, 91, "?"),
                 (91, 101, "?")])
    shift_colors = {i: VERM_DARK for i in (3, 4, 5, 6)}
    ribbon(ax, 25, 105, 30.5, 34.5, cellsB, label_colors=shift_colors,
           highlight=(55, 61, "+k"))
    ax.text(106.3, 32.5, "…", fontsize=FS_BODY, color="0.35", va="center",
            zorder=1.4)
    connectors(ax, pre, yB - 1.1, 34.5)
    post_y = [38 + 7.5 * ((x - 61) / 49.0) ** 2 for x in post]
    for x, y in zip(post, post_y):
        ax.plot([x, x], [34.5, y - 1.1], color="0.55", lw=LW_HAIR,
                ls=(0, (1, 1.4)), zorder=2)
    ax.text(24, 32.5, "shared training stream", ha="right", va="center",
            fontsize=FS_BODY, color=GRAY, style="italic", zorder=4)
    ax.text(23, 25.7, "+k: aligned prefix only; later reads and reconvergence unrestricted",
            fontsize=FS_BODY, color=GRAY, va="center", zorder=4)

    # probe box taps the SAME ribbon
    rounded_box(ax, 48.5, 46.2, 67.5, 52.2, edge=ORANGE_DARK, face="#FBE8C4",
                lw=1.1)
    ax.text(58, 49.2, "probe: k draws", ha="center", va="center",
            fontsize=FS_BODY, color=ORANGE_DARK, zorder=4.2)
    arrow(ax, (58, 46.2), (58, 34.8), ORANGE_DARK, lw=1.3, ms=8, zorder=2.8)

    # ---------------- outcomes at right -----------------------------------
    ax.text(112.2, yA, "acc A", fontsize=FS_BODY, color=BLUE, va="center",
            fontweight="bold", zorder=4)
    ax.text(112.2, 45.8, "acc B", fontsize=FS_BODY, color=VERM, va="center",
            fontweight="bold", zorder=4)
    brx = 120.8
    ax.plot([brx, brx], [45.8, yA], color=INK, lw=0.8, zorder=3)
    ax.plot([brx - 1.6, brx], [yA, yA], color=INK, lw=0.8, zorder=3)
    ax.plot([brx - 1.6, brx], [45.8, 45.8], color=INK, lw=0.8, zorder=3)
    arrow(ax, (brx, 53.9), (121.7, 53.9), INK, lw=0.8, ms=7)
    rounded_box(ax, 121.8, 47.5, 144.3, 60.5, edge=VERM_DARK,
                face="#FCEFE7", lw=1.0)
    ax.text(133.05, 56.4, "printed Δ", ha="center", va="center",
            fontsize=FS_OUTCOME, color=INK, zorder=4.2)
    ax.text(133.05, 52.2, "may shift", ha="center", va="center",
            fontsize=FS_BODY, color=VERM_DARK, fontweight="bold", zorder=4.2)
    ax.text(133.05, 48.9, "mean may agree", ha="center", va="center",
            fontsize=FS_BODY, color=VERM_DARK, zorder=4.2)

    # ---------------- inset: isolated probe via fork_rng ------------------
    rounded_box(ax, 20, 5, 144.3, 22, edge="0.55", face="0.985",
                lw=LW_HAIR, ls=(0, (4, 3)), zorder=0.5)
    ax.text(23, 19.4, "Under Prop. 1 premises", fontsize=FS_BODY,
            color="0.25", style="italic", va="center", zorder=4)

    yI = 14.5
    itick = [35, 45, 55, 65, 75, 85, 95]
    ax.plot([26, 110.6], [yI, yI], color=BLUE, lw=2.0, zorder=3,
            solid_capstyle="round")
    ax.plot([26, 110.6], [yI, yI], color=VERM, lw=1.0, ls=DASH, zorder=3.2)
    arrow(ax, (106, yI), (112, yI), BLUE, lw=1.8, ms=9)
    ticks(ax, [(x, yI) for x in itick], "0.2", half=0.85)
    ax.text(114.5, yI, "✓ bitwise identical", fontsize=FS_BODY,
            color=GREEN_DARK, fontweight="bold", va="center", zorder=4)

    cellsI = ([(30 + 10 * i, 40 + 10 * i, f"#{i + 1}") for i in range(6)]
              + [(90, 100, "…")])
    ribbon(ax, 30, 100, 9.3, 12.7, cellsI)
    connectors(ax, itick[:-1], yI - 0.85, 12.7)
    ax.text(29, 11.0, "unchanged", ha="right", va="center", fontsize=FS_BODY,
            color=GRAY, style="italic", zorder=4)

    rounded_box(ax, 55.5, 15.9, 76, 19.7, edge=ORANGE_DARK, face="#FBE8C4",
                lw=1.0)
    ax.text(65.75, 17.8, "probe (fork_rng)", ha="center", va="center",
            fontsize=FS_BODY, color=ORANGE_DARK, zorder=4.2)
    ax.add_patch(Rectangle((79, 16.3), 18, 2.6, facecolor=GREEN, alpha=0.30,
                           edgecolor=GREEN_DARK, lw=LW_HAIR, zorder=1.3))
    ax.text(88, 17.6, "forked stream", ha="center", va="center",
            fontsize=FS_BODY, color=GREEN_DARK, zorder=4)
    arrow(ax, (76, 17.8), (79, 17.6), GREEN_DARK, lw=1.0, ms=7)

    return fig


def render(output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    """Save fig1_mechanism.pdf/.png into output_dir and close the figure.

    The canvas is already the final AAAI size, so no bbox_inches="tight".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = output_dir / "fig1_mechanism.pdf"
    png_path = output_dir / "fig1_mechanism.png"
    try:
        fig.savefig(pdf_path, facecolor="white")
        fig.savefig(png_path, facecolor="white")
    finally:
        plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    pdf_path, _ = render()
    print(f"[fig1] wrote {pdf_path} and .png "
          "(pure schematic, no data inputs)")


if __name__ == "__main__":
    main()
