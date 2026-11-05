#!/usr/bin/env python3
"""Fig 2: probe-placement harvest = extreme-value statistics (Prop 2).

Panel A: paired delta vs the no-probe base run (pt) for every (seed,
placement) cell of the P13 harvest grid; per-seed max circled. Horizontal
lines: honest mean over all cells (dashed blue, negative: probes hurt on
average) vs per-seed select-then-average (solid vermillion). The gap between
the two lines is the manufactured "improvement".

Panel B: subsample growth curve. For m' = 2..6 placements, mean over seeds
and all exhaustive placement subsets of (max - mean), against the Gaussian
null c_{m'} * sigma_bar and the Student-t3 (unit variance) null. Nulls use
the same deterministic Monte Carlo as scripts/prop2_fit.py (numpy
default_rng(0), n=200000); the m'=6 constants are asserted against the c_m
values stored in the JSON.

Reads ONLY the committed JSON log logs_flip_demo/prop2_fit.json (deltas_pt,
sd_pt, harvest_max_minus_mean_pt, c_m, washout). All plotted quantities are
recomputed from deltas_pt at runtime and asserted against the stored fields.

Sized for the TMLR single-column measure: exactly 6.5 x 2.6 in (468 x 187.2
pt), panels side by side so the float arrives unscaled, all text >= 8 pt, all
lines >= 0.7 pt, data markers >= 4 pt.

Deterministic, CPU-only, no network.
Outputs figs/fig2_harvest.{pdf,png}.
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs_flip_demo" / "prop2_fit.json"
OUTDIR = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"      # honest mean / Gaussian null
SKY = "#56B4E9"       # t3 null
VERM = "#D55E00"      # selection
INK = "0.15"
SEED_MARKERS = ["o", "s", "^", "D", "v"]

BASE = 8.0            # AAAI floor: no visible text below 8 pt
FS_PANEL_LETTER = 9.0
MS_POINT = 4.2        # ordinary scatter markers
MS_GROWTH = 4.4       # growth-curve markers
MS_RING = 8.0         # selected-maximum rings
FIGSIZE = (6.5, 2.6)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.linewidth": 0.7,
    "xtick.labelsize": BASE,
    "ytick.labelsize": BASE,
    "legend.fontsize": BASE,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def mc_cm(m: int, dist: str, n: int = 200_000) -> float:
    """E[max - mean] of m i.i.d. unit-variance draws.

    Verbatim replica of scripts/prop2_fit.py::mc_cm (deterministic MC,
    numpy default_rng(0)).
    """
    rng = np.random.default_rng(0)
    if dist == "normal":
        x = rng.standard_normal((n, m))
    else:  # t3, scaled to unit variance (var of t3 = 3)
        x = rng.standard_t(3, (n, m)) / np.sqrt(3.0)
    return float((x.max(1) - x.mean(1)).mean())


def load_harvest_data(path: Path = LOG) -> dict:
    """Load the harvest log, recompute every summary, assert vs stored fields."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    P = d["placements"]
    seeds = d["seeds"]
    D = np.array(d["deltas_pt"])
    m = len(P)

    # ---- recompute everything from deltas_pt; assert vs stored fields ----
    honest = float(D.mean())
    per_seed_sel = float(D.max(axis=1).mean())
    gap = per_seed_sel - honest
    sd = D.std(axis=1, ddof=1)
    sigma_bar = float(sd.mean())
    harvest = D.max(axis=1) - D.mean(axis=1)

    assert abs(honest - d["washout"]["honest_mean"]) < 1e-12
    assert abs(per_seed_sel - d["washout"]["per_seed_select"]) < 1e-12
    assert np.allclose(sd, d["sd_pt"], atol=1e-12)
    assert np.allclose(harvest, d["harvest_max_minus_mean_pt"], atol=1e-12)

    mps = list(range(2, m + 1))
    emp, gauss, t3 = [], [], []
    for mp in mps:
        subs = list(combinations(range(m), mp))
        emp.append(float(np.mean(
            [[D[i, list(c)].max() - D[i, list(c)].mean() for c in subs]
             for i in range(len(seeds))])))
        gauss.append(mc_cm(mp, "normal") * sigma_bar)
        t3.append(mc_cm(mp, "t3") * sigma_bar)
    assert abs(mc_cm(m, "normal") - d["c_m"]["gauss"]) < 1e-9
    assert abs(mc_cm(m, "t3") - d["c_m"]["t3_unitvar"]) < 1e-9
    assert abs(emp[-1] - float(np.mean(d["harvest_max_minus_mean_pt"]))) < 1e-12

    return {
        "placements": P,
        "seeds": seeds,
        "deltas": D,
        "honest": honest,
        "selected": per_seed_sel,
        "gap": gap,
        "sigma_bar": sigma_bar,
        "m_values": mps,
        "empirical": emp,
        "gaussian": gauss,
        "student_t3": t3,
    }


def build_figure(data: dict) -> Figure:
    """Draw both panels from loaded harvest data; return the unsaved figure."""
    P = data["placements"]
    seeds = data["seeds"]
    D = data["deltas"]
    honest = data["honest"]
    per_seed_sel = data["selected"]
    gap = data["gap"]
    mps = data["m_values"]
    emp = data["empirical"]
    gauss = data["gaussian"]
    t3 = data["student_t3"]

    # Two-up, not stacked. Stacked at 3.3125in the float had to be scaled to
    # 0.62\textwidth in the single-column build, which multiplied its height to
    # about 4.6in and made it the tallest object in the body. Side by side at
    # the full 6.5in measure it arrives unscaled at 2.6in.
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=FIGSIZE, constrained_layout=True
    )

    # ---------------- Panel A: the harvest scatter ------------------------
    axA.axhline(0, color="0.85", lw=0.7, zorder=0.5)
    axA.axhline(honest, color=BLUE, ls=(0, (4, 2)), lw=1.1, zorder=2)
    axA.axhline(per_seed_sel, color=VERM, lw=1.3, zorder=2)

    for i, seed in enumerate(seeds):
        xj = [p + (i - 2) * 14 for p in P]  # deterministic seed offset
        points = axA.plot(xj, D[i], ls="none", marker=SEED_MARKERS[i],
                          ms=MS_POINT, mfc="0.4", mec="none", zorder=3)[0]
        points.set_gid(f"delta-points-seed-{seed}")
        jmax = int(D[i].argmax())
        ring = axA.plot([xj[jmax]], [D[i][jmax]], ls="none", marker="o",
                        ms=MS_RING, mfc="none", mec=VERM, mew=1.0,
                        zorder=4)[0]
        ring.set_gid(f"selected-max-seed-{seed}")

    axA.text(1700, 4.55, f"selected mean {per_seed_sel:+.2f}",
             fontsize=BASE, color=VERM, ha="right", va="center", zorder=5)
    axA.annotate(
        "per-seed max",
        xy=(398, 3.0), xytext=(95, 4.55), fontsize=BASE, color=VERM,
        va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color=VERM, lw=0.7,
                        shrinkA=1.0, shrinkB=0.5),
    )
    axA.text(250, -3.75, f"honest mean {honest:+.2f}",
             fontsize=BASE, color=BLUE, va="top", zorder=5)

    gap_arrow = axA.annotate(
        "", xy=(1470, per_seed_sel), xytext=(1470, honest),
        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9,
                        mutation_scale=7, shrinkA=0.0, shrinkB=0.0),
        zorder=4,
    )
    gap_arrow.set_gid("gap-arrow")
    axA.text(1700, -2.2, f"{gap:+.2f} manufactured", fontsize=BASE,
             color=INK, ha="right", va="top", zorder=5)

    axA.set_xlim(80, 1720)
    axA.set_ylim(-10.2, 5.4)
    axA.set_xticks(P)
    axA.set_yticks([-10, -5, 0, 5])
    axA.set_xlabel("probe placement (step)")
    axA.set_ylabel("Δ vs no-probe run (pt)")
    axA.text(0.0, 1.02, "A", transform=axA.transAxes,
             fontsize=FS_PANEL_LETTER, fontweight="bold", va="bottom",
             ha="left")

    # ---------------- Panel B: extreme-value growth curve -----------------
    gauss_line = axB.plot(mps, gauss, color=BLUE, lw=1.2, zorder=2,
                          label="Gaussian fixed-scale reference")[0]
    gauss_line.set_gid("growth-gaussian-null")
    t3_line = axB.plot(mps, t3, color=SKY, lw=1.2, ls=(0, (1.5, 1.5)),
                       zorder=2, label="Student-$t_3$ fixed-scale reference")[0]
    t3_line.set_gid("growth-student-t3-null")
    emp_line = axB.plot(mps, emp, color=INK, lw=1.0, marker="o",
                        ms=MS_GROWTH, zorder=3,
                        label="empirical (5-seed mean)")[0]
    emp_line.set_gid("growth-empirical")

    axB.text(6.05, 4.62, f"{emp[-1]:.2f} vs {gauss[-1]:.2f} fitted (in-sample)",
             fontsize=BASE, color=INK, ha="right", va="center")

    axB.set_xlim(1.8, 6.2)
    axB.set_ylim(0, 5.2)
    axB.set_xticks(mps)
    axB.set_yticks([0, 1, 2, 3, 4])
    axB.set_xlabel("m′ = placements tried")
    axB.set_ylabel("max − mean (pt)")
    handles, labels = axB.get_legend_handles_labels()
    order = [2, 0, 1]  # empirical first
    axB.legend([handles[k] for k in order], [labels[k] for k in order],
               loc="lower right", frameon=False, borderaxespad=0.2,
               handlelength=1.6, labelspacing=0.3)
    axB.text(0.0, 1.02, "B", transform=axB.transAxes,
             fontsize=FS_PANEL_LETTER, fontweight="bold", va="bottom",
             ha="left")

    return fig


def render(data: dict, output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    """Save fig2_harvest.pdf/.png into output_dir and close the figure.

    constrained_layout already fits everything inside the exact canvas, so
    no bbox_inches="tight" and no post-layout shrinking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data)
    pdf_path = output_dir / "fig2_harvest.pdf"
    png_path = output_dir / "fig2_harvest.png"
    try:
        fig.savefig(pdf_path)
        fig.savefig(png_path)
    finally:
        plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    data = load_harvest_data()
    render(data)

    P = data["placements"]
    seeds = data["seeds"]
    D = data["deltas"]
    honest = data["honest"]
    per_seed_sel = data["selected"]
    gap = data["gap"]
    sigma_bar = data["sigma_bar"]
    mps = data["m_values"]
    emp = data["empirical"]
    gauss = data["gaussian"]
    t3 = data["student_t3"]
    m = len(P)
    with open(LOG, encoding="utf-8") as f:
        d = json.load(f)

    # ------------- printed key numbers (self-review / honesty) -----------
    print(f"[fig2] placements: {P} | seeds: {seeds}")
    for i, s in enumerate(seeds):
        print(f"[fig2] seed {s}: deltas "
              + " ".join(f"{v:+.2f}" for v in D[i])
              + f" | max {D[i].max():+.2f} at p{P[int(D[i].argmax())]}")
    print(f"[fig2] Panel A lines: honest mean {honest:+.4f} pt | "
          f"per-seed select-then-average {per_seed_sel:+.4f} pt | "
          f"gap {gap:+.4f} pt (assertions vs washout fields passed)")
    print(f"[fig2] sigma_bar {sigma_bar:.4f} pt | "
          f"c_6 gauss {mc_cm(m, 'normal'):.6f} (json {d['c_m']['gauss']:.6f})"
          f" | c_6 t3 {mc_cm(m, 't3'):.6f} (json {d['c_m']['t3_unitvar']:.6f})")
    for k, mp in enumerate(mps):
        print(f"[fig2] m'={mp}: empirical {emp[k]:.3f} | gauss null "
              f"{gauss[k]:.3f} | t3 null {t3[k]:.3f}")
    print(f"[fig2] final point annotation: {emp[-1]:.2f} empirical vs "
          f"{gauss[-1]:.2f} Gaussian fitted (in-sample; not distribution validation)")
    print(f"[fig2] wrote {OUTDIR / 'fig2_harvest.pdf'} and .png")


if __name__ == "__main__":
    main()
