#!/usr/bin/env python3
"""App fig R: the detection window is early training (CIFAR-100).

Full-run grad-norm participation ratio, clean vs 30% noise, 2 seeds
(logs_monitor_study cifar100 files). Early training separates the
conditions cleanly; late in training the noisy curves collapse BELOW the
clean ones (crossing annotated), so a late-window detector reads the signal
inverted.

Reads ONLY committed JSONs. Deterministic, CPU-only, no network.
Outputs figs/app_latewindow.{pdf,png}.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "logs_monitor_study"
OUTDIR = Path(__file__).resolve().parent

C_CLEAN = "#0072B2"
C_NOISE = "#D55E00"
INK = "0.15"
SEED_LS = {1: "-", 2: (0, (4, 2))}

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 0.5,
    "ytick.labelsize": BASE - 0.5,
    "legend.fontsize": BASE - 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def series(noise: str, seed: int):
    with open(MON / f"cifar100_n{noise}_frozen_seed{seed}.json",
              encoding="utf-8") as f:
        mon = json.load(f)["monitor"]
    return [m["epoch"] for m in mon], [m["grad_norm_pr"] for m in mon]


def main() -> None:
    runs = {(n, s): series(n, s) for n in ("0.0", "0.3") for s in (1, 2)}

    # per-epoch bucket means per condition (2-seed mean), to find crossing
    cond_mean = {}
    for n in ("0.0", "0.3"):
        b = defaultdict(list)
        for s in (1, 2):
            xs, ys = runs[(n, s)]
            for x, y in zip(xs, ys):
                b[int(math.floor(x))].append(y)
        cond_mean[n] = {e: sum(v) / len(v) for e, v in sorted(b.items())}
    cross = next(e for e in sorted(cond_mean["0.0"])
                 if cond_mean["0.3"][e] < cond_mean["0.0"][e])

    fig, ax = plt.subplots(figsize=(3.3, 2.6), constrained_layout=True)
    ax.axvspan(cross, 20, color="0.92", zorder=0)
    for (n, s), (xs, ys) in runs.items():
        ax.plot(xs, ys, color=C_CLEAN if n == "0.0" else C_NOISE,
                ls=SEED_LS[s], lw=0.9, zorder=3)

    ax.axvline(cross, color=INK, ls=(0, (3, 2)), lw=0.8, zorder=2)
    ax.text(cross - 0.35, 0.75, f"curves invert\n(epoch {cross})",
            fontsize=BASE - 1.0, color=INK, va="center", ha="right")
    # four short lines fit inside the grey band; "late window:" on one line
    # touched the dashed boundary
    ax.text(cross + (20 - cross) / 2.0, 0.66, "late\nwindow:\nsignal\ninverted",
            ha="center", va="center", fontsize=BASE - 1.3, color="0.35")
    ax.text(2.6, 0.30, "early window:\nclean below noisy", ha="center",
            va="center", fontsize=BASE - 1.3, color=INK)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad-norm participation ratio")
    handles = [Line2D([], [], color=C_NOISE, lw=1.0),
               Line2D([], [], color=C_CLEAN, lw=1.0),
               Line2D([], [], color="0.4", ls=SEED_LS[1], lw=1.0),
               Line2D([], [], color="0.4", ls=SEED_LS[2], lw=1.0)]
    ax.legend(handles, ["30% noise", "clean", "seed 1", "seed 2"],
              # lower left is empty; upper right put the line-style keys
              # across the dashed epoch-16 boundary
              loc="lower left", frameon=False, borderaxespad=0.2,
              handlelength=1.5, labelspacing=0.25, ncol=2,
              columnspacing=0.8)

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_latewindow.{ext}")
    plt.close(fig)

    print(f"[appR] condition means (2 seeds) at epoch bucket 1: clean "
          f"{cond_mean['0.0'][1]:.3f} vs noise {cond_mean['0.3'][1]:.3f}")
    print(f"[appR] first epoch bucket with noise mean BELOW clean mean: "
          f"{cross} (clean {cond_mean['0.0'][cross]:.3f} vs noise "
          f"{cond_mean['0.3'][cross]:.3f})")
    print(f"[appR] final bucket 19: clean {cond_mean['0.0'][19]:.3f} vs "
          f"noise {cond_mean['0.3'][19]:.3f}")
    print(f"[appR] wrote {OUTDIR / 'app_latewindow.pdf'} and .png")


if __name__ == "__main__":
    main()
