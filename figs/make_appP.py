#!/usr/bin/env python3
"""App fig P: detector transfers — grad-PR clean vs noise on four datasets.

2x2 grid of gradient participation-ratio trajectories, clean (blue) vs 30%
label noise (vermillion), 2 seeds each (solid / dashed):
  top: CIFAR-10 and CIFAR-100 (logs_monitor_study, per-step 'grad_norm_pr'),
  bottom: CoLA and MRPC (logs_nlp_gradpr, step_log.csv 'topo_grad_pr';
  x converted to epochs via step * total_epochs / max_step).

Reads ONLY committed logs. Deterministic, CPU-only, no network.
Outputs figs/app_detector_grid.{pdf,png}.
"""
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "logs_monitor_study"
NLP = ROOT / "logs_nlp_gradpr"
OUTDIR = Path(__file__).resolve().parent

C_CLEAN = "#0072B2"
C_NOISE = "#D55E00"
SEED_LS = {1: "-", 2: (0, (4, 2))}

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE - 0.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 1.3,
    "ytick.labelsize": BASE - 1.3,
    "legend.fontsize": BASE - 1.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def vision_series(ds: str, noise: str, seed: int):
    with open(MON / f"{ds}_n{noise}_frozen_seed{seed}.json",
              encoding="utf-8") as f:
        mon = json.load(f)["monitor"]
    return [m["epoch"] for m in mon], [m["grad_norm_pr"] for m in mon]


def nlp_series(task: str, cond: str, seed: int):
    path = NLP / f"roberta_base_{task}" / f"topo_gradpr_{cond}" / \
        f"bs32_seed{seed}" / "step_log.csv"
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["topo_grad_pr"] not in ("", "None")]
    max_step = max(int(r["step"]) for r in rows)
    total_ep = max(float(r["epoch"]) for r in rows)
    xs = [int(r["step"]) * total_ep / max_step for r in rows]
    return xs, [float(r["topo_grad_pr"]) for r in rows]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.1),
                             constrained_layout=True)
    ep1 = {}

    panels = [
        ("CIFAR-10", axes[0, 0], "vision", "cifar10"),
        ("CIFAR-100", axes[0, 1], "vision", "cifar100"),
        ("CoLA", axes[1, 0], "nlp", "cola"),
        ("MRPC", axes[1, 1], "nlp", "mrpc"),
    ]
    for name, ax, kind, key in panels:
        vals_at_1 = {}
        for cond, color in (("clean", C_CLEAN), ("noise", C_NOISE)):
            for seed in (1, 2):
                if kind == "vision":
                    xs, ys = vision_series(
                        key, "0.0" if cond == "clean" else "0.3", seed)
                else:
                    xs, ys = nlp_series(
                        key, "clean" if cond == "clean" else "noise30", seed)
                ax.plot(xs, ys, color=color, ls=SEED_LS[seed], lw=0.8)
                k = min(range(len(xs)), key=lambda i: abs(xs[i] - 1.0))
                vals_at_1.setdefault(cond, []).append(ys[k])
        ep1[name] = {c: sum(v) / len(v) for c, v in vals_at_1.items()}
        ax.text(0.97, 0.90, name, transform=ax.transAxes, ha="right",
                va="top", fontsize=BASE - 1.0, color="black",
                fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        if kind == "vision":
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 10, 20])
        else:
            ax.set_xlim(0, 5.2)
            ax.set_xticks([0, 2.5, 5])
    axes[0, 0].set_ylabel("grad PR")
    axes[1, 0].set_ylabel("grad PR")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")

    handles = [Line2D([], [], color=C_NOISE, lw=1.0),
               Line2D([], [], color=C_CLEAN, lw=1.0),
               Line2D([], [], color="0.4", ls=SEED_LS[1], lw=1.0),
               Line2D([], [], color="0.4", ls=SEED_LS[2], lw=1.0)]
    labels = ["30% noise", "clean", "seed 1", "seed 2"]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=0.9)

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_detector_grid.{ext}",
                    bbox_inches="tight")
    plt.close(fig)

    for name, v in ep1.items():
        print(f"[appP] {name}: grad-PR at ~epoch 1 (2-seed mean) "
              f"clean {v['clean']:.3f} vs noise {v['noise']:.3f} "
              f"(gap {v['noise'] - v['clean']:+.3f})")
    print("[appP] keys: vision 'grad_norm_pr' (monitor list), "
          "NLP 'topo_grad_pr' (step_log.csv)")
    print(f"[appP] wrote {OUTDIR / 'app_detector_grid.pdf'} and .png")


if __name__ == "__main__":
    main()
