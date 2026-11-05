#!/usr/bin/env python3
"""App fig B: probe artifact persistence, 150-step vs full-length runs.

|Delta val acc| (pt) of the probe_cuda arm vs the paired reference: the 10
a1b 150-step seed-task pairs vs the 4 a1c full-length runs (mrpc/rte x seeds
41-42). Matched (task, seed) pairs are connected. Means per column as
horizontal ticks. The artifact shrinks but does not vanish at convergence.

Reads ONLY committed JSONs in logs_hf_pairing_probe/. Deterministic,
CPU-only, no network. Outputs figs/app_persistence.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs_hf_pairing_probe"
OUTDIR = Path(__file__).resolve().parent

VERM = "#D55E00"
INK = "0.15"

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 0.5,
    "ytick.labelsize": BASE - 0.5,
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

A1B = [("a1b", t, s) for t in ("rte", "mrpc") for s in range(41, 46)]
A1C = [("a1c", t, s) for t in ("rte", "mrpc") for s in (41, 42)]


def dacc(stage: str, task: str, seed: int) -> float:
    with open(LOGDIR / f"{stage}_{task}_s{seed}.json", encoding="utf-8") as f:
        res = json.load(f)["results"]
    return 100.0 * abs(res["probe_cuda"]["final_val_acc"]
                       - res["ref"]["final_val_acc"])


def main() -> None:
    short = {(t, s): dacc("a1b", t, s) for _, t, s in A1B}
    full = {(t, s): dacc("a1c", t, s) for _, t, s in A1C}
    mean_s = sum(short.values()) / len(short)
    mean_f = sum(full.values()) / len(full)
    max_f = max(full.values())

    fig, ax = plt.subplots(figsize=(3.3, 2.6), constrained_layout=True)

    x_s, x_f = 0.0, 1.0
    # deterministic jitter by sorted key order
    keys_s = sorted(short)
    keys_f = sorted(full)
    off_s = {k: (i - (len(keys_s) - 1) / 2.0) * (0.24 / (len(keys_s) - 1))
             for i, k in enumerate(keys_s)}
    off_f = {k: (i - (len(keys_f) - 1) / 2.0) * (0.10 / (len(keys_f) - 1))
             for i, k in enumerate(keys_f)}

    # connect matched (task, seed) pairs
    for k in keys_f:
        ax.plot([x_s + off_s[k], x_f + off_f[k]], [short[k], full[k]],
                color="0.72", lw=0.7, zorder=2)
    ax.plot([x_s + off_s[k] for k in keys_s], [short[k] for k in keys_s],
            ls="none", marker="o", ms=3.8, mfc=VERM, mec="none", zorder=3)
    ax.plot([x_f + off_f[k] for k in keys_f], [full[k] for k in keys_f],
            ls="none", marker="o", ms=3.8, mfc=VERM, mec="none", zorder=3)

    for x, m in ((x_s, mean_s), (x_f, mean_f)):
        ax.hlines(m, x - 0.20, x + 0.20, color=INK, lw=1.2, zorder=4)
    ax.text(x_s + 0.23, mean_s, f"mean {mean_s:.2f}", fontsize=BASE - 1.3,
            color=INK, va="center")
    # right of its bar, like the 150-step label: on the left a matched-pair
    # line ran through it
    ax.text(x_f + 0.23, mean_f, f"mean {mean_f:.2f}", fontsize=BASE - 1.3,
            color=INK, va="center", ha="left")
    kmax = max(full, key=full.get)
    ax.annotate(f"max {max_f:.2f}", xy=(x_f + off_f[kmax], max_f),
                xytext=(x_f + 0.12, max_f + 0.55), fontsize=BASE - 1.3,
                color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6,
                                shrinkA=1, shrinkB=2))
    ax.text(0.5, 6.55, "convergence does not\nerase the artifact",
            ha="center", va="top", fontsize=BASE - 1.0, color=INK,
            style="italic")

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0, 6.8)
    ax.set_xticks([x_s, x_f])
    ax.set_xticklabels(["150 steps\n(n=10)", "full length\n(n=4)"])
    ax.set_ylabel("|Δ val acc| vs paired ref (pt)")

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_persistence.{ext}")
    plt.close(fig)

    print("[appB] 150-step |dacc| (pt): "
          + " ".join(f"{k[0]}-s{k[1]}={v:.2f}" for k, v in sorted(short.items())))
    print("[appB] full-length |dacc| (pt): "
          + " ".join(f"{k[0]}-s{k[1]}={v:.2f}" for k, v in sorted(full.items())))
    print(f"[appB] means: 150-step {mean_s:.2f} | full {mean_f:.2f} | "
          f"full max {max_f:.2f}")
    print(f"[appB] wrote {OUTDIR / 'app_persistence.pdf'} and .png")


if __name__ == "__main__":
    main()
