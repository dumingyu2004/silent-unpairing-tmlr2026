#!/usr/bin/env python3
"""App fig S: signal-vs-update decoupling in meta-reweighting systems.

Left: MW-Net (logs_a2_mwnet/mwnet_signal_*.json, 4 runs = noise {0,0.4} x
seeds {1,2}): raw meta-gradient norm (vermillion) swings over ~3 orders of
magnitude while the induced v-net update norm (blue) stays within a small
band — log scale, per-run thin lines, pooled max/min ratios annotated.

Right: L2RW (logs_a2_mwnet/l2rw_signal_*.json, 4 runs):
grad_eps_norm / lr stays O(1); the observed pooled band is shaded.

Reads ONLY committed JSONs. Deterministic, CPU-only, no network.
Outputs figs/app_f1_decoupling.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs_a2_mwnet"
OUTDIR = Path(__file__).resolve().parent

VERM = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
INK = "0.15"
RUNS = [("0.0", 1), ("0.0", 2), ("0.4", 1), ("0.4", 2)]

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE - 0.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 1.3,
    "ytick.labelsize": BASE - 1.3,
    "legend.fontsize": BASE - 1.5,
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


def rows(kind: str, noise: str, seed: int):
    with open(LOGDIR / f"{kind}_signal_n{noise}_s{seed}.json",
              encoding="utf-8") as f:
        return json.load(f)["rows"]


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.3, 2.5),
                                   constrained_layout=True)

    meta_all, upd_all, eps_all = [], [], []
    run_ratio = {}
    for noise, seed in RUNS:
        rs = rows("mwnet", noise, seed)
        xs = [r["step"] for r in rs]
        meta = [r["meta_grad_norm"] for r in rs]
        upd = [r["vnet_update_norm"] for r in rs]
        run_ratio[(noise, seed)] = max(meta) / min(meta)
        meta_all += meta
        upd_all += upd
        axL.plot(xs, meta, color=VERM, lw=0.7, alpha=0.8)
        axL.plot(xs, upd, color=BLUE, lw=0.7, alpha=0.8)
    meta_ratio = max(run_ratio.values())  # largest within-run swing
    upd_ratio = max(upd_all) / min(upd_all)

    axL.set_yscale("log")
    axL.set_ylim(3e-3, 4e2)
    axL.set_xlabel("step")
    axL.set_ylabel("norm (log)")
    axL.text(0.97, 0.96, "MW-Net", transform=axL.transAxes, ha="right",
             va="top", fontsize=BASE - 1.0, color="black", fontweight="bold")
    axL.text(430, 4.0, f"meta-grad: up to\n~{meta_ratio:,.0f}× swing",
             fontsize=BASE - 1.8, color=VERM, ha="right", va="bottom")
    axL.text(430, 3.6e-3, f"update norm: ~{upd_ratio:.1f}× band",
             fontsize=BASE - 1.8, color=BLUE, ha="right", va="bottom")

    for noise, seed in RUNS:
        rs = rows("l2rw", noise, seed)
        xs = [r["step"] for r in rs]
        eps = [r["grad_eps_norm_over_lr"] for r in rs]
        eps_all += eps
        axR.plot(xs, eps, color=GREEN, lw=0.7, alpha=0.8)
    lo, hi = min(eps_all), max(eps_all)
    axR.axhspan(lo, hi, color=GREEN, alpha=0.12, zorder=0)

    axR.set_yscale("log")
    axR.set_ylim(3e-3, 4e2)
    axR.set_xlabel("step")
    axR.text(0.97, 0.96, "L2RW", transform=axR.transAxes, ha="right",
             va="top", fontsize=BASE - 1.0, color="black", fontweight="bold")
    axR.text(0.5, 0.30, f"O(1) band:\n{lo:.1f}–{hi:.0f} × lr",
             transform=axR.transAxes, ha="center", va="top",
             fontsize=BASE - 1.8, color="#007A5C")

    handles = [Line2D([], [], color=VERM, lw=1.0),
               Line2D([], [], color=BLUE, lw=1.0),
               Line2D([], [], color=GREEN, lw=1.0)]
    fig.legend(handles, ["MW-Net meta-grad norm", "MW-Net update norm",
                         "L2RW grad-eps / lr"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, handlelength=1.4, handletextpad=0.4,
               columnspacing=0.9)

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_f1_decoupling.{ext}",
                    bbox_inches="tight")
    plt.close(fig)

    print(f"[appS] MW-Net meta_grad_norm pooled range "
          f"{min(meta_all):.4g}..{max(meta_all):.4g} "
          f"(pooled {max(meta_all)/min(meta_all):,.0f}x)")
    print("[appS] per-run meta swings: "
          + " ".join(f"n{n}s{s}={r:,.0f}x"
                     for (n, s), r in run_ratio.items())
          + f" | annotated max within-run {meta_ratio:,.0f}x")
    print(f"[appS] MW-Net vnet_update_norm pooled range "
          f"{min(upd_all):.4g}..{max(upd_all):.4g} "
          f"=> band ~{upd_ratio:.2f}x")
    print(f"[appS] L2RW grad_eps_norm_over_lr pooled range "
          f"{lo:.3g}..{hi:.3g} (O(1) band, shaded)")
    print(f"[appS] wrote {OUTDIR / 'app_f1_decoupling.pdf'} and .png")


if __name__ == "__main__":
    main()
