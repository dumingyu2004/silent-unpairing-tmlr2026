#!/usr/bin/env python3
"""Realistic-eval harvest figure (ext21): the selection-harvest reproduced by an
UNMODIFIED standard monitoring evaluation, at a believable protocol.

One panel: paired delta vs the no-eval base run (pt) for every (seed, placement)
cell of the ext21 5-seed x 5-placement grid on MW-Net official code (45k train,
12 epochs, real 3-phase LR, held-out test). The mid-training "probe" is a
standard shuffled monitoring DataLoader whose iterator draws exactly two int64
from the global stream training consumes -- no randperm inserted. Per-seed max
circled; honest mean (dashed) vs per-seed select-then-average (solid) bracket
the manufactured gain. Only two isolated comparisons were directly checked,
both at placement 1400; their endpoint deltas are shown as two markers.
No isolated-grid selection gain or unmeasured zero is inferred.

Reads logs_moneyshot/ext21_re_verdict5.json plus the two original isolated
and reference run records. Shared-arm quantities are recomputed from deltas_A_pt and
asserted against the stored fields. Deterministic, CPU-only, no network.
Outputs figs/fig_realeval.{pdf,png}. Matches figs/make_fig2.py house style.
"""
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs_moneyshot" / "ext21_re_verdict5.json"
OUTDIR = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe palette (identical to make_fig2.py)
BLUE = "#0072B2"      # honest mean
VERM = "#D55E00"      # selection
GREEN = "#009E73"     # fork_rng cure
INK = "0.15"
SEED_MARKERS = ["o", "s", "^", "D", "v"]

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 1.0,
    "ytick.labelsize": BASE - 0.5,
    "legend.fontsize": BASE - 1.3,
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


def mc_cm(m: int, n: int = 200_000) -> float:
    """E[max - mean] of m i.i.d. standard normals (deterministic MC, rng(0))."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, m))
    return float((x.max(1) - x.mean(1)).mean())


def main() -> None:
    with open(LOG, encoding="utf-8") as f:
        d = json.load(f)
    P = d["placements"]
    seeds = d["seeds_present"]
    D = np.array(d["deltas_A_pt"])
    m = len(P)

    # ---- recompute from deltas; assert vs stored comparator fields ----
    honest = float(D.mean())
    per_seed_sel = float(D.max(axis=1).mean())
    gap = per_seed_sel - honest
    sigma_bar = float(D.std(axis=1, ddof=1).mean())
    cN = mc_cm(m)
    ratio = gap / (cN * sigma_bar)

    assert abs(honest - d["honest_mean_A"]) < 1e-9, (honest, d["honest_mean_A"])
    assert abs(per_seed_sel - d["per_seed_select_A"]) < 1e-9
    assert abs(gap - d["manufactured_gain_A"]) < 1e-9
    assert abs(cN - d["c_N"]) < 1e-6, (cN, d["c_N"])
    assert abs(sigma_bar - d["sigma_bar"]) < 1e-6
    assert abs(ratio - d["ratio"]) < 1e-3, (ratio, d["ratio"])
    # The legacy comparator's manufactured_gain_B was not a full-grid readout.
    # Verify only the two archived isolated endpoint comparisons directly.
    isolated = []
    for seed in (1, 2):
        arm = json.loads((LOG.parent / f"ext21_re_seed{seed}_iso_p1400.json").read_text())
        base = json.loads((LOG.parent / f"ext21_re_seed{seed}_rep.json").read_text())
        assert not arm["partial"] and not base["partial"]
        assert arm["args"]["probe_step"] == 1400
        assert arm["result"]["eval_fired"] == 1
        for field in ("state_sha256", "vnet_state_sha256"):
            assert arm["result"][field] == base["result"][field]
        delta = 100 * (arm["result"]["final_test_acc"] - base["result"]["final_test_acc"])
        assert delta == 0.0
        isolated.append((seed, 1400, delta))

    fig, ax = plt.subplots(figsize=(3.3, 2.45), constrained_layout=True)

    ax.axhline(0, color="0.85", lw=0.5, zorder=0.5)
    ax.axhline(honest, color=BLUE, ls=(0, (4, 2)), lw=1.1, zorder=2)
    ax.axhline(per_seed_sel, color=VERM, lw=1.3, zorder=2)

    span = max(P) - min(P)
    off = span * 0.012
    for i in range(len(seeds)):
        xj = [p + (i - 2) * off for p in P]  # deterministic seed offset
        ax.plot(xj, D[i], ls="none", marker=SEED_MARKERS[i % 5], ms=3.6,
                mfc="0.4", mec="none", zorder=3)
        jmax = int(D[i].argmax())
        ax.plot([xj[jmax]], [D[i][jmax]], ls="none", marker="o", ms=9,
                mfc="none", mec=VERM, mew=1.0, zorder=4)

    # Two verified points only; no line across unmeasured placements.
    for seed, placement, delta in isolated:
        i = seeds.index(seed)
        ax.plot(placement + (i - 2) * off, delta, ls="none",
                marker=SEED_MARKERS[i], ms=4.4, mfc="white",
                mec=GREEN, mew=1.0, zorder=5)

    # manufactured-gain bracket
    xb = max(P) + span * 0.03
    ax.annotate("", xy=(xb, per_seed_sel), xytext=(xb, honest),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9,
                                mutation_scale=7, shrinkA=0.0, shrinkB=0.0),
                zorder=4, annotation_clip=False)

    lo, hi = float(D.min()), float(D.max())
    pad = (hi - lo) * 0.28
    ax.set_ylim(lo - pad, hi + pad * 1.25)
    ax.set_xlim(min(P) - span * 0.06, max(P) + span * 0.16)

    ax.text(P[1], hi + pad * 0.95,
            f"selected mean: {per_seed_sel:+.2f} pt",
            fontsize=BASE - 1.4, color=VERM, va="center", zorder=5)
    ax.text(P[1], honest - pad * 0.30,
            f"honest mean {honest:+.2f} pt", fontsize=BASE - 1.4,
            color=BLUE, va="top", zorder=5)
    ax.text(xb, (honest + per_seed_sel) / 2, f"  {gap:+.2f} pt\n  manufactured",
            fontsize=BASE - 1.4, color=INK, ha="left", va="center", zorder=5,
            clip_on=False)
    ax.annotate(
        "fork_rng isolated:\n2/25 checked\nΔ = 0.00",
        xy=(1400 - 1.5 * off, 0.0),
        xytext=(min(P) + span * 0.07, 0.95),
        fontsize=BASE - 1.6, color=GREEN, ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7,
                        shrinkA=1.0, shrinkB=1.0),
        zorder=5,
    )
    ax.text(0.985, 0.03,
            f"standard eval = 2 int64 draws;  ratio to $c_5\\bar\\sigma$ = {ratio:.2f}",
            transform=ax.transAxes, fontsize=BASE - 1.7, color=INK,
            ha="right", va="bottom")

    ax.set_xticks(P)
    ax.set_xlabel("monitoring-eval placement (step)")
    ax.set_ylabel("Δ vs no-eval run (pt)")

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig_realeval.{ext}")
    plt.close(fig)

    # ---- printed key numbers (self-review / honesty) ----
    print(f"[fig_realeval] placements {P} | seeds {seeds}")
    for i, s in enumerate(seeds):
        print(f"[fig_realeval] seed {s}: "
              + " ".join(f"{v:+.2f}" for v in D[i])
              + f" | max {D[i].max():+.2f} at p{P[int(D[i].argmax())]}")
    print(f"[fig_realeval] honest {honest:+.4f} | per-seed-select "
          f"{per_seed_sel:+.4f} | manufactured {gap:+.4f} pt "
          f"(stored {d['manufactured_gain_A']:+.4f})")
    print(f"[fig_realeval] sigma_bar {sigma_bar:.4f} | c_5 {cN:.4f} | "
          f"ratio {ratio:.3f} (stored {d['ratio']:.3f}) | "
          f"isolated: {len(isolated)}/{D.size} directly checked endpoint pairs; "
          "no isolated-grid harvest estimated -- all assertions passed")


if __name__ == "__main__":
    main()
