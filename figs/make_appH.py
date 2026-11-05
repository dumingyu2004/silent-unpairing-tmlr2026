#!/usr/bin/env python3
"""App fig H: guard vs LAWA payout across the five under-tuned cells.

Paired bars of the mean val-2 gain over the baseline arm (pt): guard
(topo champion) vs LAWA, for NLP macro (3 tasks x 5 seeds; LAWA not run on
NLP, bar omitted), CIFAR-10 clean/noise (frozen det), CIFAR-100 clean/noise
(stratified det). Above each bar: per-unit win/zero/negative counts
(units = 15 task-seed cells for NLP, 5 seeds for vision; zero means an
EXACT 0.00 delta at metric resolution -- the paper's certified-no-op
semantics -- so the +0.02 pt seed on C100 noise counts as a win, matching
the main text's 3/2/0).

Reads ONLY committed logs: the four det JSONs in logs_local_5080_mnist/ and
the logs_nlp_payout_det/ epoch_log.csv + topo_sel_log.jsonl trees.
Deterministic, CPU-only, no network. Outputs figs/app_payout_bars.{pdf,png}.
"""
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "logs_local_5080_mnist"
NLP = ROOT / "logs_nlp_payout_det"
OUTDIR = Path(__file__).resolve().parent

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "0.15"
TASKS = ["rte", "mrpc", "cola"]
ZERO_EPS = 1e-9  # pt: zero = exactly 0.00 (certified no-op), not a band

BASE = 7.5
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.linewidth": 0.6,
    "xtick.labelsize": BASE - 1.0,
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


def wzn(deltas):
    w = sum(1 for x in deltas if x > ZERO_EPS)
    z = sum(1 for x in deltas if abs(x) <= ZERO_EPS)
    n = sum(1 for x in deltas if x < -ZERO_EPS)
    return w, z, n


def wzn_label(counts):
    w, z, n = counts
    parts = [f"{w}W"] + ([f"{z}Z"] if z else []) + ([f"{n}N"] if n else [])
    return "·".join(parts)


def vision_cell(fn):
    with open(VIS / fn, encoding="utf-8") as f:
        r = json.load(f)["results"]
    seeds = sorted(r["baseline"].keys(), key=int)
    g = [100 * (r["topo"][s] - r["baseline"][s]) for s in seeds]
    l = [100 * (r["lawa"][s] - r["baseline"][s]) for s in seeds]
    return g, l


def nlp_deltas():
    out = []
    for t in TASKS:
        for s in range(1, 6):
            ep = NLP / f"roberta_base_{t}" / "baseline_payout_det" / \
                f"bs32_seed{s}" / "epoch_log.csv"
            with open(ep, encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["phase"] == "val2"]
            base = float(rows[-1]["primary_metric"])
            sel = NLP / f"roberta_base_{t}" / "topo_ema_payout_det" / \
                f"bs32_seed{s}" / "topo_sel_log.jsonl"
            last = None
            with open(sel, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = json.loads(line)
            out.append((t, s, 100 * (last["val2_selected_primary"] - base)))
    return out


def main() -> None:
    nlp = nlp_deltas()
    per_task = {t: [d for tt, _, d in nlp if tt == t] for t in TASKS}
    nlp_macro = sum(sum(v) / len(v) for v in per_task.values()) / len(TASKS)
    nlp_flat = [d for _, _, d in nlp]

    cells = [("NLP\nmacro", nlp_macro, None, wzn(nlp_flat), None)]
    for label, fn in [("C10\nclean", "cifar10_resnet_frozen_det_n0.0.json"),
                      ("C10\nnoise", "cifar10_resnet_frozen_det_n0.3.json"),
                      ("C100\nclean", "cifar100_resnet_strat_det_n0.0.json"),
                      ("C100\nnoise", "cifar100_resnet_strat_det_n0.3.json")]:
        g, l = vision_cell(fn)
        cells.append((label, sum(g) / len(g), sum(l) / len(l),
                      wzn(g), wzn(l)))

    fig, ax = plt.subplots(figsize=(3.3, 2.7), constrained_layout=True)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    w = 0.32
    for i, (label, g, l, cg, cl) in enumerate(cells):
        xg = i if l is None else i - w / 2 - 0.02
        ax.bar(xg, g, width=w, color=BLUE, edgecolor="black", lw=0.4,
               zorder=3)
        # vertical: the labels are wider than the bars and ran into the
        # neighbouring bar when set horizontally
        ax.text(xg, g + 0.25, wzn_label(cg), ha="center", va="bottom",
                fontsize=BASE - 2.2, color=INK, rotation=90)
        if l is not None:
            xl = i + w / 2 + 0.02
            ax.bar(xl, l, width=w, color=ORANGE, edgecolor="black", lw=0.4,
                   zorder=3)
            ax.text(xl, max(l, 0) + 0.25, wzn_label(cl), ha="center",
                    va="bottom", fontsize=BASE - 2.2, color=INK, rotation=90)
        else:
            ax.text(i + w / 2 + 0.06, 0.25, "LAWA\nn/a", ha="left",
                    va="bottom", fontsize=BASE - 2.2, color="0.5")

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylabel("mean val-2 gain vs baseline (pt)")
    ax.set_ylim(-2.0, 8.6)
    ax.set_yticks([-2, 0, 2, 4, 6, 8])
    handles = [Patch(facecolor=BLUE, edgecolor="black", lw=0.4,
                     label="guard"),
               Patch(facecolor=ORANGE, edgecolor="black", lw=0.4,
                     label="LAWA")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False, handlelength=1.2, handletextpad=0.45,
              columnspacing=1.0, borderaxespad=0.0)

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_payout_bars.{ext}")
    plt.close(fig)

    print(f"[appH] NLP per-task means (pt): "
          + " ".join(f"{t}={sum(v)/len(v):+.2f}" for t, v in per_task.items())
          + f" | macro {nlp_macro:+.2f} | W/Z/N {wzn(nlp_flat)}")
    for label, g, l, cg, cl in cells[1:]:
        print(f"[appH] {label.replace(chr(10), ' ')}: guard {g:+.2f} pt "
              f"{cg} | lawa {l:+.2f} pt {cl}")
    print("[appH] zero = exact 0.00 delta (certified no-op semantics)")
    print(f"[appH] wrote {OUTDIR / 'app_payout_bars.pdf'} and .png")


if __name__ == "__main__":
    main()
