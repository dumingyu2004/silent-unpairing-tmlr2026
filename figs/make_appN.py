#!/usr/bin/env python3
"""App fig N: CoLA seed-2 case study — guard rescues a collapsed final epoch.

val-2 MCC (x100) per half-epoch for the baseline arm vs the guard (topo) arm
(deterministic pairing, same seed). The baseline's raw model collapses at
epoch 5 (final 47.83); the guard's deployed champion ends at 62.88. Final
values marked and annotated; both recomputed and asserted against
topo_sel_log.jsonl.

Reads ONLY committed logs under logs_nlp_payout_det/roberta_base_cola/.
Deterministic, CPU-only, no network. Outputs figs/app_cola_case.{pdf,png}.
"""
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "logs_nlp_payout_det" / "roberta_base_cola"
OUTDIR = Path(__file__).resolve().parent

BLUE = "#0072B2"
GRAY = "0.45"
INK = "0.15"
SEED = 2

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


def val2_curve(arm: str):
    path = CELL / f"{arm}_payout_det" / f"bs32_seed{SEED}" / "epoch_log.csv"
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "val2"]
    assert all(r["primary_metric_name"] == "mcc" for r in rows)
    return ([float(r["epoch"]) for r in rows],
            [100 * float(r["primary_metric"]) for r in rows])


def main() -> None:
    eb, yb = val2_curve("baseline")
    et, yt = val2_curve("topo_ema")

    last = None
    sel = CELL / "topo_ema_payout_det" / f"bs32_seed{SEED}" / \
        "topo_sel_log.jsonl"
    with open(sel, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    champ_final = 100 * last["val2_selected_primary"]
    base_final = yb[-1]
    assert abs(yt[-1] - champ_final) < 1e-9
    assert abs(100 * last["val2_raw_primary"] - base_final) < 1e-9

    fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)
    ax.plot(eb, yb, color=GRAY, lw=1.1, ls=(0, (4, 2)), marker="o", ms=2.6,
            label="baseline (raw model)", zorder=2)
    ax.plot(et, yt, color=BLUE, lw=1.3, marker="o", ms=2.8,
            label="guard (deployed champion)", zorder=3)

    for y, c in ((base_final, GRAY), (champ_final, BLUE)):
        ax.hlines(y, 5.06, 5.42, color=c, lw=1.2, zorder=4)
    ax.text(5.48, base_final, f"{base_final:.2f}", fontsize=BASE - 1.0,
            color=GRAY, va="center")
    ax.text(5.48, champ_final, f"{champ_final:.2f}", fontsize=BASE - 1.0,
            color=BLUE, va="center")
    ax.annotate("", xy=(5.24, champ_final - 0.7),
                xytext=(5.24, base_final + 0.7),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                mutation_scale=7))
    # right of the arrow: on the left the dashed baseline crossed the label
    ax.text(5.32, (base_final + champ_final) / 2.0,
            f"+{champ_final - base_final:.2f}", fontsize=BASE - 1.3,
            color=INK, ha="left", va="center")

    ax.set_xlim(0.3, 6.05)
    ax.set_ylim(44, 77)
    ax.set_yticks([45, 50, 55, 60, 65, 70, 75])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("epoch")
    ax.set_ylabel("val-2 MCC (×100)")
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.2,
              handlelength=1.8, labelspacing=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"app_cola_case.{ext}")
    plt.close(fig)

    print(f"[appN] CoLA seed {SEED}: baseline val2 curve "
          + " ".join(f"{v:.2f}" for v in yb))
    print(f"[appN] CoLA seed {SEED}: guard   val2 curve "
          + " ".join(f"{v:.2f}" for v in yt))
    print(f"[appN] finals: baseline {base_final:.2f} | guard champion "
          f"{champ_final:.2f} (champion={last['champion']}, "
          f"trigger: {last['trigger_reason']}) | gain "
          f"{champ_final - base_final:+.2f}")
    print(f"[appN] wrote {OUTDIR / 'app_cola_case.pdf'} and .png")


if __name__ == "__main__":
    main()
