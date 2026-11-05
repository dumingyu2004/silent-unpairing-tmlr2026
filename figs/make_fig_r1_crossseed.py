#!/usr/bin/env python3
"""Fig R1: the preregistered cross-seed held-out prediction for Proposition 2.

The harvest-law preregistration (appendix.tex, remark "Model premises and
their preregistered check") lists four analyses: "fit against exact
constants; cross-seed prediction; bootstrap subsampling curve;
variance-matched t_3 alternative". Three are reported in the paper. This
figure supplies the missing fourth one, which is the only genuinely
out-of-sample test of Proposition 2 in the whole campaign.

Test design (leave-one-seed-out, no refitting of anything else):
  for each held-out seed s,
      sigma_bar(-s) = mean of the per-seed SDs of the OTHER four seeds
      predicted_s   = c_6 * sigma_bar(-s)
      observed_s    = max_j Delta_{s,j} - mean_j Delta_{s,j}
with c_6 = 1.2672, the exact quadrature constant of appendix Table
tab:cm (row "c_m (exact)", m = 6). Nothing in the prediction touches the
held-out seed's own deltas, so predicted_s is genuinely out of sample.

A second, deliberately in-sample series is drawn for contrast:
predicted_own_s = c_6 * sd_s uses the held-out seed's OWN scale. It is
NOT a held-out test and is labelled as such on the figure; it is here
only to separate two very different failure modes -- a wrong constant
c_6 (which would move both series) from non-transportable per-seed scale
(which moves only the held-out series).

Reads ONLY the committed JSON log logs_flip_demo/prop2_fit.json. Every
per-seed SD and every observed harvest value is recomputed from
deltas_pt at runtime and asserted against the stored sd_pt /
harvest_max_minus_mean_pt fields; the leave-one-out predictions are
recomputed here and are asserted to differ from the paper's full-sample
aggregate prediction c_6 * sigma_bar(all), so nothing plotted can be a
copy of a stored aggregate.

Sized for the TMLR single column (textwidth 6.5 in): exactly
4.00 x 3.20 in, all text >= 8 pt, all lines >= 0.7 pt, markers >= 4 pt.

Deterministic, CPU-only, no network. No model is run and no number is
interpolated: every plotted value traces to a JSON key printed by main().
Outputs figs/fig_r1_crossseed.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs_flip_demo" / "prop2_fit.json"
OUTDIR = Path(__file__).resolve().parent

# Exact quadrature constant for m = 6, appendix Table tab:cm row "c_m (exact)".
C6_EXACT = 1.2672

# Okabe-Ito colorblind-safe palette, consistent with figs/make_fig2.py.
BLUE = "#0072B2"      # held-out (leave-one-seed-out) prediction
SKY = "#56B4E9"       # in-sample own-sigma reference
VERM = "#D55E00"      # identity line
INK = "0.15"
SEED_MARKERS = ["o", "s", "^", "D", "v"]
# Deterministic per-seed label placement (dy in data units, vertical align);
# seed 2 goes below its marker so it cannot crowd seed 1's marker.
LABEL_OFFSETS = [(0.22, "bottom"), (-0.24, "top"), (0.22, "bottom"),
                 (0.22, "bottom"), (0.22, "bottom")]

BASE = 8.0            # floor: no visible text below 8 pt
MS_POINT = 5.0        # held-out markers
MS_REF = 4.4          # in-sample reference markers
# Drawn at the width it is typeset at (0.50\textwidth of 6.5in). Shrinking it
# here rather than in \includegraphics is what keeps every label at the 8 pt
# floor above: scaling a 4.0in canvas down to 3.25in would land the text at
# 6.5 pt instead.
FIGSIZE = (3.05, 2.60)

# Applied both at import (matching figs/make_fig2.py) and inside
# build_figure via plt.rc_context, so the floors hold even when another
# figure module has written the global rcParams first.
RC = {
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
}
mpl.rcParams.update(RC)


def leave_one_out_sigma_bar(sd: np.ndarray) -> np.ndarray:
    """sigma_bar computed from every seed EXCEPT the held-out one.

    Returns an array whose s-th entry is mean(sd[all seeds but s]).
    """
    sd = np.asarray(sd, dtype=float)
    n = sd.size
    if n < 2:
        raise ValueError("leave-one-out needs at least 2 seeds")
    total = sd.sum()
    return (total - sd) / (n - 1)


def load_crossseed_data(path: Path = LOG, c_m: float = C6_EXACT) -> dict:
    """Load the harvest grid and build the leave-one-seed-out test.

    Fail-closed: the data contract (5 seeds x 6 placements, no NaNs) and
    every recomputed summary are asserted against the frozen fields.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    placements = list(d["placements"])
    seeds = list(d["seeds"])
    D = np.asarray(d["deltas_pt"], dtype=float)

    # ---- data contract -------------------------------------------------
    if D.shape != (len(seeds), len(placements)):
        raise AssertionError(
            f"deltas_pt shape {D.shape} != (seeds {len(seeds)}, "
            f"placements {len(placements)})")
    if D.shape != (5, 6):
        raise AssertionError(
            f"expected the 5 seeds x 6 placements grid, got {D.shape}")
    if not np.isfinite(D).all():
        raise AssertionError("deltas_pt contains non-finite entries")
    if len(set(seeds)) != len(seeds):
        raise AssertionError(f"duplicate seeds in {seeds}")

    # ---- recompute every summary from deltas_pt, assert vs stored -------
    sd = D.std(axis=1, ddof=1)
    observed = D.max(axis=1) - D.mean(axis=1)
    if not np.allclose(sd, d["sd_pt"], atol=1e-12):
        raise AssertionError(f"recomputed sd {sd} != stored {d['sd_pt']}")
    if not np.allclose(observed, d["harvest_max_minus_mean_pt"], atol=1e-12):
        raise AssertionError(
            f"recomputed harvest {observed} != stored "
            f"{d['harvest_max_minus_mean_pt']}")

    # ---- the held-out test ---------------------------------------------
    sigma_bar_loo = leave_one_out_sigma_bar(sd)
    predicted = c_m * sigma_bar_loo

    # in-sample reference (NOT held out): each seed's own scale
    predicted_own = c_m * sd

    # the paper's full-sample aggregate prediction; predictions must not
    # coincide with it, or they would not be leave-one-out at all
    sigma_bar_all = float(sd.mean())
    aggregate_prediction = c_m * sigma_bar_all
    if np.any(np.abs(predicted - aggregate_prediction) < 1e-9):
        raise AssertionError(
            "a leave-one-out prediction equals the full-sample aggregate "
            f"{aggregate_prediction}; predictions are not held out")
    if len(np.unique(np.round(predicted, 12))) != predicted.size:
        raise AssertionError("leave-one-out predictions are not distinct")

    errors = predicted - observed
    errors_own = predicted_own - observed
    n_over = int((errors > 0).sum())
    n_under = int((errors < 0).sum())
    r = float(np.corrcoef(predicted, observed)[0, 1])

    return {
        "placements": placements,
        "seeds": seeds,
        "deltas": D,
        "sd": sd,
        "observed": observed,
        "sigma_bar_loo": sigma_bar_loo,
        "predicted": predicted,
        "predicted_own": predicted_own,
        "errors": errors,
        "errors_own": errors_own,
        "mae": float(np.abs(errors).mean()),
        "mae_own": float(np.abs(errors_own).mean()),
        "mean_signed_error": float(errors.mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "n_over": n_over,
        "n_under": n_under,
        "straddles_identity": n_over > 0 and n_under > 0,
        "pearson_r": r,
        "sigma_bar_all": sigma_bar_all,
        "aggregate_prediction": aggregate_prediction,
        "observed_mean": float(observed.mean()),
        "c_m": float(c_m),
    }


def build_figure(data: dict) -> Figure:
    """Predicted (held-out) vs observed harvest, with the identity line.

    Left cell: the square predicted-vs-observed panel (box aspect 1, equal
    x and y limits, so the identity line is a true 45-degree diagonal).
    Right cell: an axis-off panel carrying the out-of-sample error
    numbers, kept outside the data area so nothing overplots a datum.
    """
    with plt.rc_context(RC):
        seeds = data["seeds"]
        observed = data["observed"]
        predicted = data["predicted"]
        predicted_own = data["predicted_own"]

        fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.38])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])
        ax_txt.set_axis_off()

        lo = 1.2
        hi = 7.9
        ident = ax.plot([lo, hi], [lo, hi], color=VERM, lw=1.0,
                        ls=(0, (4, 2)), zorder=1.5, label="identity")[0]
        ident.set_gid("identity-line")

        for i, seed in enumerate(seeds):
            # thin connector: same observed value, own-sigma vs held-out sigma
            link = ax.plot([predicted[i], predicted_own[i]],
                           [observed[i], observed[i]],
                           color="0.72", lw=0.7, zorder=2)[0]
            link.set_gid(f"sigma-transport-seed-{seed}")

            ref = ax.plot([predicted_own[i]], [observed[i]], ls="none",
                          marker=SEED_MARKERS[i], ms=MS_REF, mfc="none",
                          mec=SKY, mew=0.9, zorder=3)[0]
            ref.set_gid(f"in-sample-own-sigma-seed-{seed}")

            pt = ax.plot([predicted[i]], [observed[i]], ls="none",
                         marker=SEED_MARKERS[i], ms=MS_POINT, mfc=BLUE,
                         mec=BLUE, zorder=4)[0]
            pt.set_gid(f"held-out-prediction-seed-{seed}")

            dy, va = LABEL_OFFSETS[i]
            ax.text(predicted[i], observed[i] + dy, f"s{seed}", fontsize=BASE,
                    color=BLUE, ha="center", va=va, zorder=5)

        legend_handles = [
            plt.Line2D([], [], ls="none", marker="o", ms=MS_POINT, mfc=BLUE,
                       mec=BLUE, label="held out"),
            plt.Line2D([], [], ls="none", marker="o", ms=MS_REF, mfc="none",
                       mec=SKY, mew=0.9, label="in sample"),
            plt.Line2D([], [], color=VERM, lw=1.0, ls=(0, (4, 2)),
                       label="identity"),
        ]
        # The legend lives in the side column, not inside the axes. At this
        # canvas width the data leaves no clean interior corner: the s1/s2/s4
        # cluster fills the lower middle and s5/s3 sit against the upper left,
        # so both in-axes placements land on markers. The side column is
        # already the figure's key, so the legend belongs with the statistics.
        ax_txt.legend(handles=legend_handles, loc="lower left", frameon=False,
                      borderaxespad=0.0, handlelength=1.4, handletextpad=0.5,
                      labelspacing=0.28)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1.0)
        ax.set_xticks([2, 3, 4, 5, 6, 7])
        ax.set_yticks([2, 3, 4, 5, 6, 7])
        ax.set_xlabel("predicted $c_6\\,\\bar{\\sigma}$ (pt)")
        ax.set_ylabel("observed max − mean (pt)")

        stats = (
            "held out (LOSO)\n"
            f"  MAE  {data['mae']:.2f} pt\n"
            f"  = {100 * data['mae'] / data['observed_mean']:.0f}% of mean\n"
            f"    harvest {data['observed_mean']:.2f}\n"
            f"  bias {data['mean_signed_error']:+.2f} pt\n"
            f"  {data['n_over']} over, {data['n_under']} under\n"
            f"  $r$ = {data['pearson_r']:+.2f}\n"
            "\n"
            "in sample (own $\\sigma$)\n"
            f"  MAE  {data['mae_own']:.2f} pt\n"
            "\n"
            f"$c_6$ = {data['c_m']:.4f} exact"
        )
        ax_txt.text(0.0, 0.98, stats, transform=ax_txt.transAxes, fontsize=BASE,
                    color=INK, ha="left", va="top", linespacing=1.32)

        return fig


def render(data: dict, output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    """Save fig_r1_crossseed.pdf/.png into output_dir; close the figure.

    constrained_layout already fits everything inside the exact canvas, so
    no bbox_inches="tight" and no post-layout shrinking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data)
    pdf_path = output_dir / "fig_r1_crossseed.pdf"
    png_path = output_dir / "fig_r1_crossseed.png"
    try:
        # savefig reads pdf.fonttype and savefig.dpi at write time, so the
        # rc context has to cover the save, not just the draw.
        with plt.rc_context(RC):
            fig.savefig(pdf_path)
            fig.savefig(png_path)
    finally:
        plt.close(fig)
    return pdf_path, png_path


def close_figure(fig: Figure) -> None:
    plt.close(fig)


def main() -> None:
    data = load_crossseed_data()
    render(data)

    seeds = data["seeds"]
    rel = LOG.relative_to(ROOT).as_posix()
    print("[R1] PROVENANCE DUMP -- every plotted number -> file -> JSON key")
    print(f"[R1] source file: {rel}")
    print(f"[R1] grid: seeds {seeds} (key 'seeds') x placements "
          f"{data['placements']} (key 'placements')")
    print(f"[R1] c_6 = {data['c_m']} <- paper/appendix.tex Table tab:cm, "
          "row 'c_m (exact)', column m=6 (NOT from any log)")
    print("[R1] per-seed inputs, all recomputed from key 'deltas_pt' and "
          "asserted equal to the stored fields:")
    for i, s in enumerate(seeds):
        print(f"[R1]   seed {s}: deltas_pt[{i}] = "
              + " ".join(f"{v:+.2f}" for v in data["deltas"][i]))
        print(f"[R1]            sd (ddof=1) {data['sd'][i]:.6f} "
              f"== sd_pt[{i}]  |  observed max-mean "
              f"{data['observed'][i]:.6f} == harvest_max_minus_mean_pt[{i}]")
    print("[R1] leave-one-seed-out predictions (recomputed here; no stored "
          "field holds these):")
    for i, s in enumerate(seeds):
        others = [t for t in seeds if t != s]
        print(f"[R1]   hold out seed {s}: sigma_bar(seeds {others}) = "
              f"{data['sigma_bar_loo'][i]:.6f} -> predicted "
              f"{data['predicted'][i]:.4f} pt | observed "
              f"{data['observed'][i]:.4f} pt | error "
              f"{data['errors'][i]:+.4f} pt")
    print(f"[R1] held-out MAE {data['mae']:.4f} pt | RMSE {data['rmse']:.4f} "
          f"pt | mean signed error {data['mean_signed_error']:+.4f} pt")
    print(f"[R1] observed mean harvest {data['observed_mean']:.4f} pt -> "
          f"MAE is {100 * data['mae'] / data['observed_mean']:.1f}% of it")
    print(f"[R1] straddle: {data['n_over']} of {len(seeds)} over-predicted, "
          f"{data['n_under']} of {len(seeds)} under-predicted -> "
          f"straddles identity: {data['straddles_identity']}")
    print(f"[R1] Pearson r(predicted, observed) = {data['pearson_r']:+.4f}")
    print("[R1] in-sample reference (own sigma, NOT held out): "
          + " ".join(f"{v:.4f}" for v in data["predicted_own"])
          + f" -> MAE {data['mae_own']:.4f} pt")
    print(f"[R1] full-sample aggregate for contrast: sigma_bar(all) "
          f"{data['sigma_bar_all']:.6f} -> c_6*sigma_bar "
          f"{data['aggregate_prediction']:.4f} pt vs observed mean "
          f"{data['observed_mean']:.4f} pt "
          f"({100 * abs(data['aggregate_prediction'] - data['observed_mean']) / data['observed_mean']:.1f}% "
          "deviation; this is the paper's in-sample 4.01 vs 4.12)")
    print(f"[R1] wrote {OUTDIR / 'fig_r1_crossseed.pdf'} and .png")


if __name__ == "__main__":
    main()
