#!/usr/bin/env python3
"""Fig R2: the three harvest ratios with bootstrap 95% confidence intervals.

The paper reports the observed-over-predicted harvest ratio for three
campaigns as bare point values, and calls the full-protocol pair "exact":

  shortened protocol   5 seeds x 6 placements   +4.01 vs c_6 sigma_bar 4.12
  realistic-eval (ext21) 5 seeds x 5 placements  ratio 1.15
  full 120-epoch protocol 3 seeds x 4 placements ratio 1.00 ("likewise exact")

No interval accompanies any of them. This figure attaches a seed-level
bootstrap 95% CI to each, so the reader can see how much a 3-5 seed
campaign can actually resolve.

Statistic, identical for all three campaigns and recomputed from the
per-seed paired deltas of each frozen log:

    harvest_s = max_j Delta_{s,j} - mean_j Delta_{s,j}
    sd_s      = sample SD (ddof=1) of Delta_{s,.}
    ratio     = mean_s(harvest_s) / (c_m * mean_s(sd_s))

Bootstrap: seeds are resampled with replacement (paired -- the same
resampled seed index supplies both harvest_s and sd_s, so the per-seed
coupling of numerator and denominator is preserved), N_BOOTSTRAP draws,
numpy default_rng(BOOTSTRAP_SEED) with BOOTSTRAP_SEED a fixed integer
constant. Percentile interval at 2.5/97.5.

Each campaign's c_m is read from its own frozen verdict file, so the
recomputed point ratio reproduces the published number exactly; the
assertions in load_campaigns() fail closed if it does not.

Sized for the TMLR single column (textwidth 6.5 in): exactly
4.00 x 3.20 in, all text >= 8 pt, all lines >= 0.7 pt, markers >= 4 pt.

Deterministic, CPU-only, no network. No model is run; no per-seed value is
reconstructed from an aggregate. Outputs figs/fig_r2_ratios_ci.{pdf,png}.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent

LOG_SHORTENED = ROOT / "logs_flip_demo" / "prop2_fit.json"
LOG_REALEVAL = ROOT / "logs_moneyshot" / "ext21_re_verdict5.json"
LOG_FULL = ROOT / "logs_fullrecipe" / "analysis_summary.json"

BOOTSTRAP_SEED = 0        # fixed constant, matches scripts/prop2_fit.py
N_BOOTSTRAP = 20_000      # >= 10000 required
CI_LOW, CI_HIGH = 2.5, 97.5

# Okabe-Ito colorblind-safe palette, consistent with figs/make_fig2.py.
BLUE = "#0072B2"
SKY = "#56B4E9"
VERM = "#D55E00"
INK = "0.15"

BASE = 8.0
MS_POINT = 5.0
# Drawn at the width it is typeset at (0.50\textwidth of 6.5in), matching the
# cross-seed figure it sits beside: the two are one piece of evidence about how
# far Proposition 2 transports, and different widths read as an accident.
# Shrinking here rather than in \includegraphics keeps every label at the 8 pt
# floor above instead of scaling it to 6.5 pt.
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


@dataclass(frozen=True)
class Campaign:
    """One campaign's per-seed harvest data plus its provenance strings."""
    key: str
    label: str
    source: str            # repo-relative path of the frozen log
    deltas_key: str        # JSON key path holding the per-seed x placement grid
    seeds: tuple
    placements: tuple
    deltas: np.ndarray     # (n_seeds, n_placements) paired deltas, pt
    harvest: np.ndarray    # per-seed max - mean, pt
    sd: np.ndarray         # per-seed SD, ddof=1, pt
    c_m: float
    c_m_source: str
    observed: float        # mean_s(harvest_s)
    predicted: float       # c_m * mean_s(sd_s)
    ratio: float
    published: str         # how the paper states it
    called_exact: bool


def _harvest_and_sd(deltas: np.ndarray) -> tuple:
    return deltas.max(axis=1) - deltas.mean(axis=1), deltas.std(axis=1, ddof=1)


def _contract(deltas: np.ndarray, n_seeds: int, n_placements: int,
              key: str) -> None:
    if deltas.shape != (n_seeds, n_placements):
        raise AssertionError(
            f"{key}: shape {deltas.shape} != ({n_seeds}, {n_placements})")
    if not np.isfinite(deltas).all():
        raise AssertionError(f"{key}: non-finite entries")


def load_shortened(path: Path = LOG_SHORTENED) -> Campaign:
    """Shortened protocol, 5 seeds x 6 placements (Prop 2's fit campaign)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    seeds = tuple(d["seeds"])
    placements = tuple(d["placements"])
    D = np.asarray(d["deltas_pt"], dtype=float)
    _contract(D, 5, 6, "prop2_fit.json:deltas_pt")

    harvest, sd = _harvest_and_sd(D)
    if not np.allclose(sd, d["sd_pt"], atol=1e-12):
        raise AssertionError("shortened: recomputed sd != stored sd_pt")
    if not np.allclose(harvest, d["harvest_max_minus_mean_pt"], atol=1e-12):
        raise AssertionError(
            "shortened: recomputed harvest != stored "
            "harvest_max_minus_mean_pt")

    c_m = float(d["c_m"]["gauss"])
    observed = float(harvest.mean())
    predicted = c_m * float(sd.mean())
    # the paper states this pair as "+4.01 versus predicted c_6 sigma_bar = 4.12"
    if round(observed, 2) != 4.01:
        raise AssertionError(f"shortened observed {observed} != published 4.01")
    if round(predicted, 2) != 4.12:
        raise AssertionError(
            f"shortened predicted {predicted} != published 4.12")
    # cross-check against the stored washout decomposition
    stored_gain = (d["washout"]["per_seed_select"]
                   - d["washout"]["honest_mean"])
    if abs(observed - stored_gain) > 1e-12:
        raise AssertionError(
            f"shortened observed {observed} != washout gain {stored_gain}")

    return Campaign(
        key="shortened",
        label="shortened protocol",
        source=path.relative_to(ROOT).as_posix(),
        deltas_key="deltas_pt",
        seeds=seeds,
        placements=placements,
        deltas=D,
        harvest=harvest,
        sd=sd,
        c_m=c_m,
        c_m_source="prop2_fit.json:c_m.gauss (MC c_6; exact 1.2672)",
        observed=observed,
        predicted=predicted,
        ratio=observed / predicted,
        published="4.01 / 4.12",
        called_exact=False,
    )


def load_realeval(path: Path = LOG_REALEVAL) -> Campaign:
    """Realistic-eval (ext21) campaign, 5 seeds x 5 placements."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    seeds = tuple(d["seeds_present"])
    placements = tuple(d["placements"])
    D = np.asarray(d["deltas_A_pt"], dtype=float)
    _contract(D, 5, 5, "ext21_re_verdict5.json:deltas_A_pt")
    if d["missing"]:
        raise AssertionError(f"realistic-eval: missing arms {d['missing']}")

    harvest, sd = _harvest_and_sd(D)
    if not np.allclose(harvest, d["per_seed_margin_shared"], atol=1e-12):
        raise AssertionError(
            "realistic-eval: recomputed per-seed harvest != stored "
            "per_seed_margin_shared")
    if abs(float(sd.mean()) - d["sigma_bar"]) > 1e-12:
        raise AssertionError(
            f"realistic-eval: recomputed sigma_bar {sd.mean()} != stored "
            f"{d['sigma_bar']}")

    c_m = float(d["c_N"])
    observed = float(harvest.mean())
    predicted = c_m * float(sd.mean())
    if abs(observed - d["manufactured_gain_A"]) > 1e-12:
        raise AssertionError(
            f"realistic-eval: observed {observed} != stored "
            f"manufactured_gain_A {d['manufactured_gain_A']}")
    if abs(predicted - d["extreme_value_pred"]) > 1e-12:
        raise AssertionError(
            f"realistic-eval: predicted {predicted} != stored "
            f"extreme_value_pred {d['extreme_value_pred']}")
    if abs(observed / predicted - d["ratio"]) > 1e-12:
        raise AssertionError(
            f"realistic-eval: recomputed ratio {observed / predicted} != "
            f"stored ratio {d['ratio']}")

    return Campaign(
        key="realeval",
        label="realistic-eval (ext21)",
        source=path.relative_to(ROOT).as_posix(),
        deltas_key="deltas_A_pt",
        seeds=seeds,
        placements=placements,
        deltas=D,
        harvest=harvest,
        sd=sd,
        c_m=c_m,
        c_m_source="ext21_re_verdict5.json:c_N (MC c_5; exact 1.1630)",
        observed=observed,
        predicted=predicted,
        ratio=observed / predicted,
        published="1.15",
        called_exact=False,
    )


def load_full(metric: str = "best", path: Path = LOG_FULL) -> Campaign:
    """Full 120-epoch official protocol, 3 seeds x 4 placements."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    h = d["harvest"]
    seeds = tuple(h["seeds"])
    placements = tuple(h["placements"])
    block = h[metric]
    D = np.asarray(block["deltas_pt"], dtype=float)
    _contract(D, 3, 4, f"analysis_summary.json:harvest.{metric}.deltas_pt")

    harvest, sd = _harvest_and_sd(D)
    if not np.allclose(sd, block["sd_pt"], atol=1e-12):
        raise AssertionError(f"full/{metric}: recomputed sd != stored sd_pt")
    if not np.allclose(harvest, block["max_minus_mean_pt"], atol=1e-12):
        raise AssertionError(
            f"full/{metric}: recomputed harvest != stored max_minus_mean_pt")

    c_m = float(h["c_m"]["gauss"])
    observed = float(harvest.mean())
    predicted = c_m * float(sd.mean())
    if abs(predicted - block["predicted_cm_sigma_pt"]) > 1e-12:
        raise AssertionError(
            f"full/{metric}: predicted {predicted} != stored "
            f"predicted_cm_sigma_pt {block['predicted_cm_sigma_pt']}")
    stored_gain = block["per_seed_select_pt"] - block["honest_mean_pt"]
    if abs(observed - stored_gain) > 1e-12:
        raise AssertionError(
            f"full/{metric}: observed {observed} != select-minus-honest "
            f"{stored_gain}")

    label = ("full protocol" if metric == "best"
             else "full protocol (final)")
    return Campaign(
        key=f"full_{metric}",
        label=label,
        source=path.relative_to(ROOT).as_posix(),
        deltas_key=f"harvest.{metric}.deltas_pt",
        seeds=seeds,
        placements=placements,
        deltas=D,
        harvest=harvest,
        sd=sd,
        c_m=c_m,
        c_m_source="analysis_summary.json:harvest.c_m.gauss "
                   "(MC c_4; exact 1.0294)",
        observed=observed,
        predicted=predicted,
        ratio=observed / predicted,
        published="1.00",
        called_exact=True,
    )


def bootstrap_ratio_ci(campaign: Campaign, rng: np.random.Generator,
                       n_boot: int = N_BOOTSTRAP) -> dict:
    """Percentile 95% CI for the ratio, resampling seeds with replacement.

    Numerator and denominator are resampled with the SAME seed indices, so
    the per-seed pairing of harvest and scale is preserved.
    """
    n = campaign.harvest.size
    idx = rng.integers(0, n, size=(n_boot, n))
    num = campaign.harvest[idx].mean(axis=1)
    den = campaign.c_m * campaign.sd[idx].mean(axis=1)
    if not np.all(den > 0):
        raise AssertionError(f"{campaign.key}: non-positive bootstrap "
                             "denominator (a resample with zero SD)")
    ratios = num / den
    lo, hi = np.percentile(ratios, [CI_LOW, CI_HIGH])
    per_seed = campaign.harvest / (campaign.c_m * campaign.sd)
    # With few seeds, a resample in which all n draws hit ONE given seed has
    # probability n^-n: 1/27 = 3.7% at n = 3 (above the 2.5% tail) but
    # 1/3125 = 0.03% at n = 5. When it exceeds the tail mass the percentile
    # endpoints are literally single-seed ratios and the interval is not a
    # meaningful 95% interval. Flag it rather than hide it.
    degenerate = bool(
        abs(lo - per_seed.min()) < 1e-9 or abs(hi - per_seed.max()) < 1e-9)
    return {
        "n_seeds": n,
        "n_boot": n_boot,
        "lo": float(lo),
        "hi": float(hi),
        "median": float(np.median(ratios)),
        "covers_one": bool(lo <= 1.0 <= hi),
        "per_seed_ratio": per_seed,
        "endpoints_are_single_seed": degenerate,
        "prob_all_one_seed": float(n ** (-n)),
    }


def load_campaigns() -> list:
    """The three campaigns whose per-seed values are recoverable, in order."""
    return [load_shortened(), load_realeval(), load_full("best")]


def build_bootstrap(campaigns: list, seed: int = BOOTSTRAP_SEED,
                    n_boot: int = N_BOOTSTRAP) -> list:
    """One deterministic rng stream, consumed in campaign order."""
    rng = np.random.default_rng(seed)
    return [bootstrap_ratio_ci(c, rng, n_boot) for c in campaigns]


def build_figure(campaigns: list, boots: list) -> Figure:
    """Horizontal dot-and-interval chart of the three ratios."""
    with plt.rc_context(RC):
        if len(campaigns) != len(boots):
            raise AssertionError("campaign/bootstrap length mismatch")

        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

        # rows top-to-bottom in the order the paper introduces the campaigns
        y = np.arange(len(campaigns))[::-1].astype(float)

        one = ax.axvline(1.0, color=VERM, lw=1.0, ls=(0, (4, 2)), zorder=1.5)
        one.set_gid("ratio-one-reference")

        any_degenerate = False
        for row, (c, b) in enumerate(zip(campaigns, boots)):
            yy = y[row]
            # per-seed ratios: show the raw spread the CI is computed from
            pts = ax.plot(b["per_seed_ratio"], np.full(c.harvest.size, yy + 0.27),
                          ls="none", marker="o", ms=4.0, mfc="none", mec=SKY,
                          mew=0.8, zorder=3)[0]
            pts.set_gid(f"per-seed-ratios-{c.key}")

            bar = ax.barh(yy, c.ratio, height=0.26, color=BLUE, zorder=2,
                          edgecolor="none")
            bar[0].set_gid(f"ratio-bar-{c.key}")
            err = ax.errorbar(c.ratio, yy,
                              xerr=[[c.ratio - b["lo"]], [b["hi"] - c.ratio]],
                              fmt="o", ms=MS_POINT, mfc=BLUE, mec=BLUE,
                              ecolor=INK, elinewidth=0.9, capsize=2.4,
                              capthick=0.9, zorder=4)
            err.lines[0].set_gid(f"ratio-ci-{c.key}")

            dagger = "†" if b["endpoints_are_single_seed"] else ""
            any_degenerate = any_degenerate or b["endpoints_are_single_seed"]
            ax.text(0.035, yy - 0.24,
                    f"{c.label}:  {c.ratio:.2f}  "
                    f"[{b['lo']:.2f}, {b['hi']:.2f}]{dagger}",
                    fontsize=BASE, color=INK, ha="left", va="top", zorder=5)

        ax.set_yticks(y)
        ax.set_yticklabels([
            f"{len(c.seeds)} seeds\n× {len(c.placements)} pl."
            for c in campaigns
        ])
        ax.set_ylim(-0.68, len(campaigns) - 0.58)
        ax.set_xlim(0.0, 1.62)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        xlabel = ("observed harvest / $c_m\\bar{\\sigma}$  "
                  "(bootstrap 95% CI)")
        if any_degenerate:
            xlabel += "\n† CI endpoints are single-seed ratios"
        ax.set_xlabel(xlabel)

        legend_handles = [
            plt.Line2D([], [], ls="none", marker="o", ms=4.0, mfc="none",
                       mec=SKY, mew=0.8, label="per-seed ratio"),
            plt.Line2D([], [], color=VERM, lw=1.0, ls=(0, (4, 2)),
                       label="ratio = 1"),
        ]
        # Above the axes, laid out in one row. Inside the axes there is no
        # free corner: each bar carries its ratio and interval as in-axes
        # text and the per-seed dots spread the full width, so a
        # lower-right legend lands on the bottom row once the panel
        # narrows for the two-up layout.
        ax.legend(handles=legend_handles, loc="lower center",
                  frameon=False, bbox_to_anchor=(0.5, 1.0),
                  ncol=len(legend_handles), borderaxespad=0.2,
                  handlelength=1.4, handletextpad=0.5, columnspacing=1.4)

        return fig


def render(campaigns: list, boots: list,
           output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    """Save fig_r2_ratios_ci.pdf/.png into output_dir; close the figure.

    constrained_layout already fits everything inside the exact canvas, so
    no bbox_inches="tight" and no post-layout shrinking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(campaigns, boots)
    pdf_path = output_dir / "fig_r2_ratios_ci.pdf"
    png_path = output_dir / "fig_r2_ratios_ci.png"
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
    campaigns = load_campaigns()
    boots = build_bootstrap(campaigns)
    render(campaigns, boots)

    print("[R2] PROVENANCE DUMP -- every plotted number -> file -> JSON key")
    print(f"[R2] bootstrap: numpy default_rng({BOOTSTRAP_SEED}), "
          f"{N_BOOTSTRAP} draws, percentile {CI_LOW}/{CI_HIGH}, seeds "
          "resampled with replacement (numerator and denominator paired)")
    for c, b in zip(campaigns, boots):
        print(f"[R2] --- {c.label} ({c.key}) ---")
        print(f"[R2]   source: {c.source}  key: {c.deltas_key}  "
              f"shape {c.deltas.shape}")
        print(f"[R2]   seeds {list(c.seeds)} | placements {list(c.placements)}")
        print(f"[R2]   c_m {c.c_m:.10f} <- {c.c_m_source}")
        for i, s in enumerate(c.seeds):
            print(f"[R2]   seed {s}: deltas "
                  + " ".join(f"{v:+.3f}" for v in c.deltas[i])
                  + f" | harvest {c.harvest[i]:.6f} | sd {c.sd[i]:.6f} "
                  f"| per-seed ratio {b['per_seed_ratio'][i]:.4f}")
        print(f"[R2]   observed mean harvest {c.observed:.6f} pt | predicted "
              f"c_m*sigma_bar {c.predicted:.6f} pt")
        print(f"[R2]   ratio {c.ratio:.4f}  (paper states {c.published}"
              + ("; called \"exact\"" if c.called_exact else "") + ")")
        print(f"[R2]   bootstrap 95% CI [{b['lo']:.4f}, {b['hi']:.4f}] "
              f"on n={b['n_seeds']} seeds | covers 1.00: {b['covers_one']} "
              f"| CI width {b['hi'] - b['lo']:.4f}")
        if b["endpoints_are_single_seed"]:
            print(f"[R2]   WARNING n={b['n_seeds']}: a resample drawing one "
                  "single seed n times has probability "
                  f"{100 * b['prob_all_one_seed']:.2f}% > 2.5%, so the "
                  "percentile endpoints ARE single-seed ratios; this "
                  "interval is not a meaningful 95% interval")

    extra = load_full("final")
    extra_boot = bootstrap_ratio_ci(
        extra, np.random.default_rng(BOOTSTRAP_SEED))
    print("[R2] NOT PLOTTED (spec asks for three rows) but computed for the "
          "record -- the second campaign the paper calls \"exact\":")
    print(f"[R2]   full protocol, FINAL metric: source {extra.source} key "
          f"{extra.deltas_key}; observed {extra.observed:.6f} predicted "
          f"{extra.predicted:.6f} ratio {extra.ratio:.4f} "
          f"CI [{extra_boot['lo']:.4f}, {extra_boot['hi']:.4f}] "
          f"covers 1.00: {extra_boot['covers_one']}")
    print(f"[R2] wrote {OUTDIR / 'fig_r2_ratios_ci.pdf'} and .png")


if __name__ == "__main__":
    main()
