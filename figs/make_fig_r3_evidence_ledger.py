#!/usr/bin/env python3
"""Fig R3: verification-resolution ledger for the paper's bitwise claims.

Three claim families in the paper are phrased as universal quantifiers but
are individually verified on a subset of the cells or arms they quantify
over. This figure is the ledger: one row per claim family, the bar split
by the resolution at which each member was actually checked.

  1. Historical guard registry: 18 raw-selected metric-zero cells among
     19 metric-zero outcomes (the remaining outcome is an EMA metric tie).
     The old eight-bitwise count was inferred from table text. Recovered
     per-cell NLP records contain selection/metric logs, not independent
     weight hashes or rerun floors. Zero cells are directly hash certified
     by this registry. Metric observations remain reported as such.

  2. Full 120-epoch protocol, "returns every arm to bitwise identity
     with its never-probed baseline" (paper/sections/f2.tex). The
     campaign has 12 placement x seed arms (4 placements x 3 seeds, all
     12 confirmed to desynchronize). The fork-rng cure is registered at
     one placement only: logs_fullrecipe/analysis_summary.json cure.rows
     holds 3 rows (probe_step 23520, seeds 1-3).

  3. Realistic-eval (ext21) protocol, "returns each isolated arm ... to
     bitwise identity with the never-evaluated baseline"
     (paper/sections/f2.tex). The campaign is 5 seeds x 5 placements =
     25 arms; the isolated arm exists for 2 of them (seeds 1 and 2 at
     placement 1400). Note the paper's own sentence parenthesises the
     scope ("two seeds, one placement"), so this row documents a
     narrower claim than row 2.

Counts for rows 2 and 3 are not merely read off the prose: the
"verified bitwise" segment is confirmed here by comparing the frozen
SHA-256 strings of each isolated arm against its never-probed baseline
run, and the script fails closed if a hash pair does not match or if a
count disagrees with the corresponding verdict file.

Sized for the TMLR text width (6.5 in): exactly 6.50 x 2.60 in, all text
>= 8 pt, all lines >= 0.7 pt.

Deterministic, CPU-only, no network. No model is run. Rows whose split
cannot be determined from the text are omitted, not guessed; see
OMITTED_ROWS and the provenance dump printed by main().
Outputs figs/fig_r3_evidence_ledger.{pdf,png}.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent

LOG_FULL = ROOT / "logs_fullrecipe" / "analysis_summary.json"
LOG_REALEVAL_VERDICT = ROOT / "logs_moneyshot" / "ext21_re_verdict.json"

# Okabe-Ito colorblind-safe palette, consistent with figs/make_fig2.py.
BLUE = "#0072B2"      # verified bitwise / SHA-256
SKY = "#56B4E9"       # verified metric-exact
GRAY = "0.82"         # inferred / not individually verified
INK = "0.15"

BASE = 8.0
FIGSIZE = (6.5, 2.6)

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

SEG_LABELS = (
    "model + meta-net hashes",
    "recorded metric equality",
    "not individually checked",
)
SEG_COLORS = (BLUE, SKY, GRAY)
SEG_TEXT = ("white", INK, INK)

# Claim families whose split could NOT be determined from the stated text,
# and are therefore omitted from the figure rather than guessed.
OMITTED_ROWS = (
    ("Shortened-protocol fork-rng cure (MW-Net demo)",
     "paper/appendix.tex Table tab:floors states \"the fork_rng-isolated "
     "probe lands bitwise on the never-probed baseline in 5/5 seeds\" -- a "
     "scoped count, not a universal quantifier, and the text never states "
     "how many arms the family contains, so the denominator of the row is "
     "undetermined. (The 5 verified pairs are frozen in "
     "logs_flip_demo/cure_seed{1..5}.json vs floor_seed{1..5}_a.json.)"),
    ("HuggingFace probe experiments, \"16 floor files ... all exactly 0.0\"",
     "paper/appendix.tex Table tab:floors registers 16 floor files at "
     "resolution \"weight-level max|dW|\", but the text states no total "
     "number of arms the 16 are drawn from, so no verified/unverified "
     "split can be computed."),
)


@dataclass(frozen=True)
class LedgerRow:
    """One claim family and the resolution at which its members were checked."""
    key: str
    claim: str              # the paper's own wording, abbreviated
    unit: str               # what is being counted
    total: int
    bitwise: int
    metric_exact: int
    inferred: int
    claim_source: str
    split_source: str
    verified_source: str
    verified_hashes: tuple = field(default=())

    def __post_init__(self):
        parts = self.bitwise + self.metric_exact + self.inferred
        if parts != self.total:
            raise AssertionError(
                f"{self.key}: segments {self.bitwise}+{self.metric_exact}+"
                f"{self.inferred} = {parts} != total {self.total}")
        if min(self.bitwise, self.metric_exact, self.inferred,
               self.total) < 0:
            raise AssertionError(f"{self.key}: negative count")

    @property
    def segments(self) -> tuple:
        return (self.bitwise, self.metric_exact, self.inferred)


def _sha_pair(run_path: Path, base_path: Path) -> tuple:
    """(model_sha, vnet_sha, equal) for an isolated arm vs its baseline.

    Handles both frozen layouts in this repo: top-level hashes
    (logs_fullrecipe) and hashes nested under "result" (logs_moneyshot).
    """
    def read(p: Path) -> tuple:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "state_sha256" in d:
            node = d
        elif "result" in d and "state_sha256" in d["result"]:
            node = d["result"]
        else:
            raise AssertionError(f"{p.name}: no state_sha256 found")
        return str(node["state_sha256"]), str(node["vnet_state_sha256"])

    a_model, a_vnet = read(run_path)
    b_model, b_vnet = read(base_path)
    return a_model, b_model, (a_model == b_model and a_vnet == b_vnet)


def verify_full_protocol_cure(path: Path = LOG_FULL) -> dict:
    """Count the full-protocol arms whose fork-rng cure is SHA-verified."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    h = d["harvest"]
    n_arms = len(h["seeds"]) * len(h["placements"])
    if len(d["displacement"]["rows"]) != n_arms:
        raise AssertionError(
            f"displacement rows {len(d['displacement']['rows'])} != "
            f"{n_arms} placement x seed arms")

    rows = d["cure"]["rows"]
    for r in rows:
        if not (r["sha_equal"] and r["vnet_sha_equal"]):
            raise AssertionError(f"cure row not SHA-equal: {r}")
    steps = sorted({r["probe_step"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    # independent re-verification straight off the frozen run files
    hashes = []
    for seed in seeds:
        step = steps[0]
        iso = path.parent / f"probe_isolated_p{step}_s{seed}.json"
        base = path.parent / f"mwnet_s{seed}.json"
        a, b, equal = _sha_pair(iso, base)
        if not equal:
            raise AssertionError(
                f"full protocol seed {seed}: {iso.name} SHA {a[:12]} != "
                f"{base.name} SHA {b[:12]}")
        hashes.append((f"seed {seed} @p{step}", a[:12]))

    return {
        "n_arms": n_arms,
        "n_verified": len(rows),
        "placements_verified": steps,
        "seeds_verified": seeds,
        "hashes": tuple(hashes),
    }


def verify_realeval_cure(path: Path = LOG_REALEVAL_VERDICT) -> dict:
    """Count the realistic-eval arms whose fork-rng cure is SHA-verified."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not d["iso_sha_equal"]:
        raise AssertionError("ext21_re_verdict.json: iso_sha_equal is false")
    iso_seeds = list(d["seeds_present"])
    margins = d["per_seed_margin_isolated"]
    if len(margins) != len(iso_seeds):
        raise AssertionError(
            f"per_seed_margin_isolated length {len(margins)} != isolated "
            f"seeds {len(iso_seeds)}")
    if any(m != 0.0 for m in margins):
        raise AssertionError(
            f"isolated margins not exactly zero: {margins}")

    # the campaign total comes from the 5-seed verdict, not from this file
    with open(path.parent / "ext21_re_verdict5.json", encoding="utf-8") as f:
        d5 = json.load(f)
    n_arms = len(d5["seeds_present"]) * len(d5["placements"])

    hashes = []
    placement = 1400
    for seed in iso_seeds:
        iso = path.parent / f"ext21_re_seed{seed}_iso_p{placement}.json"
        base = path.parent / f"ext21_re_seed{seed}_rep.json"
        a, b, equal = _sha_pair(iso, base)
        if not equal:
            raise AssertionError(
                f"realistic-eval seed {seed}: {iso.name} SHA {a[:12]} != "
                f"{base.name} SHA {b[:12]}")
        hashes.append((f"seed {seed} @p{placement}", a[:12]))

    return {
        "n_arms": n_arms,
        "n_verified": len(iso_seeds),
        "placements_verified": [placement],
        "seeds_verified": iso_seeds,
        "hashes": tuple(hashes),
    }


def load_ledger() -> list:
    """Build the ledger rows, verifying every count that a log can confirm."""
    full = verify_full_protocol_cure()
    real = verify_realeval_cure()

    rows = [
        # ---- row 1: guard certificate registry -------------------------
        # 18 historical raw-selected metric-zero cells. The old eight
        # direct-hash entries were table-text inference, now withdrawn.
        LedgerRow(
            key="guard_certificate",
            claim="Historical raw-selected zeros",
            unit="guard cells",
            total=18,
            bitwise=0,
            metric_exact=18,
            inferred=0,
            claim_source="Historical Table tab:certreg; 19 metric-zero cells, 18 raw-selected and one EMA tie.",
            split_source="revision_results/evidence_ledger.json and docs/REVISION_EVIDENCE.md: old eight-hash count withdrawn; no independent per-cell weight registry.",
            verified_source="Metric/selection record only; zero independent hash certificates or inferred floors. The nineteenth EMA tie is outside this row.",
        ),
        # ---- row 2: full 120-epoch protocol cure -----------------------
        LedgerRow(
            key="full_protocol_cure",
            claim="Full-protocol isolated comparisons",
            unit="placement × seed arms",
            total=full["n_arms"],
            bitwise=full["n_verified"],
            metric_exact=0,
            inferred=full["n_arms"] - full["n_verified"],
            claim_source=(
                "paper/sections/f2.tex: \"The same draw inside a forked RNG "
                "state returns every arm to bitwise identity with its "
                "never-probed baseline\"; the 12-arm universe is the same "
                "paragraph's \"all twelve placement x seed pairs "
                "desynchronize bitwise\""),
            split_source=(
                "logs_fullrecipe/analysis_summary.json: "
                "harvest.seeds x harvest.placements = "
                f"{full['n_arms']} arms; displacement.rows confirms all "
                f"{full['n_arms']}; cure.rows holds only "
                f"{full['n_verified']} (probe_step "
                f"{full['placements_verified']}, seeds "
                f"{full['seeds_verified']})"),
            verified_source=(
                "re-verified here: probe_isolated_p23520_s{1,2,3}.json "
                "state_sha256 and vnet_state_sha256 equal mwnet_s{1,2,3}"
                ".json"),
            verified_hashes=full["hashes"],
        ),
        # ---- row 3: realistic-eval (ext21) cure ------------------------
        LedgerRow(
            key="realeval_cure",
            claim="Standard-eval isolated comparisons",
            unit="placement × seed arms",
            total=real["n_arms"],
            bitwise=real["n_verified"],
            metric_exact=0,
            inferred=real["n_arms"] - real["n_verified"],
            claim_source=(
                "paper/sections/f2.tex: \"returns each isolated arm (two "
                "seeds, one placement) to bitwise identity with the "
                "never-evaluated baseline\"; the 25-arm universe is the "
                "same paragraph's \"Across 5 seeds x 5 placements\""),
            split_source=(
                "logs_moneyshot/ext21_re_verdict5.json: seeds_present x "
                f"placements = {real['n_arms']} arms; "
                "logs_moneyshot/ext21_re_verdict.json: seeds_present "
                f"{real['seeds_verified']}, iso_sha_equal true, "
                "per_seed_margin_isolated length "
                f"{real['n_verified']}"),
            verified_source=(
                "re-verified here: ext21_re_seed{1,2}_iso_p1400.json "
                "state_sha256 and vnet_state_sha256 equal "
                "ext21_re_seed{1,2}_rep.json"),
            verified_hashes=real["hashes"],
        ),
    ]
    return rows


def build_figure(rows: list) -> Figure:
    """Horizontal stacked-bar ledger, one row per claim family."""
    with plt.rc_context(RC):
        if not rows:
            raise ValueError("no ledger rows to draw")

        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

        y = np.arange(len(rows))[::-1].astype(float)
        x_max = max(r.total for r in rows)
        x_limit = x_max * 1.34

        for row, r in enumerate(rows):
            yy = y[row]
            left = 0.0
            for seg, (count, color, tcolor) in enumerate(
                    zip(r.segments, SEG_COLORS, SEG_TEXT)):
                if count == 0:
                    left += count
                    continue
                bar = ax.barh(yy, count, left=left, height=0.52, color=color,
                              edgecolor="white", linewidth=0.7, zorder=2)
                bar[0].set_gid(f"{r.key}-segment-{seg}")
                ax.text(left + count / 2.0, yy, str(count), fontsize=BASE,
                        color=tcolor, ha="center", va="center", zorder=4)
                left += count
            if abs(left - r.total) > 1e-9:
                raise AssertionError(f"{r.key}: drawn width {left} != total "
                                     f"{r.total}")

            ax.text(r.total + 0.45 * x_max / 25.0, yy,
                    f"{r.bitwise}/{r.total} hashes", fontsize=BASE, color=INK,
                    ha="left", va="center", zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.claim}\n{r.total} {r.unit}" for r in rows])
        ax.set_ylim(-0.60, len(rows) - 0.40)
        ax.set_xlim(0, x_limit)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_xlabel("historical cells / placement arms (not full-state certificates)")

        # Figure-level legend outside the axes: it needs the full 6.5 in of
        # figure width, which an axes-anchored legend does not have.
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none",
                                 label=lab)
                   for c, lab in zip(SEG_COLORS, SEG_LABELS)]
        fig.legend(handles=handles, loc="outside upper center", ncol=3,
                   frameon=False, borderaxespad=0.15, handlelength=1.1,
                   handleheight=0.9, handletextpad=0.4, columnspacing=1.0)

        return fig


def render(rows: list, output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    """Save fig_r3_evidence_ledger.pdf/.png into output_dir; close the figure.

    constrained_layout already fits everything inside the exact canvas, so
    no bbox_inches="tight" and no post-layout shrinking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(rows)
    pdf_path = output_dir / "fig_r3_evidence_ledger.pdf"
    png_path = output_dir / "fig_r3_evidence_ledger.png"
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
    rows = load_ledger()
    render(rows)

    print("[R3] PROVENANCE DUMP -- every plotted count -> stated source")
    for r in rows:
        print(f"[R3] --- {r.key} ---")
        print(f"[R3]   claim: {r.claim}")
        print(f"[R3]   universe: {r.total} {r.unit}")
        print(f"[R3]   bitwise/SHA-256 verified: {r.bitwise} | metric-exact: "
              f"{r.metric_exact} | inferred or unverified: {r.inferred}")
        print(f"[R3]   claim source: {r.claim_source}")
        print(f"[R3]   split source: {r.split_source}")
        print(f"[R3]   verified how: {r.verified_source}")
        for tag, sha in r.verified_hashes:
            print(f"[R3]     SHA-256 match {tag}: {sha}...")
    print("[R3] OMITTED ROWS (split not determinable from the stated text):")
    for name, why in OMITTED_ROWS:
        print(f"[R3]   {name}: {why}")
    print(f"[R3] wrote {OUTDIR / 'fig_r3_evidence_ledger.pdf'} and .png")


if __name__ == "__main__":
    main()
