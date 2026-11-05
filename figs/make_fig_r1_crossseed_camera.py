#!/usr/bin/env python3
"""Camera-ready rendering of the leave-one-seed-out figure (Figure 4A).

figs/make_fig_r1_crossseed.py is hash-recorded in
revision_results/statistics.json, so it stays byte-identical. This wrapper
calls its loader and build_figure() unchanged and then edits only two pieces
of layout:

1. the s4 seed label moves from above its marker to its right, so it no
   longer collides with the s2 label printed below the s2 marker;
2. the side-panel statistics are re-set with one indentation level and a true
   minus sign. Every number comes from the same data dict as before.

Usage: python figs/make_fig_r1_crossseed_camera.py [--output-dir DIR]
(default: paper_tmlr/figs, the copy the manuscript includes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGS))
import make_fig_r1_crossseed as frozen  # noqa: E402

DEFAULT_OUT = FIGS.parent / "paper_tmlr" / "figs"
MINUS = "−"


def _signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", MINUS)


def stats_text(data: dict) -> str:
    pct = 100 * data["mae"] / data["observed_mean"]
    return "\n".join([
        "held out (LOSO)",
        f"  MAE {data['mae']:.2f} pt",
        f"  = {pct:.0f}% of mean",
        f"  harvest {data['observed_mean']:.2f} pt",
        f"  bias {_signed(data['mean_signed_error'])} pt",
        f"  {data['n_over']} over, {data['n_under']} under",
        f"  $r$ = {_signed(data['pearson_r'])}",
        "",
        "in sample (own $\\sigma$)",
        f"  MAE {data['mae_own']:.2f} pt",
        "",
        f"$c_6$ = {data['c_m']:.4f} exact",
    ])


def build_camera_figure(data: dict):
    fig = frozen.build_figure(data)
    ax, ax_txt = fig.axes[0], fig.axes[1]
    seeds = list(data["seeds"])

    moved = 0
    for text in ax.texts:
        if text.get_text() == "s4":
            i = seeds.index(4)
            text.set_position((data["predicted"][i] + 0.34, data["observed"][i]))
            text.set_horizontalalignment("left")
            text.set_verticalalignment("center")
            moved += 1
    if moved != 1:
        raise RuntimeError("expected exactly one s4 label in the frozen figure")

    blocks = [t for t in ax_txt.texts if t.get_text().startswith("held out (LOSO)")]
    if len(blocks) != 1:
        raise RuntimeError("expected exactly one statistics block in the frozen figure")
    blocks[0].set_text(stats_text(data))
    return fig


def render(data: dict, output_dir: Path = DEFAULT_OUT) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_camera_figure(data)
    pdf_path = output_dir / "fig_r1_crossseed.pdf"
    try:
        with plt.rc_context(frozen.RC):
            fig.savefig(pdf_path)
    finally:
        plt.close(fig)
    return pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    path = render(frozen.load_crossseed_data(), args.output_dir)
    print(f"[R1 camera] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
