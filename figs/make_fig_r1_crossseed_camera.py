#!/usr/bin/env python3
"""Camera-ready rendering of the leave-one-seed-out figure (Figure 4A).

figs/make_fig_r1_crossseed.py is hash-recorded in
revision_results/statistics.json, so it stays byte-identical. This wrapper
calls its loader and build_figure() unchanged and then edits only two pieces
of layout:

1. the s4 seed label moves from above its marker to its right, so it no
   longer collides with the s2 label printed below the s2 marker;
2. the side-panel statistics are re-set with one indentation level, a true
   minus sign and bold black headings. Every number comes from the same data
   dict as before.

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


def stats_lines(data: dict) -> list[tuple[str, str]]:
    """(kind, text) rows: 'head' = bold black heading, 'item' = indented entry,
    'note' = unindented entry, '' = blank spacer row."""
    pct = 100 * data["mae"] / data["observed_mean"]
    return [
        ("head", "held out (LOSO)"),
        ("item", f"MAE {data['mae']:.2f} pt"),
        ("item", f"= {pct:.0f}% of mean"),
        ("item", f"harvest {data['observed_mean']:.2f} pt"),
        ("item", f"bias {_signed(data['mean_signed_error'])} pt"),
        ("item", f"{data['n_over']} over, {data['n_under']} under"),
        ("item", f"$r$ = {_signed(data['pearson_r'])}"),
        ("", ""),
        ("head", "in sample (own $\\sigma$)"),
        ("item", f"MAE {data['mae_own']:.2f} pt"),
        ("", ""),
        ("note", f"$c_6$ = {data['c_m']:.4f} exact"),
    ]


def draw_stats(ax_txt, block, data: dict) -> None:
    """Replace the frozen single text block with one text per row, so the
    headings can be bold black while the entries keep the original style."""
    fontsize = block.get_fontsize()
    color = block.get_color()
    block.remove()
    step = fontsize * 1.32  # the frozen block's linespacing
    for row, (kind, text) in enumerate(stats_lines(data)):
        if not kind:
            continue
        head = kind == "head"
        ax_txt.annotate(
            text, xy=(0.0, 0.98), xycoords="axes fraction",
            xytext=(fontsize * 0.9 if kind == "item" else 0.0, -row * step),
            textcoords="offset points", ha="left", va="top",
            fontsize=fontsize, color="black" if head else color,
            fontweight="bold" if head else "normal",
        )


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
    draw_stats(ax_txt, blocks[0], data)
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
