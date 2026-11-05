#!/usr/bin/env python3
"""Divergence-onset heatmap from the frozen causal-trace v3 figure data.

One question, answered visually: WHERE does the same-seed pair start to
diverge?  Panel A shows the shared-stream arm --- every seed-layer cell is
exactly zero at the pre-probe checkpoint (step 1399) and nonzero from the
probe step (1400) onward.  Panel B shows the isolated arm --- exactly zero
everywhere through step 5400.  Checkpoints are unevenly spaced and drawn
categorically; the caption states this.

Data loading is delegated to figs.make_fig_causal_trace.load_figure_data
(fail-closed, contract-tested).  The first-effective-consumer annotation is
cross-checked against the frozen verdict JSON before it is drawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from figs.make_fig_causal_trace import (
    FIGURE_DATA,
    LAYER_ORDER,
    PRIMARY_SEEDS,
    TRACE_STEPS,
    load_figure_data,
)

ROOT = Path(__file__).resolve().parents[1]
VERDICT = ROOT / "logs_causal_trace" / "causal_trace_verdict.json"
OUTDIR = Path(__file__).resolve().parent

# Drawn at the width it is actually typeset at (0.76\textwidth of 6.5in), so
# every label keeps its declared point size instead of being scaled down to
# ~5.6pt by \includegraphics. Height buys room for the 35 layer rows.
FIGSIZE = (4.03, 3.30)
PROBE_STEP = 1400
FIRST_CONSUMER_LABEL = "meta-batch indices"
ZERO_COLOR = "#d9d9d9"
LAYER_LABELS = {
    "resnet_stem": "stem",
    "resnet_stage1": "stage1",
    "resnet_stage2": "stage2",
    "resnet_stage3": "stage3",
    "resnet_head": "head",
    "vnet_layer1": "vnet1",
    "vnet_layer2": "vnet2",
}


def load_first_effective_event(path: str | Path = VERDICT) -> str:
    """Return the common first effective event, requiring seed unanimity."""
    with open(path, "r", encoding="utf-8") as handle:
        verdict = json.load(handle)
    events = verdict.get("first_effective_event")
    if not isinstance(events, dict) or not events:
        raise ValueError("verdict lacks first_effective_event")
    expected = {f"seed{s}" for s in PRIMARY_SEEDS}
    if set(events) != expected:
        raise ValueError("first_effective_event does not cover exactly the frozen seeds")
    values = set(events.values())
    if values != {"meta_indices"}:
        raise ValueError(f"first_effective_event is not unanimous meta_indices: {values}")
    return "meta_indices"


def build_matrix(relative_l2: np.ndarray) -> np.ndarray:
    """Stack (seed, layer, checkpoint) into a (seed*layer, checkpoint) grid."""
    n_seeds, n_layers, n_checkpoints = relative_l2.shape
    return relative_l2.reshape(n_seeds * n_layers, n_checkpoints)


def build_figure(data: dict[str, Any], first_consumer: str) -> "object":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    if first_consumer != "meta_indices":
        raise ValueError("refusing to draw an unverified first-consumer annotation")

    shared = build_matrix(data["shared_relative_l2"])
    isolated = build_matrix(data["isolated_relative_l2"])
    n_rows, n_cols = shared.shape
    n_layers = len(LAYER_ORDER)

    with plt.rc_context(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    ):
        # Panel B carries one bit (everything is zero), so it gets a strip
        # rather than an equal half: the space belongs to Panel A, where the
        # layer rows have to be individually resolvable.
        figure, axes = plt.subplots(
            2, 1, figsize=FIGSIZE, sharex=True,
            gridspec_kw={"height_ratios": [4.2, 1.0]},
        )
        # left was 0.150, which clipped the y label: "stem" printed as "em".
        # top was 0.800, which left the bold panel titles sitting on the mesh.
        figure.subplots_adjust(
            left=0.185, right=0.790, top=0.775, bottom=0.180, hspace=0.34
        )

        positive = shared[shared > 0]
        vmin = float(np.floor(np.log10(positive.min())))
        vmax = float(np.ceil(np.log10(positive.max())))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        base_cmap = plt.get_cmap("viridis")

        mesh = None
        for axis, grid, title in (
            (
                axes[0],
                shared,
                "A   Shared stream: zero before the probe",
            ),
            (
                axes[1],
                isolated,
                "B   Isolated probe: zero everywhere",
            ),
        ):
            log_grid = np.full(grid.shape, np.nan)
            nonzero = grid > 0
            log_grid[nonzero] = np.log10(grid[nonzero])
            masked = np.ma.masked_invalid(log_grid)
            local_cmap = base_cmap.copy()
            local_cmap.set_bad(ZERO_COLOR)
            mesh = axis.pcolormesh(
                np.arange(n_cols + 1),
                np.arange(n_rows + 1),
                masked,
                cmap=local_cmap,
                norm=norm,
                rasterized=False,
                linewidth=0.7,
            )
            axis.invert_yaxis()
            # pad 6.0, not 3.0: at this canvas the title sat on the top row of
            # cells. Titles are also kept short so they stop before the
            # colorbar's rotated label at x = 0.79.
            axis.set_title(title, loc="left", weight="bold", pad=6.0, fontsize=6.8)
            axis.axvline(1.0, color="#d62728", linestyle="--", linewidth=1.4)
            # Hairline between EVERY layer row, heavier between seed blocks.
            # Without the per-layer rule the block reads as one band and the
            # claim it exists to support -- that all layers turn over at the
            # same checkpoint -- is invisible.
            for row in range(1, n_rows):
                if row % n_layers == 0:
                    axis.axhline(row, color="white", linewidth=1.4)
                else:
                    axis.axhline(row, color="white", linewidth=0.35, alpha=0.75)
            axis.set_yticks(
                [
                    seed_index * n_layers + n_layers / 2.0
                    for seed_index in range(len(PRIMARY_SEEDS))
                ]
            )
            axis.set_yticklabels([f"seed {s}" for s in PRIMARY_SEEDS])
            axis.tick_params(length=2.5, width=0.7)

        # Name the row structure on the axis instead of only in the caption:
        # the reader has to know a block is 7 layers to read "all at once".
        # One line. The layer roster moved to the caption: at this width the
        # second line was wider than the left margin and lost its first word.
        axes[0].set_ylabel("7 layers per seed", fontsize=7.0, labelpad=3.0)

        figure.text(
            0.02,
            0.982,
            "Where does the same-seed pair start to diverge?",
            fontsize=7.6,
            weight="bold",
            ha="left",
            va="top",
        )
        # Two lines: at this width the single-line form ran past the canvas.
        figure.text(
            0.02,
            0.951,
            f"dashed: probe fires at step {PROBE_STEP}\n"
            f"first divergent consumer: {FIRST_CONSUMER_LABEL} (5/5 seeds)",
            fontsize=6.2,
            linespacing=1.25,
            ha="left",
            va="top",
            color="#d62728",
        )
        # The strip is too short for five seed labels, and it does not need
        # them: nothing varies down the axis. One label names the extent.
        axes[1].set_yticks([n_rows / 2.0])
        axes[1].set_yticklabels(["5 seeds\n7 layers"], fontsize=7.0)
        axes[1].text(
            # centred right of the dashed probe line, which crossed the first
            # word when the label was centred on the whole strip
            (n_cols + 1.0) / 2.0,
            n_rows / 2.0,
            "all 35 cells exactly zero, every checkpoint",
            ha="center",
            va="center",
            # 6.4, not 7.5: at 7.5 the line was wider than the strip and ran
            # out over the tick labels on both sides.
            fontsize=6.4,
            weight="bold",
            color="#4d4d4d",
        )

        axes[1].set_xticks(np.arange(n_cols) + 0.5)
        axes[1].set_xticklabels(
            [str(step) for step in TRACE_STEPS], rotation=45, ha="right"
        )
        axes[1].set_xlabel("trace checkpoint (training step; categorical spacing)")

        # Colorbar spans Panel A only: Panel B has no magnitudes to encode.
        pos_a = axes[0].get_position()
        cax = figure.add_axes([0.815, pos_a.y0, 0.024, pos_a.height])
        colorbar = figure.colorbar(mesh, cax=cax)
        colorbar.set_label(
            "log10(relative L2 vs same-seed base)", fontsize=6.2, labelpad=2.0
        )
        colorbar.ax.tick_params(labelsize=6.2, width=0.7, length=2.0, pad=1.2)
        colorbar.outline.set_linewidth(0.7)
        colorbar.dividers.set_linewidth(0.7)
        # The gray-cell key is NOT hung off the colorbar. Anchored left of the
        # bar it overprinted Panel B's title; centred under the bar it
        # overprinted the bar's own rotated label. The header block at the top
        # left is the one place on this canvas with room, and it is where a
        # reader already looks for the figure's key.
        figure.text(
            0.02,
            0.900,
            "gray cells: exactly 0",
            fontsize=6.2,
            ha="left",
            va="top",
            color="#4d4d4d",
        )
        return figure


def close_figure(figure: "object") -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def render(
    data: dict[str, Any],
    first_consumer: str,
    output_dir: str | Path = OUTDIR,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data, first_consumer)
    pdf = output / "fig_divergence_heatmap.pdf"
    png = output / "fig_divergence_heatmap.png"
    figure.savefig(pdf, dpi=300, facecolor="white")
    figure.savefig(png, dpi=300, facecolor="white")
    close_figure(figure)
    return pdf, png


def main() -> int:
    data = load_figure_data(FIGURE_DATA)
    first_consumer = load_first_effective_event(VERDICT)
    pdf, png = render(data, first_consumer)
    shared = data["shared_relative_l2"]
    isolated = data["isolated_relative_l2"]
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(
        "shared: zeros at step 1399 =",
        int((shared[:, :, 0] == 0).sum()),
        "of",
        shared[:, :, 0].size,
        "| nonzero from 1400 on =",
        bool((shared[:, :, 1:] > 0).all()),
    )
    print("isolated: exactly zero everywhere =", bool((isolated == 0).all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
