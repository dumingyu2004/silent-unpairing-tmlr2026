#!/usr/bin/env python3
"""Render the preregistered three-panel causal-trace figure.

The data contract is NumPy-only.  Matplotlib is imported lazily by
``render_figure`` so contract validation remains dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA = ROOT / "logs_causal_trace" / "causal_trace_figure_data.npz"
OUTDIR = Path(__file__).resolve().parent

PRIMARY_SEEDS = (1, 2, 3, 4, 5)
TRACE_STEPS = (1399, 1400, 1401, 1402, 1405, 1410, 1450, 1800, 1801, 2700, 4000, 5400)
LAYER_ORDER = (
    "resnet_stem",
    "resnet_stage1",
    "resnet_stage2",
    "resnet_stage3",
    "resnet_head",
    "vnet_layer1",
    "vnet_layer2",
)
EVENT_ORDER = (
    "pre_eval",
    "post_loader_iter",
    "post_eval_internal",
    "post_isolation_context",
    "train_batch",
    "meta_model_pre_init",
    "meta_model_post_init",
    "meta_model_post_load",
    "meta_indices",
    "vnet_post_backward",
    "vnet_post_step",
    "sample_weights",
    "resnet_post_backward",
    "resnet_post_step",
)
FIGURE_CAPTION = (
    "Layer-wise magnitude localizes propagation but does not establish that a layer "
    "is responsible for the final metric change."
)

_ARRAY_KEYS = (
    "panel_a_shared_equal",
    "panel_a_isolated_equal",
    "shared_relative_l2",
    "isolated_relative_l2",
    "shared_total_relative_l2",
    "shared_accuracy_delta",
)


def _ordered_strings(value: np.ndarray, label: str) -> tuple[str, ...]:
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise ValueError(f"{label} must be a one-dimensional string array")
    return tuple(str(item) for item in value.tolist())


def _ordered_integers(value: np.ndarray, label: str) -> tuple[int, ...]:
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    return tuple(int(item) for item in value.tolist())


def _finite_numeric(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    if value.shape != shape:
        raise ValueError(f"{label} shape {value.shape} != frozen shape {shape}")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain only finite numeric values")
    return np.asarray(value, dtype=np.float64)


def panel_b_mean_log10(relative_l2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the seed mean of log10 divergences and cells containing exact zeros."""
    values = np.asarray(relative_l2, dtype=np.float64)
    expected_shape = (len(PRIMARY_SEEDS), len(LAYER_ORDER), len(TRACE_STEPS))
    if values.shape != expected_shape:
        raise ValueError(f"Panel B values shape {values.shape} != frozen shape {expected_shape}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Panel B relative L2 values must be finite and non-negative")

    exact_zero = np.any(values == 0.0, axis=0)
    mean_log10 = np.full(values.shape[1:], np.nan, dtype=np.float64)
    positive = ~exact_zero
    mean_log10[positive] = np.log10(values[:, positive]).mean(axis=0)
    return mean_log10, exact_zero


def load_figure_data(path: str | Path = FIGURE_DATA) -> dict[str, Any]:
    """Load and fail-closed validate the frozen figure-data archive."""
    required = {
        "panel_a_seed",
        "panel_a_event_order",
        "seeds",
        "layer_order",
        "checkpoint_order",
        *_ARRAY_KEYS,
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"missing figure-data field(s): {', '.join(missing)}")
        raw = {key: np.array(archive[key], copy=True) for key in required}

    panel_a_seed = raw["panel_a_seed"]
    if panel_a_seed.ndim != 0 or panel_a_seed.dtype.kind not in "iu" or int(panel_a_seed) != 1:
        raise ValueError("Panel A must use frozen seed 1")

    events = _ordered_strings(raw["panel_a_event_order"], "Panel A event order")
    if events != EVENT_ORDER:
        raise ValueError("Panel A event order does not match the frozen order")
    seeds = _ordered_integers(raw["seeds"], "seed set")
    if seeds != PRIMARY_SEEDS:
        raise ValueError("seed set must be exactly 1--5 in frozen order")
    layers = _ordered_strings(raw["layer_order"], "layer order")
    if layers != LAYER_ORDER:
        raise ValueError("layer order does not match the frozen order")
    checkpoints = _ordered_integers(raw["checkpoint_order"], "checkpoint order")
    if checkpoints != TRACE_STEPS:
        raise ValueError("checkpoint order does not match the frozen order")

    event_shape = (len(EVENT_ORDER),)
    equality = {}
    for key in ("panel_a_shared_equal", "panel_a_isolated_equal"):
        value = raw[key]
        if value.shape != event_shape or value.dtype.kind != "b":
            raise ValueError(f"{key} must be a {event_shape} boolean array")
        equality[key] = value
    training_visible_start = EVENT_ORDER.index("post_isolation_context")
    if not np.all(equality["panel_a_isolated_equal"][training_visible_start:]):
        raise ValueError("isolated arm must equal base at every training-visible event")

    cube_shape = (len(PRIMARY_SEEDS), len(LAYER_ORDER), len(TRACE_STEPS))
    shared_layers = _finite_numeric(raw["shared_relative_l2"], cube_shape, "shared_relative_l2")
    isolated_layers = _finite_numeric(raw["isolated_relative_l2"], cube_shape, "isolated_relative_l2")
    shared_total = _finite_numeric(
        raw["shared_total_relative_l2"],
        (len(PRIMARY_SEEDS), len(TRACE_STEPS)),
        "shared_total_relative_l2",
    )
    accuracy_delta = _finite_numeric(
        raw["shared_accuracy_delta"], (len(PRIMARY_SEEDS),), "shared_accuracy_delta"
    )
    if np.any(shared_layers < 0.0) or np.any(isolated_layers < 0.0) or np.any(shared_total < 0.0):
        raise ValueError("relative L2 values must be non-negative")
    if np.any(isolated_layers != 0.0):
        raise ValueError("isolated relative L2 cells must be exact zero")

    return {
        "panel_a_seed": 1,
        "panel_a_event_order": events,
        **equality,
        "seeds": seeds,
        "layer_order": layers,
        "checkpoint_order": checkpoints,
        "shared_relative_l2": shared_layers,
        "isolated_relative_l2": isolated_layers,
        "shared_total_relative_l2": shared_total,
        "shared_accuracy_delta": accuracy_delta,
    }


def _short_event_labels() -> list[str]:
    return [
        "pre-eval",
        "loader iter",
        "eval internal",
        "context exit",
        "train batch",
        "meta pre-init",
        "meta post-init",
        "meta post-load",
        "meta indices",
        "VNet grad",
        "VNet step",
        "weights",
        "ResNet grad",
        "ResNet step",
    ]


def render_figure(data: dict[str, Any], output_dir: str | Path = OUTDIR) -> None:
    """Render Panels A--C and write the frozen PDF and PNG outputs."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )

    blue = "#0072B2"
    vermillion = "#D55E00"
    green = "#009E73"
    sky = "#56B4E9"
    ink = "0.15"
    seed_markers = ("o", "s", "^", "D", "v")

    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.9, 1.2))

    # Panel A: seed-1 event equality chain.
    ax_a = figure.add_subplot(grid[0, :])
    x_events = np.arange(len(EVENT_ORDER))
    ax_a.axvspan(0.65, 2.35, color=sky, alpha=0.12, zorder=0)
    ax_a.axvspan(2.65, len(EVENT_ORDER) - 0.55, color=green, alpha=0.06, zorder=0)
    for row, key in ((1, "panel_a_shared_equal"), (0, "panel_a_isolated_equal")):
        equal = np.asarray(data[key], dtype=bool)
        ax_a.plot(x_events, np.full_like(x_events, row), color="0.75", lw=0.7, zorder=1)
        ax_a.scatter(
            x_events[equal],
            np.full(np.count_nonzero(equal), row),
            s=24,
            marker="o",
            facecolor=green,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax_a.scatter(
            x_events[~equal],
            np.full(np.count_nonzero(~equal), row),
            s=27,
            marker="X",
            facecolor=vermillion,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax_a.text(1.5, 1.46, "isolated: inside fork", color=blue, ha="center", fontsize=6.5)
    ax_a.text(7.8, 1.46, "isolated: training-visible state restored after context exit", color=green,
              ha="center", fontsize=6.5)
    ax_a.set_xticks(x_events)
    ax_a.set_xticklabels(_short_event_labels(), rotation=38, ha="right")
    ax_a.set_yticks((0, 1))
    ax_a.set_yticklabels(("isolated_eval", "shared_eval"))
    ax_a.set_ylim(-0.55, 1.62)
    ax_a.set_xlim(-0.55, len(EVENT_ORDER) - 0.45)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_title("A   Event-level causal chain (seed 1)", loc="left", weight="bold")
    ax_a.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=green, markeredgecolor="none", label="equal to base"),
            Line2D([], [], marker="X", linestyle="none", markerfacecolor=vermillion, markeredgecolor="none", label="different from base"),
        ),
        loc="upper right",
        frameon=False,
        ncol=2,
    )

    # Panel B: five-seed shared-arm layer/time mean; green dots mark isolated exact zeros.
    ax_b = figure.add_subplot(grid[1, 0])
    log_shared, shared_exact_zero = panel_b_mean_log10(data["shared_relative_l2"])
    ax_b.set_facecolor("0.92")
    image = ax_b.imshow(
        np.ma.masked_invalid(log_shared), aspect="auto", cmap="magma", interpolation="nearest"
    )
    xx, yy = np.meshgrid(np.arange(len(TRACE_STEPS)), np.arange(len(LAYER_ORDER)))
    ax_b.scatter(xx.ravel() + 0.27, yy.ravel() + 0.27, s=3.0, color=green, linewidth=0)
    zero_y, zero_x = np.nonzero(shared_exact_zero)
    ax_b.scatter(zero_x, zero_y, s=10, marker="x", color=ink, linewidth=0.65)
    ax_b.set_xticks(np.arange(len(TRACE_STEPS)))
    ax_b.set_xticklabels([str(step) for step in TRACE_STEPS], rotation=50, ha="right")
    ax_b.set_yticks(np.arange(len(LAYER_ORDER)))
    ax_b.set_yticklabels(
        [name.replace("resnet_", "R: ").replace("vnet_", "V: ") for name in LAYER_ORDER]
    )
    ax_b.set_xlabel("checkpoint (global step)")
    ax_b.set_title("B   Layer × time divergence", loc="left", weight="bold")
    colorbar = figure.colorbar(image, ax=ax_b, fraction=0.046, pad=0.03)
    colorbar.set_label(r"mean $\log_{10}$(relative L2), shared")
    ax_b.text(
        0.01,
        -0.43,
        "× shared: ≥1 exact-zero seed   • isolated_eval: exact zero in every cell",
        transform=ax_b.transAxes,
        color=green,
        fontsize=6.2,
        va="top",
    )

    # Panel C: five trajectories, median/range, and signed accuracy endpoints.
    ax_c = figure.add_subplot(grid[1, 1])
    trajectories = np.asarray(data["shared_total_relative_l2"])
    x_steps = np.arange(len(TRACE_STEPS))
    minimum = trajectories.min(axis=0)
    maximum = trajectories.max(axis=0)
    median = np.median(trajectories, axis=0)
    ax_c.fill_between(x_steps, minimum, maximum, color=blue, alpha=0.15, linewidth=0, label="seed range")
    for index, seed in enumerate(PRIMARY_SEEDS):
        ax_c.plot(x_steps, trajectories[index], color="0.55", lw=0.65, alpha=0.8,
                  marker=seed_markers[index], ms=2.4, markevery=[-1], label=f"seed {seed}")
    ax_c.plot(x_steps, median, color=blue, lw=1.5, label="median")
    ax_c.set_yscale("symlog", linthresh=1e-12)
    ax_c.set_xticks(x_steps)
    ax_c.set_xticklabels([str(step) for step in TRACE_STEPS], rotation=50, ha="right")
    ax_c.set_xlabel("checkpoint (global step)")
    ax_c.set_ylabel("total parameter relative L2")
    ax_c.set_title("C   Propagation and final endpoint", loc="left", weight="bold")

    ax_accuracy = ax_c.twinx()
    endpoint_x = np.full(len(PRIMARY_SEEDS), x_steps[-1], dtype=float)
    for index, delta in enumerate(data["shared_accuracy_delta"]):
        ax_accuracy.scatter(endpoint_x[index], delta, marker=seed_markers[index], s=24,
                            facecolor=vermillion, edgecolor="white", linewidth=0.45, zorder=5)
    ax_accuracy.axhline(0.0, color="0.75", lw=0.5, ls=(0, (2, 2)))
    ax_accuracy.set_ylabel("final signed test-accuracy Δ (pt)", color=vermillion)
    ax_accuracy.tick_params(axis="y", colors=vermillion)
    ax_accuracy.spines["right"].set_color(vermillion)
    handles, labels = ax_c.get_legend_handles_labels()
    endpoint_handle = Line2D([], [], marker="o", linestyle="none", markerfacecolor=vermillion,
                             markeredgecolor="none", label="accuracy endpoints")
    ax_c.legend(handles[:1] + [handles[-1], endpoint_handle], labels[:1] + [labels[-1], "accuracy endpoints"],
                loc="upper left", frameon=False)

    figure.text(0.5, 0.005, FIGURE_CAPTION, ha="center", va="bottom", color=ink, fontsize=6.4)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(output_path / f"fig_causal_trace.{extension}")
    plt.close(figure)


def main() -> None:
    render_figure(load_figure_data())


if __name__ == "__main__":
    main()
