#!/usr/bin/env python3
"""Validate and render the frozen HuggingFace boundary figure."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs_hf_pairing_probe"
OUTDIR = Path(__file__).resolve().parent

ARM_KEYS = ("repeat", "hf_eval", "probe_samp", "probe_cuda")
# Short horizontal labels: the long two-line labels collided at 20 degrees in
# the 3.3 in column. The caption spells out "sequential Trainer evaluation" and
# "CPU/CUDA random-batch probe".
ARM_LABELS = (
    "Repeat",
    "Trainer\neval",
    "CPU\nprobe",
    "CUDA\nprobe",
)
# Plot max|delta theta| in units of 1e-4, stated in the axis label, instead of
# a scientific offset text that crowded the panel titles.
Y_SCALE = 1e4
BOUNDARY_ARMS = ("ref", *ARM_KEYS)
REPLICATION_ARMS = ("ref", "repeat", "probe_cuda")
TASKS = ("rte", "mrpc")
SEEDS = tuple(range(41, 46))
_FIXED_TIMESTAMP = dt.datetime(2026, 7, 16, 0, 0, 0, tzinfo=dt.timezone.utc)
_FIGURE_RCPARAMS = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.0,
    "lines.markersize": 5.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_PDF_METADATA = {
    "Title": "fig3_hf_boundary",
    "Creator": "roberta_momentum_samedist_suite",
    "CreationDate": _FIXED_TIMESTAMP,
    "ModDate": _FIXED_TIMESTAMP,
}
_PNG_METADATA = {
    "Title": "fig3_hf_boundary",
    "Software": "roberta_momentum_samedist_suite",
}


@dataclass(frozen=True)
class HFBoundaryData:
    arm_keys: tuple[str, ...]
    arm_labels: tuple[str, ...]
    boundary_param_linf: np.ndarray
    pair_ids: tuple[tuple[str, int], ...]
    repeat_param_linf: np.ndarray
    cuda_param_linf: np.ndarray
    abs_accuracy_pp: np.ndarray


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid HF receipt: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"HF receipt is not an object: {path.name}")
    return value


def _require_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing HF receipt: {path.name}")
    return _read_json(path)


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_arm_set(receipt: dict[str, Any], expected: tuple[str, ...], label: str) -> dict[str, Any]:
    results = _require_dict(receipt.get("results"), f"{label} results")
    if set(results) != set(expected):
        raise ValueError(f"{label} arm set does not match the frozen contract")
    for arm in expected:
        if not isinstance(results[arm], dict):
            raise ValueError(f"{label} arm record is malformed: {arm}")
    return results


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _readonly_float64(values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _positive_finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive finite value") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite value")
    return number


def _validate_boundary_receipt(receipt: dict[str, Any]) -> np.ndarray:
    args = _require_dict(receipt.get("args"), "result_v2 args")
    if (
        args.get("model") != "roberta-base"
        or args.get("steps") != 150
        or args.get("probe_every") != 50
        or args.get("seed") != 42
    ):
        raise ValueError("result_v2 metadata does not match the frozen contract")

    results = _require_arm_set(receipt, BOUNDARY_ARMS, "result_v2")
    reference_acc = _finite_number(results["ref"].get("final_val_acc"), "result_v2 ref accuracy")
    boundary = []
    for arm in ARM_KEYS:
        result = _require_dict(results[arm], f"result_v2 {arm}")
        diff = _finite_number(result.get("max_param_diff_vs_ref"), f"result_v2 {arm} parameter diff")
        boundary.append(diff)
        if arm != "probe_cuda":
            if diff != 0.0:
                raise ValueError("result_v2 non-CUDA boundary arms must be exact zero")
            accuracy = _finite_number(result.get("final_val_acc"), f"result_v2 {arm} accuracy")
            if accuracy != reference_acc:
                raise ValueError("result_v2 non-CUDA boundary arms must preserve accuracy")
        elif diff <= 0.0:
            raise ValueError("result_v2 CUDA boundary must be positive")
    return _readonly_float64(boundary)


def _validate_replication_receipt(receipt: dict[str, Any]) -> tuple[tuple[str, int], float, float, float]:
    args = _require_dict(receipt.get("args"), "replication args")
    if (
        args.get("model") != "roberta-base"
        or args.get("steps") != 150
        or args.get("probe_every") != 50
        or args.get("arms") != "ref,repeat,probe_cuda"
    ):
        raise ValueError("replication metadata does not match the frozen contract")

    task = args.get("task")
    seed = args.get("seed")
    if task not in TASKS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("replication metadata does not match the frozen contract")

    results = _require_arm_set(receipt, REPLICATION_ARMS, "replication")
    reference_acc = _finite_number(results["ref"].get("final_val_acc"), "replication ref accuracy")
    repeat_acc = _finite_number(results["repeat"].get("final_val_acc"), "replication repeat accuracy")
    repeat_diff = _finite_number(
        results["repeat"].get("max_param_diff_vs_ref"), "replication repeat parameter diff"
    )
    cuda_acc = _finite_number(results["probe_cuda"].get("final_val_acc"), "replication CUDA accuracy")
    cuda_diff = _positive_finite_number(
        results["probe_cuda"].get("max_param_diff_vs_ref"), "CUDA parameter diff"
    )

    if repeat_diff != 0.0:
        raise ValueError("repeat parameter diff must be exact zero")
    if repeat_acc != reference_acc:
        raise ValueError("repeat accuracy must equal its paired reference")

    abs_accuracy_pp = 100.0 * abs(cuda_acc - reference_acc)
    return (task, seed), repeat_diff, cuda_diff, abs_accuracy_pp


def load_hf_boundary_data(logdir: Path = LOGDIR) -> HFBoundaryData:
    logdir = Path(logdir)
    boundary_receipt = _require_receipt(logdir / "result_v2.json")
    boundary_param_linf = _validate_boundary_receipt(boundary_receipt)

    pair_ids: list[tuple[str, int]] = []
    repeat_param_linf: list[float] = []
    cuda_param_linf: list[float] = []
    abs_accuracy_pp: list[float] = []
    replication_cuda_by_pair: dict[tuple[str, int], dict[str, Any]] = {}

    for task in TASKS:
        for seed in SEEDS:
            receipt = _require_receipt(logdir / f"a1b_{task}_s{seed}.json")
            pair_id, repeat_diff, cuda_diff, accuracy_pp = _validate_replication_receipt(receipt)
            pair_ids.append(pair_id)
            repeat_param_linf.append(repeat_diff)
            cuda_param_linf.append(cuda_diff)
            abs_accuracy_pp.append(accuracy_pp)
            replication_cuda_by_pair[pair_id] = receipt["results"]["probe_cuda"]

    if len(pair_ids) != 10 or len(set(pair_ids)) != 10:
        raise ValueError("duplicate pair IDs are not allowed")

    expected_pairs = tuple((task, seed) for task in TASKS for seed in SEEDS)
    if tuple(pair_ids) != expected_pairs:
        missing = sorted(set(expected_pairs).difference(pair_ids))
        extra = sorted(set(pair_ids).difference(expected_pairs))
        raise ValueError(f"pair IDs do not match the frozen contract: missing={missing}, extra={extra}")

    duplicate_pair = ("rte", 42)
    if boundary_receipt["results"]["probe_cuda"] != replication_cuda_by_pair.get(duplicate_pair):
        raise ValueError("probe_cuda duplicate relationship changed")

    return HFBoundaryData(
        arm_keys=ARM_KEYS,
        arm_labels=ARM_LABELS,
        boundary_param_linf=boundary_param_linf,
        pair_ids=tuple(pair_ids),
        repeat_param_linf=_readonly_float64(repeat_param_linf),
        cuda_param_linf=_readonly_float64(cuda_param_linf),
        abs_accuracy_pp=_readonly_float64(abs_accuracy_pp),
    )


def build_figure(data: HFBoundaryData):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    repeat_color = "#009E73"
    cuda_color = "#D55E00"
    mean_color = "#0072B2"
    segment_color = "0.65"
    boundary_color = "0.30"

    with mpl.rc_context(_FIGURE_RCPARAMS):
        figure = Figure(figsize=(3.3125, 3.45))
        FigureCanvasAgg(figure)
        ax_a, ax_b = figure.subplots(2, 1)
        figure.subplots_adjust(left=0.20, right=0.98, top=0.93, bottom=0.08, hspace=0.62)

        x_boundary = np.arange(len(data.arm_keys), dtype=np.float64)
        boundary = data.boundary_param_linf * Y_SCALE
        repeat_values = data.repeat_param_linf * Y_SCALE
        cuda_values = data.cuda_param_linf * Y_SCALE
        ax_a.axhline(0.0, color="0.82", linewidth=0.8, zorder=0)
        ax_a.plot(
            x_boundary,
            boundary,
            color=boundary_color,
            linewidth=1.0,
            label="boundary-path",
        )
        ax_a.scatter(
            x_boundary[:3],
            boundary[:3],
            s=30.25,
            marker="o",
            facecolors="white",
            edgecolors=repeat_color,
            linewidths=1.0,
            zorder=3,
            label="boundary-zero",
        )
        ax_a.scatter(
            [x_boundary[3]],
            [boundary[3]],
            s=30.25,
            marker="o",
            facecolors=cuda_color,
            edgecolors=cuda_color,
            linewidths=1.0,
            zorder=4,
            label="boundary-cuda",
        )
        ax_a.set_title("A   Which tested path breaks pairing?", fontsize=8.2, pad=3.0)
        ax_a.set_ylabel("max $|\\Delta \\theta|_\\infty$\n($\\times 10^{-4}$)")
        ax_a.set_xticks(x_boundary)
        ax_a.set_xticklabels(data.arm_labels, rotation=0, ha="center")
        ax_a.set_xlim(-0.45, 3.45)
        ax_a.set_ylim(-2.5e-5 * Y_SCALE, 6.9e-4 * Y_SCALE)

        x_pairs = np.arange(len(data.pair_ids), dtype=np.float64)
        labels = [f"{task.upper()[0]}{seed}" for task, seed in data.pair_ids]
        mean_value = float(cuda_values.mean())

        ax_b.axhline(0.0, color="0.82", linewidth=0.8, zorder=0)
        ax_b.plot(
            [x_pairs[0] - 0.5, x_pairs[-1] + 0.5],
            [mean_value, mean_value],
            color=mean_color,
            linestyle="--",
            linewidth=1.0,
            label="cuda-mean",
            zorder=1,
        )
        for index, ((task, seed), repeat_diff, cuda_diff) in enumerate(
            zip(data.pair_ids, repeat_values, cuda_values, strict=True)
        ):
            pair_key = f"{task}-{seed}"
            marker = "o" if task == "rte" else "s"
            marker_shape = "circle" if task == "rte" else "square"
            segment = ax_b.plot(
                [x_pairs[index], x_pairs[index]],
                [repeat_diff, cuda_diff],
                color=segment_color,
                linewidth=0.9,
                solid_capstyle="round",
                zorder=2,
                label="pair-segment",
            )[0]
            segment.set_gid(f"pair-segment-{pair_key}")
            repeat_marker = ax_b.scatter(
                [x_pairs[index]],
                [repeat_diff],
                s=30.25,
                marker=marker,
                facecolors="white",
                edgecolors=repeat_color,
                linewidths=1.0,
                zorder=3,
                label=f"repeat-{marker_shape}",
            )
            repeat_marker.set_gid(f"repeat-{pair_key}")
            cuda_marker = ax_b.scatter(
                [x_pairs[index]],
                [cuda_diff],
                s=30.25,
                marker=marker,
                facecolors=cuda_color,
                edgecolors=cuda_color,
                linewidths=1.0,
                zorder=4,
                label=f"cuda-{marker_shape}",
            )
            cuda_marker.set_gid(f"cuda-{pair_key}")
        ax_b.set_title("B   Ten unique task–seed replications", fontsize=8.2, pad=3.0)
        ax_b.set_ylabel("max $|\\Delta \\theta|_\\infty$\n($\\times 10^{-4}$)")
        ax_b.set_xticks(x_pairs)
        ax_b.set_xticklabels(labels, rotation=0)
        ax_b.set_xlim(-0.5, len(data.pair_ids) - 0.5)
        ax_b.set_ylim(-2.5e-5 * Y_SCALE, 6.9e-4 * Y_SCALE)

        for axes in (ax_a, ax_b):
            axes.tick_params(axis="both", which="major", width=0.7, length=3.5)
            for spine in axes.spines.values():
                spine.set_linewidth(0.7)

    return figure


def close_figure(figure) -> None:
    figure.clear()


def render(data: HFBoundaryData, output_dir: Path = OUTDIR) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data)
    pdf_path = output_dir / "fig3_hf_boundary.pdf"
    png_path = output_dir / "fig3_hf_boundary.png"
    try:
        figure.savefig(pdf_path, metadata=_PDF_METADATA)
        figure.savefig(png_path, dpi=300, metadata=_PNG_METADATA)
    finally:
        close_figure(figure)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=LOGDIR)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args(argv)

    data = load_hf_boundary_data(args.logdir)
    render(data, args.output_dir)
    return 0


if __name__ == "__main__":
    mpl.use("Agg")
    raise SystemExit(main())
