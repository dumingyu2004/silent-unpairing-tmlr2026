"""Frozen external-calibration readout; no validation scale fit or seed selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from revision_nanogpt_campaign import CAL_SEEDS, VAL_SEEDS, PLACEMENTS, check_seed, frozen_prediction, job_key, atomic, sha


def main(args):
    root = Path(args.root)
    rows = {p.stem: json.loads(p.read_text()) for p in (root / "results").glob("*.json")}
    for seed in CAL_SEEDS + VAL_SEEDS:
        checked = check_seed(rows, seed)
        if not checked["passes"]:
            raise RuntimeError(f"Incomplete/invalid inference set: {checked}")
    frozen = json.loads((root / "frozen_calibration_prediction.json").read_text())
    if frozen != frozen_prediction(rows):
        raise RuntimeError("Frozen calibration prediction no longer matches calibration data")
    def grid(seeds, prefix):
        return np.asarray([[rows[job_key({"seed": s, "path": "baseline", "repeat": 1})]["fixed_val_loss"] -
            rows[job_key({"seed": s, "path": f"{prefix}_{p}", "repeat": 1})]["fixed_val_loss"]
            for p in PLACEMENTS] for s in seeds])
    cal, val = grid(CAL_SEEDS, "shared"), grid(VAL_SEEDS, "shared")
    cal_sd = cal.std(axis=1, ddof=1)
    val_h = val.max(axis=1) - val.mean(axis=1)
    coefficient = frozen["c5"] / frozen["normal_sample_sd_bias_factor"]
    prediction = frozen["predicted_mean_harvest"]
    rng = np.random.default_rng(2026090801)
    cal_ix = rng.integers(0, len(CAL_SEEDS), size=(20000, len(CAL_SEEDS)))
    val_ix = rng.integers(0, len(VAL_SEEDS), size=(20000, len(VAL_SEEDS)))
    boot_prediction = coefficient * cal_sd[cal_ix].mean(axis=1)
    boot_observed = val_h[val_ix].mean(axis=1)
    discrepancy = float(val_h.mean() - prediction)
    quantiles = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    result = {"protocol": "frozen-external-calibration", "calibration_n": 16, "validation_n": 16,
              "independent_unit": "seed, not repeat or placement", "placements": PLACEMENTS,
              "predicted_mean_harvest": prediction, "observed_validation_mean_harvest": float(val_h.mean()),
              "signed_prediction_discrepancy": discrepancy,
              "relative_prediction_discrepancy": discrepancy / prediction if prediction else None,
              "calibration_scale_prediction_ci95": quantiles(boot_prediction),
              "validation_mean_harvest_ci95": quantiles(boot_observed),
              "discrepancy_ci95": quantiles(boot_observed - boot_prediction),
              "validation_harvest_per_seed": val_h.tolist(), "validation_shared_deltas": val.tolist(),
              "validation_unselected_mean_delta": float(val.mean()),
              "validation_selected_max_mean_delta": float(val.max(axis=1).mean()),
              "calibration_shared_deltas": cal.tolist(),
              "validation_isolated_deltas": grid(VAL_SEEDS, "isolated").tolist(),
              "bootstrap": {"repetitions": 20000, "seed": 2026090801, "unit": "independently resample calibration and validation seed vectors", "method": "percentile, approximate"},
              "frozen_prediction_sha256": sha(root / "frozen_calibration_prediction.json"),
              "scope": "Independent-setting calibration/holdout prediction check; no validation scale fit; conditional Gaussian model need not hold; all results retained regardless of sign or interval."}
    atomic(root / "analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
