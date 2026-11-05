# Incremental prediction-value benchmark — 2026-09-09

## Scope

This comparison was chosen after inspecting the held-out results and is
**exploratory post-readout**, not a new preregistered primary endpoint. Both
predictors use calibration seeds only. No new training, omitted seeds,
validation-scale fitting or changes to the frozen Gaussian prediction occur.

The comparator is the empirical mean calibration harvest. Unlike a zero
predictor, it is a substantive distribution-free estimate of mean harvest
under the same calibration/held-out seed-population premise. The benchmark
asks whether Gaussian scale structure provides additional prediction value
relative to this simple empirical estimate.

## Results

All scores refer to the same fixed panel; harvest H is max placement D minus
mean placement D. The 16 held-out seed mean H is 0.00343060363.

| Calibration-only predictor | Prediction | Held-out mean squared error | Mean absolute error (descriptive) |
|---|---:|---:|---:|
| Frozen corrected Gaussian P | 0.00388406019 | 0.00000189579913 | 0.00120529638 |
| Empirical calibration mean H, Q | 0.00428078845 | 0.00000241299050 | 0.00129035022 |

Both overpredict the held-out mean. Their absolute mean discrepancies are
0.00045345656 and 0.00085018482 respectively. Squared error is the relevant
mean-prediction scoring rule; MAE is descriptive, since it targets a median
rather than a mean when optimized.

The paired mean squared-error contrast, Gaussian minus empirical, is
-0.00000051719138 (negative favors Gaussian). Gaussian has about 21.4% lower
observed MSE in this sample. Two uncertainty scopes must remain separate:

- **Whole-procedure uncertainty:** independently resample calibration and
  held-out seed vectors, recompute both predictors from resampled calibration
  data only, then score the resampled held-out H. The approximate 95% interval
  is [-0.00000170060567, +0.00000016498245], including zero.
- **Conditional on these two fitted predictors:** keep P and Q fixed and
  resample only held-out seed vectors. The interval is
  [-0.00000102282925, -0.00000000538531], narrowly below zero. It does not
  propagate uncertainty in estimating P and Q from 16 calibration seeds.

Both use the original 20,000-draw RNG sequence (2026090801), calibration
indices generated first, then held-out indices. The joint procedure resamples
the same calibration rows for both competing predictors and the same
held-out rows for both losses, preserving the paired comparison.

The exact finite-sample identity

```text
mean[(H-P)^2 - (H-Q)^2] = (P-Q) * (P+Q-2*mean(H))
```

is independently verified. Both forecasts are constants across held-out
seeds; this comparison tests population-mean prediction, not learning which
individual seeds have high harvest.

## Interpretation

The Gaussian predictor performs better at the observed point estimates.
The full-procedure interval nevertheless includes zero, so the experiment
does not establish a stable advantage over empirical calibration. The
conditional interval must not be substituted for the more comprehensive
uncertainty statement. Neither result identifies Gaussian residuals, proves
equivalence, establishes universal transport, or turns the separate E1
validation-to-untouched-test result positive. Post-readout comparator choice
also rules out presenting this as an independently prespecified model contest.

## Reproduce

```powershell
python -B scripts/nanogpt_prediction_benchmark.py --root revision_results/nanogpt_final
python -B -m unittest discover -s tests -p "test_nanogpt_prediction_benchmark.py"
```

The separate result is `revision_results/nanogpt_final/prediction_benchmark.json`.
Original analysis and frozen prediction remain unchanged. This report is a
supplementary artifact; it does not silently replace the primary manuscript
endpoint or reinterpret the existing prediction interval as equivalence.
