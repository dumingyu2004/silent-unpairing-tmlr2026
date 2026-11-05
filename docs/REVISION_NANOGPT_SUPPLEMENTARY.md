# nanoGPT supplementary analysis — 2026-09-09

These analyses use the completed 16-calibration/16-held-out-seed campaign. No new training, seed exclusions, validation-scale fitting or changes to the frozen corrected primary analysis were made. Independent recomputation from all 726 archived raw records reproduced both placement grids exactly.

## Frozen primary and prespecified sensitivity

All values are fixed-panel loss units. Positive D = baseline loss minus shared-probe loss. Harvest H = max of five D values minus their mean; the baseline is not a sixth candidate.

| Analysis | Calibration-only prediction | Held-out mean H | Observed minus predicted | Joint seed-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Frozen corrected primary | 0.00388406 | 0.00343060 | -0.00045346 (-11.7%) | [-0.00134285, 0.00043370] |
| Prespecified uncorrected sensitivity | 0.00365096 | 0.00343060 | -0.00022036 (-6.0%) | [-0.00108275, 0.00064263] |

The uncorrected prediction interval is [0.00307846, 0.00422724]; the held-out H interval is [0.00279334, 0.00407564]. Both analyses use 20,000 whole-seed bootstrap resamples with frozen RNG seed 2026090801, drawing calibration indices before validation indices. Calibration and validation groups are independently resampled. The sensitivity's closer point estimate does not replace the corrected primary. Intervals containing zero discrepancy establish neither equivalence nor the iid Gaussian assumptions, individual-seed accuracy or cross-domain transport.

## Exploratory selected and unselected outcomes

| Held-out quantity | Mean | Whole-seed bootstrap 95% interval |
|---|---:|---:|
| Unselected mean placement D | -0.00018085 | [-0.00165377, 0.00128459] |
| Selected best placement D | 0.00324976 | [0.00149894, 0.00490740] |
| H: selected minus unselected | 0.00343060 | [0.00279334, 0.00407564] |

The selected mean's exploratory interval excludes zero. This is an apparent improvement on the same fixed scoring panel used to select placements, not an untouched-test result or an identified estimate of pure selection bias. H is nonnegative by construction. These post-readout intervals do not replace the frozen prediction-discrepancy endpoint. E1's validation-to-untouched-test result remains inconclusive.

## Exploratory placement diagnostics

Placement order is 200, 500, 900, 1300, 1700 updates. Calibration means are approximately [0.00264404, 0.00075536, -0.00105386, -0.00039134, -0.00022961]; held-out means are [-0.00005906, -0.00037103, 0.00035027, -0.00035512, -0.00046929]. Full means, marginal intervals, standard deviations and covariance/correlation matrices are in supplementary.json.

Raw off-diagonal correlations range from 0.382 to 0.881 in calibration and 0.210 to 0.876 held out. Shared baselines and seed components can contribute to these correlations, so these values alone do not reject a common-component Gaussian model. Row-centered contrast correlations range from -0.694 to 0.791 and -0.690 to 0.684 respectively; row centering itself imposes zero sums and singular covariance. Late-placement contrast correlation is 0.791/0.684. These are descriptive heterogeneity/dependence diagnostics, not formal independence or Gaussianity tests. The JSON also verifies the exact finite-sample decomposition of mean within-seed sample variance into marginal variance, off-diagonal covariance and between-placement mean components. No general claim that placement effects preserve the Gaussian calibration is made.

## Exploratory influence and per-seed plot

Omitting one calibration seed at a time gives corrected predictions 0.00372860–0.00400198. Omitting one held-out seed gives mean H 0.00329137–0.00355852; against the unchanged full-calibration prediction, discrepancies remain negative (-0.00059269 to -0.00032554). This describes influence only: every seed is retained and no exclusion rule or validation refit is introduced.

The figure at `revision_results/nanogpt_final/nanogpt_per_seed.pdf` shows calibration and held-out H against both frozen predictions and the held-out selected/unselected D pairs. Individual seed scatter is not a prediction interval for the population mean.

## Reproduce

Run from the revision root:

```powershell
python -B scripts/nanogpt_supplementary.py --root revision_results/nanogpt_final
python -B scripts/nanogpt_supplementary_plot.py
python -B -m unittest discover -s tests -p "test_nanogpt*.py"
```

Frozen analysis.json, frozen_calibration_prediction.json and the raw archive are preserved; input and code hashes are recorded in supplementary.json. The anonymous package separately records original and redacted-copy hashes and verifies numerical parity after extraction. Historical retraining/timestamp gaps are not resolved by this fresh campaign.
