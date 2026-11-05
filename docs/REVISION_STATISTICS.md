# Reproducible statistical revision (2026-09-08)

This is a new, reviewer-requested analysis of existing completed runs. It does
not change the frozen E1 preregistration, comparator, training code, or logs.
It strengthens reproducibility and assumption checking; simulations and
resampling are not additional experiments.

## Reproduce and inspect

```powershell
python -m unittest discover -s tests -p test_revision_reanalysis.py -v
python scripts/revision_reanalysis.py --draws 100000 --seed 20260908
```

The canonical machine-readable output is `revision_results/statistics.json`.
The accompanying `harvest_summary.csv`, `harvest_seed_readout.csv`,
`e1_seed_readout.csv`, and `lineage.csv` expose the report and individual seed
values. Lineage records SHA-256 and byte size for 163 historical inputs and
protocol/comparator sources. All 35 shortened-protocol baseline/placement
logs are reconciled against `logs_flip_demo/prop2_fit.json`; all 120 E1 arm
logs are loaded and reconciled against `logs_moneyshot/e1_verdict.json`.
The other harvest grids are explicitly sourced from their archived aggregate
JSON matrices, not claimed to be independently reconstructed raw training.

## Harvest estimand and scale

For each complete seed row, `H_s = max_j D_sj - mean_j D_sj` and
`s_s = sd_j(D_sj, ddof=1)`. Both are unchanged by an arbitrary common offset
within that seed. The ratio is the ratio of means,
`R = mean_s H_s / (c_m mean_s s_s)`, not the mean of per-seed ratios.
`c_m` is obtained by deterministic normal-density quadrature. The archived
Monte Carlo constants and their original ratios are retained separately.

For m independent normal placements, the ordinary sample SD has expectation
`c4(m) sigma`, where `c4(m) = sqrt(2/(m-1)) Gamma(m/2)/Gamma((m-1)/2)`.
Consequently the normal-specific unbiased scale estimate is `s_s/c4(m)`;
the corrected prediction is larger and the corrected ratio is `c4(m) R`.
This corrects scale bias under a specified model, not misspecification,
placement offsets, correlated residuals, or heavy tails.

| Campaign | Seeds x placements | Mean H | Normal plug-in | c4-corrected prediction | R | Corrected R |
|---|---:|---:|---:|---:|---:|---:|
| Shortened | 5 x 6 | 4.010667 | 4.120255 | 4.330124 | 0.973402 | 0.926224 |
| Realistic eval | 5 x 5 | 0.448400 | 0.391187 | 0.416163 | 1.146255 | 1.077463 |
| Full, best | 3 x 4 | 0.159167 | 0.157986 | 0.171478 | 1.007476 | 0.928206 |
| Full, final | 3 x 4 | 0.125000 | 0.124571 | 0.135210 | 1.003443 | 0.924489 |

All accuracy differences/predictions are percentage points. Best and final
are two endpoints of the same three-seed full-protocol campaign, not two
independent replications.

Whole seed rows supply numerator and denominator jointly in every resample.
For n=5 all 3,125 ordered n-out-of-n samples are enumerated; for n=3 all 27
are enumerated. Percentiles use NumPy's linear quantile convention.

| Campaign | R percentile 95% interval | Corrected R percentile 95% interval |
|---|---:|---:|
| Shortened | [0.735951, 1.172299] | [0.700282, 1.115481] |
| Realistic eval | [0.951588, 1.277547] | [0.894479, 1.200876] |
| Full, best | [0.940537, 1.061085] | [0.866534, 0.977596] |
| Full, final | [0.717611, 1.351506] | [0.661148, 1.245166] |

These are discrete small-n bootstrap sensitivity intervals, not validated
nominal-coverage statements. In particular, the corrected best-endpoint
interval excluding one is not a reliable three-seed rejection of a Gaussian
model. It must not be concealed, nor interpreted without the full sampling
simulation below. The best-endpoint uncorrected interval is relatively
narrow because its three H/SD ratios are similar; it is incorrect to call
every full-protocol interval wide.

## Leave-one-seed-out definitions and placement diagnostics

Two historical implementations differ. `scripts/prop2_fit.py` flattens all
remaining rows before calculating one SD; `figs/make_fig_r1_crossseed.py`
averages the remaining seeds' within-row SDs. Flattening additionally mixes
between-seed means into the scale and is not invariant to seed-specific
offsets. We preserve both definitions and use exact quadrature consistently
in the new comparison, while retaining archived constants separately.

For shortened-protocol data:

| Predictor | MAE (pt) | RMSE (pt) | Signed bias (pt) | Pearson r |
|---|---:|---:|---:|---:|
| LOSO mean other within-row SD | 2.165339 | 2.389829 | 0.109589 | -0.867186 |
| Same, c4-corrected | 2.220961 | 2.425434 | 0.319458 | -0.867186 |
| LOSO flatten other rows | 2.244069 | 2.440499 | 0.542148 | -0.968298 |
| Own within-row SD, in sample | 0.825567 | 1.077363 | 0.109589 | 0.867186 |

No improvement in aggregate agreement makes these per-seed predictions
successful. For the mean-other-SD definition, predictions are affine
decreasing functions of held-out SD, so their correlation with H is the
negative of the observed correlation of SD and H (when nondegenerate).
It does not follow without that observed positive SD-H association that
the LOSO correlation must be negative under every dataset/model.

Placement means and both raw and row-centered contrast covariance matrices
are fully exported. In placement order, shortened means are
`[-3.956,-0.872,1.248,-0.840,-4.676,-1.816]`; realistic-eval means are
`[-0.072,-0.240,-0.136,-0.010,-0.004]`. They are descriptive diagnostics of
possible placement structure, not precise population offset estimates.
Raw/contrast covariance ranks are 4/4 for both five-seed campaigns and 2/2
for both full endpoints. These data cannot validate an unrestricted
placement covariance model or independent/exchangeable residual assumptions.

## Full plug-in simulation, including scale re-estimation

For each design, 100,000 complete n-by-m matrices are generated using
`numpy.default_rng(20260908)`. The generator draws the normal batch first,
then the Student-t3 batch scaled by `1/sqrt(3)` to match variance. H and all
sample SDs are recomputed in every simulated dataset, then the complete
ratio-of-means statistic is recomputed. The JSON includes means, Monte Carlo
standard errors, and 2.5/97.5 sampling-reference percentiles. Reusing the
same seed/design gives the same unit-scale references for the full best and
final endpoints; this does not treat them as independent.

| Design | Normal corrected R mean | Normal reference 95% | t3 corrected R mean | t3 reference 95% |
|---|---:|---:|---:|---:|
| 5 x 6 | 0.999698 | [0.808659,1.197895] | 1.017699 | [0.725144,1.301353] |
| 5 x 5 | 1.000113 | [0.804615,1.194834] | 1.008921 | [0.732355,1.266337] |
| 3 x 4 | 1.000077 | [0.751016,1.231722] | 1.000817 | [0.695418,1.263871] |

Uncorrected R Monte Carlo standard errors are respectively
normal/t3: 0.000333/0.000492, 0.000338/0.000464, and
0.000432/0.000516. Corrected-R MCSEs are these multiplied by c4(m).
These are simulation errors for the reported mean, not uncertainty about
the empirical training experiment or model validity.

An additional fitted-scale sensitivity uses each observed row's `s_s/c4(m)`
as a fixed scale, applies the same scales to normal and variance-matched t3
draws, and again re-estimates SDs inside each replicate. These are expressly
plug-in fitted scales, not known population scales. Corrected means
normal/t3 are 0.999828/1.017695 (shortened), 1.000166/1.008769
(realistic), 0.999892/1.000776 (full best), and 0.999584/1.000715
(full final). Full reference intervals and MCSEs are in the JSON.

The normal-specific c4 multiplier is deliberately held fixed as part of the
same statistic in the t3 sensitivity; it is not an unbiased t3 correction.
The near-overlap shows why comparing only population expected-max constants
while holding a noisy estimated scale fixed is insufficient to discriminate
normal from t3 residuals. The legacy t3 constant-only underprediction is not
distributional validation. This conclusion is stronger than just removing
the word "exact": it identifies and tests the nuisance-scale mechanism.

## E1: preserve the frozen primary, separate the reviewer analysis

The complete 20 seeds 101--120, baseline plus placements
400/1400/2700/4000/5200, are used. Selection consults validation only, with
baseline first and ascending placements thereafter; first maximum wins.
17 seeds select a placement and three select baseline. All 20 selected test
deltas, including every exact zero, remain in the primary mean. Seed 117
selects p5200 but has zero test delta, so four primary deltas are zero even
though only three champions are baseline.

| Quantity | Estimate | Frozen historical interval | New seed-cluster percentile interval |
|---|---:|---:|---:|
| Selected test delta, primary | +0.124500 pt | t: [-0.040600,+0.289600] | [-0.018013,+0.283000] |
| Pooled correlation, 100 nested pairs | 0.455904 | Fisher-z: [0.285005,0.598712] | [0.226817,0.637196] |
| Within-seed centered correlation | 0.438599 | Not preregistered | [0.222714,0.615243] |

The frozen t interval is reproduced with the comparator's critical value
2.093; exact `t.ppf(.975,19)` gives [-0.040602,+0.289602], a negligible
rounding difference that is exposed instead of silently replacing the
historical calculation. Frozen Fisher-z treats the 100 pairs as independent;
it is retained for provenance and the frozen decision ladder, not endorsed
as cluster-robust inference. The frozen outcome remains rung 2, with the
primary mean interval including zero. The new primary and correlation
intervals resample 20 whole seed clusters 100,000 times with fixed seed
20260908 and recompute each statistic in each replicate. They are
reviewer-requested sensitivity analyses, not retroactively preregistered.

The historical interval conflict has distinct provenance:

- `logs_moneyshot/e1_verdict.json` and `scripts/moneyshot/e1_compare.py`
  exactly reproduce the appendix table's [-0.041,+0.290] t interval.
- The old `paper/sections/f2.tex` passage labels [-0.019,+0.279] a
  seed-clustered interval. The available historical analysis scripts do not
  supply its resampling seed/draw specification; that exact interval has
  not been reproduced. It is not the frozen t interval. The new deterministic
  bootstrap above replaces unverifiable bootstrap precision while retaining
  the primary t readout; no interval is chosen for a more favorable result.

## Exact covariance decomposition, not a baseline attribution

Under the uniform empirical distribution over 20 seeds and five placements
(all covariance terms use ddof=0), the identity is

`Cov(DeltaVal,DeltaTest) = mean_s Cov_j(DeltaVal,DeltaTest|s)
                          + Cov_s(mean_j DeltaVal,mean_j DeltaTest)`.

Numerically, in squared percentage points:

`0.0626699 = 0.0348025 + 0.0278674`.

Validation variance similarly splits as `0.1906306875 = 0.1076300 +
0.0830006875`; test variance as `0.09912376 = 0.0584996 + 0.04062416`.
The between-seed term is 44.47% of total covariance, but it contains both
seed-level effects and shared-baseline effects and is not an independently
identified baseline contribution. Comparing the two Pearson correlations
(0.456 versus 0.439) cannot justify "baseline contributes under 4%" because
the variances and normalization also change.

Positive pooled or centered Pearson correlation does not establish the
conditional regression structure, identical candidate distributions, or
independence needed for `rho c5 sigma_Test`. The frozen derived value
0.133495 pt is preserved only as a historical heuristic. E1 chooses among
baseline plus five placements, and baseline is the shared random reference;
neither treating this as five exchangeable candidates nor merely replacing
c5 by c6 repairs that mismatch. The directly measured selected-test primary
is the relevant bridge estimand, with a currently inconclusive mean interval.

## Verification and scope

Ten tests pass, including exact constants, joint seed bootstrap,
seed-offset invariance, distinction between LOSO definitions, covariance
identity, tie/zero handling, full E1 reproduction, deterministic simulation,
common-scale invariance of the fitted-scale simulation, and appendix
separation of frozen and reviewer-requested intervals. The script fails
closed on incomplete E1 rows, wrong seed/protocol, non-finite grids,
nonpositive row scales, or degenerate bootstrap correlations. No training,
remote action, commit, or historical artifact rewrite is performed.
