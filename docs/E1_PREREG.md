# E1 Pre-registration — Val→Test Causal Bridge (Silent Unpairing, AAAI)

**Date frozen:** 2026-07-11 (before any GPU run of E1; analysis code frozen at
this commit).
**Runner:** `scripts/moneyshot/e1_valselect_mwnet.py` (derived, per the
provenance-freeze discipline, from the frozen
`scripts/moneyshot/realistic_eval_mwnet.py`, sha256
`704456a431e6dac2106b1815ddfb4c407eb69e5a430adfdfe8c9e121fdda2243`).
**Comparator:** `scripts/moneyshot/e1_compare.py` (deterministic; no RNG;
exact-table constants).

## Motivation — the bridge question

ext21 established that a shared-stream monitoring evaluation desynchronizes
same-seed pairs and that selecting the best of m probe placements **on the
reported metric itself** manufactures a gain ≈ c_m·σ (extreme-value law;
+0.45 pt measured at m=5, exactly 0.00 under `fork_rng`). That evidence does
not yet answer the sharper question a skeptic will ask: real practitioners do
not select on the reported test metric — they select by a **validation set**
and then report an untouched test set. E1 closes this gap causally: per seed,
the champion among {base + 5 placements} is chosen by a post-training
validation accuracy (val1), and the *reported* quantity is the untouched
official CIFAR-10 test accuracy. Only the genuine model-quality component of
the placement-induced divergence can transfer to the test set; the
val-measurement-noise component cannot. E1 therefore measures how much of the
manufactured selection gain survives the val→test bridge.

## Design

| Item | Value |
|---|---|
| Seeds | 20 seeds = 101..120 |
| Arms per seed | `base` (probe never fires) + `evalcb` at placements **400, 1400, 2700, 4000, 5200** |
| Training protocol | **ext21's, bit-compatible**: MW-Net (official `resnet.py`, pinned `3f1800ff`), CIFAR-10, 45,000 train / 1,000 meta, 40% uniform label noise, batch 100, 12 epochs (5,400 steps), lr 0.1 3-phase (×0.1 at epochs 8 and 10), SGD m=0.9 wd=5e-4 nesterov, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Desync source | ext21's real monitoring eval: `DataLoader(val, shuffle=True)`, 4 batches × 100 on the 2,000-example monitoring split; fires once at the placement step; 2 global-stream int64 draws per fire |
| Post-training holdout 1 (selection) | **val1 = 4,000 examples** = `perm[46000:50000]` of the per-seed split permutation — the *entire* remaining train-pool after 1k meta + 45k train (its first 2,000 are the monitoring split, read-only during training; its last 2,000 are touched by nothing). A 5,000-example val1 is infeasible under the frozen layout — 4,000 is the maximum, and is fixed here. Clean labels. Evaluated once, post-training, sequentially (no loader, no RNG). |
| Post-training holdout 2 (reported) | **official CIFAR-10 test split, 10,000 examples**, untouched during training and disjoint from val1 by torchvision split; evaluated exactly once at the very end, sequentially (no loader, no RNG) |
| Invariant | Both holdout evaluations run strictly after training and consume zero RNG, so E1's treatment arms are the **same experiment** as ext21 (bitwise: `state_sha256(e1) == state_sha256(frozen)` at equal CLI args, verified in the CPU smoke) |
| Output naming | `logs_moneyshot/e1_seed{S}_baseline.json`, `logs_moneyshot/e1_seed{S}_evalcb_p{P}.json` |
| Runner CLI (per arm) | `python -X utf8 scripts/moneyshot/e1_valselect_mwnet.py --repo $MWNET_REPO --data_root $VISION_DATA --epochs 12 --train_size 45000 --val_size 2000 --val1_size 4000 --batch_size 100 --noise 0.4 --lr 0.1 --eval_batch_size 100 --eval_batches 4 --seed $S --arm {base\|evalcb} --probe_step {999999\|$P} --out $OUT` |
| Analysis CLI | `python -X utf8 scripts/moneyshot/e1_compare.py --logs_dir logs_moneyshot --prefix e1 --seeds 101,...,120 --placements 400,1400,2700,4000,5200 --out logs_moneyshot/e1_verdict.json` — placements must be passed ascending; the tie-break's "earlier placement" is positional (argmax first-match) |

## Estimands (frozen)

**PRIMARY.** Mean over seeds of **[Test(champion_by_Val1) − Test(base)]**,
where the champion is selected *per seed* among {base + 5 placements} by
highest val1 accuracy (ties broken toward base, then earlier placement;
val1-only selection — the test metric is never consulted). Seeds where the
champion is base contribute exactly 0 **by construction — that is part of the
estimand; they are not excluded.** Reported as the paired mean with a t-based
95% CI over the 20 seeds (df = 19).

**SECONDARY (high-power).** Transfer coefficient **ρ** = Pearson correlation
over all (seed, placement) pairs of (ΔVal1_j, ΔTest_j), each delta taken
against the same-seed base: 20 × 5 = 100 paired points. Reported with a
Fisher-z 95% CI. Additionally: σ_Test = mean over seeds of the per-seed std
(ddof=1) of test deltas; c₅ = 1.1630 from the exact constant table
(E[max of 5 iid N(0,1)]); **derived check:** predicted mean test harvest
≈ ρ·c₅·σ_Test, printed next to the primary.

Also reported (descriptive, not gating): selection rate (fraction of seeds
with champion ≠ base), per-seed champion table, honest mean of test deltas
(no selection).

## Interpretation ladder (frozen BEFORE any result exists)

1. **Primary 95% CI entirely > 0** → "val-driven selection manufactures
   held-out test gains at this protocol" enters the paper.
2. **Primary CI contains 0 but ρ CI entirely > 0** → the paper reports the
   transfer coefficient and scopes the claim to *partial transfer,
   underpowered mean at this σ*.
3. **ρ CI contains 0** → boundary paragraph: no detectable transfer at this
   protocol; report honestly.

Rungs are evaluated top-down; the first matching rung applies. Any CI lying
entirely below 0 is read as its rung-3 outcome (no supported transfer;
reported honestly with its sign).

## Commitments

- **No estimator changes after results are read.** The two estimands, the
  tie-break rule, the CI constructions, and the constants above are final at
  this commit; the analysis code (`e1_compare.py`) is frozen at the prereg
  commit and may only be touched for crash bugs that do not alter estimator
  definitions (any such fix must be committed before re-reading results).
- **Budget:** ~39 h GPU solo (120 arms × 5,400 steps at the measured
  0.216 s/step), ~13–16 h at the usual concurrency. Launch when the running
  full-recipe campaign frees the GPU; resumable/skip-if-exists like ext21.
- **Hard kill-switch:** if E1 is not verified + launched by **2026-07-14
  EOD**, it is dropped from the submission (camera-ready material only).
- **Partial completion:** if any arms are unrecoverable by the kill-switch
  date, the analysis set is all COMPLETE seeds (a seed is complete iff its
  base and all five placement arms finished), with n and df adjusting
  accordingly; the incompleteness and the missing-arm list are disclosed in
  the paper. If fewer than 15 complete seeds exist, E1 is dropped from the
  submission.
