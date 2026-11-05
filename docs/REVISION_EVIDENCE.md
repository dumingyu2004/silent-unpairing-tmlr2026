# Evidence-resolution audit, 2026-09-08

`revision_results/evidence_ledger.json` records each locally recovered NLP
selection cell and individually checked MW-Net weight-hash pairs. It separates
within-arm selected/raw metric equality, cross-arm weight equality,
same-configuration repeat floors, and full-state equality. Artifact checksums
are not training-state hashes. Equality of two recorded model hashes does not
establish equality of optimizer state, random generators, or data cursors.

## Guard certificate correction

The old figure script's eight NLP “hash-verified” cells came from interpreting
the manuscript table, not a per-cell hash registry. Its source explicitly
acknowledged this at `figs/make_fig_r3_evidence_ledger.py:267-295` before the
revision. Recovered `logs_nlp_gateb` and `logs_nlp_payout_det` contain **24**
nonempty per-cell selection logs (9 tuned, 15 under-tuned). These logs record
champion identities and selected/raw metrics, but contain no independently
verified weight-hash pairs or same-configuration reruns. Thus the evidence
ledger certifies **zero** of these cells at weight or full-state level.

All historical metrics remain reportable at their recorded resolution. Do not
delete their outcomes merely because their bitwise certification is withdrawn.
The eight previously inferred bitwise NLP entries must move to the
metric/selection-log category; missing independent floors are not inferred
from zero deltas. The tenth/vision-family cells also require a direct per-cell
registry before any independent hash-verification claim.

## What direct hashes do establish

The ledger directly compares full-recipe isolated arms at placement 23520,
seeds 1–3, with their never-probed MW-Net baselines, using the model and meta-net
`state_sha256` fields. It also compares the three same-configuration baseline
repeat files separately. The standard-evaluation campaign has two recovered
isolated arms at placement 1400 (seeds 1–2), compared with their reference files.
These support the **recorded model/meta-network weights and buffers** only.
They must not be counted as full optimizer/RNG/cursor trajectory verification.

The ledger does not turn manuscript-level campaign summaries into new direct
hash evidence. HF parameter-difference logs and causal-trace archives have
different resolutions and should retain their own explicit definitions; no
NLP certificate count is borrowed from those unrelated campaigns.

No new GPU training or NLP rerun was performed by this audit. The repair here
is an evidence correction and scope reduction. Run `python
scripts/revision_evidence.py --check` to verify that the ledgers still match
the recovered records and cited source files.
