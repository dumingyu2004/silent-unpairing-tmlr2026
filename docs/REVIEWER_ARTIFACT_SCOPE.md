# Reviewer artifact: current scope

The package manifest is the authoritative list of included files. Older local
search inventories are not shipped as current claims of absence. This release
contains analysis inputs and code, recovered historical material and explicitly
labelled revision-generated material; those categories remain distinct.

| Reviewer-requested material | Included evidence and scope |
|---|---|
| Raw results and exact configurations | Historical harvest and E1 inputs, per-cell evidence records, and all 726 new nanoGPT pilot/main records. Some older harvest grids survive as aggregate matrices rather than separately recovered runs; the statistical lineage identifies them. |
| Seeds and split/noise files | Recorded historical seeds/configurations, retained E1 runner and twenty recipe-checked reconstructed E1 input files. The original E1 runner did not save these arrays separately. Clean dataset labels remain an external input. |
| Repair diffs and sources | Three revision-generated explanatory diffs against documented source bases, plus the two recovered historical MW-Net moneyshot runners. The source note separates original recovered bytes from anonymized release copies and explanatory changes. |
| Repeat and model comparison evidence | Fifteen original MW-Net endpoint files across five baseline/repeat/isolated triplets, recorded historical comparisons, and the new nanoGPT per-path repeat records. Stored model/meta-network tensor bytes can be compared without deserializing the checkpoint. They are not complete optimizer/RNG/cursor snapshots. |
| Protocols and timestamps | Retained protocols, freeze/execution records and a redacted E1 chronology placing the retained protocol about 51 hours before launch. These are local records, not independent external registration. |
| Upstream versions and attribution | Exact source pins/citations, available licensed source excerpts and notices. Six source roots without recovered license text remain metadata-only references, explicitly marked in the release source-availability report. |
| Additional statistics | Full fit-and-select simulations, E1 decomposition, nanoGPT supplementary diagnostics, per-seed figure and post-readout Gaussian-versus-empirical prediction benchmark, with code and numerical outputs. |

## Verification boundaries

The main README lists the checks that run from an unpacked copy. The release
manifest describes actual released bytes. Historical manifests retain original
file digests; anonymized JSON/source copies are not silently relabelled as
originals. The checkpoint-only release check validates the untouched binaries
and their storage comparisons while explicitly skipping original-sidecar byte
identity. Released sidecar integrity is covered separately by the package
manifest and regenerated analysis/evidence checks.

This meets the practical request for inspectable reported results, inputs,
repair explanations and verification code. It is not a claim that every
historical training run can be reproduced with no external data or dependencies,
that all original checkpoints survive, or that local timestamps are externally
attested. The full original repair history is not manufactured from new diffs.

This release does not include the seventeen original MW-Net trace NPZ
archives or the separate `historical_trace_payloads.zip` companion. The
retained trace manifest and historical release note describe those archives
but do not make their tensor bytes available for an independent dense-trace
check from this package. The much larger auxiliary CaffeNet archive is not
used as a new validation experiment or included here. Only files listed in
this release's manifest are delivered with this ZIP.
