# Historical MW-Net trace payload availability

This release does not include the seventeen original MW-Net trace NPZ files
or the separate `historical_trace_payloads.zip` companion. The retained trace
manifest describes the historical baseline, shared-evaluation and isolated-
evaluation triplets for seeds 1-5 and the baseline repeats for seeds 1 and 2.
It is a provenance record, not a substitute for the tensor payloads.

The historical companion was designed to contain those NPZ files, anonymized
result records, an original-payload manifest and a NumPy-only checker. That
companion is not delivered with this package, so an independent comparison of
the dense-trace tensor bytes cannot be reproduced from this release alone.
The recorded dense-trace findings remain historical results, rather than a
new check performed when preparing this release.

The recorded model/meta-network tensors and trace arrays do not establish
every-update capture of complete optimizer, RNG or cursor state. They are not
new experimental runs and do not increase the number of independent seeds.
The current package manifest is the authoritative list of delivered files.
