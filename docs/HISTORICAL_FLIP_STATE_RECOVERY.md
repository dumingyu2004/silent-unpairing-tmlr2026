# Historical MW-Net Flip-State Recovery

This directory curates the fifteen final-state checkpoints from the historical
July 9 handoff archive that correspond to the five shortened-protocol repeat
floors and five isolated-probe comparisons:

- `artifacts/historical_flip_states/floor_seed{1..5}_a_mwnet.pt`
- `artifacts/historical_flip_states/floor_seed{1..5}_b_mwnet.pt`
- `artifacts/historical_flip_states/cure_seed{1..5}_mwnet_probe_isolated.pt`

The source archive is identified in the manifest by basename, byte count and
SHA-256 only. Its machine-local path is intentionally not recorded. Recovery
selected those fifteen exact ZIP members from the 3,399-member archive; it did
not extract the archive wholesale.

## Safe verification

`scripts/verify_historical_flip_states.py` treats every `.pt` file as an opaque
nested ZIP. It validates ZIP CRCs and hashes the raw bytes of all 161 numbered
tensor-storage members in each checkpoint. It does not import PyTorch, execute
pickle opcodes, instantiate a model, or call `pickle.load`/`torch.load`.
It also uses `pickletools.genops` to inspect string operands in `data.pkl`
without constructing objects. All fifteen scans contain no author-machine
paths, user identifiers, or the known author-run hostnames checked by the
verifier. The original checkpoint bytes are therefore retained unchanged.

For every seed, all 161 storage members are byte-identical across floor A,
floor B and the isolated-probe cure. Whole `.pt` hashes differ because PyTorch
records container-specific metadata and root names. The storage comparison is
therefore the relevant container-independent byte check.

The verifier also establishes that:

- the fifteen associated `logs_flip_demo/*.json` records are byte-identical to
  their historical archive members;
- their recorded final accuracy, model `state_sha256`, and meta-network
  `vnet_state_sha256` agree within each seed triplet, and every cure record says
  `probe_fired: true`;
- `scripts/flip_demo_mwnet.py` is byte-identical to its historical archive
  member; its save contract contains only `model.state_dict()` and
  `vnet.state_dict()`.

Run the full private-workspace check, including exact original JSON/source
sidecar hashes, with:

```text
python scripts/verify_historical_flip_states.py
```

Anonymous packaging may redact machine-local paths inside copied JSON/source
sidecars. Those transformed copies must not be represented as byte-identical to
the originals, and the original digests in `manifest.json` must not be silently
rewritten. A released package should preserve the fifteen `.pt` files and this
manifest byte-for-byte, then use the checkpoint-only mode:

```text
python scripts/verify_historical_flip_states.py --checkpoint-only-release
```

That mode still verifies every checkpoint file hash, every raw tensor-storage
member, all five within-seed triplets, and the non-executing pickle-string
anonymity scan. It intentionally reports zero original JSON records and no
original source copy as verified because release redaction may have changed
those sidecars. If release-sidecar verification is desired, record their
post-redaction hashes in a separate release manifest rather than replacing the
historical digests here.

The complete file, JSON, source and per-seed storage fingerprints are in
`artifacts/historical_flip_states/manifest.json`.

## Evidence boundary

These payloads support final model and meta-network parameter/buffer equality
for the five tested shortened-protocol seeds. They do **not** contain or verify
optimizer state, RNG state, data-cursor state, or an every-update trajectory.
They do not upgrade this evidence to full training-state identity, recover
other campaigns, or provide independent preregistration attestation.
