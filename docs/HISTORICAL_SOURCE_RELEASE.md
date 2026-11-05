# Historical source and repair-delta release

This note distinguishes recovered historical material from files generated
during the revision. It does not claim an independent timestamp, a complete
historical repository, or the ability to retrain every historical campaign.
No training was rerun to prepare this release.

## Provenance terms

- **Exact local-record reconstruction** means that a complete retained Write
  payload was decoded byte for byte and its SHA-256 was corroborated by a
  separate retained local hash record. These are two local records, not
  independent external attestation.
- **Exact historical Git source** means that the released source matches a
  surviving file blob in a retained historical commit.
- **Revision-generated explanatory patch** means a minimal diff prepared now
  against an identified preserved base. It is not represented as the
  contemporaneous patch used to launch the historical run.
- **Release copy** means the copy in the anonymous reviewer archive. Its digest
  is recorded in `PACKAGE_MANIFEST.json`. When anonymization changes an
  author-local path, the historical-original digest below intentionally does
  not describe the released bytes.

## Recovered runner sources

| Path | Provenance | UTF-8 bytes | Historical-original SHA-256 |
|---|---|---:|---|
| `scripts/moneyshot/moneyshot_mwnet.py` | exact local-record reconstruction | 10,091 | `d8b82d301767f0f2fa4ebf3d954d2ab9781afd18a7f29cd8c4a2e14080d863eb` |
| `scripts/moneyshot/moneyshot_compare.py` | exact local-record reconstruction | 9,783 | `f6de95db6f1ac21190a5a5e1180ab1bb4533b4af211836ec3aa03e25009ed98c` |

The complete payloads and the separate hash record were retained in local
session histories. Full, non-truncated Read records also agree with the
reconstructed contents. The original temporary runtime files no longer exist,
so this is source-content recovery rather than recovery of the original
filesystem objects.

The exact MW-Net runner contains an author-local default data path. The private
source retains its historical bytes; the anonymous packaging step replaces
that path in the released copy. Consequently, the table above remains the
historical-original record while `PACKAGE_MANIFEST.json` is authoritative for
the released-copy digest. The repair hunk is elsewhere in the file and remains
applicable to the anonymized copy.

## Curated repair deltas

All three files below are **revision-generated explanatory patches**. Their
machine-readable hashes and base information are in
`revision_results/repair_patches/manifest.json`.

| Patch | Preserved base | What the minimal diff expresses | Evidence and scope boundary |
|---|---|---|---|
| `mwnet_probe_rng_isolation.patch` | exact reconstructed `scripts/moneyshot/moneyshot_mwnet.py`, SHA-256 `d8b82d301767f0f2fa4ebf3d954d2ab9781afd18a7f29cd8c4a2e14080d863eb` | Wrap the unused global-CPU `randperm` probe in `torch.random.fork_rng`. | The recovered runner already contains a separate isolated arm implementing this mechanism. The patch records the call-site repair, not the full history of the runner or upstream MW-Net. |
| `bhn_test_loader_generator.patch` | `revision_sources/bhn/BHN.py`, upstream commit `2b09d3a376494371b9dd852e4742aef756b8fd93`, SHA-256 `e37c95d6b90b32667d48bd4b3f3217c6c460f57efba44f1a2b862fc8e97be8d6` | Give the test loader an independent, deterministically seeded `torch.Generator`. | `logs_flip_demo/bhn_witness.json` records the exact stock/isolated lines, duplicate floors within each configuration, and divergence beginning at epoch 2. The loader iterator obtains a base seed even when evaluation is unshuffled. The historical harness also had common portability, determinism, and state-dump changes; they are not silently attributed to this one-line patch. |
| `dividemix_gmm_random_state.patch` | `revision_sources/dividemix/Train_cifar.py`, upstream commit `d9d3058fa69a952463b896f84730378cdee6ec39`, SHA-256 `3fa169263c3f3b9cbd6b031017394a66d0d999c7e2f53e5cab01d2d241f5df89` | Add `random_state=0` to the witnessed `GaussianMixture` call. | `scripts/run_ext10_dividemix_witness.sh` records this as the sole arm contrast plus common harness changes; the four logs show equal within-configuration final hashes and different cross-configuration hashes. This is a configuration contrast: pinning `random_state` changes GMM initialization as well as global NumPy stream consumption, so it is not a pure no-op isolation result. |

The three patches apply cleanly to the stated bases. A local verification
reported these historical-original post-apply SHA-256 values:

| Patch | Locally verified post-apply SHA-256 |
|---|---|
| MW-Net | `b4513f2c9cffe04863b71e67b7a31ae1329535217f3dcdd7ef0fc25f480fe2e6` |
| BHN | `b954131157e5193846d247831dce887d5945df4ad343acd82979735dff4e9add` |
| DivideMix | `ca489f2bbe004eddcb5dd1077b94223d7c72932f73a16450c2475140b40092e2` |

These post-apply digests are verification records for the stated local base
bytes. Anonymous release validation must recompute the base and applied
digests from the extracted archive, because path redaction changes the MW-Net
base bytes. The released base and patch bytes remain independently covered by
`PACKAGE_MANIFEST.json`.

## Surviving HF repair source

`scripts/hf_pairing_probe_iso.py` is not a reconstruction. It is byte-identical
to the file blob added by historical commit
`7d424aabe22da6abb18bc219976da26ce98206e0`:

- Git blob: `4f3d5f1bfa0fb536eeb9c9925a4c01a0fdd7fdfc`
- File bytes: 6,418
- SHA-256: `11f5e9e04aaeb9638454bfbd885ca8d9718730d06b8661183975b3df49b604c8`

That commit added a sixth-arm `fork_rng` source while leaving the frozen
five-arm runner unchanged. Because the exact historical source survives, no
fourth reconstructed patch is presented.

## Verification

Source syntax can be checked without importing training dependencies:

```bash
python -B -c "import ast,pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ('scripts/moneyshot/moneyshot_mwnet.py','scripts/moneyshot/moneyshot_compare.py','scripts/verify_repair_patches.py')]"
```

The patch checker uses temporary copies only; it does not import model code or
run training:

```bash
python -B scripts/verify_repair_patches.py
```

It runs `git apply --check`, applies each patch in an isolated temporary
repository, and prints the actual base and post-apply digests. On the anonymous
archive it reports whether the MW-Net base differs from the historical
original; package provenance, rather than this checker, identifies the reason
for that difference. The two third-party bases must retain their declared
hashes.

## What this release does and does not close

It closes the source-content gap for the two missing money-shot scripts and
supplies three small, testable, source-backed repair deltas. It also preserves
the stronger Git provenance available for the HF isolation source.

It does **not** recover contemporaneous original patch files for MW-Net, BHN,
or DivideMix; prove that the minimal diffs were the only historical
implementation changes; supply every common harness edit, dataset, environment,
optimizer/RNG/cursor checkpoint, or split array; establish independent
registration; or make complete historical retraining available.
