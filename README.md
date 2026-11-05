# Silent Unpairing

Code, evidence and manuscript source for

> **Silent Unpairing: How Mid-Training Probes Break Same-Seed Comparisons and Can Enable Selection Bias**
> Mingyu Du, Lu Han, Zhaoqi He, Jingyao Zhang, Junming Huang, Yun Li, Zerun Niu, Nguyen H. Tran, Ken-Tye Yong.
> *Transactions on Machine Learning Research (TMLR)*, 10/2026.
> Paper: https://openreview.net/forum?id=Xr662CduGR

A mid-training probe (an evaluation pass, a sampling callback, a monitoring
statistic) that draws from the same random stream as training silently shifts
every later draw. Two runs that share a seed and differ only in whether the
probe fires stop being a paired comparison, and a selection step that reads the
shifted metric can turn that into a one-sided gain. This repository holds the
source audit of 28 pinned codebases, the paired-run evidence, the CPU
reanalysis that regenerates the paper's statistics, and the LaTeX source.

## Quick start

Python 3.12. The analysis was recorded with NumPy 2.4.4 and SciPy 1.18.0.

```bash
pip install -r requirements.txt
python -B -m unittest discover -s tests -p "test_revision_*.py"
python -B -m unittest discover -s tests -p "test_nanogpt*.py"
```

Everything below runs on a CPU in a few minutes and needs no training:

```bash
python revision_results/nanogpt_final/revision_nanogpt_analyze.py --root revision_results/nanogpt_final
python scripts/nanogpt_supplementary.py --root revision_results/nanogpt_final
python scripts/nanogpt_prediction_benchmark.py --root revision_results/nanogpt_final
python scripts/revision_reanalysis.py --draws 100000 --seed 20260908
python scripts/revision_evidence.py --check
python scripts/revision_e1_inputs.py --check
python scripts/verify_historical_flip_states.py --checkpoint-only-release
python scripts/verify_repair_patches.py        # needs git, runs no training
python scripts/revision_render_statistics.py
python scripts/revision_generate_appendix.py
```

The PyTorch stream premises behind Proposition 1 need a CUDA device and
PyTorch (the paper used 2.11.0):

```bash
python scripts/rng_premise_check.py            # 13 assertions; exits non-zero on any failure
```

To build the paper:

```bash
cd paper_tmlr
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`paper_tmlr/appendix_body.tex` is generated from `paper/appendix.tex` by
`scripts/revision_generate_appendix.py`; edit the source, not the output.

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/`, `paper_tmlr/` | Manuscript source (sections shared by both builds; the TMLR build is `paper_tmlr/main.tex`) and figures |
| `scripts/` | Reanalysis, evidence and verification scripts, the RNG premise check, and the historical witness runners |
| `figs/` | Figure generators |
| `tests/` | Contracts for the reanalysis and evidence files |
| `revision_results/` | Registered statistical inputs, the nanoGPT campaign (726 sanitized run records, frozen analyzer and prediction), repair patches, ledgers |
| `revision_sources/` | Cited files from the audited codebases at their pinned commits, with their licenses; `external_source_index.json` holds pins and hashes |
| `artifacts/` | Reconstructed E1 inputs, historical MW-Net endpoint files, causal-trace manifest |
| `logs_*` | Per-run logs and configurations for the vision, MW-Net and NLP campaigns |
| `docs/` | Preregistration, provenance and scope notes for each evidence family |

## What is and is not included

This release is the artifact verified during review, with the changes listed
in [`RELEASE_CHANGES.md`](RELEASE_CHANGES.md). `PACKAGE_MANIFEST.json` records
the SHA-256 of every file as reviewed. Machine-local paths and host names in
copied records were replaced by `/artifact/local-path`; scientific values,
seeds, model hashes, third-party pins and attribution are unchanged.

- **Included:** canonical TeX and figure inputs; all 163 registered
  statistical inputs; per-cell NLP selection logs and configurations; model
  and meta-network hash records; cited files and license notices for the
  redistributable audited sources; the E1 preregistration text and a
  pre-launch chronology (`docs/E1_PROVENANCE.md`); the CPU reanalysis and
  evidence scripts with their tests.
- **Historical material:** two recovered MW-Net runners, three explanatory
  repair diffs with documented bases, and fifteen original model/meta-network
  endpoint files for five seeds (`docs/HISTORICAL_SOURCE_RELEASE.md`,
  `docs/HISTORICAL_FLIP_STATE_RECOVERY.md`). The endpoint verifier reads
  storage bytes without unpickling. These are endpoints, not full
  optimizer/RNG/cursor checkpoints. The `scripts/run_ext*.sh` files are launch
  records kept for provenance; their working directories are not part of the
  release, so they do not run as-is.
- **Not included:** the seventeen MW-Net dense-trace NPZ payloads
  (`docs/HISTORICAL_CAUSAL_TRACE_RELEASE.md`); the complete historical patch
  series; immutable split/noise arrays for every campaign; full
  optimizer/RNG/cursor checkpoints; externally attested preregistration
  timestamps. E1's runner generated its split/noise arrays in memory; twenty
  explicitly labelled reconstructions and a NumPy-only checker are included
  (`docs/E1_RECONSTRUCTED_INPUTS.md`).
- **NLP guard families** have no independent same-configuration repeat floors
  or weight-hash registry. Their metrics are descriptive and count as zero
  verified certificates.
- **Source audit is static.** Cited excerpts and pins do not execute the
  external codebases. This release is not a package for retraining every
  historical campaign.

## Third-party code

`revision_sources/` redistributes cited files from MIT, BSD-3-Clause and
Apache-2.0 projects, each next to its original license file. Six audited
repositories had no recoverable license text (Co-teaching, DISC, InfoBatch,
L2RW PyTorch, ProMix, Sel-CL), so their code is not redistributed; their
commit pins, cited line ranges and file hashes remain in
`revision_sources/external_source_index.json`. Those hashes were computed over
CRLF line endings: normalize a file fetched from GitHub to CRLF before
comparing.

## Citation

```bibtex
@article{du2026silent,
  title   = {Silent Unpairing: How Mid-Training Probes Break Same-Seed Comparisons and Can Enable Selection Bias},
  author  = {Du, Mingyu and Han, Lu and He, Zhaoqi and Zhang, Jingyao and Huang, Junming and Li, Yun and Niu, Zerun and Tran, Nguyen H. and Yong, Ken-Tye},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Xr662CduGR}
}
```

## License

Code written for this project is released under the [MIT License](LICENSE).
Files under `revision_sources/` keep their original licenses.
