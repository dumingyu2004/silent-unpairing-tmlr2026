# Changes from the reviewed artifact

`PACKAGE_MANIFEST.json` lists the SHA-256 of every file in the artifact that
was verified during review. This public release differs from it only as listed
here; every other file is byte-identical to its manifest entry. No scientific
input, run record, frozen analyzer, model hash or registered statistic changed,
and every test and check in the README passes on the release.

## Manuscript: code citations aligned with the code

Every `file:line`, identifier and command quoted in the paper was rechecked
against the pinned sources (the redistributed files, the non-redistributed
files at their pinned commits, and transformers 5.12.1 / accelerate 1.14.0 /
torch 2.11.0). Corrections, applied to `paper/appendix.tex` and
`paper/sections/f2.tex` (with `paper_tmlr/appendix_body.tex` regenerated):

| Location | Was | Now | Reason |
| --- | --- | --- | --- |
| litgpt row | `finetune/lora.py:81` | `lora.py:82`; default `lora_dropout=0.05`, `finetune/lora.py:58` | line 81 is an unrelated argument; the `nn.Dropout` is in `litgpt/lora.py` |
| DISC row | `DISC.py:308`, `main.py:189` | `losses/loss_mixup.py:27`, called from `DISC.py:308`; `main.py:204` | the `randperm` lives in `Mixup.forward`; line 189 is the WebVision branch, 204 the default path |
| ProMix row | `(:427)` after `dataloader_cifarn.py` | `Train_promix.py:427` | `dataloader_cifarn.py` has 305 lines; the loop is in `Train_promix.py` |
| Sel-CL row | `torch.rand(50000,128)` | `torch.rand(len(trainloader.dataset), args.low_dim)`, 50,000 x 128 on CIFAR | quote the code, state the evaluated shape separately |
| CleanRL row | `qdagger_dqn_...:362` | `qdagger_dqn_atari_impalacnn.py:362` | full file name |
| accelerate row | `initial_seed + epoch` | `self.epoch + self.initial_seed` | as written at `accelerate/data_loader.py:99` |
| cross-seed remark | `prop2_fit.py`, `make_fig_r1_crossseed.py` | `scripts/prop2_fit.py`, `figs/make_fig_r1_crossseed.py` | repository paths |
| Appendix D.1, Sec. 2.5 | `fork_rng(devices=[0])` | `fork_rng(devices=[device])` (`devices=[]` on CPU) | as written in the MW-Net runners and repair patch |
| Sec. 2.5 | `torch.Generator().manual_seed(s)` | `torch.Generator().manual_seed(0)` | as written in the BHN repair patch |
| Appendix D.1, deterministic kernels | every paired run sets `allow_tf32 = False` on both paths | deterministic algorithms (`warn_only=True` in the DivideMix and BHN witnesses) and `CUBLAS_WORKSPACE_CONFIG` for every run; TF32 explicitly disabled only in the nanoGPT campaign, PyTorch defaults elsewhere, with the repeat floors measured under those settings | only the nanoGPT arm code sets the TF32 flags, and Sec. 2.5 already reports a CIFAR-100 regime run with TF32 |
| Appendix D.1, NLP harness | "reaches the same state through `enable_full_determinism(seed)`" | reaches deterministic execution through it, which sets `CUBLAS_WORKSPACE_CONFIG=":16:8"` itself | `transformers` 5.12.1 `enable_full_determinism` overwrites the variable with `:16:8`, not `:4096:8` |

## Manuscript: camera-ready wording

- The anonymous-supplement wording now points to this repository: the
  introduction and Appendix D.6 give its URL, D.6 says it is the reviewed
  artifact plus the changes in this file, "freshly unpacked copy" became
  "fresh clone", and D.5 names `revision_results/nanogpt_final/supplementary.json`
  instead of "the supplemental JSON". The abstract ends with the code URL.
- Leftovers from the AAAI supplement build now say "appendix": the appendix
  opening sentence, "supplement-scoped" (now "appendix-scoped"), and two
  "in the supplement" pointers. The "reviewer-requested" analysis keeps its
  name because the code and `revision_results/statistics.json` call it
  `reviewer_bootstrap`.
- PDF metadata (title, authors, subject) set with `\hypersetup`.
- An epigraph (Feynman, "Cargo Cult Science", 1974) opens the introduction,
  set with standard LaTeX (no new package).

## Figures: overlapping labels fixed

No plotted value changed; each regenerated figure was first checked to
reproduce the reviewed PDF pixel for pixel before its layout was edited.

- Figure 1 (`figs/make_fig1.py`): the seed connectors leave the seed box at
  its top and bottom so they no longer cut through "training RNG stream";
  the Run B title is set on two lines so the probe arrow no longer crosses it;
  "counterfactual A path" moved onto the dashed line's extension, clear of
  the step connectors.
- Figure 2: its generator `figs/make_fig3_hf_boundary.py` and the eleven
  probe records it reads (`logs_hf_pairing_probe/result_v2.json`,
  `a1b_{rte,mrpc}_s41..45.json`) were missing from the reviewed artifact and
  are restored from the authors' archive. Their data validation is unchanged.
  The rotated x labels in panel A overlapped and are now short horizontal
  labels (the caption names each path in full); the 1e-4 scale moved into
  the y labels; "task--seed" now prints an en dash.
- Figure 4a: new wrapper `figs/make_fig_r1_crossseed_camera.py` calls the
  hash-recorded `figs/make_fig_r1_crossseed.py` unchanged, moves the s4
  label to the right of its marker (it overlapped s2) and sets the side
  statistics with uniform indentation and a true minus sign.

## Code

- `scripts/rng_premise_check.py`: the 13 premise checks now run as assertions
  with a pass count and a non-zero exit on failure (previously they printed
  booleans only). The checks themselves are unchanged; 13/13 pass on
  torch 2.11.0+cu128.
- Lint-level cleanup (unused imports, import order, redundant encoding
  declarations) in files that no frozen record hashes: the figure
  generators except `figs/make_fig_r1_crossseed.py`, `scripts/revision_*.py`
  (except `revision_reanalysis.py`), `scripts/verify_*.py`,
  `scripts/nanogpt_supplementary_plot.py`, and `tests/`. Behaviour unchanged.
- `scripts/run_ext10_dividemix_witness.sh`, `scripts/run_ext11_bhn_witness.sh`:
  comments translated to English and marked as non-runnable launch records.
  Commands unchanged.
- Files whose SHA-256 is recorded by a frozen analysis, provenance record or
  repair manifest were left byte-identical, including the historical MW-Net
  runners and the nanoGPT campaign code.

## Repository files

- Removed three figures no source file referenced:
  `paper_tmlr/figs/app_hf_fivearm.pdf`, `fig3_detector.pdf`, `fig4_minefield.pdf`.
- `paper_tmlr/main.tex` switched from the anonymous submission mode to
  `[accepted]`, with the author block, the 10/2026 header and the OpenReview
  link filled in; `paper_tmlr/main.pdf` rebuilt from this source (45 pages;
  the author block adds one page before the References).
- `README.md` rewritten for the public release; added `LICENSE`,
  `CITATION.cff`, `requirements.txt`, `.gitignore`, `.gitattributes` and this
  file. `.gitattributes` stores every file byte-for-byte so that checkouts
  match the hashes in `PACKAGE_MANIFEST.json` and the provenance records.
