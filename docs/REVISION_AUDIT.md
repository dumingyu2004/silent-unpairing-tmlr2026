# Pinned-source re-audit, 2026-09-08

The full machine-readable register is `revision_results/audit_ledger.json`.
Each row identifies the recovered commit, exact file/line anchors, source-file
SHA-256, original verdict, revised verdict, and scope. Source hashes establish
provenance, not a training experiment. All 28 entries were checked against the
recovered sources. This is a static source-chain audit; external repositories
were not executed.

The count changes from **21 exposed / 5 guarded / 2 conditional** to
**22 exposed / 4 guarded / 2 conditional**. These counts concern the selected
paths and the RNG/cursor channel, not population prevalence or complete state
purity. A guarded RNG path can still mutate model state.

| Repository | Before | After | Rechecked mechanism / boundary |
|---|---|---|---|
| L2RW TF | exposed | exposed | Shared training iterator is consumed by evaluation |
| L2RW PyTorch | exposed | exposed | Unconditional shuffled-loader override |
| Meta-Weight-Net | exposed | exposed | Shared meta iterator and model initialization |
| DivideMix | exposed | exposed | Global NumPy GMM and mixup; intervention also changes GMM initialization |
| Co-teaching | exposed | exposed | Functional dropout active during evaluation |
| MAML | exposed | exposed | Mid-training shuffled evaluation loader |
| nanoGPT | exposed | exposed | Same CPU randint in training and evaluation |
| RENT | exposed | exposed | On-device multinomial; method operation, not assumed no-op |
| BHN | exposed | exposed | Evaluation iterator base seed and subsequent training loader |
| TopoFilter | exposed | exposed | Feature extraction reuses shuffled trainloader |
| CRUST | exposed | exposed | Evaluation loader base seed plus structural coreset changes |
| Sel-CL | exposed | exposed | Pair-selection rand and WeightedRandomSampler share CPU RNG |
| MLC | exposed | exposed | Shuffled validation and training loaders |
| ProMix | exposed | exposed | Nonpersistent evaluation iterator base seed, despite shuffle=False |
| UNICON | exposed | exposed | JSD evaluation iterator base seed, despite shuffle=False |
| DISC | exposed | exposed | Per-epoch evaluation loader and training loader |
| DDPM | exposed | exposed | Periodic sampling and training diffusion noise |
| timm | exposed | exposed | First evaluation iterator construction affects nondistributed training shuffle; persistent reuse is not a fresh draw every epoch |
| CleanRL | exposed | exposed | QDagger evaluation/training share Python random |
| LitGPT | exposed | exposed | Default nonzero LoRA dropout and example generation |
| InfoBatch | guarded | **exposed** | NumPy pruning reseed does not protect main-process torch augmentation from evaluation iterator base seed |
| DCGAN | guarded | guarded | Fixed-noise RNG path only; BatchNorm state mutation prevents state-purity claim |
| SB3 | guarded | guarded | Default deterministic policy, separate private-RNG environment; custom/global-RNG components and stochastic actions excluded |
| TRL | conditional | conditional | Backend and enabled evaluation matter; train shuffle must not be counted as an eval draw |
| OLMo | guarded | guarded | Private PCG data order and eval-mode forward; no blanket k=0 claim |
| SDXL | exposed | exposed | Image sampling and diffusion training noise |
| MMSelfSup | guarded | guarded | Private sampler and worker seed override on shipped path; extraction can consume CPU seed and hook mutates method state |
| diffusers | conditional | conditional | Specified pipeline sampling generator depends on supplied seed |

## Loader correction and genuine verdict change

`shuffle=False` only constrains the sampler. A newly constructed PyTorch
DataLoader iterator can draw a base seed even with zero workers. Reusing an
existing persistent-worker iterator is not fresh construction. Explicit
worker initialization may overwrite worker seeds; it does not restore the
main-process generator.

InfoBatch's `research/cifar_example.py:51-54` uses random flip/crop; its train
loader (`70-71`) leaves workers at the default zero and its test loader
(`76-77`) is unshuffled without a generator. Test runs between epochs
(`145-150,174-176`). The next training augmentation consumes main-process
CPU randomness displaced by that iterator construction. The pruning sampler's
`np.random.seed` (`research/infobatch.py:80-81`) protects only its own NumPy
sequence. The maintained example repeats this chain at
`examples/cifar_example.py:106-108,138-143,271`. Thus the original general
guarded verdict is incorrect for these shipped CIFAR paths.

MMSelfSup retains a scoped RNG guard: sampler uses its own generator,
augmentation runs in workers, and mmengine's worker initialization explicitly
sets NumPy, Python and torch seeds (`mmengine/dataset/utils.py:34-39`, pinned
dependency commit `a2e410bd48890c54f16c527d67c371ed8562facd`). Its hook also
changes labels and initializes the classifier; these are genuine method
operations and do not satisfy a state-pure probe assumption.

OLMo's `data_loader.py:544-564` dedicated torch generator belongs to
**get_mock_batch**, not a generic evaluation DataLoader. The actual audited
data-order mechanism is the private PCG RNG at `670-688`; eval disables
training-only forward draws. This distinction prevents an unrelated generator
from being offered as proof of evaluation isolation.

DCGAN's fixed-noise visualization (`dcgan/main.py:268`) calls a train-mode
generator containing BatchNorm (`133-145`). Keep its RNG guard, but do not call
that operation state-pure or infer identical full state. SB3's separate eval
environment must own its randomness; merely being a separate Python object
does not prove isolation of an arbitrary custom environment.

Rebuild/check: `python scripts/revision_evidence.py`, then
`python scripts/revision_evidence.py --check`.
