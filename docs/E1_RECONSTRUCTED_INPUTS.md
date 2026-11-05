# E1 split and corruption inputs

`artifacts/e1_reconstructed_inputs/` supplies twenty newly reconstructed input
files, one per registered seed 101–120. They contain the permutation, meta/train/
monitoring/selection IDs, corruption mask and nonzero class offsets from the
retained runner's NumPy recipe. All 120 run configurations were checked against
this recipe. The original runner created these arrays in memory, not as files.

These are **reconstructions, not recovered originals**. The manifest records
NumPy version and the array contents. Exact historical array equality cannot be
independently checked because no original array files were saved by that workflow.
No training was rerun and no result is changed by this export.

The official CIFAR-10 training ordering and clean labels are required to obtain
noisy labels: at each training index, add its offset modulo ten if its mask is
true; otherwise retain its clean label. Monitoring/selection labels stay clean.
Monitoring is the first 2,000 entries of the 4,000-entry selection split, as in
the original design. The official test split is separate. Dataset redistribution
and historical endpoint reproduction are not implied by these exported draws.

Run `python -B scripts/revision_e1_inputs.py --check` to compare every released
array against the recipe and validate the registered configurations. Omitting
`--check` exports the files, refusing to replace differing existing arrays.
