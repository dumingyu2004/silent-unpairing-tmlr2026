# E1 protocol and execution provenance

The retained historical records place the E1 protocol before the recorded
campaign launch. Replaying its initial Write and four unique Edit events
reproduces `docs/E1_PREREG.md` exactly after normalizing line endings. Replaying
the runner's Write and one docstring Edit similarly reproduces the original
retained `scripts/moneyshot/e1_valselect_mwnet.py`. These comparisons were performed
before anonymous-release path sanitization; the released runner may therefore
differ in local default paths, not in its experiment logic.

The protocol's last retained edit is dated **10 July 2026, 14:56:39.888 UTC**
(11 July in Sydney). The recorded launch command is dated **12 July 2026,
17:39:29.898 UTC**, approximately **50.7 hours later**. Successful launch output
was recorded at 17:40:03.093 UTC and includes the repeat-floor gate.

The anonymous event ledger in `revision_results/e1_provenance.json` contains
the relevant write/edit times and these minimal launch excerpts:

```text
nohup bash e1_launch.sh 12 > e1_master.log 2>&1 &
E1 LAUNCHED, master pid 17055
== phase 0: repeat-floor gate on THIS machine ==
```

Remote account/address information and surrounding filesystem commands are
omitted. Full historical conversations are not distributed. This ledger is an
author-maintained extract, not an independently timestamped registration or
external certification. It documents the available chronology without claiming
independent verification of the original clocks or records.

## Split and corruption inputs

The retained runner creates split indices and label-corruption arrays in memory
and writes configuration/result JSON, rather than separate array files. The
available launcher contains no separate array-export step. No original array
payload was found in the expanded search. Thus the documented workflow did not
save these arrays separately; they should not be described as recovered files.
The source and recorded seeds/settings describe their generation. Any future
reconstruction must be labelled as such and checked against the original input
data and software environment. The supplement now supplies explicitly labelled
reconstructed split IDs and corruption draws; see `docs/E1_RECONSTRUCTED_INPUTS.md`.
These do not change the original experiment records or establish independent
equality to unsaved historical arrays.

## Checks

`python -B scripts/revision_e1_provenance.py --check` checks the released ledger's
chronological consistency and referenced-file presence without private inputs.
It does **not** authenticate the underlying historical records. The same script
accepts `--source-history` and `--launch-history` for the authors' original-record
replay; those private inputs are not required for reproducing the paper's
statistical analyses. Its output explicitly scopes source equality to the
original retained files before release sanitization.
