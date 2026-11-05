#!/bin/bash
# ext17: MW-Net shared-stream-probe HARVEST money-shot at a believable protocol.
#
# Goal: upgrade the shortened-protocol harvest (10k subset / 20 epochs) into a
#   convincing demonstration that a realistic model-selection decision (select
#   best-of-N probe placements on a shared-stream validation) MANUFACTURES a
#   quantifiable apparent gain that VANISHES under fork_rng isolation -- run on
#   the FULL CIFAR-10 train set (49k, num_meta=1000, like MW-Net.py) with a
#   real 3-phase LR schedule + held-out TEST-set eval.
#
# Framing discipline: this is NOT a claim that MW-Net's published result is
#   wrong. Claim = "selection over the shared-stream probe manufactures +X pt of
#   apparent gain; under fork_rng it drops to ~0; the isolated comparison is the
#   trustworthy one." NO claim about MW-Net's true ranking vs SGD.
#
# PRE-REGISTERED comparison (moneyshot_compare.py, fixed before launch):
#   Condition A (shared stream / contaminated): Delta[s,p]=100*(acc_probe - acc_base);
#     manufactured_gain = mean_s max_p Delta - mean_{s,p} Delta.
#   Condition B (fork_rng isolated / cure): same selection; must be ~0 (each
#     isolated placement is bitwise-identical to the un-probed baseline).
#   Extreme-value prediction: c_N * sigma_bar (c_N = E[max-mean of N iid N(0,1)]).
#   Verdicts: V1 floor mwnet==mwnet_rep bitwise; V2 cure iso==baseline bitwise;
#     V3 harvest gain>0 & within [0.7,1.4]x c_N*sigma; V4 headline = A gain -> B ~0.
#
# Budget (measured 0.23 s/step on this RTX-5080 Laptop, deterministic kernels):
#   default EPOCHS=10 -> 4900 steps/arm ~19 min/arm; grid below = 19 meta-arms +
#   3 sgd ~= 6.2 h dedicated (~8 h if GPU shared). EPOCHS=12 -> ~7.5 h dedicated.
#   The FULL official 120-epoch run would be ~3.8 h/ARM (~86 h for this grid) --
#   infeasible overnight; documented as out of scope.
#
# Determinism: runner sets use_deterministic_algorithms(True); CUBLAS_WORKSPACE_
#   CONFIG exported below; every arm is single-process + atomic JSON write, so
#   this chain is crash-safe and resumable (skips any arm whose JSON exists).
set -u
cd "/c/artifact/local-path" || exit 1
exec >> logs_nlp_gateb/overnight_p0_console.log 2>&1
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- knobs (controller may edit before launch) --------------------------------
SEEDS="1 2 3"
PLACEMENTS="200 800 2000 4000"     # N=4 probe steps, all < EPOCHS*490 steps
TRAIN_SIZE=49000                    # full train (1000 meta held out = 50000)
EPOCHS=10                           # 10 -> ~6.2 h; 12 -> ~7.5 h (dedicated GPU)
ISO_PLACE=200                       # placement used for the per-seed cure witness
FLOOR_SEED=1                        # seed that also runs mwnet_rep (bitwise floor)
NOISE=0.4
# -------------------------------------------------------------------------------

MSDIR="/artifact/local-path"
REPO="/artifact/local-path"
DATA="/artifact/local-path"  # explicit Windows path: the inline integrity check embeds this in a python -c string, where git-bash does NOT auto-mangle /c/ -> C:\ (bare args to python.exe are mangled, string contents are not)
OUT_DIR="logs_flip_demo/moneyshot"
OUT_JSON="$OUT_DIR/moneyshot_verdict.json"
RUNNER="$MSDIR/moneyshot_mwnet.py"
CMP="$MSDIR/moneyshot_compare.py"

mkdir -p "$OUT_DIR"
echo "=== EXT17 MW-NET MONEY-SHOT START $(date) ==="
echo "    SEEDS=[$SEEDS] PLACEMENTS=[$PLACEMENTS] TRAIN_SIZE=$TRAIN_SIZE EPOCHS=$EPOCHS"

if [ -f "$OUT_JSON" ]; then
  echo "=== EXT17 SKIP: $OUT_JSON already exists $(date) ==="; exit 0
fi

# data integrity check (no download)
python -X utf8 -c "from torchvision import datasets; datasets.CIFAR10('$DATA',train=True,download=False); datasets.CIFAR10('$DATA',train=False,download=False); print('EXT17 CIFAR10 integrity OK')"
echo "--- DATA READY exit=$? $(date) ---"

# run one arm; skip if its JSON already exists (resumable)
run_arm () {  # $1=seed $2=arm $3=probe_step $4=out_basename
  local out="$OUT_DIR/$4.json"
  if [ -f "$out" ]; then echo "SKIP $4 (exists)"; return 0; fi
  echo "--- RUN $4  seed$1 $2 p$3 $(date) ---"
  python -X utf8 "$RUNNER" --repo "$REPO" --data_root "$DATA" \
    --seed "$1" --arm "$2" --probe_step "$3" \
    --epochs "$EPOCHS" --train_size "$TRAIN_SIZE" --noise "$NOISE" \
    --out "$out"
  echo "--- DONE $4 exit=$? $(date) ---"
}

for s in $SEEDS; do
  # honest no-probe baseline (this IS Condition B's every-placement result too)
  run_arm "$s" mwnet 999999999 "ms_seed${s}_baseline"
  # bitwise repeat floor (one seed is enough; add more by widening FLOOR_SEED)
  if [ "$s" = "$FLOOR_SEED" ]; then run_arm "$s" mwnet_rep 999999999 "ms_seed${s}_rep"; fi
  # SGD comparison arm (cheap; for the apparent-gain-vs-baseline narrative)
  run_arm "$s" sgd_baseline 999999999 "ms_seed${s}_sgd"
  # Condition A: shared-stream probe at each placement
  for p in $PLACEMENTS; do run_arm "$s" mwnet_probe "$p" "ms_seed${s}_probe_p${p}"; done
  # Condition B: fork_rng-isolated cure witness (one placement per seed)
  run_arm "$s" mwnet_probe_isolated "$ISO_PLACE" "ms_seed${s}_iso_p${ISO_PLACE}"
done

echo "=== EXT17 COMPARATOR (pre-registered verdict) $(date) ==="
CMP_PLACE="$(echo $PLACEMENTS | tr ' ' ',')"
CMP_SEEDS="$(echo $SEEDS | tr ' ' ',')"
python -X utf8 "$CMP" --logs_dir "$OUT_DIR" --prefix ms \
  --seeds "$CMP_SEEDS" --placements "$CMP_PLACE" --out "$OUT_JSON"
echo "--- COMPARATOR exit=$? $(date) ---"
echo "=== EXT17 ALL DONE $(date) ==="
