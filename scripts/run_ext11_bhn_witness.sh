#!/bin/bash
# ext11: BHN (ICML 2023) four-arm pairing-divergence witness. A mid-training probe
# consumes the global CPU RNG stream that training shares, silently desynchronizing
# same-seed paired runs. Mirrors ext10 (DivideMix).
#
# HISTORICAL LAUNCH RECORD, NOT RUNNABLE AS-IS. Machine-local paths were redacted
# to /artifact/local-path for release, and the per-arm working copies and
# compare_ext11.py are not included. Kept as a configuration/provenance record
# (see README); the comparator output is logs_flip_demo/bhn_witness.json.
#
# Layout at run time: <workdir>/bhn_arms/{A,B,C,D}, comparator compare_ext11.py
#   (preregistered P1-P4, atomic JSON write).
# Arm naming (as in ext10: first pair = treated/ISO, second pair = stock):
#   A/B = ISO: BHN.py loaders() test_loader gets
#         generator=torch.Generator().manual_seed(0) (the only differing line).
#   C/D = STOCK. Verified by byte diff: A vs C differ in that one BHN.py line;
#         utility.py and run_arm.py have identical sha256 across all four arms.
# Mechanism: at the end of every epoch PvalueTrainer.test() creates iter(test_loader),
#   which draws one int64 _base_seed from the global CPU generator and shifts the
#   next epoch's shuffle seed. The ISO line gives test_loader a private generator and
#   removes exactly that draw (with num_workers=0 and shuffle=False its outputs are
#   bitwise identical at equal weights).
# Patches shared by all arms (byte-identical): set_bestgpu pinned to GPU0 (Windows),
#   short set_outdir path (MAX_PATH), cudnn.deterministic=True / benchmark=False +
#   use_deterministic_algorithms(True, warn_only=True), EXT11_EPOCH_SHA / EXT11_FINAL_SHA256
#   prints, and a run_arm.py wrapper (CUBLAS_WORKSPACE_CONFIG=:4096:8 set before importing
#   torch; warnings hooked into EXT11_DET_WARNING lines).
# Preregistered verdicts: P1 A==B final SHA floor; P2 C==D; P3 C!=A; P4 onset: epoch-1
#   SHA A==C, epoch-2 differs (the divergent draw first runs at the end of epoch 1, and
#   the first weight-affecting difference is the epoch-2 shuffle seed), with the floor
#   holding at every epoch (A==B and C==D per epoch). Final test accuracy per arm and
#   the stock-minus-iso gap are printed unconditionally.
# Budget: 12 epochs per arm ~ 4-5 min (~6 s/epoch + 2-3 min start-up), ~20-25 GPU min total.
cd "/c/artifact/local-path" || exit 1
exec >> logs_nlp_gateb/overnight_p0_console.log 2>&1
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

WD="/artifact/local-path"
ARMS="$WD/bhn_arms"
SUITE_DIR="$(pwd)"

echo "=== EXT11 WAITING FOR EXT12 $(date) ==="
until grep -q "EXT12 ALL DONE" logs_nlp_gateb/overnight_p0_console.log 2>/dev/null; do
  sleep 60
done
echo "=== EXT11 STARTING: BHN FOUR-ARM PAIRING-DIVERGENCE WITNESS $(date) ==="

mkdir -p logs_flip_demo

# Data staging: BHN uses root='~/data'. Nothing is downloaded; the existing
# cifar-10-batches-py copy is staged there, then torchvision checks its integrity.
if [ ! -f "/artifact/local-path" ]; then
  mkdir -p "/artifact/local-path"
  cp -r "$WD/data/cifar-10-batches-py" "/artifact/local-path"
fi
python -X utf8 -c "import torchvision,os; torchvision.datasets.CIFAR10(root=os.path.expanduser('~/data'), train=True); torchvision.datasets.CIFAR10(root=os.path.expanduser('~/data'), train=False); print('EXT11 CIFAR10 integrity OK')"
echo "--- DATA READY exit=$? $(date) ---"

for arm in A B C D; do
  echo "=== EXT11 ARM ${arm} $(date) ==="
  ( cd "$ARMS/$arm" && python -X utf8 run_arm.py --seed 1 --n_epochs 12 ) 2>&1 \
    | tee "logs_flip_demo/bhn_arm${arm}.log"
  echo "--- DONE bhn arm ${arm} exit=${PIPESTATUS[0]} $(date) ---"
done

echo "=== EXT11 COMPARATOR (pre-registered P1-P4) $(date) ==="
python -X utf8 "$ARMS/compare_ext11.py" \
  --logs-dir "$SUITE_DIR/logs_flip_demo" \
  --arms-root "$ARMS" \
  --out-json "$SUITE_DIR/logs_flip_demo/bhn_witness.json" \
  --log-prefix bhn_arm
echo "--- COMPARATOR exit=$? $(date) ---"
echo "=== EXT11 ALL DONE $(date) ==="
