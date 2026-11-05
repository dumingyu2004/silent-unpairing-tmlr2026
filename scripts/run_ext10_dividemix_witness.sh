#!/bin/bash
# ext10: DivideMix four-arm dynamic witness.
#
# HISTORICAL LAUNCH RECORD, NOT RUNNABLE AS-IS. Machine-local paths were redacted
# to /artifact/local-path for release, and the per-arm working copies are not
# included. Kept as a configuration/provenance record (see README).
#
# Layout at run time: <workdir>/arms/{A,B,C,D} and <workdir>/data.
# The only difference between arms (verified by byte diff): arms A/B add
# random_state=0 to the GMM at Train_cifar.py:186; arms C/D are stock.
# Patches shared by all arms: iter.next() -> next(iter) (x2), num_workers 5 -> 0,
#   cudnn.benchmark=False, and a final-state dump after the loop
#   (checkpoint/final_{arm}.pt + "EXT10_FINAL_SHA256 <hex>" + "EXT10_NP_STREAM_POS <int>").
# Wrapper run_arm.py (byte-identical across arms): CUBLAS_WORKSPACE_CONFIG=:4096:8,
#   OMP_NUM_THREADS=4, np.random.seed(777), cudnn.deterministic=True,
#   use_deterministic_algorithms(True, warn_only=True), visdom stub (torchnet installed with --no-deps).
# Preregistered verdicts: P1 A==B SHA floor; P2 C==D SHA floor; P3 C!=A artifact
#   (pinning the GMM changes the final weights); P4 accuracy-log prefix byte-identical
#   through epoch 9 (the GMM does not fire during warm-up; divergence expected at epoch 10).
# Budget: ~35 GPU minutes (12 epochs x 4 arms; 10 warm-up + 3 SSL).
cd "/c/artifact/local-path" || exit 1
exec >> logs_nlp_gateb/overnight_p0_console.log 2>&1
export FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

WD="/artifact/local-path"
ARMS="$WD/arms"
DATA_ROOT="$WD/data"
# dataloader_cifar does not download; it unpickles data_batch_* directly, so --data_path
# must point at cifar-10-batches-py. The noise file (0.5_sym.json) is written there by
# Train_cifar on the first arm and shared by all four arms.
DATA_PATH="$DATA_ROOT/cifar-10-batches-py"

echo "=== EXT10 WAITING FOR EXT9 $(date) ==="
until grep -q "EXT9 ALL DONE" logs_nlp_gateb/overnight_p0_console.log 2>/dev/null; do
  sleep 60
done
echo "=== EXT10 STARTING: DIVIDEMIX FOUR-ARM WITNESS $(date) ==="

mkdir -p "$DATA_ROOT" logs_flip_demo
python -X utf8 -c "import torchvision; torchvision.datasets.CIFAR10(root='$DATA_ROOT', download=True); print('CIFAR10 ready')"
echo "--- DATA READY exit=$? $(date) ---"

for arm in A B C D; do
  echo "=== EXT10 ARM ${arm} $(date) ==="
  ( cd "$ARMS/$arm" && python -X utf8 run_arm.py --data_path "$DATA_PATH" --dataset cifar10 \
      --r 0.5 --noise_mode sym --num_epochs 12 --seed 123 ) 2>&1 | tee "logs_flip_demo/dividemix_arm${arm}.log"
  echo "--- DONE dividemix arm ${arm} exit=${PIPESTATUS[0]} $(date) ---"
done

echo "=== EXT10 COMPARATOR (pre-registered P1-P4) $(date) ==="
python -X utf8 - "$ARMS" <<'PYEOF'
import os, re, sys
arms_root = sys.argv[1]
sha, pos = {}, {}
for a in 'ABCD':
    try:
        t = open('logs_flip_demo/dividemix_arm%s.log' % a, encoding='utf-8', errors='replace').read()
    except OSError:
        t = ''
    m = re.findall(r'EXT10_FINAL_SHA256 ([0-9a-f]{64})', t)
    p = re.findall(r'EXT10_NP_STREAM_POS (\d+)', t)
    sha[a] = m[-1] if m else None
    pos[a] = p[-1] if p else None
for a in 'ABCD':
    print('EXT10 OBS arm %s sha=%s np_pos=%s' % (a, sha[a], pos[a]))
def verdict(name, cond, desc):
    print('EXT10 %s %s: %s' % (name, desc, 'PASS' if cond else 'FAIL'))
ok = all(sha[a] for a in 'ABCD')
verdict('P1', ok and sha['A'] == sha['B'], '(floor A==B, GMM random_state=0)')
verdict('P2', ok and sha['C'] == sha['D'], '(floor C==D, stock GMM + seeded global numpy)')
verdict('P3', ok and sha['C'] != sha['A'], '(artifact C!=A, GMM pinning shifts final weights)')
def prefix9(a):  # byte prefix of the accuracy log through the epoch-9 line; epochs 0-9 are warm-up (GMM idle)
    fp = os.path.join(arms_root, a, 'checkpoint', 'cifar10_0.5_sym_acc.txt')
    try:
        b = open(fp, 'rb').read()
    except OSError:
        return None
    i = b.find(b'Epoch:9 ')
    if i < 0:
        return None
    j = b.find(b'\n', i)
    return b[:j + 1] if j >= 0 else None
p9 = dict((a, prefix9(a)) for a in 'ABCD')
verdict('P4', all(p9[a] is not None for a in 'ABCD') and len(set(p9.values())) == 1,
        '(acc-log prefix byte-identical through epoch 9 across all arms; divergence only at epoch>=10)')
PYEOF
echo "--- COMPARATOR exit=$? $(date) ---"
echo "=== EXT10 ALL DONE $(date) ==="
