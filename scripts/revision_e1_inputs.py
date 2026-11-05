"""Export labelled reconstructions of E1 split IDs and corruption draws.

No dataset pickle, model, GPU, or training is loaded. These are newly generated
inputs from retained code/settings, not recovered historical array payloads.
Clean CIFAR-10 labels plus these offsets/masks determine the noisy labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def arrays(seed):
    perm = np.random.default_rng(seed).permutation(50000)
    noise_rng = np.random.default_rng(seed * 7919)
    return {'permutation': perm, 'meta_idx': perm[:1000],
            'train_idx': perm[1000:46000], 'monitor_idx': perm[46000:48000],
            'selection_idx': perm[46000:50000],
            'corruption_mask': noise_rng.random(45000) < 0.4,
            'corruption_offsets': noise_rng.integers(1, 10, size=45000)}

def validate_records(root):
    files = sorted((root/'logs_moneyshot').glob('e1_seed*.json'))
    assert len(files) == 120
    seen = set()
    for path in files:
        item = json.loads(path.read_text(encoding='utf-8')); a = item['args']
        assert not item['partial']
        assert (a['train_size'], a['val_size'], a['val1_size'], a['noise']) == (45000,2000,4000,0.4)
        assert a['seed'] in range(101,121)
        key = (a['seed'], a['arm'], a['probe_step']); assert key not in seen; seen.add(key)
    expected = {(s,'base',999999) for s in range(101,121)}
    expected |= {(s,'evalcb',p) for s in range(101,121) for p in [400,1400,2700,4000,5200]}
    assert seen == expected

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def build(root):
    validate_records(root)
    dest = root/'artifacts/e1_reconstructed_inputs'; dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in range(101,121):
        data=arrays(seed); path=dest/f'seed{seed}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as prior:
                assert set(prior.files)==set(data)
                assert all(np.array_equal(prior[k],v) for k,v in data.items()), 'Refusing to replace differing inputs'
        else: np.savez_compressed(path,**data)
        rows.append({'seed':seed,'path':path.relative_to(root).as_posix(),
                     'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} for k,v in data.items()}})
    manifest={'status':'reconstructed_from_retained_recipe_not_original_saved_arrays',
              'numpy_version':np.__version__,'bit_generator':'PCG64 via default_rng',
              'source':'scripts/moneyshot/e1_valselect_mwnet.py',
              'source_lines':'316-344','config_records':120,'seeds':20,
              'labels':'For each train_idx, use (clean_label+offset)%10 where mask is true; otherwise retain clean_label. Clean labels are not included.',
              'historical_array_equality_independently_verified':False,'training_rerun':False,'files':rows}
    (dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

def check(root):
    validate_records(root)
    m=json.loads((root/'artifacts/e1_reconstructed_inputs/MANIFEST.json').read_text())
    assert m['historical_array_equality_independently_verified'] is False
    for row in m['files']:
        expected=arrays(row['seed'])
        with np.load(root/row['path'],allow_pickle=False) as data:
            assert set(data.files)==set(expected)
            for k,v in expected.items():
                assert np.array_equal(data[k],v)
                assert digest(data[k])==row['arrays'][k]['sha256']
    return {'reconstructed_seed_files_checked':len(m['files']),'historical_originals_claimed':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--check',action='store_true'); args=ap.parse_args()
    if not args.check: build(ROOT)
    print(json.dumps(check(ROOT)))
