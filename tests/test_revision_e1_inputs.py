import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e1_inputs',ROOT/'scripts/revision_e1_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class E1InputsTests(unittest.TestCase):
    def test_recipe_and_disjointness(self):
        a=m.arrays(101); p=np.random.default_rng(101).permutation(50000)
        self.assertTrue(np.array_equal(a['permutation'],p))
        self.assertEqual(len(set(a['meta_idx'])|set(a['train_idx'])|set(a['selection_idx'])),50000)
        self.assertTrue(np.array_equal(a['monitor_idx'],a['selection_idx'][:2000]))
        r=np.random.default_rng(101*7919)
        self.assertTrue(np.array_equal(a['corruption_mask'],r.random(45000)<0.4))
        self.assertTrue(np.array_equal(a['corruption_offsets'],r.integers(1,10,size=45000)))
    def test_released_inputs_match_recipe(self):
        self.assertEqual(m.check(ROOT)['reconstructed_seed_files_checked'],20)
    def test_registered_settings(self):
        m.validate_records(ROOT)

if __name__=='__main__': unittest.main()
