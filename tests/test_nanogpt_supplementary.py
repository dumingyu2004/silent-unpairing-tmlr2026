import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/nanogpt_supplementary.py'

class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'supplementary implementation is missing')
        spec = importlib.util.spec_from_file_location('supp', SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_empirical_variance_identity_and_contrast_constraint(self):
        x = np.array([[0.,1.,4.],[2.,0.,3.],[8.,4.,2.],[1.,3.,9.]])
        ix = np.random.default_rng(1).integers(0,4,(100,4))
        d = self.mod.placement_diagnostics(x, ix)
        v = d['variance_identity']
        self.assertAlmostEqual(v['mean_within_seed_sample_variance'], v['sum_components'])
        self.assertGreater(v['between_placement_mean_component'], 0)
        np.testing.assert_allclose(np.asarray(d['contrast_covariance']).sum(axis=1), 0, atol=1e-12)

    def test_full_data_reproduces_primary_and_joint_bootstrap(self):
        root = ROOT / 'revision_results/nanogpt_final'
        result = self.mod.build(root)
        a = json.loads((root/'analysis.json').read_text())
        f = json.loads((root/'frozen_calibration_prediction.json').read_text())
        self.assertTrue(result['primary_reproduction']['all_checks_passed'])
        u = result['uncorrected_sensitivity']
        self.assertAlmostEqual(u['predicted_mean_harvest'], f['uncorrected_prediction_sensitivity'])
        self.assertAlmostEqual(u['signed_prediction_discrepancy'], a['observed_validation_mean_harvest']-u['predicted_mean_harvest'])
        v = result['selected_unselected_harvest']['validation']
        np.testing.assert_allclose(np.array(v['selected']['per_seed'])-v['unselected']['per_seed'], v['harvest']['per_seed'])
        np.testing.assert_allclose(v['harvest']['ci95'], a['validation_mean_harvest_ci95'])
        self.assertEqual(len(result['influence']['validation']['per_seed']),16)
        json.dumps(result, allow_nan=False)

    def test_zero_variance_correlation_is_null_not_nan(self):
        x = np.ones((4,3))
        d = self.mod.placement_diagnostics(x,np.array([[0,1,2,3]]))
        self.assertEqual(d['raw_correlation'], [[None]*3]*3)
        json.dumps(d,allow_nan=False)

    def test_anonymous_package_without_budget_and_cli_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            root = package/'revision_results/nanogpt_final'
            root.mkdir(parents=True)
            for name in ['analysis.json','frozen_calibration_prediction.json','manifest.json']:
                shutil.copyfile(ROOT/'revision_results/nanogpt_final'/name,root/name)
            analyzer = ROOT/'scripts/revision_nanogpt_analyze.py'
            if not analyzer.exists():
                analyzer = ROOT/'revision_results/nanogpt_final/revision_nanogpt_analyze.py'
            shutil.copyfile(analyzer,root/'revision_nanogpt_analyze.py')
            script = package/'nanogpt_supplementary.py'
            shutil.copyfile(SCRIPT,script)
            output = package/'supplementary.json'
            proc = subprocess.run([sys.executable,'-B',str(script),'--root',str(root),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual(saved,json.loads(proc.stdout))
            original = self.mod.build(ROOT/'revision_results/nanogpt_final')
            saved.pop('provenance')
            original.pop('provenance')
            self.assertEqual(saved,original)

if __name__ == '__main__':
    unittest.main()
