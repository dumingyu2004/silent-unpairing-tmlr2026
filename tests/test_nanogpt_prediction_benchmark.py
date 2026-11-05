import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/nanogpt_prediction_benchmark.py'

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(),'benchmark implementation missing')
        spec=importlib.util.spec_from_file_location('benchmark',SCRIPT)
        self.mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_real_data_matches_independent_paired_bootstrap_calculation(self):
        root=ROOT/'revision_results/nanogpt_final'
        result=self.mod.build(root)
        a=json.loads((root/'analysis.json').read_text())
        f=json.loads((root/'frozen_calibration_prediction.json').read_text())
        cal=np.asarray(a['calibration_shared_deltas'])
        h=np.asarray(a['validation_harvest_per_seed'])
        ch=cal.max(axis=1)-cal.mean(axis=1)
        g=f['predicted_mean_harvest']; e=ch.mean()
        delta=(h-g)**2-(h-e)**2
        rng=np.random.default_rng(2026090801)
        ci=rng.integers(0,16,(20000,16)); vi=rng.integers(0,16,(20000,16))
        gb=f['c5']/f['normal_sample_sd_bias_factor']*cal.std(axis=1,ddof=1)[ci].mean(axis=1)
        eb=ch[ci].mean(axis=1)
        full=((h[vi]-gb[:,None])**2-(h[vi]-eb[:,None])**2).mean(axis=1)
        self.assertAlmostEqual(result['predictors']['empirical']['prediction'],e)
        self.assertAlmostEqual(result['contrast']['mse_gaussian_minus_empirical'],delta.mean())
        np.testing.assert_allclose(result['contrast']['fixed_predictor_ci95'],np.percentile(delta[vi].mean(axis=1),[2.5,97.5]))
        np.testing.assert_allclose(result['contrast']['full_procedure_ci95'],np.percentile(full,[2.5,97.5]))
        np.testing.assert_allclose([r['squared_error_difference'] for r in result['per_seed']],delta)
        self.assertEqual(result['stage'],'exploratory_post_readout')
        json.dumps(result,allow_nan=False)

    def test_mse_difference_equals_squared_mean_discrepancy_difference(self):
        result=self.mod.build(ROOT/'revision_results/nanogpt_final')
        g,e=[result['predictors'][name] for name in ['gaussian','empirical']]
        self.assertAlmostEqual(result['contrast']['mse_gaussian_minus_empirical'],g['signed_mean_discrepancy']**2-e['signed_mean_discrepancy']**2)

    def test_portable_cli_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['analysis.json','frozen_calibration_prediction.json','manifest.json']:
                shutil.copyfile(ROOT/'revision_results/nanogpt_final'/name,root/name)
            script=root/'benchmark.py'
            shutil.copyfile(SCRIPT,script)
            output=root/'output.json'
            command=[sys.executable,'-B',str(script),'--root',str(root),'--output',str(output)]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            content=output.read_bytes()
            self.assertEqual(json.loads(first.stdout),json.loads(content))
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(output.read_bytes(),content)

if __name__=='__main__':
    unittest.main()
