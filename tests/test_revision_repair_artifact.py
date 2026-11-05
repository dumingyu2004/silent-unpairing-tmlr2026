"""Catch transcription errors in the published explanatory-patch provenance."""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_checker', ROOT / 'scripts/verify_repair_patches.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class RepairArtifactTests(unittest.TestCase):
    def test_patch_bytes_and_post_apply_provenance(self):
        manifest = json.loads((ROOT / 'revision_results/repair_patches/manifest.json').read_text())
        rows = {row['name']: row for row in manifest['patches']}
        self.assertEqual(set(rows), {row['name'] for row in checker.SPECS})
        for config in checker.SPECS:
            row = rows[config['name']]
            patch = (ROOT / row['path']).read_bytes()
            self.assertEqual(hashlib.sha256(patch).hexdigest(), row['patch_sha256'])
            result = checker.verify_one(config)
            if result['base_sha256'] == row['historical_original_base_sha256']:
                self.assertEqual(result['applied_sha256'], row['local_verified_applied_sha256'])
            self.assertEqual(result['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
