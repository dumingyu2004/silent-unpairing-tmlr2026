import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
class SourceAvailabilityTest(unittest.TestCase):
    def test_omitted_sources_are_explicit_not_successful_hash_checks(self):
        index=json.loads((ROOT/'revision_sources/external_source_index.json').read_text())
        ledger=json.loads((ROOT/'revision_results/audit_ledger.json').read_text())
        cites=[c for row in ledger['repositories'] for c in row['citations']]
        marked=[c for c in cites if c.get('release_verification')=='metadata_only_source_omitted']
        self.assertEqual(len(marked),len(index))
        self.assertEqual(len(index),11)
        self.assertTrue(all(not (ROOT/c['path']).exists() for c in marked))
        self.assertTrue(all((ROOT/c['path']).exists() for c in cites if c not in marked))
