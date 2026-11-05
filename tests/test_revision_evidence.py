"""Evidence regression checks: prevent manuscript text from becoming hashes."""
import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('revision_evidence',Path(__file__).resolve().parents[1]/'scripts/revision_evidence.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class EvidenceAuditTests(unittest.TestCase):
    def test_complete_source_register_and_real_correction(self):
        audit=module.build_audit()
        self.assertEqual(len(audit['repositories']),28)
        self.assertEqual(audit['after_counts'],{'exposed':22,'guarded':4,'conditional':2})
        changed=[r['name'] for r in audit['repositories'] if r['before']!=r['after']]
        self.assertEqual(changed,['infobatch'])

    def test_no_inferred_nlp_hash_certificates(self):
        evidence=module.build_evidence()
        self.assertEqual(len(evidence['nlp_cells']),24)
        self.assertTrue(all(not c['certificate_eligible'] for c in evidence['nlp_cells']))
        self.assertEqual(evidence['nlp_summary']['direct_weight_verified'],0)

    def test_direct_pairs_not_promoted_to_full_state(self):
        pairs=module.build_evidence()['individually_checked_weight_pairs']
        self.assertEqual(len(pairs),8)
        self.assertTrue(all(p['equal'] for p in pairs))
        self.assertTrue(all(not p['full_state_equality_verified'] for p in pairs))
        self.assertEqual(sum(p['comparison_kind']=='independent_repeat_floor' for p in pairs),3)

if __name__=='__main__': unittest.main()
