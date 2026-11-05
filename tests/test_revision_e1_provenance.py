import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e1_provenance", ROOT / "scripts/revision_e1_provenance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class E1ProvenanceTests(unittest.TestCase):
    def test_public_ledger(self):
        result = module.check_release(ROOT)
        self.assertTrue(result["public_chronology_consistent"])
        self.assertFalse(result["private_records_reverified"])

    def test_replay_and_normalization(self):
        def record(name, args):
            return {"timestamp": "2026-07-10T00:00:00Z", "message": {"content": [
                {"type": "tool_use", "name": name, "input": args}]}}
        records = [record("Write", {"content": "a\r\nb\r\n"}),
                   record("Edit", {"old_string": "b\n", "new_string": "c\n"})]
        text, events = module.replay(records, [1, 2])
        self.assertEqual(text, "a\nc\n")
        self.assertEqual(len(events), 2)

    def test_release_has_no_private_identifiers(self):
        text = (ROOT / "revision_results/e1_provenance.json").read_text(encoding="utf-8")
        for token in ["C:\\", "/home/", "100.109.", "sessionId", "tool_use_id"]:
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
