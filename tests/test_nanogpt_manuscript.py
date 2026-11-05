"""Small regression contract for the canonical nanoGPT manuscript integration."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NanoGPTManuscriptTests(unittest.TestCase):
    def test_primary_numbers_and_scopes(self):
        text = (ROOT / "paper/sections/f2.tex").read_text(encoding="utf-8")
        for token in ("0.00388406", "0.00343060", "-0.00045346",
                      "-0.00134285,0.00043370", "whole seed vectors",
                      "neither equivalence", "not validation-to-test transfer",
                      "sec:nanogpt-supplement", "$+4.01$"):
            self.assertIn(token, text)
        self.assertNotIn("Every harvest number in this paper comes from", text)
        self.assertNotIn("For nanoGPT we ran no", text)

    def test_supplement_numbers_and_figure(self):
        text = (ROOT / "paper/appendix.tex").read_text(encoding="utf-8")
        for token in ("prespecified uncorrected", "0.00365096",
                      "-0.00108275,0.00064263", "0.00149894,0.00490740",
                      "-0.00165377,0.00128459", "Exploratory",
                      "common seed component", "sum-zero constraint",
                      "influence ranges, not uncertainty", "remains inconclusive",
                      "figs/nanogpt_per_seed.pdf"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
