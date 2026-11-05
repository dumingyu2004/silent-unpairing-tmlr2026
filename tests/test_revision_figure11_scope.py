"""Prevent two isolated endpoints from being presented as a full-grid repair."""
import unittest
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


class Figure11ScopeTests(unittest.TestCase):
    def test_figure_text_has_explicit_coverage(self):
        text = PdfReader(ROOT / 'paper_tmlr/figs/fig_realeval.pdf').pages[0].extract_text()
        self.assertIn('2/25 checked', text)
        self.assertNotIn('all arms', text)

    def test_no_full_grid_isolation_line_or_aggregate_readout(self):
        source = (ROOT / 'figs/make_fig_realeval.py').read_text(encoding='utf-8')
        self.assertNotIn('ax.hlines(', source)
        self.assertNotIn('d["manufactured_gain_B"]', source)
        self.assertIn('for seed in (1, 2):', source)

    def test_appendix_does_not_extrapolate_two_endpoints(self):
        for filename in ('paper/appendix.tex', 'paper_tmlr/appendix_body.tex'):
            text = (ROOT / filename).read_text(encoding='utf-8')
            start = text.index(r'\label{sec:extended-harvest}')
            stop = text.index(r'\section{Environment Manifest}', start)
            section = text[start:stop]
            self.assertIn('remaining 23 repaired comparisons', section)
            self.assertIn('do not establish zero', section)
            self.assertNotIn('the selectable gain exactly', section)


if __name__ == '__main__':
    unittest.main()
