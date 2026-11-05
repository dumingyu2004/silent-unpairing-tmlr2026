import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SupplementPlotTests(unittest.TestCase):
    def test_plot_contract(self):
        path = ROOT/'scripts/nanogpt_supplementary_plot.py'
        self.assertTrue(path.exists(), 'Approved per-seed plot is not implemented')
        spec = importlib.util.spec_from_file_location('supp_plot', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = json.loads((ROOT/'revision_results/nanogpt_final/analysis.json').read_text())
        frozen = json.loads((ROOT/'revision_results/nanogpt_final/frozen_calibration_prediction.json').read_text())
        fig = module.build_figure(data, frozen)
        self.assertEqual(len(fig.axes), 3)
        for ax in fig.axes[:2]:
            self.assertEqual(len(ax.collections[0].get_offsets()), 16)
            self.assertTrue(any('Frozen corrected' in line.get_label() for line in ax.lines))
            self.assertTrue(any('Uncorrected' in line.get_label() for line in ax.lines))
        grid = np.asarray(data['validation_shared_deltas'])
        np.testing.assert_allclose(fig.axes[2].collections[0].get_offsets()[:, 1], grid.mean(axis=1)*1000)
        np.testing.assert_allclose(fig.axes[2].collections[1].get_offsets()[:, 1], grid.max(axis=1)*1000)
        self.assertIn('Held-out', fig.axes[1].get_title())
        self.assertIn('reported score', fig.axes[2].get_title())
        module.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
