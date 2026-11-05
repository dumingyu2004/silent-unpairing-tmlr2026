"""Regression contracts for the reviewer-requested, seed-level reanalysis."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'revision_reanalysis.py'


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('revision_reanalysis', SCRIPT)
            cls.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.mod)

    def test_implementation_exists(self):
        self.assertTrue(SCRIPT.exists(), 'new reanalysis implementation is required')

    def test_normal_constants(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        self.assertAlmostEqual(self.mod.normal_cm(2), 1 / np.sqrt(np.pi), places=10)
        self.assertAlmostEqual(self.mod.normal_cm(6), 1.2672063606, places=9)
        self.assertAlmostEqual(self.mod.normal_c4(2), np.sqrt(2 / np.pi), places=12)

    def test_seed_shift_invariance(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        x = np.array([[0., 1., 2.], [1., 3., 7.], [0., 4., 9.]])
        a, b = self.mod.harvest_summary(x), self.mod.harvest_summary(x + np.array([[100.], [-90.], [44.]]))
        for key in ['harvest_pt', 'within_row_sd_pt', 'ratio_of_means']:
            np.testing.assert_allclose(a[key], b[key])
        self.assertEqual(a['bootstrap']['resamples'], 27)
        indices = np.array(list(__import__('itertools').product(range(3), repeat=3)))
        h, sd = x.max(1) - x.mean(1), x.std(1, ddof=1)
        expected = np.quantile(h[indices].mean(1) / (self.mod.normal_cm(3) * sd[indices].mean(1)), [.025, .975])
        np.testing.assert_allclose(a['bootstrap']['ratio_ci95'], expected)

    def test_loso_variants_distinct_and_corrected(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        x = np.array([[0., 1., 2.], [10., 11., 12.], [20., 21., 22.]])
        s = self.mod.harvest_summary(x)
        a, b = s['loso']['mean_other_within_sd'], s['loso']['flatten_other_rows_sd']
        self.assertGreater(b['predicted_pt'][0], a['predicted_pt'][0])
        np.testing.assert_allclose(s['loso']['mean_other_within_sd_c4']['predicted_pt'], np.array(a['predicted_pt']) / self.mod.normal_c4(3))

    def test_baseline_and_early_ties_zeros_retained(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        val = np.array([[1., 1., 0.], [0., 1., 1.], [3., 2., 1.]])
        test = np.array([[4., 8., 0.], [2., 5., 8.], [9., 7., 2.]])
        i, delta = self.mod.select_champions(val, test)
        np.testing.assert_array_equal(i, [0, 1, 0])
        np.testing.assert_array_equal(delta, [0, 3, 0])
        self.assertEqual(delta.mean(), 1.)

    def test_within_between_covariance_identity(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        x = np.array([[0., 1., 2.], [5., 7., 9.], [2., 3., 5.]])
        y = np.array([[3., 4., 5.], [2., 1., 0.], [6., 2., 1.]])
        z = self.mod.covariance_decomposition(x, y)
        self.assertAlmostEqual(z['total_covariance'], np.cov(x.ravel(), y.ravel(), ddof=0)[0, 1])
        self.assertAlmostEqual(z['total_covariance'], z['within_covariance'] + z['between_covariance'])

    def test_simulation_reproducible(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        a = self.mod.plugin_simulation(3, 4, draws=200, seed=12)
        self.assertEqual(a, self.mod.plugin_simulation(3, 4, draws=200, seed=12))
        self.assertIn('t3_unit_variance', a)
        self.assertEqual(a['normal']['draws'], 200)

    def test_simulation_fitted_scales_preserve_common_scale_invariance(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        a = self.mod.plugin_simulation(3, 4, draws=100, seed=3, seed_scales=[1, 2, 4])
        b = self.mod.plugin_simulation(3, 4, draws=100, seed=3, seed_scales=[3, 6, 12])
        for kind in ['normal', 't3_unit_variance']:
            self.assertAlmostEqual(a[kind]['ratio_mean'], b[kind]['ratio_mean'])
            self.assertAlmostEqual(a[kind]['harvest_mean'] * 3, b[kind]['harvest_mean'])

    def test_e1_complete_and_historical_provenance(self):
        if not SCRIPT.exists(): self.skipTest('implementation absent')
        s, inputs = self.mod.e1_analysis(ROOT, draws=100, seed=8)
        self.assertEqual(len(inputs), 120)
        self.assertEqual(s['n_seeds'], 20)
        self.assertEqual(s['selected_count'], 17)
        self.assertEqual(s['baseline_count'], 3)
        self.assertAlmostEqual(s['primary_mean_pt'], .1245, places=8)
        self.assertLess(s['historical_primary_t_ci95_pt'][0], 0)

    def test_appendix_distinguishes_frozen_and_reviewer_intervals(self):
        app = (ROOT / 'paper/appendix.tex').read_text(encoding='utf-8')
        self.assertNotIn('contributes under $4', app)
        self.assertNotIn('All intervals here are percentile', app)
        self.assertIn('Original primary, seed-level $t$', app)
        self.assertIn('Reviewer primary, seed-cluster percentile', app)
        self.assertIn('0.0348025+0.0278674', app)
        self.assertIn('[0.8665,0.9776]', app)


if __name__ == '__main__':
    unittest.main()
