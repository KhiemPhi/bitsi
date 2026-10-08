"""Regression checks: python -m fitpart.scripts.test_partial_sections (from parent)."""
import unittest
import numpy as np
from fitpart.partial_sections import observed_radii, masked_partition, masked_piece_cost


class PartialSectionsTest(unittest.TestCase):
    def test_missing_bins_are_unknown(self):
        y = observed_radii(np.ones(8), np.zeros(8), np.zeros(8, int), 3)
        self.assertEqual(np.isfinite(y).sum(), 1)
        self.assertEqual(y[0, 12], 1)

    def test_smooth_profile_across_gap(self):
        x = np.linspace(-1, 1, 30)
        y = np.tile((3 + x + x*x)[:, None], (1, 6))
        y[10:18] = np.nan
        self.assertEqual(masked_partition(y)[0], [])

    def test_observed_step_survives(self):
        y = np.zeros((30, 6))
        y[15:] = 30
        self.assertEqual(masked_partition(y)[0], [14])

    def test_disjoint_views_are_not_boundary_evidence(self):
        y = np.full((30, 6), np.nan)
        y[:15, :3] = 0
        y[15:, 3:] = 30
        self.assertEqual(masked_partition(y)[0], [])

    def test_unknown_channel_has_no_cost(self):
        y = np.arange(20.)[:, None]
        z = np.c_[y, np.full(20, np.nan)]
        self.assertAlmostEqual(masked_piece_cost(y, 0, 20), masked_piece_cost(z, 0, 20))

    def test_empty_support(self):
        self.assertEqual(masked_partition(np.full((12, 24), np.nan)), ([], 0.0))

    def test_opt_out_matches_v4_and_fit_resets_trace(self):
        from fitpart.bitsi4 import BITSI4
        from fitpart.bitsi5 import BITSI5
        phi, z = np.meshgrid(np.linspace(-np.pi, np.pi, 40, endpoint=False), np.linspace(0, 4, 30))
        p = np.c_[np.cos(phi.ravel()), np.sin(phi.ravel()), z.ravel()]
        n = np.c_[p[:, :2], np.zeros(len(p))]
        kw = dict(up=(0, 0, 1), max_depth=0)
        baseline, _ = BITSI4(**kw).fit(p, n)
        m = BITSI5(axis_rule="residual", spine_coding="free", **kw)
        np.testing.assert_array_equal(m.fit(p, n)[0], baseline)
        m.partial_cuts = True
        labels, _ = m.fit(p, n)
        self.assertEqual(len(labels), len(p))
        self.assertTrue(np.all(labels >= 0))
        first = len(m.trace)
        m.fit(p, n)
        self.assertEqual(len(m.trace), first)


if __name__ == "__main__":
    unittest.main()
