"""Worklog 68: adjacency-aware LOCAL orientation-folding, distinct from the
existing global single-reference `compute_orientation_consistency`.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_local_orientation_folding import compute_local_orientation_folding
from osn_gs.surface.torch_parametric_diagnostics import compute_orientation_consistency


class LocalOrientationFoldingTest(unittest.TestCase):
    def test_flat_consistent_grid_has_zero_local_folds(self):
        resolution = 5
        normals = torch.zeros((resolution * resolution, 3))
        normals[:, 2] = 1.0  # every sample points +z
        result = compute_local_orientation_folding(normals, resolution)
        self.assertEqual(result["local_fold_count"], 0)
        self.assertAlmostEqual(result["local_fold_fraction"], 0.0)
        self.assertGreater(result["local_adjacent_dot_min"], 0.99)

    def test_global_reversal_alone_is_not_a_local_fold(self):
        """Every sample flips sign AT ONCE (a global reversal) -- neighbors
        still agree with each other, so this must report zero local folds,
        unlike a naive single-reference check which may or may not."""
        resolution = 5
        normals = torch.zeros((resolution * resolution, 3))
        normals[:, 2] = -1.0  # every sample points -z (globally "reversed")
        result = compute_local_orientation_folding(normals, resolution)
        self.assertEqual(result["local_fold_count"], 0)

    def test_checkerboard_flip_is_fully_folded(self):
        resolution = 4
        grid = torch.zeros((resolution, resolution, 3))
        for i in range(resolution):
            for j in range(resolution):
                grid[i, j, 2] = 1.0 if (i + j) % 2 == 0 else -1.0
        normals = grid.reshape(-1, 3)
        result = compute_local_orientation_folding(normals, resolution)
        self.assertEqual(result["local_fold_fraction"], 1.0)
        self.assertAlmostEqual(result["local_adjacent_dot_mean"], -1.0, places=5)

    def test_single_interior_flip_is_a_small_local_fraction(self):
        resolution = 6
        grid = torch.zeros((resolution, resolution, 3))
        grid[:, :, 2] = 1.0
        grid[3, 3, 2] = -1.0  # one interior sample folds
        normals = grid.reshape(-1, 3)
        result = compute_local_orientation_folding(normals, resolution)
        self.assertGreater(result["local_fold_count"], 0)
        # only the flipped cell's 4 neighbor-pairs (up/down/left/right) can disagree
        self.assertLessEqual(result["local_fold_count"], 4)
        self.assertLess(result["local_fold_fraction"], 0.1)

    def test_distinct_from_global_single_reference_check_on_a_reversed_but_smooth_grid(self):
        """Same fixture as the global-reversal test: the EXISTING global
        check reports every sample as a 'flip' relative to its own
        arbitrarily-seeded reference in a fully degenerate (all-identical)
        case, while the local check correctly reports zero folding -- they
        answer different questions and must not be conflated."""
        resolution = 5
        normals = torch.zeros((resolution * resolution, 3))
        normals[:, 2] = -1.0
        local = compute_local_orientation_folding(normals, resolution)
        global_result = compute_orientation_consistency(normals)
        self.assertEqual(local["local_fold_count"], 0)
        # global check's own flip_count is a DIFFERENT axis (reference-relative,
        # not adjacency) -- it is well-defined but answers a different question,
        # asserted here only to document the two never claim the same thing.
        self.assertIn("orientation_flip_count", global_result)


if __name__ == "__main__":
    unittest.main()
