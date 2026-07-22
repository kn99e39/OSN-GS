from __future__ import annotations

"""Unit tests for osn_gs/data/vendor/graphdeco_scene_split.py -- the vendored
(verbatim-ported) Graphdeco held-out test-camera split and resolution-decision
logic, used for the same-condition OSN-GS vs. baseline 3DGS A/B (TODO.md)."""

import unittest

from osn_gs.data.vendor.graphdeco_scene_split import (
    resolve_graphdeco_resolution,
    select_llff_holdout_test_names,
)


class LlffHoldoutSplitTest(unittest.TestCase):
    def test_every_nth_sorted_name_is_held_out(self):
        # Unsorted input; the function must sort internally (matching
        # upstream) before indexing by llffhold.
        names = [f"img_{i:03d}.jpg" for i in range(20)]
        shuffled = names[::-1]
        test_names = select_llff_holdout_test_names(shuffled, eval=True, llffhold=4)
        expected = [names[i] for i in range(0, 20, 4)]
        self.assertEqual(test_names, expected)

    def test_eval_false_returns_empty(self):
        names = [f"img_{i:03d}.jpg" for i in range(20)]
        self.assertEqual(select_llff_holdout_test_names(names, eval=False, llffhold=8), [])

    def test_default_llffhold_matches_upstream_default(self):
        names = [f"img_{i:03d}.jpg" for i in range(16)]
        test_names = select_llff_holdout_test_names(names, eval=True)
        self.assertEqual(test_names, [names[0], names[8]])

    def test_360_in_path_forces_llffhold_8(self):
        names = [f"img_{i:03d}.jpg" for i in range(16)]
        # llffhold=2 requested, but a "360" scene path should force 8 (upstream quirk).
        test_names = select_llff_holdout_test_names(
            names, scene_path="/data/mip360_scene", eval=True, llffhold=2
        )
        self.assertEqual(test_names, [names[0], names[8]])


class ResolutionDecisionTest(unittest.TestCase):
    def test_auto_downscales_above_1600(self):
        # Matches the real DATASET scene verified directly against upstream's
        # own loadCam computation (5187x3361 -> 1600x1036, scale=3.241875).
        width, height, scale = resolve_graphdeco_resolution(5187, 3361, resolution=-1, resolution_scale=1.0)
        self.assertEqual((width, height), (1600, 1036))
        self.assertAlmostEqual(scale, 3.241875, places=5)

    def test_no_downscale_below_threshold(self):
        width, height, scale = resolve_graphdeco_resolution(800, 600, resolution=-1, resolution_scale=1.0)
        self.assertEqual((width, height), (800, 600))
        self.assertEqual(scale, 1.0)

    def test_explicit_power_of_two_resolution(self):
        width, height, scale = resolve_graphdeco_resolution(1600, 1200, resolution=2, resolution_scale=1.0)
        self.assertEqual((width, height), (800, 600))
        self.assertEqual(scale, 2.0)

    def test_resolution_scale_compounds_with_auto_downscale(self):
        width, height, scale = resolve_graphdeco_resolution(3200, 2400, resolution=-1, resolution_scale=2.0)
        # global_down = 3200/1600 = 2.0; scale = 2.0*2.0 = 4.0
        self.assertEqual((width, height), (800, 600))
        self.assertAlmostEqual(scale, 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
