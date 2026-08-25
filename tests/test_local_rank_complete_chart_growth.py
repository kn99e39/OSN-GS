from __future__ import annotations

"""Worklog 114 -- Local Rank-Complete NURBS Chart Growth.

Pure-logic (CPU, no CUDA needed) focused tests for
`osn_gs/surface/torch_local_rank_complete_chart_growth.py`: deterministic
rank-closed growth, full-rank closure, canonical-component isolation,
insufficient-support preservation, hole/gap non-crossing, and deterministic
replay -- directive section "TESTING".
"""

import unittest

import torch

from osn_gs.surface.torch_local_rank_complete_chart_growth import (
    REASON_INSUFFICIENT_RANK_CLOSURE,
    REASON_RUNTIME_CAP_SKIPPED,
    REASON_TOO_FEW_PIXELS,
    grow_local_rank_complete_charts,
)

_RES_U, _RES_V, _DEG = 8, 4, 2
_FULL_CAPACITY = _RES_U * _RES_V  # 32


def _maps(h: int, w: int, component_value: int = 0, mask=None):
    """Build (component_id_map, representative_id_map, world_points) for a
    single rectangular region, optionally restricted by a boolean `mask`
    (H, W) -- pixels outside the mask are marked invalid (-1), simulating an
    image-space hole or a component boundary."""

    comp = torch.full((h, w), component_value, dtype=torch.int64)
    rep = torch.arange(h * w, dtype=torch.int64).reshape(h, w)
    if mask is not None:
        comp[~mask] = -1
        rep[~mask] = -1
    row, col = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    world = torch.stack([col, row, torch.zeros_like(row)], dim=-1)
    return comp, rep, world


class DeterministicRankClosedGrowthTest(unittest.TestCase):
    def test_large_planar_blob_decomposes_into_multiple_full_rank_local_charts(self) -> None:
        comp, rep, world = _maps(40, 40)
        charts, unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertGreater(len(charts), 1, "a 1600-pixel blob should decompose into more than one local chart")
        for chart in charts:
            self.assertEqual(chart.rank, _FULL_CAPACITY)
            self.assertGreaterEqual(chart.pixel_rows.shape[0], _FULL_CAPACITY)

    def test_full_rank_closure_holds_for_every_returned_chart(self) -> None:
        comp, rep, world = _maps(20, 20)
        charts, _unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertGreater(len(charts), 0)
        for chart in charts:
            self.assertEqual(chart.rank, chart.full_capacity)

    def test_deterministic_replay_yields_identical_charts_and_unresolved(self) -> None:
        comp, rep, world = _maps(30, 25)
        charts_a, unresolved_a = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        charts_b, unresolved_b = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertEqual(len(charts_a), len(charts_b))
        for chart_a, chart_b in zip(charts_a, charts_b):
            self.assertTrue((chart_a.pixel_rows == chart_b.pixel_rows).all())
            self.assertTrue((chart_a.pixel_cols == chart_b.pixel_cols).all())
        self.assertEqual(len(unresolved_a), len(unresolved_b))


class ComponentIdentityPreservationTest(unittest.TestCase):
    def test_two_adjacent_components_never_share_a_local_chart(self) -> None:
        h, w = 20, 40
        comp = torch.zeros((h, w), dtype=torch.int64)
        comp[:, w // 2 :] = 1  # left half component 0, right half component 1, directly adjacent
        rep = torch.arange(h * w, dtype=torch.int64).reshape(h, w)
        row, col = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        world = torch.stack([col, row, torch.zeros_like(row)], dim=-1)
        charts, _unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertGreater(len(charts), 0)
        for chart in charts:
            cols_side = chart.pixel_cols < (w // 2)
            self.assertTrue(bool(cols_side.all()) or bool((~cols_side).all()), "a chart's pixels crossed the component boundary")


class InsufficientSupportPreservationTest(unittest.TestCase):
    def test_small_region_below_full_capacity_stays_unresolved_not_fitted(self) -> None:
        comp, rep, world = _maps(4, 4)  # 16 pixels < 32
        charts, unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertEqual(len(charts), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].reason, REASON_TOO_FEW_PIXELS)
        self.assertEqual(int(unresolved[0].pixel_rows.shape[0]), 16)

    def test_collinear_strip_with_enough_pixels_but_no_2d_spread_stays_unresolved(self) -> None:
        # a 1-pixel-wide strip has >= 32 pixels but can never fill an 8x4
        # tensor-product grid's column rank (no variation along one axis).
        comp, rep, world = _maps(1, 50)
        charts, unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertEqual(len(charts), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].reason, REASON_INSUFFICIENT_RANK_CLOSURE)


class HoleAndGapNonCrossingTest(unittest.TestCase):
    def test_ring_shaped_blob_never_produces_a_chart_pixel_inside_its_own_hole(self) -> None:
        h, w = 30, 30
        mask = torch.ones((h, w), dtype=torch.bool)
        mask[10:20, 10:20] = False  # interior hole, never part of the blob
        comp, rep, world = _maps(h, w, mask=mask)
        charts, _unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertGreater(len(charts), 0)
        for chart in charts:
            for row, col in zip(chart.pixel_rows.tolist(), chart.pixel_cols.tolist()):
                self.assertFalse(10 <= row < 20 and 10 <= col < 20, "a chart pixel fell inside the hole")

    def test_globally_occluded_gap_between_two_blobs_is_never_crossed(self) -> None:
        h, w = 20, 41
        comp = torch.zeros((h, w), dtype=torch.int64)
        comp[:, w // 2] = -1  # a 1-pixel invalid gap column separates the two halves
        comp[:, w // 2 + 1 :] = 0  # SAME component id on both sides -- only the gap separates them physically
        rep = torch.arange(h * w, dtype=torch.int64).reshape(h, w)
        rep[:, w // 2] = -1
        row, col = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        world = torch.stack([col, row, torch.zeros_like(row)], dim=-1)
        charts, _unresolved = grow_local_rank_complete_charts(0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG)
        self.assertGreater(len(charts), 0)
        for chart in charts:
            cols_side = chart.pixel_cols < (w // 2)
            self.assertTrue(bool(cols_side.all()) or bool((~cols_side).all()), "a chart crossed the occluded gap column")


class RuntimeSafetyValveTest(unittest.TestCase):
    def test_max_patches_per_blob_stops_further_extraction_and_reports_it_separately(self) -> None:
        comp, rep, world = _maps(40, 40)
        charts, unresolved = grow_local_rank_complete_charts(
            0, comp, rep, world, _RES_U, _RES_V, _DEG, _DEG, max_patches_per_blob=1
        )
        self.assertEqual(len(charts), 1)
        self.assertTrue(any(u.reason == REASON_RUNTIME_CAP_SKIPPED for u in unresolved))


if __name__ == "__main__":
    unittest.main()
