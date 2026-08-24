from __future__ import annotations

"""Worklog 111 required synthetic contracts A-F (directive section 11).

Pure-logic / pure-torch: no CUDA, no renderer. Chart domains are built
directly from synthetic (component_id_map, representative_id_map) pairs
exactly as the real-scene script derives them from
`render_with_pixel_representative` + the frozen Worklog 107/109 subset_ids
-- only the renderer call itself is out of scope for these fixtures.
"""

import unittest

import torch

from osn_gs.surface.torch_camera_observed_chart_domains import (
    build_view_chart_candidates,
    valid_chart_mask,
)
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_lsq, project_torch_points_to_nurbs

_RESOLUTION_U = 8
_RESOLUTION_V = 4
_DEGREE = 2
_MIN_MEMBERS = _RESOLUTION_U * _RESOLUTION_V  # directive section 7: derived, not tuned


def _grid_positions(rows: int, cols: int, z_fn) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(rows, dtype=torch.float32), torch.arange(cols, dtype=torch.float32), indexing="ij")
    x = ii / max(rows - 1, 1)
    y = jj / max(cols - 1, 1)
    z = z_fn(x, y)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


class ChartA_PlanarSheetTest(unittest.TestCase):
    def test_planar_sheet_yields_valid_chart_and_accurate_fit(self):
        rows, cols = 10, 10
        positions = _grid_positions(rows, cols, lambda x, y: torch.zeros_like(x))
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)

        vc = build_view_chart_candidates(0, comp, rep)
        self.assertTrue(bool(valid_chart_mask(vc, _MIN_MEMBERS).all()))

        member_positions = positions[vc.member_representative_id]
        surface, uv = fit_torch_visible_surface_lsq(
            member_positions, resolution_u=_RESOLUTION_U, resolution_v=_RESOLUTION_V,
            degree_u=_DEGREE, degree_v=_DEGREE, initial_uv=vc.member_uv,
        )
        fitted = surface.evaluate(uv)
        residual = (fitted - member_positions).norm(dim=-1)
        self.assertLess(float(residual.max().item()), 0.05)


class ChartB_CurvedSheetTest(unittest.TestCase):
    def test_curved_sheet_fits_continuously_despite_rotating_normals(self):
        rows, cols = 12, 12
        positions = _grid_positions(rows, cols, lambda x, y: 0.3 * torch.sin(x * 3.14159))
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)

        vc = build_view_chart_candidates(0, comp, rep)
        self.assertTrue(bool(valid_chart_mask(vc, _MIN_MEMBERS).all()))
        member_positions = positions[vc.member_representative_id]
        surface, uv = fit_torch_visible_surface_lsq(
            member_positions, resolution_u=_RESOLUTION_U, resolution_v=_RESOLUTION_V,
            degree_u=_DEGREE, degree_v=_DEGREE, initial_uv=vc.member_uv, correction_rounds=3,
        )
        fitted = surface.evaluate(uv)
        residual = (fitted - member_positions).norm(dim=-1)
        self.assertLess(float(residual.mean().item()), 0.05)
        normals = surface.normals(uv)
        # Rotating normals: first-row vs last-row normals must differ (curved, not flat).
        self.assertGreater(float((normals[0] - normals[-1]).norm().item()), 0.1)


class ChartC_MultiViewSameComponentTest(unittest.TestCase):
    def test_one_component_two_views_yields_two_charts_same_component_id(self):
        rows, cols = 9, 9
        positions = _grid_positions(rows, cols, lambda x, y: torch.zeros_like(x))
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep_view0 = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
        # Second view observes the SAME component with a shifted representative
        # id space, simulating a distinct camera's own representative capture.
        rep_view1 = rep_view0 + (rows * cols)

        vc0 = build_view_chart_candidates(0, comp, rep_view0)
        vc1 = build_view_chart_candidates(1, comp, rep_view1)
        self.assertEqual(vc0.blob_count, 1)
        self.assertEqual(vc1.blob_count, 1)
        self.assertEqual(int(vc0.blob_component_id[0].item()), int(vc1.blob_component_id[0].item()))
        self.assertTrue(bool(valid_chart_mask(vc0, _MIN_MEMBERS).all()))
        self.assertTrue(bool(valid_chart_mask(vc1, _MIN_MEMBERS).all()))


class ChartD_OverlappingChartsTest(unittest.TestCase):
    def test_overlapping_charts_agree_in_overlap(self):
        rows, cols = 10, 10
        positions = _grid_positions(rows, cols, lambda x, y: 0.2 * torch.sin(x * 3.14159))
        comp = torch.zeros((rows, cols), dtype=torch.int64)
        rep = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)

        # View 0: natural row/col UV. View 1: transposed UV (a different
        # camera observing the same surfels with different image-space layout).
        vc0 = build_view_chart_candidates(0, comp, rep)
        vc1 = build_view_chart_candidates(1, comp.T.contiguous(), rep.T.contiguous())

        member_positions0 = positions[vc0.member_representative_id]
        surface0, uv0 = fit_torch_visible_surface_lsq(
            member_positions0, resolution_u=_RESOLUTION_U, resolution_v=_RESOLUTION_V,
            degree_u=_DEGREE, degree_v=_DEGREE, initial_uv=vc0.member_uv,
        )
        member_positions1 = positions[vc1.member_representative_id]
        surface1, uv1 = fit_torch_visible_surface_lsq(
            member_positions1, resolution_u=_RESOLUTION_U, resolution_v=_RESOLUTION_V,
            degree_u=_DEGREE, degree_v=_DEGREE, initial_uv=vc1.member_uv,
        )

        # Same representative set observed by both charts (full overlap here).
        shared = torch.isin(vc1.member_representative_id, vc0.member_representative_id)
        shared_ids = vc1.member_representative_id[shared]
        ground_truth = positions[shared_ids]

        uv_on_0 = project_torch_points_to_nurbs(ground_truth, surface0)
        uv_on_1 = project_torch_points_to_nurbs(ground_truth, surface1)
        point_on_0 = surface0.evaluate(uv_on_0)
        point_on_1 = surface1.evaluate(uv_on_1)
        disagreement = (point_on_0 - point_on_1).norm(dim=-1)
        self.assertLess(float(disagreement.mean().item()), 0.08)


class ChartE_DifferentComponentsNeverShareChartTest(unittest.TestCase):
    def test_two_different_components_adjacent_in_image_space_never_share_a_chart(self):
        comp = torch.tensor([[0, 0, 1, 1]] * 6, dtype=torch.int64)
        rep = torch.arange(24, dtype=torch.int64).reshape(6, 4)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertEqual(vc.blob_count, 2)
        for blob_id, rep_id in zip(vc.blob_of_member.tolist(), vc.member_representative_id.tolist()):
            column = rep_id % 4
            expected_blob_component = 0 if column < 2 else 1
            self.assertEqual(int(vc.blob_component_id[blob_id].item()), expected_blob_component)


class ChartF_OccludedGapSplitTest(unittest.TestCase):
    def test_occluded_gap_splits_into_two_components_no_chart_crosses(self):
        # Column 3 is a gap (-1, no representative anywhere) splitting an
        # otherwise-contiguous surface into two canonical components.
        comp = torch.tensor([
            [0, 0, 0, -1, 1, 1, 1],
        ] * 5, dtype=torch.int64)
        rep = torch.arange(35, dtype=torch.int64).reshape(5, 7)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertEqual(vc.blob_count, 2)
        components = set(vc.blob_component_id.tolist())
        self.assertEqual(components, {0, 1})
        # No member representative id from the gap column (3) appears anywhere.
        gap_ids = set(rep[:, 3].tolist())
        self.assertFalse(gap_ids & set(vc.member_representative_id.tolist()))


if __name__ == "__main__":
    unittest.main()
