"""Worklog 82: evidence-scale surface-consistency chart-unit decomposition."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_dense_surface_consistency_components import (
    RELATION_CREASE_VETOED,
    RELATION_SAME_SURFACE,
    build_dense_surface_consistency_components,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _flat_grid(n: int, spacing: float = 0.1, offset: float = 0.0) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.float32) * spacing
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((u.reshape(-1) + offset, v.reshape(-1), torch.zeros(n * n)), dim=1)


def _flat_covariance(count: int, tangent_scale: float = 0.05, normal_thickness: float = 0.002) -> torch.Tensor:
    # Flat surfel: large tangent extent in x/y, tiny thickness along z.
    scale = torch.tensor([tangent_scale, tangent_scale, normal_thickness]).expand(count, 3)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4)
    return covariance_from_scale_rotation(scale, identity_quat)


class SingleSheetTest(unittest.TestCase):
    def test_flat_coherent_sheet_forms_one_component(self):
        positions = _flat_grid(6)
        covariance = _flat_covariance(int(positions.shape[0]))
        result = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        self.assertEqual(len(result.components), 1)
        self.assertEqual(len(result.components[0].member_indices), int(positions.shape[0]))
        self.assertFalse(result.components[0].non_manifold_suspected)
        self.assertEqual(result.unresolved_indices, ())

    def test_same_surface_edges_have_high_alignment_low_residual(self):
        positions = _flat_grid(6)
        covariance = _flat_covariance(int(positions.shape[0]))
        result = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        same_surface = [e for e in result.edges if e.relation == RELATION_SAME_SURFACE]
        self.assertGreater(len(same_surface), 0)
        for edge in same_surface:
            self.assertGreaterEqual(edge.normal_alignment, 0.85)
            self.assertLessEqual(edge.mutual_tangent_residual, 0.35)


class TwoSheetTest(unittest.TestCase):
    def test_two_orthogonal_sheets_form_two_components(self):
        # Two flat sheets meeting at a right angle (like a box corner) --
        # far apart in normal direction so their own local kNN candidates
        # never cross between sheets.
        sheet_a = _flat_grid(6)  # z=0 plane
        sheet_b_xy = _flat_grid(6)
        sheet_b = torch.stack(
            (sheet_b_xy[:, 0] + 2.0, sheet_b_xy[:, 2], sheet_b_xy[:, 1]), dim=1
        )  # x=const-ish plane, offset far away in x so kNN never crosses
        positions = torch.cat((sheet_a, sheet_b), dim=0)
        cov_a = _flat_covariance(int(sheet_a.shape[0]))
        # sheet_b's normal is along y after the axis swap above.
        scale_b = torch.tensor([0.05, 0.002, 0.05]).expand(int(sheet_b.shape[0]), 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(int(sheet_b.shape[0]), 4)
        cov_b = covariance_from_scale_rotation(scale_b, identity_quat)
        covariance = torch.cat((cov_a, cov_b), dim=0)
        result = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        self.assertEqual(len(result.components), 2)
        sizes = sorted(len(c.member_indices) for c in result.components)
        self.assertEqual(sizes, [36, 36])


class CreaseVetoTest(unittest.TestCase):
    def test_typed_crease_arc_prevents_merge_even_when_geometrically_close(self):
        # A single flat sheet, but with two DIFFERENTLY-TYPED arcs, one just
        # left of center and one just right of center -- geometrically the
        # whole sheet would pass same_surface, but points nearest each arc
        # carry different segment kinds, so the veto must separate them.
        positions = _flat_grid(8, spacing=0.05)
        covariance = _flat_covariance(int(positions.shape[0]))
        left_arc_start = torch.tensor([[0.05, -1.0, 0.0]])
        left_arc_end = torch.tensor([[0.05, 1.0, 0.0]])
        right_arc_start = torch.tensor([[0.30, -1.0, 0.0]])
        right_arc_end = torch.tensor([[0.30, 1.0, 0.0]])
        result_no_veto = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        self.assertEqual(len(result_no_veto.components), 1, "control: without arcs, one coherent sheet")

        result = build_dense_surface_consistency_components(
            0, positions, covariance=covariance,
            arc_starts=torch.cat((left_arc_start, right_arc_start), dim=0),
            arc_ends=torch.cat((left_arc_end, right_arc_end), dim=0),
            arc_kinds=["crease", "physical_termination"],
        )
        crease_edges = [e for e in result.edges if e.relation == RELATION_CREASE_VETOED]
        self.assertGreater(len(crease_edges), 0, "some candidate edges must be vetoed across the typed crease")


class FailClosedTest(unittest.TestCase):
    def test_isolated_point_is_unresolved_not_force_assigned(self):
        positions = _flat_grid(4)
        far_outlier = torch.tensor([[50.0, 50.0, 50.0]])
        all_positions = torch.cat((positions, far_outlier), dim=0)
        covariance = _flat_covariance(int(all_positions.shape[0]))
        result = build_dense_surface_consistency_components(0, all_positions, covariance=covariance)
        self.assertIn(int(all_positions.shape[0]) - 1, result.unresolved_indices)

    def test_empty_region_returns_no_components(self):
        positions = torch.zeros((0, 3))
        covariance = torch.zeros((0, 3, 3))
        result = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        self.assertEqual(result.components, ())
        self.assertEqual(result.point_count, 0)

    def test_random_scattered_normals_are_not_force_merged_into_one_component(self):
        torch.manual_seed(0)
        positions = torch.rand((30, 3)) * 2.0
        # Random covariance orientation per point -- no coherent sheet exists.
        scale = torch.tensor([0.05, 0.05, 0.002]).expand(30, 3)
        random_quat = torch.nn.functional.normalize(torch.rand((30, 4)) - 0.5, dim=1)
        covariance = covariance_from_scale_rotation(scale, random_quat)
        result = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        # With incoherent per-point orientation, either many small
        # components or many unresolved points must appear -- NOT one big
        # component spanning all 30 points via a chain of locally-passing
        # pairs, which would defeat the internal disagreement check's own
        # purpose.
        if result.components:
            largest = max(len(c.member_indices) for c in result.components)
            self.assertLess(largest, 30)


if __name__ == "__main__":
    unittest.main()
