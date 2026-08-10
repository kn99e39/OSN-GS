"""Worklog 83: chart-scale assembly over worklog 82 micro-components."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_dense_chart_unit_assembly import (
    RELATION_ACCEPTED,
    RELATION_AMBIGUOUS,
    RELATION_CREASE_VETOED,
    build_chart_unit_assembly,
)
from osn_gs.surface.torch_dense_surface_consistency_components import (
    build_dense_surface_consistency_components,
)
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation


def _flat_grid(n: int, spacing: float = 0.05, offset_x: float = 0.0) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.float32) * spacing
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((u.reshape(-1) + offset_x, v.reshape(-1), torch.zeros(n * n)), dim=1)


def _flat_covariance(count: int, tangent_scale: float = 0.05, normal_thickness: float = 0.002) -> torch.Tensor:
    scale = torch.tensor([tangent_scale, tangent_scale, normal_thickness]).expand(count, 3)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4)
    return covariance_from_scale_rotation(scale, identity_quat)


class AdjacentSplitSheetTest(unittest.TestCase):
    """Two halves of one flat sheet that worklog 82's kNN degree cap split
    into two separate micro-components should still be assemblable at
    chart scale, since the aggregate evidence supports continuation."""

    def _two_touching_halves(self):
        left = _flat_grid(6, spacing=0.05)
        right = _flat_grid(6, spacing=0.05, offset_x=0.30)  # a small gap, still close
        positions = torch.cat((left, right), dim=0)
        covariance = _flat_covariance(int(positions.shape[0]))
        return positions, covariance

    def test_two_close_coherent_halves_are_accepted_into_one_chart_unit(self):
        positions, covariance = self._two_touching_halves()
        # Force two micro-components directly (as if worklog 82 had split
        # them) rather than depending on its own kNN behavior for this
        # fixture -- this test is about ASSEMBLY, not about re-deriving
        # worklog 82's own component boundaries.
        left_idx = tuple(range(36))
        right_idx = tuple(range(36, 72))
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=(left_idx, right_idx), non_manifold_flags=(False, False),
            full_evidence_spacing=0.05,
        )
        self.assertEqual(result.chart_unit_count, 1)
        self.assertEqual(len(result.chart_units[0].member_indices), 72)
        self.assertEqual(result.edges[0].relation, RELATION_ACCEPTED)


class DistantUnrelatedComponentsTest(unittest.TestCase):
    def test_far_apart_components_are_not_even_candidates(self):
        left = _flat_grid(6, spacing=0.05)
        far = _flat_grid(6, spacing=0.05, offset_x=50.0)
        positions = torch.cat((left, far), dim=0)
        covariance = _flat_covariance(int(positions.shape[0]))
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=(tuple(range(36)), tuple(range(36, 72))), non_manifold_flags=(False, False),
            full_evidence_spacing=0.05,
        )
        self.assertEqual(result.chart_unit_count, 2)
        self.assertEqual(len(result.edges), 0, "far-apart components must not even become a scored candidate pair")


class CreaseVetoTest(unittest.TestCase):
    def test_crease_arc_between_close_components_prevents_assembly(self):
        left = _flat_grid(6, spacing=0.05)
        right = _flat_grid(6, spacing=0.05, offset_x=0.30)
        positions = torch.cat((left, right), dim=0)
        covariance = _flat_covariance(int(positions.shape[0]))
        # Two differently-typed arcs straddling the gap between the halves.
        left_arc_start = torch.tensor([[0.20, -1.0, 0.0]])
        left_arc_end = torch.tensor([[0.20, 1.0, 0.0]])
        right_arc_start = torch.tensor([[0.35, -1.0, 0.0]])
        right_arc_end = torch.tensor([[0.35, 1.0, 0.0]])
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=(tuple(range(36)), tuple(range(36, 72))), non_manifold_flags=(False, False),
            full_evidence_spacing=0.05,
            arc_starts=torch.cat((left_arc_start, right_arc_start), dim=0),
            arc_ends=torch.cat((left_arc_end, right_arc_end), dim=0),
            arc_kinds=["crease", "physical_termination"],
        )
        self.assertEqual(result.chart_unit_count, 2)
        self.assertEqual(result.edges[0].relation, RELATION_CREASE_VETOED)


class AmbiguousSignalTest(unittest.TestCase):
    def test_close_but_orientation_incoherent_components_are_ambiguous_not_merged(self):
        left = _flat_grid(6, spacing=0.05)
        left_cov = _flat_covariance(int(left.shape[0]))
        # A close but ORTHOGONALLY oriented patch -- proximity gate passes,
        # but normal/residual/occupancy should not jointly justify a merge.
        right_xy = _flat_grid(6, spacing=0.05)
        right = torch.stack((right_xy[:, 0] + 0.30, right_xy[:, 2], right_xy[:, 1]), dim=1)
        right_scale = torch.tensor([0.05, 0.002, 0.05]).expand(int(right.shape[0]), 3)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(int(right.shape[0]), 4)
        right_cov = covariance_from_scale_rotation(right_scale, identity_quat)
        positions = torch.cat((left, right), dim=0)
        covariance = torch.cat((left_cov, right_cov), dim=0)
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=(tuple(range(36)), tuple(range(36, 72))), non_manifold_flags=(False, False),
            full_evidence_spacing=0.05,
        )
        self.assertEqual(result.chart_unit_count, 2)
        self.assertIn(result.edges[0].relation, (RELATION_AMBIGUOUS, RELATION_CREASE_VETOED))


class NonManifoldExclusionTest(unittest.TestCase):
    def test_non_manifold_flagged_component_is_excluded_from_assembly(self):
        left = _flat_grid(6, spacing=0.05)
        right = _flat_grid(6, spacing=0.05, offset_x=0.30)
        positions = torch.cat((left, right), dim=0)
        covariance = _flat_covariance(int(positions.shape[0]))
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=(tuple(range(36)), tuple(range(36, 72))), non_manifold_flags=(True, False),
            full_evidence_spacing=0.05,
        )
        self.assertEqual(result.chart_unit_count, 1)
        self.assertEqual(result.excluded_non_manifold_component_count, 1)
        self.assertEqual(result.chart_units[0].micro_component_indices, (1,))


class NeverMergesOnFitQualityTest(unittest.TestCase):
    def test_assembly_signature_takes_no_fit_quality_input(self):
        import inspect

        from osn_gs.surface.torch_dense_chart_unit_assembly import build_chart_unit_assembly as fn

        params = inspect.signature(fn).parameters
        for forbidden in ("fit", "nurbs", "jacobian", "p95", "held_out", "extrapolat"):
            for name in params:
                self.assertNotIn(forbidden, name.lower())


class IntegrationWithWorklog82Test(unittest.TestCase):
    def test_real_worklog82_component_output_is_a_valid_input(self):
        positions = _flat_grid(6, spacing=0.05)
        covariance = _flat_covariance(int(positions.shape[0]))
        consistency = build_dense_surface_consistency_components(0, positions, covariance=covariance)
        micro_components = tuple(c.member_indices for c in consistency.components)
        flags = tuple(c.non_manifold_suspected for c in consistency.components)
        result = build_chart_unit_assembly(
            0, positions, covariance=covariance,
            micro_components=micro_components, non_manifold_flags=flags,
            full_evidence_spacing=0.05,
        )
        self.assertEqual(result.micro_component_count, len(micro_components))
        total_assembled = sum(len(u.member_indices) for u in result.chart_units)
        total_eligible = sum(len(c) for c, f in zip(micro_components, flags) if not f)
        self.assertEqual(total_assembled, total_eligible)


if __name__ == "__main__":
    unittest.main()
