from __future__ import annotations

"""Phase B isolated artificial patch-boundary reconciliation tests."""

import unittest

import torch

from osn_gs.surface.torch_boundary_reconciliation import (
    PatchEdgePair,
    fit_reconciled_patch_graph,
)
from osn_gs.surface.torch_patch_boundary import BOUNDARY_RECONCILED_INTERNAL, BOUNDARY_UNCLASSIFIED


def _adjacent_patch_samples(kind: str = "coplanar", gap: float = 0.0, count: int = 320):
    generator = torch.Generator().manual_seed(41)
    uv_a = torch.rand((count, 2), generator=generator)
    uv_b = torch.rand((count, 2), generator=generator)
    points_a = torch.stack((uv_a[:, 0], uv_a[:, 1], torch.zeros(count)), dim=1)
    if kind == "orthogonal":
        points_b = torch.stack((torch.ones(count), uv_b[:, 1], uv_b[:, 0]), dim=1)
    else:
        points_b = torch.stack((1.0 + float(gap) + uv_b[:, 0], uv_b[:, 1], torch.zeros(count)), dim=1)
    return [points_a, points_b], [uv_a, uv_b]


class BoundaryReconciliationTest(unittest.TestCase):
    def test_coplanar_artificial_seam_is_jointly_reconciled(self):
        points, uv = _adjacent_patch_samples()
        result = fit_reconciled_patch_graph(
            points,
            uv,
            [PatchEdgePair("seam", 0, "u1", 1, "u0")],
            resolution_u=7,
            resolution_v=6,
        )
        self.assertTrue(result.used_joint_fit)
        self.assertEqual(result.decisions[0].state, BOUNDARY_RECONCILED_INTERNAL)
        self.assertEqual(result.decisions[0].reason, "joint_shared_control_valid")
        self.assertLess(result.decisions[0].post_fit.gap_max, 1e-5)
        self.assertTrue(all(item["valid"] for item in result.jacobian_validity))

    def test_orthogonal_shared_edge_is_not_rejected_by_normal_angle(self):
        points, uv = _adjacent_patch_samples(kind="orthogonal")
        result = fit_reconciled_patch_graph(
            points,
            uv,
            [PatchEdgePair("right_angle", 0, "u1", 1, "u0")],
            resolution_u=7,
            resolution_v=6,
        )
        self.assertGreater(result.decisions[0].pre_fit.normal_angle_deg_mean, 80.0)
        self.assertTrue(result.used_joint_fit)
        self.assertEqual(result.decisions[0].state, BOUNDARY_RECONCILED_INTERNAL)

    def test_curved_seam_is_deterministic_across_repeat_and_pair_order(self):
        generator = torch.Generator().manual_seed(53)
        patch_points, patch_uv = [], []
        for patch_id in range(3):
            uv = torch.rand((260, 2), generator=generator)
            x = float(patch_id) + uv[:, 0]
            y = uv[:, 1]
            z = 0.06 * torch.sin(0.8 * x) * torch.cos(1.2 * y)
            patch_points.append(torch.stack((x, y, z), dim=1))
            patch_uv.append(uv)
        adjacency = [
            PatchEdgePair("seam_01", 0, "u1", 1, "u0"),
            PatchEdgePair("seam_12", 1, "u1", 2, "u0"),
        ]
        first = fit_reconciled_patch_graph(
            patch_points, patch_uv, adjacency, resolution_u=7, resolution_v=6
        )
        second = fit_reconciled_patch_graph(
            patch_points, patch_uv, list(reversed(adjacency)), resolution_u=7, resolution_v=6
        )
        self.assertTrue(first.used_joint_fit and second.used_joint_fit)
        self.assertEqual([item.pair_id for item in first.decisions], ["seam_01", "seam_12"])
        self.assertEqual([item.pair_id for item in second.decisions], ["seam_01", "seam_12"])
        for surface_a, surface_b in zip(first.surfaces, second.surfaces):
            self.assertTrue(torch.equal(surface_a.control_grid, surface_b.control_grid))
        for decision in first.decisions:
            self.assertEqual(decision.state, BOUNDARY_RECONCILED_INTERNAL)
    def test_disconnected_gap_stays_unclassified_without_joint_fit(self):
        points, uv = _adjacent_patch_samples(gap=0.2)
        result = fit_reconciled_patch_graph(
            points,
            uv,
            [PatchEdgePair("gap", 0, "u1", 1, "u0")],
            resolution_u=7,
            resolution_v=6,
            max_normalized_gap=0.05,
        )
        self.assertFalse(result.used_joint_fit)
        self.assertEqual(result.decisions[0].state, BOUNDARY_UNCLASSIFIED)
        self.assertEqual(result.decisions[0].reason, "scale_normalized_gap")
        self.assertGreater(result.decisions[0].pre_fit.normalized_gap_rms, 0.05)


if __name__ == "__main__":
    unittest.main()