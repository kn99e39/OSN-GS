"""Read-only covariance-footprint attribution for Worklog 89 failures."""

import unittest

import torch

from osn_gs.surface.torch_chart_unit_surface_topology_attribution import (
    CENTER_UNDERSAMPLING,
    GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH,
    MULTILAYER_OR_VOLUMETRIC,
    RELATION_FALSE_NEGATIVE,
    TRUE_SUPPORT_GAP,
    attribute_failed_chart_unit_surface_topology,
)
from osn_gs.surface.torch_dense_surface_consistency_components import (
    RELATION_AMBIGUOUS,
    RELATION_SAME_SURFACE,
    DenseConsistencyEdge,
)


def _flat_covariance(count: int, tangent: float = 0.2, thickness: float = 0.01) -> torch.Tensor:
    covariance = torch.zeros((count, 3, 3), dtype=torch.float32)
    covariance[:, 0, 0] = tangent * tangent
    covariance[:, 1, 1] = tangent * tangent
    covariance[:, 2, 2] = thickness * thickness
    return covariance


class SurfaceTopologyAttributionTest(unittest.TestCase):
    def test_overlap_missing_from_center_graph_is_center_undersampling(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        result = attribute_failed_chart_unit_surface_topology(points, _flat_covariance(2), [0, 1], ())
        self.assertEqual(result.primary_cause, CENTER_UNDERSAMPLING)
        self.assertEqual(result.missing_center_graph_pair_count, 1)
        self.assertTrue(result.valid_local_surface_complex_plausible)

    def test_compatible_ambiguous_candidate_is_relation_false_negative(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        edges = (DenseConsistencyEdge(0, 1, RELATION_AMBIGUOUS, 0.0, 0.0),)
        result = attribute_failed_chart_unit_surface_topology(points, _flat_covariance(2), [0, 1], edges)
        self.assertEqual(result.primary_cause, RELATION_FALSE_NEGATIVE)
        self.assertEqual(result.rejected_relation_pair_count, 1)

    def test_no_footprint_continuation_is_true_support_gap(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = attribute_failed_chart_unit_surface_topology(points, _flat_covariance(2), [0, 1], ())
        self.assertEqual(result.primary_cause, TRUE_SUPPORT_GAP)
        self.assertEqual(result.true_gap_node_count, 2)
        self.assertFalse(result.valid_local_surface_complex_plausible)

    def test_overlapping_but_depth_incompatible_support_is_multilayer(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        covariance = _flat_covariance(2)
        covariance[1] = torch.diag(torch.tensor([0.0001, 0.04, 0.04]))
        result = attribute_failed_chart_unit_surface_topology(points, covariance, [0, 1], ())
        self.assertEqual(result.primary_cause, MULTILAYER_OR_VOLUMETRIC)
        self.assertEqual(result.layer_conflict_pair_count, 1)

    def test_dense_accepted_center_support_with_failed_face_is_graph_mismatch(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.15, 0.0]])
        edges = (
            DenseConsistencyEdge(0, 1, RELATION_SAME_SURFACE, 1.0, 0.0),
            DenseConsistencyEdge(1, 2, RELATION_SAME_SURFACE, 1.0, 0.0),
            DenseConsistencyEdge(0, 2, RELATION_SAME_SURFACE, 1.0, 0.0),
        )
        result = attribute_failed_chart_unit_surface_topology(points, _flat_covariance(3), [0, 1, 2], edges)
        self.assertEqual(result.primary_cause, GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH)
        self.assertEqual(result.accepted_same_surface_pair_count, 3)
        self.assertTrue(result.valid_local_surface_complex_plausible)


if __name__ == "__main__":
    unittest.main()
