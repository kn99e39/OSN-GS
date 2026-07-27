from types import SimpleNamespace
import unittest
import torch

from osn_gs.surface.torch_boundary_planar_partition import assess_non_overlapping_planar_partition


def loop(label, nested, points):
    return SimpleNamespace(label=label, nested_in_outer_label=nested, ordered_boundary_world_points=points)


class BoundaryPlanarPartitionTest(unittest.TestCase):
    def test_opposite_oriented_nested_loops_require_materialization_not_duplication(self):
        frame = SimpleNamespace(apply=lambda values, clamp=False: torch.as_tensor(values)[:, :2])
        outer = loop(1, None, [[0., 0., 0.], [0., 3., 0.], [3., 3., 0.], [3., 0., 0.], [0., 0., 0.]])
        hole_a = loop(2, 1, [[.5, .5, 0.], [1., .5, 0.], [1., 1., 0.], [.5, 1., 0.], [.5, .5, 0.]])
        hole_b = loop(3, 1, [[2., 2., 0.], [2.5, 2., 0.], [2.5, 2.5, 0.], [2., 2.5, 0.], [2., 2., 0.]])
        result = assess_non_overlapping_planar_partition(SimpleNamespace(frame=frame, outer_loops=[outer], hole_loops=[hole_a, hole_b]))
        self.assertEqual(result.state, "review_required")
        self.assertEqual(result.reason, "partition_materialization_required")
        self.assertEqual(result.provenance["outer_boundary_owner_count"], 1)


if __name__ == "__main__":
    unittest.main()