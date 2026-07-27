from types import SimpleNamespace
import unittest

import torch

from osn_gs.surface.torch_boundary_source_fidelity import measure_observed_boundary_source_fidelity


class BoundarySourceFidelityTest(unittest.TestCase):
    def test_closed_duplicate_sample_is_not_counted_twice(self):
        curve = SimpleNamespace(world=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]))
        points = curve.world[:-1].clone()
        first = measure_observed_boundary_source_fidelity(curve, points)
        second = measure_observed_boundary_source_fidelity(curve, points)
        self.assertEqual(first.payload(), second.payload())
        self.assertEqual(first.boundary_sample_count, 4)
        self.assertEqual(first.source_point_count, 4)
        self.assertEqual(first.local_spacing_median, 1.0)
        self.assertEqual(first.boundary_to_source_maximum, 0.0)
        self.assertEqual(first.normalized_median_distance, 0.0)


if __name__ == "__main__":
    unittest.main()