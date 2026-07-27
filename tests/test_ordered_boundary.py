import unittest
import torch

from osn_gs.surface.torch_nurbs import uv_frame_from_axes
from osn_gs.surface.torch_ordered_boundary import ordered_closed_boundary_world_loops


class OrderedBoundaryTest(unittest.TestCase):
    def _frame(self):
        points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        return uv_frame_from_axes(points, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]))

    def test_closed_square_is_stitched_deterministically(self):
        mask = torch.zeros((8, 8), dtype=torch.bool)
        mask[2:6, 2:6] = True
        first = ordered_closed_boundary_world_loops(mask, self._frame())
        second = ordered_closed_boundary_world_loops(mask, self._frame())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertGreaterEqual(len(first[0]), 4)

    def test_open_mask_is_not_invented_as_closed_loop(self):
        mask = torch.zeros((8, 8), dtype=torch.bool)
        mask[0:4, 2:6] = True
        self.assertEqual(ordered_closed_boundary_world_loops(mask, self._frame()), ())


if __name__ == '__main__':
    unittest.main()