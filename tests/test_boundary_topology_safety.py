"""Worklog 40 (task section 8): topology safety of the worklog 39
direct-or-2-hop boundary support certificate.

The certificate lets two boundary candidates be adjacent when they share a
NON-candidate interior node in the region's accepted graph. This file proves
that rule recovers real perimeter adjacency without creating shortcuts across
concavities, holes, narrow necks, or between near-touching loops.
"""

from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_boundary_self_intersection import validate_simple_closed_loop
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

_HALF_EXTENT = 0.12 * 6  # _masked_sheet default grid half-width


def _masked_sheet(mask, count_per_axis: int = 13, spacing: float = 0.12, seed: int = 0):
    """Flat surfel sheet keeping only grid cells accepted by ``mask``."""
    generator = torch.Generator().manual_seed(seed)
    lin = (torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0) * spacing
    half = spacing * (count_per_axis - 1) / 2.0
    points = [
        (float(lin[i]), float(lin[j]), 0.0)
        for i in range(count_per_axis) for j in range(count_per_axis)
        if mask(float(lin[i]), float(lin[j]), half)
    ]
    positions = torch.tensor(points) + 0.001 * torch.randn(len(points), 3, generator=generator)
    scale = torch.tensor([0.05, 0.05, 0.002]).expand(len(points), 3).clone()
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(len(points), 4).clone()
    return positions, covariance_from_scale_rotation(scale, quaternion)


def _construct(positions, covariances):
    return construct_visible_nurbs_from_gaussians(
        positions, covariance=covariances, stable_ids=tuple(range(positions.shape[0])),
    )


def _closed_loops(result):
    return [c for c in result.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]


class ConcavityShortcutTest(unittest.TestCase):
    """A U-shaped sheet's boundary must follow the notch walls, not cut across."""

    @staticmethod
    def _u_mask(x, y, half):
        return not (abs(x) < half * 0.34 and y > -half * 0.15)

    def test_u_shape_recovers_one_simple_loop(self):
        positions, covariances = _masked_sheet(self._u_mask)
        result = _construct(positions, covariances)
        loops = _closed_loops(result)
        self.assertEqual(len(loops), 1)
        lookup = {i: tuple(positions[i].tolist()) for i in range(positions.shape[0])}
        report = validate_simple_closed_loop([lookup[s] for s in loops[0].ordered_source_ids])
        self.assertTrue(report.is_simple_polygon, report.reasons)

    def test_u_shape_loop_never_enters_the_removed_notch(self):
        positions, covariances = _masked_sheet(self._u_mask)
        result = _construct(positions, covariances)
        loops = _closed_loops(result)
        self.assertEqual(len(loops), 1)
        for stable_id in loops[0].ordered_source_ids:
            x, y = float(positions[stable_id][0]), float(positions[stable_id][1])
            inside_notch = abs(x) < _HALF_EXTENT * 0.34 and y > -_HALF_EXTENT * 0.15
            self.assertFalse(inside_notch, (stable_id, x, y))


class HoleShortcutTest(unittest.TestCase):
    @staticmethod
    def _hole_mask(x, y, half):
        return (x * x + y * y) ** 0.5 > half * 0.30

    def test_hole_yields_separate_outer_and_inner_loops(self):
        positions, covariances = _masked_sheet(self._hole_mask)
        result = _construct(positions, covariances)
        loops = _closed_loops(result)
        self.assertGreaterEqual(len(loops), 2)
        sizes = sorted(len(c.ordered_source_ids) for c in loops)
        # The inner hole loop must stay distinct from the outer perimeter --
        # a shortcut across the hole would fuse them into one loop.
        self.assertGreater(sizes[-1], sizes[0])

    def test_no_loop_spans_the_hole(self):
        positions, covariances = _masked_sheet(self._hole_mask)
        result = _construct(positions, covariances)
        lookup = {i: tuple(positions[i].tolist()) for i in range(positions.shape[0])}
        for component in _closed_loops(result):
            report = validate_simple_closed_loop([lookup[s] for s in component.ordered_source_ids])
            self.assertTrue(report.is_simple_polygon, report.reasons)


class NarrowNeckShortcutTest(unittest.TestCase):
    @staticmethod
    def _neck_mask(x, y, half):
        return (abs(y) < half * 0.12) or (abs(x) > half * 0.45)

    def test_narrow_neck_does_not_fuse_the_two_lobes(self):
        positions, covariances = _masked_sheet(self._neck_mask)
        result = _construct(positions, covariances)
        loops = _closed_loops(result)
        self.assertGreaterEqual(len(loops), 2)
        lookup = {i: tuple(positions[i].tolist()) for i in range(positions.shape[0])}
        for component in _closed_loops(result):
            report = validate_simple_closed_loop([lookup[s] for s in component.ordered_source_ids])
            self.assertTrue(report.is_simple_polygon, report.reasons)


class NearTouchingLoopsTest(unittest.TestCase):
    @staticmethod
    def _two_patches(gap: float, count_per_axis: int = 7, spacing: float = 0.12, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        lin = (torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0) * spacing
        offset = spacing * (count_per_axis - 1) / 2.0 + gap / 2.0
        points = [
            (float(lin[i]) + sign * offset, float(lin[j]), 0.0)
            for sign in (-1.0, 1.0)
            for i in range(count_per_axis) for j in range(count_per_axis)
        ]
        positions = torch.tensor(points) + 0.001 * torch.randn(len(points), 3, generator=generator)
        scale = torch.tensor([0.05, 0.05, 0.002]).expand(len(points), 3).clone()
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(len(points), 4).clone()
        return positions, covariance_from_scale_rotation(scale, quaternion)

    def test_near_touching_patches_never_share_a_loop(self):
        for gap in (0.30, 0.45, 0.60):
            positions, covariances = self._two_patches(gap)
            result = _construct(positions, covariances)
            loops = _closed_loops(result)
            self.assertEqual(len(loops), 2, gap)
            for component in loops:
                xs = [float(positions[s][0]) for s in component.ordered_source_ids]
                self.assertFalse(
                    min(xs) < 0.0 < max(xs),
                    f"a loop spanned both patches at gap={gap}",
                )


if __name__ == "__main__":
    unittest.main()
