from __future__ import annotations

"""Adversarial region-formation fixtures for Worklog 117/124.

They contain positions and covariance only; neither scene names nor labels are
read by the formation code.
"""

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    GaussianReliabilityScene,
    _flat_grid,
    make_gaussian_reliability_scene,
)


def make_cylinder_phase_alias_scene(*, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124: a genuine volumetric phase-alias stress fixture. A
    cylinder's side wall is periodic in the circumferential direction, so two
    points on OPPOSITE sides of the ring have exactly anti-parallel normals --
    under the orientation-insensitive ``abs(dot(n_i, n_j))`` comparison used
    throughout this pipeline, that reads as PERFECTLY aligned (1.0), even
    though the two points are on opposite faces of the real object and
    geodesically far apart along the surface. A wide candidate radius is
    needed to even surface these long-range pairs as candidates; this replaces
    the earlier ad hoc sine-sheet "long shortcut" fixture with a real solid's
    own periodicity."""
    return make_gaussian_reliability_scene("cylinder", seed=seed)


def make_genuine_narrow_connection_scene() -> GaussianReliabilityScene:
    """Two planar lobes joined by a multi-sample, locally continuous neck."""
    left, covariance = _flat_grid(5, 0.10, origin=(-0.42, 0.0, 0.0))
    right, right_covariance = _flat_grid(5, 0.10, origin=(0.42, 0.0, 0.0), seed=1)
    # Three parallel rows make multiple local paths through the neck rather
    # than a single pairwise bridge.  Its covariance remains the same smooth
    # planar frame as both lobes.
    neck_x = torch.linspace(-0.22, 0.22, 7)
    neck = torch.cat([torch.stack((neck_x, torch.full_like(neck_x, y), torch.zeros_like(neck_x)), 1) for y in (-0.1, 0.0, 0.1)])
    neck_covariance = covariance[:1].expand(neck.shape[0], -1, -1).clone()
    positions = torch.cat((left, neck, right), 0)
    covariances = torch.cat((covariance, neck_covariance, right_covariance), 0)
    labels = ("left",) * len(left) + ("neck",) * len(neck) + ("right",) * len(right)
    return GaussianReliabilityScene("genuine_narrow_connection", positions, covariances, "Two broad lobes joined by a multi-edge smooth neck.", labels)
