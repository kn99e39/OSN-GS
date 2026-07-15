"""Analytic ground-truth surface + a ground-truth NURBS the renderer can overlay.

The synthetic scenes know their true surface ``z = f(x, y)`` and true patch
topology. This module turns that knowledge into:

- ``gt_surface_points`` / ``observed_gt_surface_points``: dense samples on the
  true surface, used by the GT-based accuracy and support metrics.
- ``gt_nurbs_payload``: a degree-1 NURBS (one patch per ground-truth region,
  e.g. two for ``crease``) written next to the generated ``nurbs_surface.json``
  as ``nurbs_surface_gt.json`` in the exact renderer format, so the true and
  reconstructed surfaces can be overlaid visually.
"""

from __future__ import annotations

from typing import Any

import torch

from .scenes import SyntheticGaussianScene
from .support_domains import mask_on_grid
from .support_domains import mask_on_grid

_DOMAIN = (-1.0, 1.0)


def _grid_xy(x_range: tuple[float, float], y_range: tuple[float, float], nu: int, nv: int) -> torch.Tensor:
    xs = torch.linspace(x_range[0], x_range[1], nu)
    ys = torch.linspace(y_range[0], y_range[1], nv)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def gt_surface_points(scene: SyntheticGaussianScene, grid_n: int = 128) -> torch.Tensor:
    """Dense ``(grid_n^2, 3)`` samples on the true surface over the whole domain."""

    xy = _grid_xy(_DOMAIN, _DOMAIN, grid_n, grid_n)
    xy = xy[scene.support_predicate(xy)]
    xy = xy[scene.support_predicate(xy)]
    z = scene.surface_fn(xy)
    return torch.cat([xy, z.reshape(-1, 1)], dim=1)


def observed_gt_surface_points(
    scene: SyntheticGaussianScene, grid_n: int = 128, radius: float | None = None
) -> torch.Tensor:
    """True-surface samples restricted to the region actually covered by input data.

    The observed surface only exists where Gaussians are, so support/coverage
    metrics must ignore true-surface area with no nearby observation. A grid
    sample is kept when its ``xy`` is within ``radius`` of some input point.
    """

    pts = gt_surface_points(scene, grid_n)
    if radius is None:
        return pts
    input_xy = scene.points[:, :2]
    d = torch.cdist(pts[:, :2], input_xy).min(dim=1).values
    return pts[d <= radius]


def _gt_patch_ranges(scene: SyntheticGaussianScene) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """xy sub-domains for each ground-truth patch."""

    if scene.gt_patch_count == 2:  # crease: split at the ridge x = 0
        return [((_DOMAIN[0], 0.0), _DOMAIN), ((0.0, _DOMAIN[1]), _DOMAIN)]
    return [(_DOMAIN, _DOMAIN)]


def gt_nurbs_control_grids(scene: SyntheticGaussianScene, nu: int = 12, nv: int = 12) -> list[torch.Tensor]:
    """Degree-1 control grids that lie exactly on the true surface at their nodes."""

    grids = []
    for x_range, y_range in _gt_patch_ranges(scene):
        xy = _grid_xy(x_range, y_range, nu, nv)
        z = scene.surface_fn(xy)
        grids.append(torch.cat([xy, z.reshape(-1, 1)], dim=1).reshape(nu, nv, 3))
    return grids


def gt_nurbs_payload(scene: SyntheticGaussianScene, nu: int = 12, nv: int = 12) -> dict[str, Any]:
    """Renderer-format payload for the ground-truth NURBS (mirrors
    ``nurbs_intermediate_payload`` so the renderer parses it identically).
    """

    grids = gt_nurbs_control_grids(scene, nu, nv)
    support_mask = mask_on_grid(scene.support_predicate, max(nu, nv))
    uv_support = {"resolution": [int(support_mask.shape[0]), int(support_mask.shape[1])], "mask": support_mask.tolist(), "coordinate_space": "xy"}
    support_mask = mask_on_grid(scene.support_predicate, max(nu, nv))
    uv_support = {"resolution": [int(support_mask.shape[0]), int(support_mask.shape[1])], "mask": support_mask.tolist(), "coordinate_space": "xy"}
    patches = [
        {
            "patch_id": patch_id,
            "control_grid_shape": list(grid.shape),
            "control_grid": grid.tolist(),
            "weights": torch.ones(grid.shape[0], grid.shape[1]).tolist(),
            "degree_u": 1,
            "degree_v": 1,
            "uv_support": uv_support,
            "uv_support": uv_support,
        }
        for patch_id, grid in enumerate(grids)
    ]
    primary = grids[0]
    return {
        "type": "ground_truth_nurbs_surface",
        "iteration": 0,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "degree_u": 1,
        "degree_v": 1,
        "observed_v_max": 1.0,
        "control_grid_shape": list(primary.shape),
        "control_grid": primary.tolist(),
        "weights": torch.ones(primary.shape[0], primary.shape[1]).tolist(),
        "uv_support": uv_support,
        "uv_support": uv_support,
        "base_curves": [],
        "occlusion_curves": [],
        "patches": patches,
        "metadata": {
            "source": "nurbs_constructor_benchmark_ground_truth",
            "scene": scene.name,
            "gt_patch_count": int(scene.gt_patch_count),
            "support_domain": scene.support_name,
            "support_domain": scene.support_name,
        },
    }
