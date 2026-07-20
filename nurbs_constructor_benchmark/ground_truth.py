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

_DOMAIN = (-1.0, 1.0)


def _grid_xy(x_range: tuple[float, float], y_range: tuple[float, float], nu: int, nv: int) -> torch.Tensor:
    xs = torch.linspace(x_range[0], x_range[1], nu)
    ys = torch.linspace(y_range[0], y_range[1], nv)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _sheet_fns(scene: SyntheticGaussianScene):
    return scene.sheet_fns if getattr(scene, "sheet_fns", None) else (scene.surface_fn,)


def gt_surface_points(scene: SyntheticGaussianScene, grid_n: int = 128) -> torch.Tensor:
    """Dense samples on the true surface (union of all sheets) over the domain."""

    xy = _grid_xy(_DOMAIN, _DOMAIN, grid_n, grid_n)
    xy = xy[scene.support_predicate(xy)]
    return torch.cat(
        [torch.cat([xy, fn(xy).reshape(-1, 1)], dim=1) for fn in _sheet_fns(scene)], dim=0
    )


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


"""Boundary-conformal ground-truth charts.

The GT NURBS is the *ideal target representation* for OSN-GS, so its
parameterization must follow the surface's intrinsic structure: support
boundaries (outer contours and holes) are chart boundaries, never raster trim
masks, and iso-lines follow the geometry (concentric circles / radial lines on
the annulus, a swept strip along the U, ...). Rectangular-support scenes keep
plain rectangular charts. No GT patch carries a ``uv_support`` mask anymore.

Charts use clamped uniform knots only (renderer/payload constraint), so circular
boundaries are dense degree-2 approximations (sag < 0.5% of the radius at the
chosen column counts) rather than exact rational circles.
"""


def _lift(xy: torch.Tensor, height_fn, nu: int, nv: int) -> torch.Tensor:
    z = height_fn(xy.reshape(-1, 2))
    return torch.cat([xy.reshape(-1, 2), z.reshape(-1, 1)], dim=1).reshape(nu, nv, 3)


def _rect_chart(x_range, y_range, height_fn, nu: int = 12, nv: int = 12):
    xy = _grid_xy(x_range, y_range, nu, nv).reshape(nu, nv, 2)
    return _lift(xy, height_fn, nu, nv), 1, 1, "rect"


def _polar_chart(inner_radius_fn, outer_radius_fn, height_fn, nu: int = 40, nv: int = 4):
    """u = angle 0..2pi (C0-closed at the seam), v = inner -> outer boundary."""

    theta = torch.linspace(0.0, 2.0 * torch.pi, nu)
    unit = torch.stack([theta.cos(), theta.sin()], dim=1)
    inner = inner_radius_fn(theta)
    outer = outer_radius_fn(theta)
    v = torch.linspace(0.0, 1.0, nv)
    radius = inner[:, None] * (1.0 - v[None, :]) + outer[:, None] * v[None, :]
    xy = unit[:, None, :] * radius[..., None]
    return _lift(xy, height_fn, nu, nv), 2, 1, "polar"


def _triangle_chart(height_fn, nu: int = 12, nv: int = 12):
    """Degenerate-corner chart: v sweeps from the bottom edge to the hypotenuse."""

    u = torch.linspace(0.0, 1.0, nu)
    v = torch.linspace(0.0, 1.0, nv)
    x = (-1.0 + 2.0 * u)[:, None].expand(nu, nv)
    y = -1.0 + v[None, :] * (x + 1.0)
    xy = torch.stack([x, y], dim=-1)
    return _lift(xy, height_fn, nu, nv), 1, 1, "triangle_degenerate"


def _u_shape_chart(height_fn, per_segment: int = 10, nv: int = 4):
    """Strip chart swept along the U path, v = inner -> outer boundary polyline.

    Degree 1 along u with the polyline corners as control points keeps the
    blocky corners exact. The chart covers the quadrilateral strip between the
    two polylines; the predicate's small notched outer corners differ by a few
    cells and are accepted (documented) rather than masked.
    """

    inner_polyline = [(-0.55, 0.9), (-0.55, -0.45), (0.55, -0.45), (0.55, 0.9)]
    outer_polyline = [(-1.0, 0.9), (-1.0, -1.0), (1.0, -1.0), (1.0, 0.9)]

    def _sample(polyline: list[tuple[float, float]]) -> torch.Tensor:
        points = []
        for start, end in zip(polyline[:-1], polyline[1:]):
            for step in range(per_segment):
                t = step / per_segment
                points.append((start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])))
        points.append(polyline[-1])
        return torch.tensor(points)

    inner = _sample(inner_polyline)
    outer = _sample(outer_polyline)
    nu = int(inner.shape[0])
    v = torch.linspace(0.0, 1.0, nv)
    xy = inner[:, None, :] * (1.0 - v[None, :, None]) + outer[:, None, :] * v[None, :, None]
    return _lift(xy, height_fn, nu, nv), 1, 1, "swept_strip"


def _crescent_inner_radius(theta: torch.Tensor) -> torch.Tensor:
    # Ray from the origin (inside the cutout) exits the cutout circle
    # |x - (0.28, 0)| = 0.48 at this radius, so the crescent is exactly
    # {(theta, r): r_exit(theta) <= r <= 0.95} — a conformal polar chart.
    projection = 0.28 * theta.cos()
    return projection + torch.sqrt(projection.square() + 0.48 ** 2 - 0.28 ** 2)


def gt_nurbs_charts(scene: SyntheticGaussianScene) -> list[tuple[torch.Tensor, int, int, str]]:
    """Boundary-conformal ``(control_grid, degree_u, degree_v, chart_kind)`` per GT patch."""

    sheets = _sheet_fns(scene)
    if len(sheets) > 1:  # multi-sheet scenes: one full-domain chart per sheet
        return [_rect_chart(_DOMAIN, _DOMAIN, fn) for fn in sheets]
    if scene.gt_patch_count == 2:  # crease: split at the ridge x = 0
        return [
            _rect_chart((_DOMAIN[0], 0.0), _DOMAIN, scene.surface_fn),
            _rect_chart((0.0, _DOMAIN[1]), _DOMAIN, scene.surface_fn),
        ]
    if scene.support_name == "annulus":
        return [_polar_chart(lambda t: torch.full_like(t, 0.32), lambda t: torch.full_like(t, 0.9), scene.surface_fn)]
    if scene.support_name == "crescent":
        return [_polar_chart(_crescent_inner_radius, lambda t: torch.full_like(t, 0.95), scene.surface_fn)]
    if scene.support_name == "u_shape":
        return [_u_shape_chart(scene.surface_fn)]
    if scene.support_name == "triangle":
        return [_triangle_chart(scene.surface_fn)]
    if scene.support_name == "elongated_rect":
        return [_rect_chart(_DOMAIN, (-0.28, 0.28), scene.surface_fn)]
    return [_rect_chart(_DOMAIN, _DOMAIN, scene.surface_fn)]


def gt_nurbs_control_grids(scene: SyntheticGaussianScene, nu: int = 12, nv: int = 12) -> list[torch.Tensor]:
    """Control grids of the conformal GT charts (legacy-named accessor)."""

    return [grid for grid, _, _, _ in gt_nurbs_charts(scene)]


def gt_nurbs_payload(scene: SyntheticGaussianScene, nu: int = 12, nv: int = 12) -> dict[str, Any]:
    """Renderer-format payload for the ground-truth NURBS (mirrors
    ``nurbs_intermediate_payload`` so the renderer parses it identically).

    Boundary-conformal: support topology lives in the chart parameterization
    itself (e.g. the annulus hole is the chart's inner boundary), so no patch
    carries a ``uv_support`` trim mask.
    """

    charts = gt_nurbs_charts(scene)
    patches = [
        {
            "patch_id": patch_id,
            "control_grid_shape": list(grid.shape),
            "control_grid": grid.tolist(),
            "weights": torch.ones(grid.shape[0], grid.shape[1]).tolist(),
            "degree_u": degree_u,
            "degree_v": degree_v,
            "uv_support": None,
            "chart_kind": chart_kind,
        }
        for patch_id, (grid, degree_u, degree_v, chart_kind) in enumerate(charts)
    ]
    primary, primary_degree_u, primary_degree_v, _ = charts[0]
    return {
        "type": "ground_truth_nurbs_surface",
        "iteration": 0,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "degree_u": primary_degree_u,
        "degree_v": primary_degree_v,
        "observed_v_max": 1.0,
        "control_grid_shape": list(primary.shape),
        "control_grid": primary.tolist(),
        "weights": torch.ones(primary.shape[0], primary.shape[1]).tolist(),
        "uv_support": None,
        "base_curves": [],
        "occlusion_curves": [],
        "patches": patches,
        "metadata": {
            "source": "nurbs_constructor_benchmark_ground_truth",
            "scene": scene.name,
            "gt_patch_count": int(scene.gt_patch_count),
            "support_domain": scene.support_name,
            "parameterization": "boundary_conformal",
        },
    }
