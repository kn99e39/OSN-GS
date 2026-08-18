from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_curve_tracer import (
    propagate_tangent_onto_plane,
    sample_segment_continuous_support,
    trace_bidirectional,
    trace_curve,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _bowl_support(n: int = 20, extent: float = 3.0, noise: float = 0.01):
    torch.manual_seed(0)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    points = torch.stack([uu, vv, zz], dim=1)
    return build_latent_surface_support(points)


def test_propagate_tangent_projects_onto_new_plane_without_sign_flip():
    normal = torch.tensor([0.0, 0.0, 1.0])
    previous = torch.tensor([1.0, 0.0, 0.1])
    propagated = propagate_tangent_onto_plane(previous, normal)
    assert propagated is not None
    # The propagated direction stays aligned (positive dot) with the
    # original -- no arbitrary sign inversion.
    assert float((propagated * previous).sum()) > 0
    assert abs(float((propagated * normal).sum())) < 1e-5


def test_propagate_tangent_degenerate_direction_returns_none():
    normal = torch.tensor([0.0, 0.0, 1.0])
    previous = torch.tensor([0.0, 0.0, 5.0])  # parallel to normal
    assert propagate_tangent_onto_plane(previous, normal) is None


def test_trace_curve_stays_on_surface_and_terminates_on_step_count():
    support = _bowl_support()
    start = torch.tensor([0.0, 0.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    result = trace_curve(start, direction, support, step_count=10)
    assert result.terminated_reason == "step_count_reached"
    assert int(result.points.shape[0]) >= 2
    check = support.query_batch(result.points)
    assert bool(check.supported.all())


def test_trace_curve_walking_off_support_terminates_early():
    support = _bowl_support(n=10, extent=1.0)  # small, bounded support
    start = torch.tensor([0.0, 0.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    result = trace_curve(start, direction, support, step_count=200, step_size=support.median_spacing * 2.0)
    assert result.terminated_reason == "unsupported"
    check = support.query_batch(result.points)
    assert bool(check.supported.all())


def test_trace_bidirectional_produces_curve_through_start_point():
    support = _bowl_support()
    start = torch.tensor([0.0, 0.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    result = trace_bidirectional(start, direction, support, step_count=8)
    check = support.query_batch(result.points)
    assert bool(check.supported.all())
    assert int(result.points.shape[0]) >= 2


def _strip(x_values: torch.Tensor, width: float = 1.0, rows: int = 6) -> torch.Tensor:
    # A genuine (locally 2D, well-planarized) strip along X, needed so the
    # weighted-PCA local plane fit is well-defined (a bare 1-D line has no
    # meaningful surface normal and is correctly rejected as unsupported by
    # the planarity gate -- not what these tests want to exercise).
    ys = torch.linspace(-width / 2, width / 2, rows)
    xx, yy = torch.meshgrid(x_values, ys, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros_like(xx.reshape(-1))], dim=1)


def test_segment_with_supported_endpoints_but_unsupported_interior_is_rejected():
    # Build a support cloud with a real gap in the middle so the straight
    # chord between two well-supported endpoints crosses unsupported space.
    left = _strip(torch.linspace(-3, -1, 10))
    right = _strip(torch.linspace(1, 3, 10))
    points = torch.cat([left, right], dim=0)
    support = build_latent_surface_support(points)
    a = torch.tensor([-1.5, 0.0, 0.0])
    b = torch.tensor([1.5, 0.0, 0.0])
    check_a = support.query_batch(a.reshape(1, 3))
    check_b = support.query_batch(b.reshape(1, 3))
    assert bool(check_a.supported[0]) and bool(check_b.supported[0])
    _points, fully_supported = sample_segment_continuous_support(support, a, b, steps=8)
    assert fully_supported is False


def test_segment_fully_within_support_is_accepted():
    points = _strip(torch.linspace(-3, 3, 40))
    support = build_latent_surface_support(points)
    a = torch.tensor([-0.5, 0.0, 0.0])
    b = torch.tensor([0.5, 0.0, 0.0])
    _points, fully_supported = sample_segment_continuous_support(support, a, b, steps=6)
    assert fully_supported is True
