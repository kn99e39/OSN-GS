from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_curve_network import (
    STATUS_CURVE_NETWORK,
    STATUS_NO_ELIGIBLE_SEED_CHART,
    STATUS_NO_SUPPORTED_SEED_POINTS,
    build_latent_surface_curve_network,
)
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support


class _FakeSegment:
    def __init__(self, kind: str) -> None:
        self.segment_kind = kind


class _FakeChart:
    def __init__(self, node_ids, segments, status: str = "eligible_parametric_chart_boundary") -> None:
        self.ordered_node_ids = node_ids
        self.segments = segments
        self.status = status


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


def _square_chart(half_extent: float = 2.5):
    corners = [(-half_extent, -half_extent), (half_extent, -half_extent),
               (half_extent, half_extent), (-half_extent, half_extent)]
    positions = []
    for x, y in corners:
        z = 0.05 * (x * x + y * y)
        positions.append([x, y, z])
    representative_positions = torch.tensor(positions)
    representative_index = {index: index for index in range(4)}
    node_ids = [0, 1, 2, 3]
    segments = [_FakeSegment("physical_termination") for _ in range(4)]
    return _FakeChart(node_ids, segments), representative_positions, representative_index


def test_eligible_chart_on_supported_surface_produces_curve_network():
    support = _bowl_support()
    chart, representative_positions, representative_index = _square_chart()
    result = build_latent_surface_curve_network(0, chart, representative_positions, representative_index, support)
    assert result.status == STATUS_CURVE_NETWORK
    assert result.has_curve_network
    assert len(result.seed_segments) == 4
    assert len(result.transversal_curves) > 0
    assert result.all_points is not None
    assert int(result.all_points.shape[0]) >= 4


def test_ineligible_chart_status_fails_closed_without_fallback():
    support = _bowl_support()
    chart, representative_positions, representative_index = _square_chart()
    chart.status = "parametric_chart_topology_open_or_branching"
    result = build_latent_surface_curve_network(0, chart, representative_positions, representative_index, support)
    assert result.status == STATUS_NO_ELIGIBLE_SEED_CHART
    assert not result.has_curve_network
    assert result.all_points is None


def test_seed_chart_far_outside_support_fails_closed():
    support = _bowl_support()
    chart, _, representative_index = _square_chart()
    # Move every "representative position" far outside the support cloud's
    # reach -- the seed walk must fail closed, never invent a curve.
    representative_positions = torch.tensor([
        [500.0, 500.0, 500.0], [501.0, 500.0, 500.0], [501.0, 501.0, 500.0], [500.0, 501.0, 500.0],
    ])
    result = build_latent_surface_curve_network(0, chart, representative_positions, representative_index, support)
    assert result.status in (STATUS_NO_SUPPORTED_SEED_POINTS, STATUS_NO_ELIGIBLE_SEED_CHART)
    assert not result.has_curve_network


def test_every_curve_point_reports_as_individually_supported():
    support = _bowl_support()
    chart, representative_positions, representative_index = _square_chart()
    result = build_latent_surface_curve_network(0, chart, representative_positions, representative_index, support)
    assert result.has_curve_network
    check = support.query_batch(result.all_points)
    # Every retained point must itself pass the support test -- the curve
    # network never contains a point it wouldn't also accept on direct query.
    assert bool(check.supported.all())


def test_never_mutates_representative_positions():
    support = _bowl_support()
    chart, representative_positions, representative_index = _square_chart()
    original = representative_positions.clone()
    build_latent_surface_curve_network(0, chart, representative_positions, representative_index, support)
    assert torch.equal(representative_positions, original)
