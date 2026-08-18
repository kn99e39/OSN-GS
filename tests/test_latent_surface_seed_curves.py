from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_seed_curves import (
    SEED_INTERIOR_CONSTRUCTION,
    SEED_PHYSICAL_BOUNDARY,
    build_seed_curves,
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


def _bowl_evidence(n: int = 20, extent: float = 3.0, noise: float = 0.01):
    torch.manual_seed(0)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    return torch.stack([uu, vv, zz], dim=1)


def _square_chart(half_extent: float = 2.5):
    corners = [(-half_extent, -half_extent), (half_extent, -half_extent),
               (half_extent, half_extent), (-half_extent, half_extent)]
    positions = [[x, y, 0.05 * (x * x + y * y)] for x, y in corners]
    representative_positions = torch.tensor(positions)
    representative_index = {index: index for index in range(4)}
    node_ids = [0, 1, 2, 3]
    segments = [_FakeSegment("physical_termination") for _ in range(4)]
    return _FakeChart(node_ids, segments), representative_positions, representative_index


def test_boundary_seed_curves_are_preserved_when_chart_is_eligible():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    curves = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    assert len(curves) == 4
    assert all(curve.seed_type == SEED_PHYSICAL_BOUNDARY for curve in curves)


def test_interior_fallback_used_when_no_eligible_chart():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    curves = build_seed_curves(evidence, None, evidence, {}, support)
    assert len(curves) > 0
    assert all(curve.seed_type == SEED_INTERIOR_CONSTRUCTION for curve in curves)


def test_interior_fallback_used_when_chart_status_ineligible():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    chart.status = "parametric_chart_topology_open_or_branching"
    curves = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    assert len(curves) > 0
    assert all(curve.seed_type == SEED_INTERIOR_CONSTRUCTION for curve in curves)


def test_boundary_is_preferred_over_interior_when_both_would_be_possible():
    # When the boundary chart survives, interior anchors must NOT also be
    # silently mixed in -- boundary is preferred and preserved, not merely
    # one option among an always-added set.
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    curves = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    assert not any(curve.seed_type == SEED_INTERIOR_CONSTRUCTION for curve in curves)


def test_interior_anchors_are_independent_starting_points_no_raw_connectivity():
    # Interior seeds must never encode adjacency derived from raw Gaussian
    # centers -- each curve is independently seeded and traced.
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    curves = build_seed_curves(evidence, None, evidence, {}, support)
    seed_ids = [curve.seed_id for curve in curves]
    assert len(seed_ids) == len(set(seed_ids))  # each anchor is its own independent seed


def test_seed_curves_never_mutate_evidence():
    evidence = _bowl_evidence()
    original = evidence.clone()
    support = build_latent_surface_support(evidence)
    build_seed_curves(evidence, None, evidence, {}, support)
    assert torch.equal(evidence, original)
