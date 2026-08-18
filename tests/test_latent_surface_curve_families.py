from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_curve_families import (
    MIN_CORRESPONDENCE_DEPTH_COUNT,
    MIN_FAMILY_CURVE_COUNT,
    build_curve_network_blocks,
)
from osn_gs.surface.torch_latent_surface_seed_curves import build_seed_curves
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


def _bowl_evidence(n: int = 24, extent: float = 3.0, noise: float = 0.01):
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


def test_boundary_seeds_produce_multiple_contract_satisfying_blocks():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    seeds = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    assert len(blocks) == len(seeds)
    satisfying = [block for block in blocks if block.satisfies_contract]
    assert len(satisfying) >= 2  # multiple independent patch candidates from one region


def test_interior_seeds_also_produce_independent_blocks():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    seeds = build_seed_curves(evidence, None, evidence, {}, support)
    blocks = build_curve_network_blocks(seeds, support)
    satisfying = [block for block in blocks if block.satisfies_contract]
    assert len(satisfying) >= 2


def test_satisfying_block_meets_the_fixed_contract_exactly():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    seeds = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    satisfying = [block for block in blocks if block.satisfies_contract]
    assert satisfying
    for block in satisfying:
        assert len(block.transversal_traces) >= MIN_FAMILY_CURVE_COUNT
        depths = {rung.depth for rung in block.rungs}
        assert len(depths) >= MIN_CORRESPONDENCE_DEPTH_COUNT
        assert block.all_points is not None
        assert int(block.all_points.shape[0]) > 0


def test_every_rung_point_is_independently_supported():
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    chart, rep_positions, rep_index = _square_chart()
    seeds = build_seed_curves(evidence, chart, rep_positions, rep_index, support)
    blocks = build_curve_network_blocks(seeds, support)
    for block in blocks:
        for rung in block.rungs:
            check = support.query_batch(rung.points)
            assert bool(check.supported.all())


def test_block_partition_is_purely_structural_no_fit_dependency():
    # build_curve_network_blocks never imports or calls anything from the
    # NURBS fitter -- partitioning must be decidable before any fit.
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_curve_families as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() for name in imported_names)


def test_isolated_seed_with_no_neighbors_does_not_satisfy_contract():
    # A single seed curve with only one usable sample (so no adjacent pair
    # of transversal traces exists to form a rung) must not be promoted to
    # a materializable block.
    evidence = _bowl_evidence()
    support = build_latent_surface_support(evidence)
    from osn_gs.surface.torch_latent_surface_seed_curves import SeedCurve

    tiny_seed = SeedCurve("tiny", "interior_construction", evidence[:1], "test")
    blocks = build_curve_network_blocks((tiny_seed,), support)
    assert len(blocks) == 1
    assert blocks[0].satisfies_contract is False
    assert blocks[0].all_points is None
