from __future__ import annotations

import torch

from osn_gs.surface.torch_curve_network_uv_parameterization import build_curve_network_uv
from osn_gs.surface.torch_latent_surface_curve_families import CurveNetworkBlock, RungSegment, TransversalTrace
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, SeedCurve


def _line(points_xyz: list[list[float]]) -> torch.Tensor:
    return torch.tensor(points_xyz, dtype=torch.float32)


def _consistent_block() -> CurveNetworkBlock:
    # A simple, geometrically consistent 3x3 grid: seed curve along X at
    # y=0, two transversal traces both walking in the SAME (+Y) direction,
    # rungs connecting them at every shared depth.
    seed_points = _line([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, seed_points, "test")

    trace_a = TransversalTrace(0, _line([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]))
    trace_b = TransversalTrace(2, _line([[2.0, 0.0, 0.0], [2.0, 0.5, 0.0], [2.0, 1.0, 0.0]]))

    rungs = tuple(
        RungSegment(depth, 0, 2, _line([
            [0.0, depth * 0.5, 0.0], [1.0, depth * 0.5, 0.0], [2.0, depth * 0.5, 0.0],
        ]))
        for depth in range(3)
    )
    all_points = torch.cat(
        [seed_points, trace_a.points, trace_b.points] + [rung.points for rung in rungs], dim=0,
    )
    return CurveNetworkBlock("seed", SEED_INTERIOR_CONSTRUCTION, seed, (trace_a, trace_b), rungs, True, all_points)


def test_chord_length_parameterization_along_ordered_curve():
    result = build_curve_network_uv(_consistent_block())
    assert result.valid
    # Trace A sits at seed sample_index 0 -> u should be 0; trace B at the
    # far end of the (2-unit-long) seed curve -> u should be 1.
    trace_a_mask = torch.tensor([tag == "trace_family:0" for tag in result.provenance])
    trace_b_mask = torch.tensor([tag == "trace_family:2" for tag in result.provenance])
    assert torch.allclose(result.uv[trace_a_mask][:, 0], torch.zeros(int(trace_a_mask.sum())), atol=1e-5)
    assert torch.allclose(result.uv[trace_b_mask][:, 0], torch.ones(int(trace_b_mask.sum())), atol=1e-5)


def test_monotonic_transverse_parameter_assignment():
    result = build_curve_network_uv(_consistent_block())
    assert result.valid
    trace_a_mask = torch.tensor([tag == "trace_family:0" for tag in result.provenance])
    v_values = result.uv[trace_a_mask][:, 1]
    assert bool((v_values[1:] - v_values[:-1] >= 0).all())
    assert float(v_values.max() - v_values.min()) > 0


def test_consistent_uv_at_rung_intersections():
    # A rung point exactly halfway between the two traces (in chord-length
    # position) at a given depth should get u close to the midpoint of the
    # two traces' own u values, at that depth's shared v.
    result = build_curve_network_uv(_consistent_block())
    assert result.valid
    rung_mask = torch.tensor([tag.startswith("rung_family:depth1") for tag in result.provenance])
    rung_uv = result.uv[rung_mask]
    # The middle rung sample (u=1.0 point in 3D) should land near u=0.5.
    middle = rung_uv[torch.argmin((result.points[rung_mask][:, 0] - 1.0).abs())]
    assert abs(float(middle[0]) - 0.5) < 0.1


def test_invariant_to_rigid_rotation_of_entire_network():
    block = _consistent_block()
    baseline = build_curve_network_uv(block)
    assert baseline.valid

    torch.manual_seed(0)
    angle = torch.tensor(0.7)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

    def _rotate(points: torch.Tensor) -> torch.Tensor:
        return points @ rotation.T

    rotated_seed = SeedCurve(block.seed_curve.seed_id, block.seed_curve.seed_type, _rotate(block.seed_curve.points), "test")
    rotated_traces = tuple(
        TransversalTrace(t.sample_index, _rotate(t.points)) for t in block.transversal_traces
    )
    rotated_rungs = tuple(
        RungSegment(r.depth, r.a_sample_index, r.b_sample_index, _rotate(r.points)) for r in block.rungs
    )
    rotated_block = CurveNetworkBlock(
        block.seed_id, block.seed_type, rotated_seed, rotated_traces, rotated_rungs, True,
        _rotate(block.all_points),
    )
    rotated = build_curve_network_uv(rotated_block)
    assert rotated.valid
    assert torch.allclose(baseline.uv, rotated.uv, atol=1e-4)


def test_no_pca_dependency():
    import ast
    import inspect

    from osn_gs.surface import torch_curve_network_uv_parameterization as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_rejects_contradictory_transversal_direction():
    seed_points = _line([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, seed_points, "test")
    trace_a = TransversalTrace(0, _line([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]]))
    # trace_b walks the OPPOSITE transverse direction (-Y) from trace_a's
    # (+Y) -- a contradictory correspondence.
    trace_b = TransversalTrace(2, _line([[2.0, 0.0, 0.0], [2.0, -0.5, 0.0], [2.0, -1.0, 0.0]]))
    rungs = tuple(
        RungSegment(depth, 0, 2, _line([
            [0.0, depth * 0.5, 0.0], [1.0, 0.0, 0.0], [2.0, -depth * 0.5, 0.0],
        ]))
        for depth in range(3)
    )
    all_points = torch.cat(
        [seed_points, trace_a.points, trace_b.points] + [rung.points for rung in rungs], dim=0,
    )
    block = CurveNetworkBlock("seed", SEED_INTERIOR_CONSTRUCTION, seed, (trace_a, trace_b), rungs, True, all_points)
    result = build_curve_network_uv(block)
    assert result.valid is False
    assert result.invalid_reason == "inconsistent_transversal_curve_direction"


def test_rejects_degenerate_seed_chord_length():
    # A "seed curve" that never actually moves has zero chord length --
    # cannot define a U parameter extent.
    seed_points = _line([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, seed_points, "test")
    trace_a = TransversalTrace(0, _line([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))
    trace_b = TransversalTrace(2, _line([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))
    rungs = (RungSegment(0, 0, 2, _line([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),)
    block = CurveNetworkBlock(
        "seed", SEED_INTERIOR_CONSTRUCTION, seed, (trace_a, trace_b), rungs, True,
        torch.cat([seed_points, trace_a.points, trace_b.points], dim=0),
    )
    result = build_curve_network_uv(block)
    assert result.valid is False
    assert result.invalid_reason == "degenerate_seed_chord_length"


def test_block_not_satisfying_contract_is_rejected_without_repair():
    seed_points = _line([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, seed_points, "test")
    block = CurveNetworkBlock("seed", SEED_INTERIOR_CONSTRUCTION, seed, (), (), False, None)
    result = build_curve_network_uv(block)
    assert result.valid is False
    assert result.invalid_reason == "block_does_not_satisfy_contract"
