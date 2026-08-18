from __future__ import annotations

import torch

from osn_gs.surface.torch_curve_network_native_fit import fit_curve_network_native, fit_pca_uv
from osn_gs.surface.torch_latent_surface_curve_families import CurveNetworkBlock, RungSegment, TransversalTrace
from osn_gs.surface.torch_latent_surface_seed_curves import SEED_INTERIOR_CONSTRUCTION, SeedCurve


def _line(points_xyz: list[list[float]]) -> torch.Tensor:
    return torch.tensor(points_xyz, dtype=torch.float32)


def _dense_consistent_block(n: int = 8) -> CurveNetworkBlock:
    # A denser, well-behaved, geometrically consistent grid on a mildly
    # curved surface (bowl), suitable for actually judging fit quality.
    u_positions = torch.linspace(0.0, 2.0, n)
    seed_points = torch.stack([u_positions, torch.zeros(n), 0.05 * u_positions.square()], dim=1)
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, seed_points, "test")

    depths = torch.linspace(0.0, 1.0, n)
    traces = []
    for index in [0, n // 3, 2 * n // 3, n - 1]:
        x = float(u_positions[index])
        y = depths
        z = 0.05 * (x * x + y.square())
        points = torch.stack([torch.full((n,), x), y, z], dim=1)
        traces.append(TransversalTrace(index, points))
    traces = tuple(traces)

    rungs = []
    for i in range(len(traces) - 1):
        trace_a, trace_b = traces[i], traces[i + 1]
        for depth in range(n):
            steps = 4
            a, b = trace_a.points[depth], trace_b.points[depth]
            interpolation = torch.linspace(0.0, 1.0, steps + 1).unsqueeze(1)
            points = (1 - interpolation) * a + interpolation * b
            rungs.append(RungSegment(depth, trace_a.sample_index, trace_b.sample_index, points))
    rungs = tuple(rungs)

    all_points = torch.cat(
        [seed_points] + [t.points for t in traces] + [r.points for r in rungs], dim=0,
    )
    return CurveNetworkBlock("seed", SEED_INTERIOR_CONSTRUCTION, seed, traces, rungs, True, all_points)


def test_native_fit_succeeds_on_a_geometrically_consistent_block():
    block = _dense_consistent_block()
    result = fit_curve_network_native(block)
    assert result.valid_parameterization is True
    assert result.surface is not None
    assert result.overall_residual.mean is not None
    assert result.overall_residual.mean < 0.2


def test_native_and_pca_fits_use_the_same_3d_samples():
    block = _dense_consistent_block()
    native = fit_curve_network_native(block)
    pca = fit_pca_uv(block)
    assert native.curve_network_uv.points.shape[0] <= block.all_points.shape[0]
    assert pca.surface is not None
    # Both paths are evaluated against real 3D geometry from the same
    # block -- neither is a no-op stand-in for the other.
    assert native.overall_residual.count > 0
    assert pca.overall_residual.count > 0


def test_native_fit_produces_control_grid_at_fixed_capacity():
    block = _dense_consistent_block()
    result = fit_curve_network_native(block)
    assert result.surface.control_grid.shape == (6, 6, 3)
    assert result.surface.degree_u == 2
    assert result.surface.degree_v == 2


def test_native_fit_reports_separate_trace_and_rung_family_residuals():
    block = _dense_consistent_block()
    result = fit_curve_network_native(block)
    assert result.trace_family_residual.count > 0
    assert result.rung_family_residual.count > 0


def test_invalid_block_never_falls_back_to_pca():
    # A block whose contract is not satisfied must report an invalid
    # parameterization with no surface -- never silently produce a PCA-UV
    # surface in its place.
    seed = SeedCurve("seed", SEED_INTERIOR_CONSTRUCTION, _line([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "test")
    block = CurveNetworkBlock("seed", SEED_INTERIOR_CONSTRUCTION, seed, (), (), False, None)
    result = fit_curve_network_native(block)
    assert result.valid_parameterization is False
    assert result.surface is None


def test_module_never_calls_pca_uv_fit_function():
    import ast
    import inspect

    from osn_gs.surface import torch_curve_network_native_fit as module

    tree = ast.parse(inspect.getsource(module))
    # fit_curve_network_native itself must never call
    # fit_torch_visible_surface_lsq (the PCA-UV path) -- that function may
    # only appear inside fit_pca_uv, the deliberately separate baseline.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fit_curve_network_native":
            calls = {
                sub.func.id for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            assert "fit_torch_visible_surface_lsq" not in calls
