from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_tangent_frame_field import (
    HolonomyEdge,
    TangentFrameFieldComponent,
)
from osn_gs.surface.torch_parametric_domain_validity import (
    assess_parametric_domain_validity,
    cycle_position_drift_p95,
)


def _grid_component(uv_fn, *, size: int = 12, curved: bool = False) -> TangentFrameFieldComponent:
    """Build a fully-connected (4-neighborhood grid graph) synthetic
    component with a real synchronized tangent frame (constant in-plane
    basis for a flat sheet), so tests exercise the corrected validator's
    exact contract (source-graph adjacency + synchronized frame) without
    depending on the full field-construction pipeline."""

    coords = torch.linspace(-2.0, 2.0, size)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x_flat, y_flat = xx.reshape(-1), yy.reshape(-1)
    if curved:
        z_flat = 0.1 * torch.sin(x_flat) * torch.cos(y_flat)
    else:
        z_flat = torch.zeros_like(x_flat)
    positions = torch.stack([x_flat, y_flat, z_flat], dim=1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    e_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(positions.shape[0], 1)
    e_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)

    u_vals, v_vals = uv_fn(x_flat, y_flat)

    # 4-neighborhood grid adjacency, split deterministically into a
    # spanning tree (row-major snake) plus remaining edges reported as
    # holonomy_edges (all marked consistent -- coherence is not under test
    # here, only the validator's own fold/orientation logic).
    def index_of(i: int, j: int) -> int:
        return i * size + j

    all_edges: set[tuple[int, int]] = set()
    for i in range(size):
        for j in range(size):
            if i + 1 < size:
                all_edges.add(tuple(sorted((index_of(i, j), index_of(i + 1, j)))))
            if j + 1 < size:
                all_edges.add(tuple(sorted((index_of(i, j), index_of(i, j + 1)))))

    tree_edges: list[tuple[int, int]] = []
    seen = {0}
    for i in range(size):
        for j in range(size):
            node = index_of(i, j)
            if node in seen:
                continue
            # connect to an already-seen neighbor
            for ni, nj in ((i - 1, j), (i, j - 1)):
                if 0 <= ni < size and 0 <= nj < size:
                    neighbor = index_of(ni, nj)
                    if neighbor in seen:
                        tree_edges.append((neighbor, node))
                        seen.add(node)
                        break
    tree_edge_set = {tuple(sorted(edge)) for edge in tree_edges}
    holonomy_edges = tuple(
        HolonomyEdge(a, b, True, 0.0) for a, b in sorted(all_edges) if (a, b) not in tree_edge_set
    )

    return TangentFrameFieldComponent(
        node_indices=tuple(range(positions.shape[0])),
        positions=positions, normals=normals, e_u=e_u, e_v=e_v,
        u=u_vals, v=v_vals,
        tree_edges=tuple(tree_edges), holonomy_edges=holonomy_edges,
        singularities=(), coherent=True, incoherence_reason=None, anchor_seed_type=None,
    )


def test_clean_planar_uv_map_is_valid():
    component = _grid_component(lambda x, y: ((x + 2.0) / 4.0, (y + 2.0) / 4.0))
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(component, uv, median_spacing=4.0 / 11.0)
    assert report.valid is True
    assert report.fold_fraction == 0.0
    assert report.duplicate_incompatible_count == 0
    assert report.global_orientation_flip_applied is False


def _folded_uv(x, y):
    # A genuine SPATIAL fold: u increases with x on the left half, then
    # reverses (decreases with x) on the right half, offset so the two
    # halves never land on the exact same u value (no duplicate-cell
    # collision) -- orientation disagrees between adjacent columns near
    # the fold line, which is not a whole-chart flip.
    u = torch.where(x >= 0, 6.0 - x, x + 2.0)
    return u / 4.0, (y + 2.0) / 4.0


def test_true_local_fold_is_detected():
    component = _grid_component(_folded_uv)
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(component, uv, median_spacing=4.0 / 11.0)
    assert report.valid is False
    assert "uv_orientation_reversal_or_foldover" in report.invalid_reasons
    assert report.fold_fraction > 0.0


def test_global_orientation_flip_is_not_a_fold():
    # A single whole-chart flip (u -> -u everywhere) is gauge-equivalent,
    # not a real fold -- must be canonicalized away, not reported as one.
    component = _grid_component(lambda x, y: (-(x + 2.0) / 4.0, (y + 2.0) / 4.0))
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(component, uv, median_spacing=4.0 / 11.0)
    assert report.global_orientation_flip_applied is True
    assert report.valid is True
    assert report.fold_fraction == 0.0


def test_fold_classification_invariant_to_global_axis_sign_flip():
    component = _grid_component(_folded_uv)
    uv = torch.stack([component.u, component.v], dim=1)
    flipped_uv = uv.clone()
    flipped_uv[:, 0] = -flipped_uv[:, 0]

    report = assess_parametric_domain_validity(component, uv, median_spacing=4.0 / 11.0)
    flipped_report = assess_parametric_domain_validity(component, flipped_uv, median_spacing=4.0 / 11.0)
    assert report.valid is False
    assert flipped_report.valid is False
    assert flipped_report.fold_fraction > 0.0


def test_source_neighborhood_not_uv_neighborhood_used_for_folds():
    # Construct a UV map that, if judged by UV-space proximity, would put
    # two DISTANT source points next to each other (a classic UV-kNN
    # confound), but is otherwise a clean, fold-free, monotone planar map
    # under genuine SOURCE-graph adjacency. The corrected validator must
    # report this as valid because it only ever looks at source-graph
    # neighbors, never UV-space kNN.
    component = _grid_component(lambda x, y: ((x + 2.0) / 4.0, (y + 2.0) / 4.0), size=8)
    uv = torch.stack([component.u, component.v], dim=1)
    # Scale U enormously and V by almost nothing -- in UV-space this makes
    # far-apart-in-source rows appear "close" along v, which would corrupt
    # a UV-space kNN neighbor search, but must not affect a source-graph
    # based validator.
    distorted_uv = torch.stack([uv[:, 0] * 1000.0, uv[:, 1] * 1e-4], dim=1)
    report = assess_parametric_domain_validity(component, distorted_uv, median_spacing=4.0 / 7.0)
    assert report.fold_fraction == 0.0


def test_uses_synchronized_frame_not_raw_pca_normal():
    # Flip every stored `normals` entry (simulating an independently-signed
    # PCA normal) while keeping the synchronized e_u/e_v frame (and hence
    # n_sync = e_u x e_v) exactly as before -- a fold-free clean planar map
    # must still be reported valid, since orientation is judged against
    # n_sync, never against `normals`.
    component = _grid_component(lambda x, y: ((x + 2.0) / 4.0, (y + 2.0) / 4.0))
    flipped_normals_component = TangentFrameFieldComponent(
        node_indices=component.node_indices, positions=component.positions,
        normals=-component.normals, e_u=component.e_u, e_v=component.e_v,
        u=component.u, v=component.v, tree_edges=component.tree_edges,
        holonomy_edges=component.holonomy_edges, singularities=component.singularities,
        coherent=component.coherent, incoherence_reason=component.incoherence_reason,
        anchor_seed_type=component.anchor_seed_type,
    )
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(flipped_normals_component, uv, median_spacing=4.0 / 11.0)
    assert report.valid is True
    assert report.fold_fraction == 0.0


def test_degenerate_extent_is_detected():
    component = _grid_component(lambda x, y: (torch.zeros_like(x), torch.zeros_like(y)))
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(component, uv, median_spacing=1.0)
    assert report.valid is False
    assert "degenerate_uv_extent" in report.invalid_reasons


def test_area_and_shear_distortion_reported():
    component = _grid_component(lambda x, y: ((x + 2.0) / 4.0, (y + 2.0) / 4.0))
    uv = torch.stack([component.u, component.v], dim=1)
    report = assess_parametric_domain_validity(component, uv, median_spacing=4.0 / 11.0)
    assert report.area_distortion_p95 is not None
    assert report.shear_distortion_p95 is not None
    assert report.shear_distortion_p95 == pytest_approx(1.0)


def pytest_approx(value, tol=1e-2):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()


def test_cycle_position_drift_zero_on_perfectly_flat_field():
    from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
    from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field

    coords = torch.linspace(-2.0, 2.0, 10)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(100)], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    drift = cycle_position_drift_p95(component, support.median_spacing)
    assert drift is not None
    assert drift < 1e-2


def test_never_uses_pca():
    import ast
    import inspect

    from osn_gs.surface import torch_parametric_domain_validity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_no_fit_dependency():
    import ast
    import inspect

    from osn_gs.surface import torch_parametric_domain_validity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() for name in imported_names)
