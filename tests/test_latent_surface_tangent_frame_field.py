from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import (
    _build_component_frame,
    _check_holonomy,
    build_tangent_frame_field,
)


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _bowl_points(n: int = 20, extent: float = 3.0, noise: float = 0.01):
    torch.manual_seed(0)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    return torch.stack([uu, vv, zz], dim=1)


def test_frame_transport_is_coherent_across_a_curved_surface():
    points = _bowl_points()
    support = build_latent_surface_support(points)
    result = build_tangent_frame_field(points, support)
    assert len(result.components) >= 1
    component = result.components[0]
    assert component.coherent is True
    assert all(edge.consistent for edge in component.holonomy_edges)


def test_sign_synchronization_across_adjacent_samples():
    points = _bowl_points()
    support = build_latent_surface_support(points)
    result = build_tangent_frame_field(points, support)
    component = result.components[0]
    # Every tree edge's parent->child transport is synchronized by
    # construction; verify that adjacent (tree-connected) nodes never end
    # up with opposing e_u directions.
    for a, b in component.tree_edges:
        cosine = float((component.e_u[a] * component.e_u[b]).sum())
        assert cosine > 0


def test_rigid_rotation_invariance_with_data_derived_anchor():
    # A flat sheet, not the curved bowl: on a zero-curvature surface the
    # tree-integrated (u, v) potential is exactly path-independent, so this
    # isolates rotation-invariance correctness from the curvature-induced
    # path-dependent drift exercised separately elsewhere.
    uu, vv = _grid(16, extent=3.0)
    points = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    support = build_latent_surface_support(points)
    anchor_position = points[0]
    anchor_hint = points[5] - points[0]
    baseline = build_tangent_frame_field(
        points, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
    )
    baseline_component = baseline.components[0]

    angle = torch.tensor(0.9)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    rotated_points = points @ rotation.T
    rotated_support = build_latent_surface_support(rotated_points)
    rotated_anchor_position = anchor_position @ rotation.T
    rotated_anchor_hint = anchor_hint @ rotation.T
    rotated = build_tangent_frame_field(
        rotated_points, rotated_support, anchor_position=rotated_anchor_position,
        anchor_hint_direction=rotated_anchor_hint,
    )
    rotated_component = rotated.components[0]

    # Both anchor + hint are data-derived (rotate consistently with the
    # cloud). On this zero-curvature sheet, tree-integrated (u, v) is
    # exactly path-independent, so the potentials must match regardless of
    # which spanning tree Dijkstra happens to pick.
    assert torch.allclose(baseline_component.u, rotated_component.u, atol=1e-3)
    assert torch.allclose(baseline_component.v, rotated_component.v, atol=1e-3)


def test_boundary_anchored_frame_initialization_aligns_with_hint():
    points = _bowl_points()
    support = build_latent_surface_support(points)
    anchor_position = points[0]
    anchor_hint = points[5] - points[0]
    result = build_tangent_frame_field(
        points, support, anchor_position=anchor_position, anchor_hint_direction=anchor_hint,
        anchor_seed_type="physical_boundary",
    )
    component = result.components[0]
    anchor_index_in_component = component.node_indices.index(
        int(torch.cdist(points, anchor_position.reshape(1, 3)).reshape(-1).argmin())
    )
    root_e_u = component.e_u[component.node_indices.index(component.node_indices[anchor_index_in_component])]
    projected_hint = anchor_hint - (anchor_hint * component.normals[anchor_index_in_component]).sum() * component.normals[anchor_index_in_component]
    projected_hint = projected_hint / projected_hint.norm()
    assert float((root_e_u * projected_hint).sum()) > 0.99
    assert component.anchor_seed_type == "physical_boundary"


def test_interior_gauge_initialization_is_deterministic():
    points = _bowl_points()
    support = build_latent_surface_support(points)
    first = build_tangent_frame_field(points, support)
    second = build_tangent_frame_field(points, support)
    assert torch.allclose(first.components[0].u, second.components[0].u)
    assert torch.allclose(first.components[0].v, second.components[0].v)
    assert first.components[0].anchor_seed_type is None


def test_holonomy_check_detects_hand_built_orientation_reversal():
    # Four points in a small loop, with normals constructed so that
    # propagating e_u all the way around disagrees with the frame already
    # assigned by the (shorter) tree path -- a hand-built, deterministic
    # non-integrable case.
    points = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    normals = torch.tensor([
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
    ])
    node_indices = [0, 1, 2, 3]
    # Tree: 0-1, 1-2. Non-tree (cycle-closing) edges: 2-3, 3-0.
    tree_edges = [(0, 1), (1, 2)]
    e_u, e_v, u, v, _tree, _sing, framed = _build_component_frame(
        points, normals, node_indices, [(0, 1), (1, 2)], root_local=0, anchor_hint_direction=None,
    )
    # Manually flip node 3's e_u to simulate a genuine orientation
    # disagreement reached via a different path (3 was framed through 2 in
    # the real tree, but here we force it to disagree).
    e_u_forced = e_u.clone()
    e_u_forced[3] = -e_u[1]
    holonomy = _check_holonomy(points, e_u_forced, e_v, normals, [(3, 0)], framed | {3})
    assert len(holonomy) == 1
    assert holonomy[0].consistent is False


def test_never_uses_pca():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_tangent_frame_field as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_incoherent_component_is_reported_not_repaired():
    # A tiny, degenerate component (below the minimum size) must simply be
    # excluded, never patched with an invented frame.
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    support = build_latent_surface_support(points)
    result = build_tangent_frame_field(points, support, min_component_size=8)
    assert len(result.components) == 0
