from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _noisy_bowl(n: int = 20, extent: float = 3.0, noise: float = 0.02, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    return torch.stack([uu, vv, zz], dim=1)


def test_query_near_surface_is_supported_and_projects_close_to_true_height():
    points = _noisy_bowl()
    support = build_latent_surface_support(points)
    query = torch.tensor([[0.5, 0.5, 0.3]])
    result = support.query_batch(query)
    assert bool(result.supported[0])
    expected_z = 0.05 * (0.5**2 + 0.5**2)
    assert abs(float(result.positions[0, 2]) - expected_z) < 0.1


def test_query_far_from_any_support_point_is_unsupported():
    points = _noisy_bowl()
    support = build_latent_surface_support(points)
    query = torch.tensor([[50.0, 50.0, 50.0]])
    result = support.query_batch(query)
    assert not bool(result.supported[0])


def test_query_far_off_surface_in_normal_direction_is_unsupported():
    points = _noisy_bowl()
    support = build_latent_surface_support(points)
    query = torch.tensor([[0.5, 0.5, 100.0]])
    result = support.query_batch(query)
    assert not bool(result.supported[0])


def test_flat_sheet_normal_points_along_z():
    uu, vv = _grid(10)
    points = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    support = build_latent_surface_support(points)
    result = support.query_batch(torch.tensor([[0.0, 0.0, 0.05]]))
    assert bool(result.supported[0])
    # Normal should be close to +-Z for a flat XY sheet.
    z_alignment = float(result.normals[0, 2].abs())
    assert z_alignment > 0.9


def test_never_mutates_support_points():
    points = _noisy_bowl()
    original = points.clone()
    support = build_latent_surface_support(points)
    support.query_batch(torch.tensor([[0.5, 0.5, 0.3]]))
    assert torch.equal(points, original)
    assert torch.equal(support.support_points, original)


def test_batch_query_matches_single_query_semantics():
    points = _noisy_bowl()
    support = build_latent_surface_support(points)
    single = support.query(torch.tensor([0.2, -0.3, 0.05]))
    batch = support.query_batch(torch.tensor([[0.2, -0.3, 0.05]]))
    assert torch.allclose(single.positions, batch.positions)
    assert single.supported[0] == batch.supported[0]
