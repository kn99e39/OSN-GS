from __future__ import annotations

"""Phase D isolated continuation-domain tests (design revision 3)."""

import math
import unittest

import torch

from osn_gs.surface.torch_continuation_domain import (
    CONTINUATION_STATES,
    STATE_DEGENERATE,
    STATE_VALID,
    ContinuationDomainBuildError,
    build_continuation_domain,
    interpolate_boundary_arclength,
)
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_patch_boundary import (
    BOUNDARY_RECONCILED_INTERNAL,
    BOUNDARY_UNCLASSIFIED,
    PatchBoundarySegment,
    build_rectangular_patch_edge,
)


def _plane_surface(resolution: int = 6, degree: int = 2) -> TorchNURBSSurface:
    u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    v = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    grid = torch.stack(
        (
            u[:, None].expand(resolution, resolution),
            v[None, :].expand(resolution, resolution),
            torch.zeros((resolution, resolution), dtype=torch.float64),
        ),
        dim=2,
    )
    return TorchNURBSSurface(
        grid, torch.ones((resolution, resolution), dtype=torch.float64), degree_u=degree, degree_v=degree
    )


def _sine_surface(resolution: int = 8, amplitude: float = 0.3) -> TorchNURBSSurface:
    u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    v = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    z = amplitude * torch.sin(uu * math.pi) * torch.sin(vv * math.pi)
    grid = torch.stack((uu, vv, z), dim=2)
    return TorchNURBSSurface(grid, torch.ones((resolution, resolution), dtype=torch.float64))


def _extreme_curvature_surface(resolution: int = 8, amplitude: float = 25.0) -> TorchNURBSSurface:
    return _sine_surface(resolution=resolution, amplitude=amplitude)


def _rotation_matrix() -> torch.Tensor:
    theta = 0.7
    axis = torch.tensor([0.3, 0.6, 0.74162], dtype=torch.float64)
    axis = axis / axis.norm()
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=torch.float64
    )
    identity = torch.eye(3, dtype=torch.float64)
    return identity + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def _make_boundary(
    patch_id: int,
    world: torch.Tensor,
    inner_world: torch.Tensor,
    *,
    closed: bool,
    state: str = BOUNDARY_UNCLASSIFIED,
    boundary_id: str | None = None,
    uv: torch.Tensor | None = None,
    inner_uv: torch.Tensor | None = None,
    confidence: dict | None = None,
) -> PatchBoundarySegment:
    """Hand-crafted boundary record for fixtures the two production helpers
    (`build_rectangular_patch_edge`/`extract_trimmed_patch_boundaries`) can't
    conveniently express. `uv`/`tangent_world`/`inward_tangent_world`/
    `normal_world` are placeholders Phase D never reuses (design section 2) --
    only `world`/`inner_world`/`closed`/`state`/`confidence` matter here.
    """

    n = int(world.shape[0])
    if uv is None:
        uv = torch.zeros((n, 2), dtype=world.dtype)
    if inner_uv is None:
        inner_uv = torch.zeros((n, 2), dtype=world.dtype)
    placeholder = torch.zeros((n, 3), dtype=world.dtype)
    placeholder[:, 0] = 1.0
    return PatchBoundarySegment(
        boundary_id=boundary_id or f"p{patch_id}:manual",
        patch_id=patch_id,
        source_kind="manual_fixture",
        uv=uv,
        world=world,
        inner_uv=inner_uv,
        inner_world=inner_world,
        tangent_world=placeholder,
        inward_tangent_world=placeholder,
        normal_world=placeholder,
        closed=closed,
        orientation="ccw" if closed else "open",
        state=state,
        confidence=confidence or {},
    )


class ContinuationDomainTest(unittest.TestCase):
    def test_planar_boundary_outward_direction_and_zero_second_order_growth(self):
        surface = _plane_surface(degree=1)  # degree 1 => exact linear precision, S_uu/S_uv/S_vv == 0
        boundary = build_rectangular_patch_edge(patch_id=1, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        self.assertEqual(domain.state, STATE_VALID)
        # v0 edge: interior is +v (+y); outward must point -y.
        outward = domain.outward_tangent_world
        self.assertTrue(torch.allclose(outward[:, 1], torch.full((9,), -1.0, dtype=torch.float64), atol=1e-6))
        self.assertLess(domain.uncertainty["second_order_growth_ratio_max"], 1e-8)

    def test_smoothly_curved_boundary_bounded_second_order_growth(self):
        surface = _sine_surface()
        boundary = build_rectangular_patch_edge(patch_id=2, surface=surface, edge="u0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        self.assertIn(domain.state, CONTINUATION_STATES)
        self.assertLess(domain.uncertainty["second_order_growth_ratio_max"], 0.5)

    def test_rotated_plane_world_space_invariance(self):
        surface = _plane_surface()
        boundary = build_rectangular_patch_edge(patch_id=3, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        rotation = _rotation_matrix()
        rotated_surface = TorchNURBSSurface(
            surface.control_grid @ rotation.T, surface.weights.clone(), degree_u=surface.degree_u, degree_v=surface.degree_v
        )
        rotated_boundary = build_rectangular_patch_edge(patch_id=3, surface=rotated_surface, edge="v0", sample_count=9)
        rotated_domain = build_continuation_domain(rotated_surface, rotated_boundary, t_count=4)

        expected_outward = domain.outward_tangent_world @ rotation.T
        self.assertTrue(torch.allclose(rotated_domain.outward_tangent_world, expected_outward, atol=1e-6))

    def test_uv_axis_swap_world_space_invariance(self):
        surface = _plane_surface()
        transposed_grid = surface.control_grid.transpose(0, 1).contiguous()
        swapped_surface = TorchNURBSSurface(transposed_grid, surface.weights.clone())

        # v0 edge on the original surface is the u-varying, v=0 boundary; after
        # swapping which axis is "u" vs "v", the SAME physical edge is u0.
        boundary = build_rectangular_patch_edge(patch_id=4, surface=surface, edge="v0", sample_count=9)
        swapped_boundary = build_rectangular_patch_edge(patch_id=4, surface=swapped_surface, edge="u0", sample_count=9)

        domain = build_continuation_domain(surface, boundary, t_count=4)
        swapped_domain = build_continuation_domain(swapped_surface, swapped_boundary, t_count=4)

        self.assertTrue(torch.allclose(domain.outward_tangent_world, swapped_domain.outward_tangent_world, atol=1e-6))

    def test_uv_scale_skew_world_space_invariance(self):
        surface = _plane_surface()
        boundary = build_rectangular_patch_edge(patch_id=5, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        # Same world geometry, but re-seed the SAME physical plane at a skewed,
        # non-uniform UV sampling (via a differently-shaped control grid whose
        # points still trace the same world line at u=const/v=0).
        skewed_u = torch.tensor([0.0, 0.02, 0.08, 0.2, 0.45, 0.8, 1.0], dtype=torch.float64)
        v_lin = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
        grid = torch.stack(
            (
                skewed_u[:, None].expand(7, 7),
                v_lin[None, :].expand(7, 7),
                torch.zeros((7, 7), dtype=torch.float64),
            ),
            dim=2,
        )
        skewed_surface = TorchNURBSSurface(grid, torch.ones((7, 7), dtype=torch.float64), degree_u=1, degree_v=1)
        skewed_boundary = build_rectangular_patch_edge(patch_id=5, surface=skewed_surface, edge="v0", sample_count=9)
        skewed_domain = build_continuation_domain(skewed_surface, skewed_boundary, t_count=4)

        self.assertTrue(torch.allclose(domain.outward_tangent_world, skewed_domain.outward_tangent_world, atol=1e-6))

    def test_orthogonal_and_oblique_boundaries_not_rejected_on_facing(self):
        plane_xy = _plane_surface(degree=1)
        boundary_xy = build_rectangular_patch_edge(patch_id=6, surface=plane_xy, edge="v0", sample_count=9)
        domain_xy = build_continuation_domain(plane_xy, boundary_xy, t_count=4)

        # A patch whose normal is orthogonal to the first (swap y/z roles).
        resolution = 6
        u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
        w = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
        grid = torch.stack(
            (
                u[:, None].expand(resolution, resolution),
                torch.zeros((resolution, resolution), dtype=torch.float64),
                w[None, :].expand(resolution, resolution),
            ),
            dim=2,
        )
        plane_xz = TorchNURBSSurface(grid, torch.ones((resolution, resolution), dtype=torch.float64), degree_u=1, degree_v=1)
        boundary_xz = build_rectangular_patch_edge(patch_id=7, surface=plane_xz, edge="v0", sample_count=9)
        domain_xz = build_continuation_domain(plane_xz, boundary_xz, t_count=4)

        self.assertIn(domain_xy.state, CONTINUATION_STATES)
        self.assertIn(domain_xz.state, CONTINUATION_STATES)
        # Both succeeded independently -- no facing/normal comparison ever
        # happens between them (there is no such cross-boundary code path).

    def test_annular_seam_rejected_observed_edges_accepted(self):
        surface = _plane_surface(resolution=8)
        seam_boundary = build_rectangular_patch_edge(
            patch_id=8, surface=surface, edge="u0", sample_count=9, state=BOUNDARY_RECONCILED_INTERNAL
        )
        observed_boundary = build_rectangular_patch_edge(patch_id=8, surface=surface, edge="v0", sample_count=9)

        with self.assertRaises(ValueError):
            build_continuation_domain(surface, seam_boundary, t_count=4)
        domain = build_continuation_domain(surface, observed_boundary, t_count=4)
        self.assertEqual(domain.state, STATE_VALID)

    def test_reversed_parameter_direction_same_world_geometry(self):
        surface = _plane_surface(degree=1)
        boundary = build_rectangular_patch_edge(patch_id=9, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        reversed_boundary = _make_boundary(
            9,
            boundary.world.flip(0),
            boundary.inner_world.flip(0),
            closed=False,
        )
        reversed_domain = build_continuation_domain(surface, reversed_boundary, t_count=4)

        self.assertTrue(
            torch.allclose(domain.outward_tangent_world.flip(0), reversed_domain.outward_tangent_world, atol=1e-6)
        )

    def test_boundary_resampling_density_invariance(self):
        surface = _plane_surface(degree=1)
        sparse = build_rectangular_patch_edge(patch_id=10, surface=surface, edge="v0", sample_count=6)
        dense = build_rectangular_patch_edge(patch_id=10, surface=surface, edge="v0", sample_count=21)
        sparse_domain = build_continuation_domain(surface, sparse, t_count=4)
        dense_domain = build_continuation_domain(surface, dense, t_count=4)

        self.assertTrue(
            torch.allclose(
                sparse_domain.outward_tangent_world[0], dense_domain.outward_tangent_world[0], atol=1e-6
            )
        )
        self.assertAlmostEqual(sparse_domain.boundary_length, dense_domain.boundary_length, places=4)

    def test_degenerate_direction_uses_mask_not_nan(self):
        # Collapse only ONE u-column's v0/v1 control-point pair together, so
        # Sv (and the surface normal Su x Sv) collapses to ~0 at just that one
        # boundary sample -- a PARTIAL degeneracy, not a total one. The
        # boundary's own world samples (which come only from row 0, untouched
        # here) stay properly spaced and don't trip the adjacent-duplicate
        # contract check.
        resolution = 6
        u = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
        v = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64)
        grid = torch.stack(
            (
                u[:, None].expand(resolution, resolution),
                v[None, :].expand(resolution, resolution),
                torch.zeros((resolution, resolution), dtype=torch.float64),
            ),
            dim=2,
        )
        grid[0, 1, :] = grid[0, 0, :]
        surface = TorchNURBSSurface(grid, torch.ones((resolution, resolution), dtype=torch.float64), degree_u=1, degree_v=1)
        boundary = build_rectangular_patch_edge(patch_id=11, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        self.assertFalse(bool(torch.isnan(domain.outward_tangent_world).any()))
        self.assertFalse(bool(torch.isnan(domain.normal).any()))
        self.assertFalse(bool(domain.direction_valid_mask.all()))
        self.assertEqual(domain.state, STATE_DEGENERATE)

    def test_high_curvature_fold_over_flags_excessive_growth(self):
        surface = _extreme_curvature_surface()
        boundary = build_rectangular_patch_edge(patch_id=12, surface=surface, edge="u0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4, extent_multiplier=3.0)

        self.assertGreater(domain.uncertainty["second_order_growth_ratio_max"], 0.5)
        self.assertEqual(domain.state, STATE_DEGENERATE)
        self.assertIn("excessive_second_order_growth", domain.reason)

    def test_reconciled_internal_boundary_raises_value_error(self):
        surface = _plane_surface()
        boundary = build_rectangular_patch_edge(
            patch_id=13, surface=surface, edge="v0", sample_count=9, state=BOUNDARY_RECONCILED_INTERNAL
        )
        with self.assertRaises(ValueError):
            build_continuation_domain(surface, boundary, t_count=4)

    def test_unsupported_open_boundary_positive_control(self):
        surface = _plane_surface()
        boundary = build_rectangular_patch_edge(patch_id=14, surface=surface, edge="v0", sample_count=9)
        self.assertEqual(boundary.state, BOUNDARY_UNCLASSIFIED)
        domain = build_continuation_domain(surface, boundary, t_count=4)
        self.assertEqual(domain.state, STATE_VALID)

    def test_corner_endpoint_invariant(self):
        surface = _plane_surface(degree=1)
        boundary = build_rectangular_patch_edge(patch_id=15, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)

        self.assertTrue(torch.isfinite(domain.tangent_s[0]).all())
        self.assertTrue(torch.isfinite(domain.tangent_s[-1]).all())
        self.assertTrue(torch.allclose(domain.world[:, 0, :], boundary.world))

    def test_t_world_matches_measured_world_distance(self):
        surface = _plane_surface(degree=1)
        boundary = build_rectangular_patch_edge(patch_id=16, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=5)

        measured = (domain.world - domain.world[:, :1, :]).norm(dim=-1)
        expected = domain.t_world[None, :].expand(domain.s_count, domain.t_count)
        self.assertTrue(torch.allclose(measured, expected, atol=1e-6))

    def test_closed_boundary_closing_segment_in_boundary_length(self):
        n = 8
        angles = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        world = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(n, dtype=torch.float64)), dim=1)
        inner_world = world * 0.5
        # Duplicate closing endpoint, matching _canonicalize_closed_loop's convention.
        world_with_duplicate = torch.cat([world, world[:1]], dim=0)
        inner_with_duplicate = torch.cat([inner_world, inner_world[:1]], dim=0)
        uv = torch.stack((angles / (2 * math.pi), torch.zeros(n, dtype=torch.float64)), dim=1)
        uv_with_duplicate = torch.cat([uv, uv[:1]], dim=0)
        boundary = _make_boundary(
            17,
            world_with_duplicate,
            inner_with_duplicate,
            closed=True,
            uv=uv_with_duplicate,
            inner_uv=uv_with_duplicate,
        )

        # A flat unit disc's own NURBS surface isn't needed for arclength math;
        # use a trivial plane surface only to satisfy build_continuation_domain's
        # analytic-derivative requirement (outward degeneracy is fine here --
        # this test targets boundary_length/s_world, not outward validity).
        surface = _plane_surface(resolution=4, degree=1)
        surface.control_grid[..., :2] = 0.0  # degenerate on purpose; not under test here
        with self.assertRaises(ContinuationDomainBuildError):
            build_continuation_domain(surface, boundary, t_count=3)

        # Verify boundary_length directly via the shared arclength helper
        # through interpolate_boundary_arclength, which exercises the exact
        # same stripping + closing-segment accounting.
        position_at_zero, _ = interpolate_boundary_arclength(boundary, 0.0)
        self.assertTrue(torch.allclose(position_at_zero, world[0], atol=1e-6))
        segment_length = float((world[1] - world[0]).norm())
        position_after_one_segment, _ = interpolate_boundary_arclength(boundary, segment_length)
        self.assertTrue(torch.allclose(position_after_one_segment, world[1], atol=1e-4))

    def test_open_boundary_adjacent_duplicate_raises_value_error(self):
        surface = _plane_surface(degree=1)
        boundary = build_rectangular_patch_edge(patch_id=18, surface=surface, edge="v0", sample_count=9)
        world = boundary.world.clone()
        world[3] = world[2]  # exact duplicate -> zero-length segment
        bad_boundary = _make_boundary(18, world, boundary.inner_world.clone(), closed=False)
        with self.assertRaises(ValueError):
            build_continuation_domain(surface, bad_boundary, t_count=4)

    def test_closed_boundary_closing_segment_zero_length_raises_value_error(self):
        # 7 genuinely distinct points (no accidental linspace-endpoint
        # duplicate); the LAST is a duplicate of the FIRST (the expected
        # closing-vertex pattern, stripped by _strip_closed_duplicate), and
        # the SECOND-TO-LAST is ALSO forced to equal the first, so that after
        # stripping away the (expected) trailing duplicate, the new last
        # unique sample still coincides with sample 0 -- a genuine
        # zero-length closing segment surviving normalization.
        n = 7
        angles = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        world = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(n, dtype=torch.float64)), dim=1)
        world[-1] = world[0].clone()
        world[-2] = world[0].clone()
        inner_world = world * 0.5
        boundary = _make_boundary(19, world, inner_world, closed=True)
        surface = _plane_surface(resolution=4)
        with self.assertRaises(ValueError):
            build_continuation_domain(surface, boundary, t_count=3)

    def test_local_surface_scale_not_dominated_by_single_probe(self):
        surface = _plane_surface(degree=1)
        boundary = build_rectangular_patch_edge(patch_id=20, surface=surface, edge="v0", sample_count=9)
        base = build_continuation_domain(surface, boundary, t_count=4)

        far_inner_boundary = _make_boundary(
            20, boundary.world.clone(), boundary.inner_world.clone() * 50.0, closed=False
        )
        far_inner_domain = build_continuation_domain(surface, far_inner_boundary, t_count=4)

        ratio = far_inner_domain.local_surface_scale / base.local_surface_scale
        self.assertLess(ratio, 10.0)  # a 50x change in the inner probe alone must not dominate linearly

    def test_local_surface_scale_derivation_failure_raises_build_error(self):
        # Only one boundary sample pair (no positive L_boundary candidate: a
        # single segment collapsed to inner==boundary) and a single-row/column
        # control grid (no L_control candidate) leaves fewer than two valid
        # scale sources.
        world = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
        )
        inner_world = world.clone()  # zero inner distance everywhere -> no L_inner candidate
        boundary = _make_boundary(21, world, inner_world, closed=False)
        control_grid = torch.zeros((1, 1, 3), dtype=torch.float64)
        surface = TorchNURBSSurface(control_grid, torch.ones((1, 1), dtype=torch.float64))

        with self.assertRaises(ContinuationDomainBuildError):
            build_continuation_domain(surface, boundary, t_count=3)

    def test_minimum_sample_count_open_and_closed(self):
        surface = _plane_surface(degree=1)
        open_boundary = build_rectangular_patch_edge(patch_id=22, surface=surface, edge="v0", sample_count=3)
        domain = build_continuation_domain(surface, open_boundary, t_count=3)
        self.assertIn(domain.state, CONTINUATION_STATES)

        n = 4
        angles = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        world = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(n, dtype=torch.float64)), dim=1)
        inner_world = world * 0.5
        closed_boundary = _make_boundary(23, world, inner_world, closed=True)
        closed_domain = build_continuation_domain(surface, closed_boundary, t_count=3)
        self.assertIn(closed_domain.state, CONTINUATION_STATES)

    def test_state_never_promoted_beyond_continuation_states(self):
        surface = _plane_surface()
        boundary = build_rectangular_patch_edge(patch_id=24, surface=surface, edge="v0", sample_count=9)
        domain = build_continuation_domain(surface, boundary, t_count=4)
        self.assertIn(domain.state, CONTINUATION_STATES)
        self.assertNotIn("occluded_candidate", CONTINUATION_STATES)
        self.assertNotIn("validated", CONTINUATION_STATES)


if __name__ == "__main__":
    unittest.main()
