from __future__ import annotations

"""Synthetic Gaussian scenes for the covariance-guided structural
reliability / manifold-affinity / surface-region foundation (worklog
111/113-123, volumetric replacement worklog 124).

Worklog 124 discards the earlier flat-plane/sine-sheet dataset entirely.
Every scene here is now sampled from the surface of a genuine 3D volumetric
solid (box, cylinder, sphere) rather than an isolated infinite flat patch --
real closed/near-closed manifolds with real multi-face corners, continuous
curvature, and (for the sphere) no boundary anywhere. This is deliberately
NOT the NURBS-fitting benchmark scenes in ``scenes.py`` (those are raw point
clouds without covariance). Each scene here returns explicit per-Gaussian
``positions`` AND ``covariances`` so the reliability/affinity/region-formation
modules -- which only consume covariance-guided structural evidence, never a
raster/KDE mask -- can be exercised end to end without any renderer/trainer/
model dependency.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

GAUSSIAN_RELIABILITY_SCENE_NAMES = (
    "box_face",
    "box",
    "cylinder",
    "sphere",
    "thin_slab",
    "box_isolated_floater",
    "box_isotropic_contamination",
    "box_with_bridge",
)


@dataclass
class GaussianReliabilityScene:
    name: str
    positions: Any  # (N, 3)
    covariances: Any  # (N, 3, 3)
    description: str
    # Optional per-Gaussian labels for test assertions only -- never consumed
    # by the reliability/affinity/region-formation modules themselves.
    group_labels: tuple[str, ...] = ()


def _identity_quaternion(count: int) -> Any:
    return torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4).clone()


def _quaternion_aligning_z_to(target: Any) -> Any:
    """Unit quaternion rotating the +z axis onto ``target`` (unit vector)."""
    z_axis = torch.tensor([0.0, 0.0, 1.0])
    target = target / target.norm().clamp_min(1e-12)
    dot = float((z_axis * target).sum())
    if dot > 1.0 - 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return torch.tensor([0.0, 1.0, 0.0, 0.0])
    axis = torch.linalg.cross(z_axis, target)
    axis = axis / axis.norm().clamp_min(1e-12)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    half = angle / 2.0
    return torch.tensor([math.cos(half), axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half)])


def _quaternions_aligning_z_to_batch(targets: Any) -> Any:
    return torch.stack([_quaternion_aligning_z_to(targets[i]) for i in range(targets.shape[0])], dim=0)


def _flat_grid(
    count_per_axis: int, spacing: float, *, normal: tuple = (0.0, 0.0, 1.0), origin: tuple = (0.0, 0.0, 0.0),
    surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0, position_noise: float = 0.001,
) -> tuple[Any, Any]:
    """A regular grid of planar surfel Gaussians tangent to the given plane --
    this is exactly what ONE FACE of a box looks like; box/cylinder/sphere
    scenes below compose several of these (or their curved equivalent) into a
    genuine closed or near-closed volumetric solid."""
    generator = torch.Generator().manual_seed(seed)
    lin = (torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0) * spacing
    grid_u, grid_v = torch.meshgrid(lin, lin, indexing="ij")
    normal_t = torch.tensor(normal, dtype=torch.float32)
    reference = torch.tensor([1.0, 0.0, 0.0]) if abs(normal_t[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    axis_u = torch.linalg.cross(normal_t, reference)
    axis_u = axis_u / axis_u.norm().clamp_min(1e-12)
    axis_v = torch.linalg.cross(normal_t, axis_u)
    origin_t = torch.tensor(origin, dtype=torch.float32)
    positions = (
        origin_t
        + grid_u.reshape(-1, 1) * axis_u
        + grid_v.reshape(-1, 1) * axis_v
        + position_noise * torch.randn(count_per_axis * count_per_axis, 3, generator=generator)
    )
    count = positions.shape[0]
    scale = torch.tensor([surfel_scale, surfel_scale, surfel_thickness]).expand(count, 3).clone()
    quaternion = _quaternion_aligning_z_to(normal_t).expand(count, 4).clone()
    covariances = covariance_from_scale_rotation(scale, quaternion)
    return positions, covariances


def _box_faces(
    half_extent: tuple[float, float, float], count_per_axis: int, *,
    surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0,
    faces: tuple[str, ...] = ("px", "nx", "py", "ny", "pz", "nz"),
) -> tuple[Any, Any, tuple[str, ...]]:
    """Sample the requested outward-facing faces of an axis-aligned box.
    Adjacent faces meet at genuine 90-degree edges/corners (a real box has 12
    edges and 8 corners where 3 faces meet -- far richer crease structure
    than a single perpendicular-pair fixture)."""
    hx, hy, hz = half_extent
    face_specs = {
        "px": ((1.0, 0.0, 0.0), (hx, 0.0, 0.0), hy, hz),
        "nx": ((-1.0, 0.0, 0.0), (-hx, 0.0, 0.0), hy, hz),
        "py": ((0.0, 1.0, 0.0), (0.0, hy, 0.0), hx, hz),
        "ny": ((0.0, -1.0, 0.0), (0.0, -hy, 0.0), hx, hz),
        "pz": ((0.0, 0.0, 1.0), (0.0, 0.0, hz), hx, hy),
        "nz": ((0.0, 0.0, -1.0), (0.0, 0.0, -hz), hx, hy),
    }
    all_positions = []
    all_covariances = []
    all_labels: list[str] = []
    for face_index, face in enumerate(faces):
        normal, origin, extent_u, extent_v = face_specs[face]
        spacing = (2.0 * max(extent_u, extent_v)) / max(count_per_axis - 1, 1)
        positions, covariances = _flat_grid(
            count_per_axis, spacing, normal=normal, origin=origin,
            surfel_scale=surfel_scale, surfel_thickness=surfel_thickness, seed=seed + face_index,
        )
        all_positions.append(positions)
        all_covariances.append(covariances)
        all_labels.extend([f"face_{face}"] * positions.shape[0])
    return torch.cat(all_positions, dim=0), torch.cat(all_covariances, dim=0), tuple(all_labels)


def _cylinder_surface(
    radius: float, height: float, *, angular_count: int = 24, height_count: int = 9,
    surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0,
    include_caps: bool = True, cap_count_per_axis: int = 7,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Side surface (curved circumferentially, flat axially -- genuine
    anisotropic curvature) plus optional flat top/bottom caps meeting the side
    at a real circular crease."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.linspace(0.0, 2.0 * math.pi, angular_count + 1)[:-1]
    heights = (torch.arange(height_count, dtype=torch.float32) - (height_count - 1) / 2.0) * (height / max(height_count - 1, 1))
    grid_angle, grid_height = torch.meshgrid(angles, heights, indexing="ij")
    grid_angle = grid_angle.reshape(-1)
    grid_height = grid_height.reshape(-1)
    x = radius * torch.cos(grid_angle)
    y = radius * torch.sin(grid_angle)
    z = grid_height
    positions = torch.stack((x, y, z), dim=1)
    positions = positions + 0.001 * torch.randn_like(positions, generator=generator)
    normals = torch.stack((torch.cos(grid_angle), torch.sin(grid_angle), torch.zeros_like(grid_angle)), dim=1)
    count = positions.shape[0]
    scale = torch.tensor([surfel_scale, surfel_scale, surfel_thickness]).expand(count, 3).clone()
    quaternions = _quaternions_aligning_z_to_batch(normals)
    covariances = covariance_from_scale_rotation(scale, quaternions)
    labels = ["side"] * count

    if include_caps:
        top_positions, top_cov = _flat_grid(
            cap_count_per_axis, (2.0 * radius) / max(cap_count_per_axis - 1, 1),
            normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, height / 2.0),
            surfel_scale=surfel_scale, surfel_thickness=surfel_thickness, seed=seed + 101,
        )
        bottom_positions, bottom_cov = _flat_grid(
            cap_count_per_axis, (2.0 * radius) / max(cap_count_per_axis - 1, 1),
            normal=(0.0, 0.0, -1.0), origin=(0.0, 0.0, -height / 2.0),
            surfel_scale=surfel_scale, surfel_thickness=surfel_thickness, seed=seed + 202,
        )
        # Caps are round; drop cap surfels that fall outside the cylinder radius
        # so the cap does not extend past the side wall.
        top_mask = (top_positions[:, 0] ** 2 + top_positions[:, 1] ** 2) <= radius ** 2
        bottom_mask = (bottom_positions[:, 0] ** 2 + bottom_positions[:, 1] ** 2) <= radius ** 2
        top_positions, top_cov = top_positions[top_mask], top_cov[top_mask]
        bottom_positions, bottom_cov = bottom_positions[bottom_mask], bottom_cov[bottom_mask]
        positions = torch.cat((positions, top_positions, bottom_positions), dim=0)
        covariances = torch.cat((covariances, top_cov, bottom_cov), dim=0)
        labels = labels + ["top_cap"] * top_positions.shape[0] + ["bottom_cap"] * bottom_positions.shape[0]

    return positions, covariances, tuple(labels)


def _sphere_surface(
    radius: float, *, count: int = 200, surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0,
) -> tuple[Any, Any, tuple[str, ...]]:
    """A closed, boundary-free spherical manifold via Fibonacci-sphere
    sampling (near-uniform density, no pole singularity artifacts)."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(count, dtype=torch.float32) + 0.5
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z_unit = 1.0 - 2.0 * indices / count
    radial = torch.sqrt(torch.clamp(1.0 - z_unit * z_unit, min=0.0))
    theta = golden_angle * indices
    x_unit = radial * torch.cos(theta)
    y_unit = radial * torch.sin(theta)
    unit = torch.stack((x_unit, y_unit, z_unit), dim=1)
    unit = unit / unit.norm(dim=1, keepdim=True).clamp_min(1e-12)
    positions = radius * unit
    positions = positions + 0.001 * torch.randn_like(positions, generator=generator)
    quaternions = _quaternions_aligning_z_to_batch(unit)
    scale = torch.tensor([surfel_scale, surfel_scale, surfel_thickness]).expand(count, 3).clone()
    covariances = covariance_from_scale_rotation(scale, quaternions)
    return positions, covariances, ("sphere",) * count


def _spherical_patch(
    radius: float, *, patch_half_extent: float = 0.5, count_per_axis: int = 11,
    surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0,
) -> tuple[Any, Any]:
    """A LOCAL patch of a sphere of the given ``radius``, parametrized so its
    physical (arc-length) extent and point density stay FIXED as ``radius``
    varies -- unlike sampling the whole closed sphere, whose point density
    would collapse at large radius for a fixed point count. As radius -> inf
    this patch flattens toward the same flat box-face limit used elsewhere."""
    generator = torch.Generator().manual_seed(seed)
    lin = torch.linspace(-patch_half_extent, patch_half_extent, count_per_axis)
    grid_u, grid_v = torch.meshgrid(lin, lin, indexing="ij")
    # Angle subtended at the sphere center by an arc-length of `u` at this radius.
    theta_u = grid_u.reshape(-1) / radius
    theta_v = grid_v.reshape(-1) / radius
    x = radius * torch.sin(theta_u)
    y = radius * torch.sin(theta_v) * torch.cos(theta_u)
    z = radius * (torch.cos(theta_u) * torch.cos(theta_v))
    positions = torch.stack((x, y, z - radius), dim=1)  # patch centered near origin
    positions = positions + 0.001 * torch.randn_like(positions, generator=generator)
    unit_normal = torch.nn.functional.normalize(torch.stack((x, y, z), dim=1), dim=1)
    count = positions.shape[0]
    scale = torch.tensor([surfel_scale, surfel_scale, surfel_thickness]).expand(count, 3).clone()
    quaternions = _quaternions_aligning_z_to_batch(unit_normal)
    covariances = covariance_from_scale_rotation(scale, quaternions)
    return positions, covariances


def make_curvature_sweep_scene(radius: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 curvature sweep: a genuine spherical PATCH (not a sine
    height-field, and not the whole closed sphere -- density/extent are kept
    fixed across the sweep) with continuously-varying curvature via
    ``radius``. Smaller radius = more curvature; larger radius = a near-flat
    local patch, matching the flat box-face limit as radius -> infinity."""
    positions, covariances = _spherical_patch(radius, seed=seed)
    return GaussianReliabilityScene(
        f"curvature_sweep_radius_{radius}", positions, covariances,
        f"Local spherical patch at radius={radius}, fixed physical extent -- curvature sweep fixture "
        "(smaller radius = more curvature).",
    )


def make_density_variation_scene(kind: str, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 density-variation matrix, sampled on ONE FACE of a box.

    ``kind`` in: uniform, center_dense_boundary_sparse, gradual_gradient,
    abrupt_transition, sparse_but_continuous.
    """
    generator = torch.Generator().manual_seed(seed)
    count_per_axis = 11
    base_spacing = 0.12
    lin_index = torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0

    if kind == "uniform":
        positions, covariances = _flat_grid(count_per_axis, base_spacing, seed=seed)
        return GaussianReliabilityScene(kind, positions, covariances, "Uniform-density box face -- density-variation baseline.")

    grid_u, grid_v = torch.meshgrid(lin_index, lin_index, indexing="ij")
    if kind == "center_dense_boundary_sparse":
        radius = torch.sqrt(grid_u.square() + grid_v.square()) / (count_per_axis / 2.0)
        # Stretch must stay well inside candidate range (scale_radius_multiplier
        # * tangent_major_scale) even at the boundary, or density variation alone
        # accidentally exceeds candidate support and manufactures a fake
        # fragmentation that has nothing to do with the density-variation signal
        # the fixture is meant to test (worklog 114 finding: 1.5 coefficient pushed
        # boundary spacing to the threshold edge and fragmented into 9 regions).
        stretch = 1.0 + 0.8 * radius
    elif kind == "gradual_gradient":
        stretch = 1.0 + 0.6 * (grid_u - lin_index.min()) / (lin_index.max() - lin_index.min()).clamp_min(1e-6)
    elif kind == "abrupt_transition":
        stretch = torch.where(grid_u >= 0, torch.full_like(grid_u, 2.2), torch.full_like(grid_u, 1.0))
    elif kind == "sparse_but_continuous":
        stretch = torch.full_like(grid_u, 1.8)
    else:
        raise ValueError(f"Unknown density-variation kind: {kind!r}")

    positions_xy = torch.stack((grid_u * base_spacing * stretch, grid_v * base_spacing * stretch), dim=-1).reshape(-1, 2)
    positions = torch.cat((positions_xy, torch.zeros((positions_xy.shape[0], 1))), dim=1)
    positions = positions + 0.001 * torch.randn_like(positions, generator=generator)
    count = positions.shape[0]
    scale = torch.tensor([0.05, 0.05, 0.002]).expand(count, 3).clone()
    quaternion = _identity_quaternion(count)
    covariances = covariance_from_scale_rotation(scale, quaternion)
    return GaussianReliabilityScene(kind, positions, covariances, f"Density-variation fixture on a box face: {kind}.")


def make_position_noise_scene(noise_std: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 position-noise sweep: a clean box face with additive
    Gaussian position jitter of the given standard deviation (in scene units)."""
    generator = torch.Generator().manual_seed(seed)
    positions, covariances = _flat_grid(9, 0.12, seed=seed)
    positions = positions + noise_std * torch.randn(positions.shape, generator=generator)
    return GaussianReliabilityScene(
        f"position_noise_{noise_std}", positions, covariances, f"Box face with position noise std={noise_std}.",
    )


def make_orientation_noise_scene(noise_degrees: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 covariance-orientation-noise sweep: centers FIXED, each
    Gaussian's covariance rotated by a random small angle about a random axis."""
    generator = torch.Generator().manual_seed(seed)
    positions, _ = _flat_grid(9, 0.12, seed=seed)
    count = positions.shape[0]
    scale = torch.tensor([0.05, 0.05, 0.002]).expand(count, 3).clone()
    base_quaternion = _identity_quaternion(count)
    if noise_degrees <= 0:
        covariances = covariance_from_scale_rotation(scale, base_quaternion)
        return GaussianReliabilityScene("orientation_noise_0", positions, covariances, "Box face, zero orientation noise (control).")
    axes = torch.nn.functional.normalize(torch.randn((count, 3), generator=generator), dim=1)
    angles = math.radians(noise_degrees) * torch.rand((count,), generator=generator)
    half = angles / 2.0
    noise_quaternion = torch.cat((torch.cos(half).unsqueeze(1), axes * torch.sin(half).unsqueeze(1)), dim=1)

    def _quat_mul(q1: Any, q2: Any) -> Any:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    perturbed_quaternion = _quat_mul(noise_quaternion, base_quaternion)
    covariances = covariance_from_scale_rotation(scale, perturbed_quaternion)
    return GaussianReliabilityScene(
        f"orientation_noise_{noise_degrees}", positions, covariances,
        f"Box face, centers fixed, covariance rotated by up to {noise_degrees} degrees of random-axis noise per Gaussian.",
    )


def make_gap_sweep_scene(gap: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 thin-gap/close-parallel sweep: the top and bottom faces of
    a THIN SLAB (a real box collapsed along one axis), with OPPOSITE outward
    normals (+z / -z) as a genuine physical front/back of a solid panel --
    the configurable ``gap`` is the slab thickness."""
    top_positions, top_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, gap / 2.0), seed=seed)
    bottom_positions, bottom_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, -1.0), origin=(0.0, 0.0, -gap / 2.0), seed=seed + 1)
    positions = torch.cat((top_positions, bottom_positions), dim=0)
    covariances = torch.cat((top_cov, bottom_cov), dim=0)
    labels = ("top",) * top_positions.shape[0] + ("bottom",) * bottom_positions.shape[0]
    return GaussianReliabilityScene(
        f"gap_sweep_{gap}", positions, covariances,
        f"Top/bottom faces of a thin slab with thickness (gap)={gap} -- opposite normals, same tangent plane.", labels,
    )


def make_missing_support_gap_scene(gap_fraction: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124: a box face with a rectangular hole (missing Gaussians)
    covering the central ``gap_fraction`` of the grid -- a small sampling gap
    should not be treated the same as a genuine surface discontinuity."""
    count_per_axis = 11
    spacing = 0.1
    positions, covariances = _flat_grid(count_per_axis, spacing, seed=seed)
    lin = torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0
    grid_u, grid_v = torch.meshgrid(lin, lin, indexing="ij")
    half_width = (count_per_axis / 2.0) * gap_fraction
    keep_mask = ~((grid_u.abs() <= half_width) & (grid_v.abs() <= half_width))
    keep_mask = keep_mask.reshape(-1)
    return GaussianReliabilityScene(
        f"missing_support_gap_{gap_fraction}", positions[keep_mask], covariances[keep_mask],
        f"Box face with a central missing-support gap covering fraction={gap_fraction} of the grid.",
    )


def make_shape_ratio_sweep_scene(minor_major_ratio: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114/124 needle-like/near-isotropic sweep: a small cluster of
    Gaussians sharing one continuous eigenvalue-ratio point. This is a pure
    per-Gaussian covariance-shape test (no surface/topology claim), so it is
    NOT redefined in volumetric terms. ``minor_major_ratio`` in [0, 1]:
    0.0 -> both minor axes tiny relative to the major axis (needle_like),
    1.0 -> all three axes equal (isotropic); values in between sweep through
    ``ambiguous_shape`` continuously (both minor axes move together, so this
    sweep never passes through a clean planar_surfel point by construction)."""
    generator = torch.Generator().manual_seed(seed)
    count = 7
    minor_scale = 0.05 + minor_major_ratio * (1.0 - 0.05)
    positions = 0.05 * torch.randn((count, 3), generator=generator)
    scale = torch.tensor([1.0, minor_scale, minor_scale]).expand(count, 3).clone()
    quaternion = _identity_quaternion(count)
    covariances = covariance_from_scale_rotation(scale, quaternion)
    return GaussianReliabilityScene(
        f"shape_ratio_sweep_{minor_major_ratio}", positions, covariances,
        f"Continuous needle-like(0.0) -> isotropic(1.0) eigenvalue-ratio sweep at ratio={minor_major_ratio}.",
    )


def make_anisotropic_planar_bridge_scene(*, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124: a PLANAR (not isotropic) oversized bridge Gaussian sitting
    in the interior of a box, whose orientation resembles one face, but whose
    tangent footprint is large enough to span toward an adjacent face --
    must not rely on isotropic rejection alone to block the bridge."""
    scene = make_gaussian_reliability_scene("box", seed=seed)
    bridge_position = torch.tensor([[0.18, 0.0, 0.18]])  # in the box interior, near the pz/px edge
    bridge_scale = torch.tensor([[0.5, 0.45, 0.01]])  # planar, but tangent footprint >> ordinary surfels
    bridge_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # normal ~ +z, like face_pz
    bridge_cov = covariance_from_scale_rotation(bridge_scale, bridge_quaternion)
    positions = torch.cat((scene.positions, bridge_position), dim=0)
    covariances = torch.cat((scene.covariances, bridge_cov), dim=0)
    labels = scene.group_labels + ("anisotropic_bridge",)
    return GaussianReliabilityScene(
        "anisotropic_planar_bridge", positions, covariances,
        "Closed box plus one large PLANAR (face-aligned) Gaussian spanning toward an adjacent face -- "
        "must be blocked by footprint/scale reasoning, not isotropic rejection.",
        labels,
    )


def make_contamination_regression_scene(*, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 124 (was 114 §10): one box face with all 6 named contaminant
    types inserted near (not far from) its own Gaussians, so their effect on
    SURROUNDING normal Gaussians -- not just the contaminant's own label --
    can be checked."""
    plane_positions, plane_cov = _flat_grid(9, 0.12, seed=seed)
    labels = ["face"] * plane_positions.shape[0]

    extra_positions = []
    extra_scales = []
    extra_quaternions = []

    # isolated floater -- far from everything, no real neighbors.
    extra_positions.append([3.0, 3.0, 3.0])
    extra_scales.append([0.05, 0.05, 0.002])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("floater")

    # isotropic Gaussian -- hovering just above the face center, no orientation evidence at all.
    extra_positions.append([0.0, 0.0, 0.05])
    extra_scales.append([0.05, 0.05, 0.05])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("isotropic")

    # wrong-normal planar Gaussian -- planar (reliable shape) but oriented perpendicular to the face.
    extra_positions.append([0.24, 0.24, 0.0])
    extra_scales.append([0.05, 0.05, 0.002])
    extra_quaternions.append(_quaternion_aligning_z_to(torch.tensor([1.0, 0.0, 0.0])).tolist())
    labels.append("wrong_normal")

    # oversized planar Gaussian -- correctly oriented, but tangent footprint far larger than its neighbors.
    extra_positions.append([-0.24, -0.24, 0.0])
    extra_scales.append([0.3, 0.3, 0.002])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("oversized")

    # tiny-scale Gaussian -- correctly oriented and shaped, but far smaller than its neighbors.
    extra_positions.append([0.24, -0.24, 0.0])
    extra_scales.append([0.005, 0.005, 0.0005])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("tiny_scale")

    positions = torch.cat((plane_positions, torch.tensor(extra_positions)), dim=0)
    covariances = torch.cat(
        (plane_cov, covariance_from_scale_rotation(torch.tensor(extra_scales), torch.tensor(extra_quaternions))), dim=0,
    )

    # nearby second surface -- a small perpendicular patch sharing an edge with one corner of the face
    # (i.e. an adjacent box face), same as a real box corner.
    second_positions, second_cov = _flat_grid(3, 0.12, normal=(1.0, 0.0, 0.0), origin=(0.48, 0.0, 0.06), seed=seed + 1)
    positions = torch.cat((positions, second_positions), dim=0)
    covariances = torch.cat((covariances, second_cov), dim=0)
    labels.extend(["second_surface"] * second_positions.shape[0])

    return GaussianReliabilityScene(
        "contamination_regression", positions, covariances,
        "Box face plus all 6 worklog 114/124 §10 contaminant types, inserted near (not far from) the face's own Gaussians.",
        tuple(labels),
    )


def make_gaussian_reliability_scene(name: str, seed: int = 0) -> GaussianReliabilityScene:
    if name not in GAUSSIAN_RELIABILITY_SCENE_NAMES:
        raise ValueError(f"Unknown Gaussian reliability scene: {name!r}")

    if name == "box_face":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "Single flat face of a box (one surfel patch); all Gaussians expected reliable.",
        )

    if name == "box":
        positions, covariances, labels = _box_faces((0.36, 0.36, 0.36), 7, seed=seed)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A genuine closed 3D box: 6 faces, 12 edges, 8 corners -- expect a crease at every "
            "shared edge and each face to form its own stable same_surface region.",
            labels,
        )

    if name == "cylinder":
        positions, covariances, labels = _cylinder_surface(radius=0.3, height=0.6, seed=seed)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A closed cylinder: circumferentially-curved side wall plus two flat end caps meeting "
            "the side at a circular crease -- tests curved+flat mixed topology.",
            labels,
        )

    if name == "sphere":
        positions, covariances, labels = _sphere_surface(radius=0.3, count=200, seed=seed)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A closed sphere: fully curved manifold with NO flat region and NO boundary anywhere -- "
            "must not manufacture a fake boundary or fragment purely from gradual curvature.",
            labels,
        )

    if name == "thin_slab":
        return make_gap_sweep_scene(0.15, seed=seed)

    if name == "box_isolated_floater":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        floater_position = torch.tensor([[3.0, 3.0, 3.0]])
        floater_scale = torch.tensor([[0.05, 0.05, 0.002]])
        floater_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        floater_cov = covariance_from_scale_rotation(floater_scale, floater_quaternion)
        all_positions = torch.cat((positions, floater_position), dim=0)
        all_covariances = torch.cat((covariances, floater_cov), dim=0)
        labels = ("face",) * positions.shape[0] + ("floater",)
        return GaussianReliabilityScene(
            name, all_positions, all_covariances,
            "A box face plus one Gaussian far from every neighbor -- must not be treated as reliable "
            "boundary/interior evidence.",
            labels,
        )

    if name == "box_isotropic_contamination":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        count = positions.shape[0]
        isotropic_indices = torch.tensor([count // 2, count // 2 + 1])
        blob_scale = torch.full((2, 3), 0.05)
        blob_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 4).clone()
        covariances = covariances.clone()
        covariances[isotropic_indices] = covariance_from_scale_rotation(blob_scale, blob_quaternion)
        labels = tuple("isotropic" if i in isotropic_indices.tolist() else "face" for i in range(count))
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A box face with two Gaussians replaced by isotropic (spherical) covariance at the same "
            "positions -- no reliable normal evidence.",
            labels,
        )

    if name == "box_with_bridge":
        scene = make_gaussian_reliability_scene("box", seed=seed)
        bridge_position = torch.tensor([[0.18, 0.0, 0.18]])  # box interior, near the pz/px edge
        bridge_scale = torch.tensor([[0.4, 0.4, 0.4]])
        bridge_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        bridge_cov = covariance_from_scale_rotation(bridge_scale, bridge_quaternion)
        positions = torch.cat((scene.positions, bridge_position), dim=0)
        covariances = torch.cat((scene.covariances, bridge_cov), dim=0)
        labels = scene.group_labels + ("bridge",)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A closed box plus one huge isotropic-ish Gaussian sitting in its interior -- must not "
            "bridge two adjacent faces into one same_surface region.",
            labels,
        )

    raise AssertionError("unreachable")
