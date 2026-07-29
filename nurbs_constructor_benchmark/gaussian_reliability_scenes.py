from __future__ import annotations

"""Small synthetic Gaussian scenes for the covariance-guided structural
reliability / manifold-affinity foundation (worklog 111/113).

These are deliberately NOT the NURBS-fitting benchmark scenes in
``scenes.py`` (those are raw point clouds without covariance). Each scene here
returns explicit per-Gaussian ``positions`` AND ``covariances`` so the
reliability/affinity modules -- which only consume covariance-guided
structural evidence, never a raster/KDE mask -- can be exercised end to end
without any renderer/trainer/model dependency.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation

GAUSSIAN_RELIABILITY_SCENE_NAMES = (
    "plane",
    "two_perpendicular_surfaces",
    "close_parallel_sheets",
    "smooth_curved_sheet",
    "isolated_floater",
    "isotropic_blob",
    "oversized_bridge",
)


@dataclass
class GaussianReliabilityScene:
    name: str
    positions: Any  # (N, 3)
    covariances: Any  # (N, 3, 3)
    description: str
    # Optional per-Gaussian labels for test assertions only -- never consumed
    # by the reliability/affinity modules themselves.
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


def _flat_grid(
    count_per_axis: int, spacing: float, *, normal: tuple = (0.0, 0.0, 1.0), origin: tuple = (0.0, 0.0, 0.0),
    surfel_scale: float = 0.05, surfel_thickness: float = 0.002, seed: int = 0,
) -> tuple[Any, Any]:
    """A regular grid of planar surfel Gaussians tangent to the given plane."""
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
        + 0.001 * torch.randn(count_per_axis * count_per_axis, 3, generator=generator)
    )
    count = positions.shape[0]
    scale = torch.tensor([surfel_scale, surfel_scale, surfel_thickness]).expand(count, 3).clone()
    quaternion = _quaternion_aligning_z_to(normal_t).expand(count, 4).clone()
    covariances = covariance_from_scale_rotation(scale, quaternion)
    return positions, covariances


def _curved_sheet(*, amplitude: float, frequency: float, count_per_axis: int, spacing: float, seed: int = 0) -> tuple[Any, Any]:
    """A sine-wave sheet ``z = amplitude * sin(frequency * x)`` with analytic
    per-point tangent frame -- used by the curvature sweep (worklog 114 §9)."""
    generator = torch.Generator().manual_seed(seed)
    lin = (torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0) * spacing
    grid_x, grid_y = torch.meshgrid(lin, lin, indexing="ij")
    z = amplitude * torch.sin(frequency * grid_x)
    positions = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), z.reshape(-1)), dim=1)
    positions = positions + 0.001 * torch.randn_like(positions, generator=generator)
    dzdx = amplitude * frequency * torch.cos(frequency * grid_x).reshape(-1)
    tangent_x = torch.stack((torch.ones_like(dzdx), torch.zeros_like(dzdx), dzdx), dim=1)
    tangent_x = tangent_x / tangent_x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    tangent_y = torch.tensor([0.0, 1.0, 0.0]).expand_as(tangent_x)
    normal = torch.linalg.cross(tangent_x, tangent_y)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-12)
    count = positions.shape[0]
    scale = torch.tensor([0.05, 0.05, 0.002]).expand(count, 3).clone()
    quaternions = torch.stack([_quaternion_aligning_z_to(normal[i]) for i in range(count)], dim=0)
    covariances = covariance_from_scale_rotation(scale, quaternions)
    return positions, covariances


def make_curvature_sweep_scene(amplitude: float, *, frequency: float = 1.2, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 curvature sweep: same sheet, increasing ``amplitude``
    (0 = flat, larger = more curved, eventually a real fold)."""
    positions, covariances = _curved_sheet(amplitude=amplitude, frequency=frequency, count_per_axis=9, spacing=0.12, seed=seed)
    return GaussianReliabilityScene(
        f"curvature_sweep_amplitude_{amplitude}", positions, covariances,
        f"Sine sheet at amplitude={amplitude}, frequency={frequency} -- curvature sweep fixture.",
    )


def make_density_variation_scene(kind: str, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 density-variation matrix.

    ``kind`` in: uniform, center_dense_boundary_sparse, gradual_gradient,
    abrupt_transition, sparse_but_continuous.
    """
    generator = torch.Generator().manual_seed(seed)
    count_per_axis = 11
    base_spacing = 0.12
    lin_index = torch.arange(count_per_axis, dtype=torch.float32) - (count_per_axis - 1) / 2.0

    if kind == "uniform":
        positions, covariances = _flat_grid(count_per_axis, base_spacing, seed=seed)
        return GaussianReliabilityScene(kind, positions, covariances, "Uniform-density plane -- density-variation baseline.")

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
    return GaussianReliabilityScene(kind, positions, covariances, f"Density-variation fixture: {kind}.")


def make_position_noise_scene(noise_std: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 position-noise sweep: clean plane with additive Gaussian
    position jitter of the given standard deviation (in scene units)."""
    generator = torch.Generator().manual_seed(seed)
    positions, covariances = _flat_grid(9, 0.12, seed=seed)
    positions = positions + noise_std * torch.randn(positions.shape, generator=generator)
    return GaussianReliabilityScene(
        f"position_noise_{noise_std}", positions, covariances, f"Flat plane with position noise std={noise_std}.",
    )


def make_orientation_noise_scene(noise_degrees: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 covariance-orientation-noise sweep: centers FIXED, each
    Gaussian's covariance rotated by a random small angle about a random axis."""
    generator = torch.Generator().manual_seed(seed)
    positions, _ = _flat_grid(9, 0.12, seed=seed)
    count = positions.shape[0]
    scale = torch.tensor([0.05, 0.05, 0.002]).expand(count, 3).clone()
    base_quaternion = _identity_quaternion(count)
    if noise_degrees <= 0:
        covariances = covariance_from_scale_rotation(scale, base_quaternion)
        return GaussianReliabilityScene("orientation_noise_0", positions, covariances, "Flat plane, zero orientation noise (control).")
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
        f"Flat plane, centers fixed, covariance rotated by up to {noise_degrees} degrees of random-axis noise per Gaussian.",
    )


def make_anisotropic_planar_bridge_scene(*, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9: a PLANAR (not isotropic) oversized bridge Gaussian
    whose orientation resembles the floor, but whose tangent footprint is
    large enough to span the floor/wall gap -- must not rely on isotropic
    rejection alone to block the bridge."""
    scene = make_gaussian_reliability_scene("two_perpendicular_surfaces", seed=seed)
    bridge_position = torch.tensor([[0.0, 0.0, 0.18]])
    bridge_scale = torch.tensor([[0.5, 0.45, 0.01]])  # planar, but tangent footprint >> ordinary surfels
    bridge_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # normal ~ +z, like the floor
    bridge_cov = covariance_from_scale_rotation(bridge_scale, bridge_quaternion)
    positions = torch.cat((scene.positions, bridge_position), dim=0)
    covariances = torch.cat((scene.covariances, bridge_cov), dim=0)
    labels = scene.group_labels + ("anisotropic_bridge",)
    return GaussianReliabilityScene(
        "anisotropic_planar_bridge", positions, covariances,
        "Floor+wall plus one large PLANAR (floor-aligned) Gaussian spanning the gap -- must be blocked by footprint/scale reasoning, not isotropic rejection.",
        labels,
    )


def make_gap_sweep_scene(gap: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 thin-gap/close-parallel sweep: two parallel sheets with
    a configurable normal-direction gap (in scene units)."""
    lower_positions, lower_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), seed=seed)
    upper_positions, upper_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, gap), seed=seed + 1)
    positions = torch.cat((lower_positions, upper_positions), dim=0)
    covariances = torch.cat((lower_cov, upper_cov), dim=0)
    labels = ("lower",) * lower_positions.shape[0] + ("upper",) * upper_positions.shape[0]
    return GaussianReliabilityScene(f"gap_sweep_{gap}", positions, covariances, f"Two parallel sheets with gap={gap}.", labels)


def make_missing_support_gap_scene(gap_fraction: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9: a flat plane with a rectangular hole (missing Gaussians)
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
        f"Flat plane with a central missing-support gap covering fraction={gap_fraction} of the grid.",
    )


def make_shape_ratio_sweep_scene(minor_major_ratio: float, *, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §9 needle-like/near-isotropic sweep: a small cluster of
    Gaussians sharing one continuous eigenvalue-ratio point, rather than a
    handful of hand-picked extreme fixtures. ``minor_major_ratio`` in [0, 1]:
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


def make_contamination_regression_scene(*, seed: int = 0) -> GaussianReliabilityScene:
    """Worklog 114 §10: one clean plane with all 6 named contaminant types
    inserted near (not far from) its own Gaussians, so their effect on
    SURROUNDING normal Gaussians -- not just the contaminant's own label --
    can be checked."""
    plane_positions, plane_cov = _flat_grid(9, 0.12, seed=seed)
    labels = ["plane"] * plane_positions.shape[0]

    extra_positions = []
    extra_scales = []
    extra_quaternions = []

    # isolated floater -- far from everything, no real neighbors.
    extra_positions.append([3.0, 3.0, 3.0])
    extra_scales.append([0.05, 0.05, 0.002])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("floater")

    # isotropic Gaussian -- hovering just above the plane center, no orientation evidence at all.
    extra_positions.append([0.0, 0.0, 0.05])
    extra_scales.append([0.05, 0.05, 0.05])
    extra_quaternions.append([1.0, 0.0, 0.0, 0.0])
    labels.append("isotropic")

    # wrong-normal planar Gaussian -- planar (reliable shape) but oriented perpendicular to the plane.
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

    # nearby second surface -- a small perpendicular patch sharing an edge with one corner of the plane.
    second_positions, second_cov = _flat_grid(3, 0.12, normal=(1.0, 0.0, 0.0), origin=(0.48, 0.0, 0.06), seed=seed + 1)
    positions = torch.cat((positions, second_positions), dim=0)
    covariances = torch.cat((covariances, second_cov), dim=0)
    labels.extend(["second_surface"] * second_positions.shape[0])

    return GaussianReliabilityScene(
        "contamination_regression", positions, covariances,
        "Clean plane plus all 6 worklog 114 §10 contaminant types, inserted near (not far from) the plane's own Gaussians.",
        tuple(labels),
    )


def make_gaussian_reliability_scene(name: str, seed: int = 0) -> GaussianReliabilityScene:
    if name not in GAUSSIAN_RELIABILITY_SCENE_NAMES:
        raise ValueError(f"Unknown Gaussian reliability scene: {name!r}")

    if name == "plane":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        return GaussianReliabilityScene(name, positions, covariances, "Single flat surfel plane; all Gaussians expected reliable.")

    if name == "two_perpendicular_surfaces":
        floor_positions, floor_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), seed=seed)
        wall_positions, wall_cov = _flat_grid(7, 0.12, normal=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.36), seed=seed + 1)
        positions = torch.cat((floor_positions, wall_positions), dim=0)
        covariances = torch.cat((floor_cov, wall_cov), dim=0)
        labels = ("floor",) * floor_positions.shape[0] + ("wall",) * wall_positions.shape[0]
        return GaussianReliabilityScene(
            name, positions, covariances,
            "Floor (z=0, normal +z) meeting a wall (x=0, normal +x) sharing an edge -- expect a crease, not same_surface.",
            labels,
        )

    if name == "close_parallel_sheets":
        lower_positions, lower_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), seed=seed)
        upper_positions, upper_cov = _flat_grid(7, 0.12, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.15), seed=seed + 1)
        positions = torch.cat((lower_positions, upper_positions), dim=0)
        covariances = torch.cat((lower_cov, upper_cov), dim=0)
        labels = ("lower",) * lower_positions.shape[0] + ("upper",) * upper_positions.shape[0]
        return GaussianReliabilityScene(
            name, positions, covariances,
            "Two parallel flat sheets separated by a small gap along their shared normal -- same orientation, must NOT merge into one same_surface region.",
            labels,
        )

    if name == "smooth_curved_sheet":
        positions, covariances = _curved_sheet(amplitude=0.05, frequency=1.2, count_per_axis=9, spacing=0.12, seed=seed)
        return GaussianReliabilityScene(name, positions, covariances, "Mildly curved sine sheet; adjacent normals differ gradually and should still connect.")

    if name == "isolated_floater":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        floater_position = torch.tensor([[3.0, 3.0, 3.0]])
        floater_scale = torch.tensor([[0.05, 0.05, 0.002]])
        floater_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        floater_cov = covariance_from_scale_rotation(floater_scale, floater_quaternion)
        all_positions = torch.cat((positions, floater_position), dim=0)
        all_covariances = torch.cat((covariances, floater_cov), dim=0)
        labels = ("plane",) * positions.shape[0] + ("floater",)
        return GaussianReliabilityScene(
            name, all_positions, all_covariances,
            "A flat plane plus one Gaussian far from every neighbor -- must not be treated as reliable boundary/interior evidence.",
            labels,
        )

    if name == "isotropic_blob":
        positions, covariances = _flat_grid(9, 0.12, seed=seed)
        count = positions.shape[0]
        isotropic_indices = torch.tensor([count // 2, count // 2 + 1])
        blob_scale = torch.full((2, 3), 0.05)
        blob_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 4).clone()
        covariances = covariances.clone()
        covariances[isotropic_indices] = covariance_from_scale_rotation(blob_scale, blob_quaternion)
        labels = tuple("isotropic" if i in isotropic_indices.tolist() else "plane" for i in range(count))
        return GaussianReliabilityScene(
            name, positions, covariances,
            "A flat plane with two Gaussians replaced by isotropic (spherical) covariance at the same positions -- no reliable normal evidence.",
            labels,
        )

    if name == "oversized_bridge":
        scene = make_gaussian_reliability_scene("two_perpendicular_surfaces", seed=seed)
        bridge_position = torch.tensor([[0.0, 0.0, 0.18]])  # in the gap between floor and wall
        bridge_scale = torch.tensor([[0.4, 0.4, 0.4]])
        bridge_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        bridge_cov = covariance_from_scale_rotation(bridge_scale, bridge_quaternion)
        positions = torch.cat((scene.positions, bridge_position), dim=0)
        covariances = torch.cat((scene.covariances, bridge_cov), dim=0)
        labels = scene.group_labels + ("bridge",)
        return GaussianReliabilityScene(
            name, positions, covariances,
            "Floor+wall (as two_perpendicular_surfaces) plus one huge isotropic-ish Gaussian sitting in the gap -- must not bridge floor to wall as one same_surface region.",
            labels,
        )

    raise AssertionError("unreachable")
