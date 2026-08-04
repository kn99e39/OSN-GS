"""Deterministic synthetic Gaussian scenes with analytic surface oracles."""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
from typing import Callable

import torch

from .support_domains import (
    SupportPredicate,
    annulus,
    annulus_elliptical,
    annulus_off_center,
    crescent,
    elongated_rect,
    full_square,
    sample_in_domain,
    triangle,
    u_shape,
)


Oracle = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
HeightFn = Callable[[torch.Tensor], torch.Tensor]
LabelFn = Callable[[torch.Tensor], torch.Tensor]
ParamFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GroundTruthFace:
    """One true, exact analytic face of a genuinely 3D (volumetric) ground-truth
    solid -- e.g. one of a box's 6 planar faces, a cylinder's curved side or a
    flat cap, or a sphere's single closed lat-long patch.

    Unlike the single ``surface_fn: z = f(x, y)`` height-field model above
    (which cannot represent a face oriented away from the shared xy view),
    each face carries its OWN local ``[-1, 1]^2`` parameter domain and its own
    analytic world map + inverse + normal -- genuinely 3D, not a projection.
    """

    face_id: int
    name: str
    to_world: ParamFn  # (N, 2) local uv in [-1, 1]^2 -> (N, 3) world position
    to_local: ParamFn  # (N, 3) world position -> (N, 2) local uv (analytic inverse)
    normal_fn: ParamFn  # (N, 2) local uv -> (N, 3) true analytic outward normal
    support_predicate: SupportPredicate = full_square
    support_name: str = "square"


@dataclass(frozen=True)
class SyntheticGaussianScene:
    """Observed Gaussian centers plus an analytic reference surface.

    The production pipeline creates the actual ``TorchGaussianModel`` from these
    centers and colors.  The oracle is intentionally test-only: it supplies
    ground truth residuals and normals that a real COLMAP scene cannot provide.

    Ground-truth surface knowledge (beyond the pointwise oracle) is exposed for
    the GT-based metrics that separate the three NURBS-construction concerns:

    - ``surface_fn`` maps ``xy`` in the ``[-1, 1]^2`` domain to the true height
      ``z``. Sampling it densely is the exact ground-truth surface used for
      accuracy (Chamfer) and support (coverage / extrapolation) metrics.
    - ``gt_patch_count`` / ``gt_patch_label`` define the ground-truth patch
      topology (e.g. ``crease`` is two planes split at ``x = 0``) used for the
      topology-agreement metric.
    """

    name: str
    points: torch.Tensor
    colors: torch.Tensor
    # Linear scale and wxyz rotation from the observed Gaussian set.
    covariance_scales: torch.Tensor
    covariance_rotations: torch.Tensor
    covariance_normals: torch.Tensor
    oracle: Oracle
    description: str
    surface_fn: HeightFn
    gt_patch_count: int
    gt_patch_label: LabelFn
    support_predicate: SupportPredicate
    support_name: str
    # Multi-sheet ground truth (e.g. two close parallel planes). ``None`` means
    # the single ``surface_fn`` height field describes the whole true surface;
    # otherwise each entry is one sheet's height field and GT samples are the
    # union of all sheets. ``surface_fn`` stays equal to ``sheet_fns[0]``.
    sheet_fns: tuple[HeightFn, ...] | None = None
    # Genuinely 3D (volumetric) ground truth: when set, this scene is a real
    # multi-face solid (box/cylinder/sphere) and ``ground_truth.py``/
    # ``metrics.py`` use THIS face-aware path instead of the single-height-
    # field ``surface_fn``/``support_predicate`` path above, which cannot
    # represent a face oriented away from the shared xy view. ``None`` for
    # every legacy height-field scene -- their existing behavior/tests are
    # untouched.
    faces: tuple[GroundTruthFace, ...] | None = None


def _colors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5, 0.55 + 0.25 * x * y], dim=1).clamp(0.0, 1.0)


def _quaternion_from_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Robust (Shepperd's method) wxyz quaternion from a batch of proper
    rotation matrices ``(N, 3, 3)``.

    The naive single-branch ``qw = sqrt(1 + trace) / 2`` formula this used to
    use is numerically UNSTABLE whenever ``trace`` is close to -1 (a
    180-degree-class rotation): both ``qw`` and the off-diagonal-difference
    numerators for qx/qy/qz collapse toward 0 simultaneously, so the ratio
    silently returns garbage (observed: exactly antipodal input normals like
    ``(0, 0, -1)`` decoded back to the IDENTITY quaternion instead of the
    correct 180-degree rotation). Volumetric solids (box/cylinder) hit this
    exactly, at scale, via their exact axis-aligned face normals -- height-
    field scenes never did, since their normals were continuous small
    perturbations around +z. Fixed by branching on whichever of
    ``trace``/``m00``/``m11``/``m22`` is largest, the standard robust
    extraction algorithm.
    """
    m00, m01, m02 = rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2]
    m10, m11, m12 = rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2]
    m20, m21, m22 = rotation[:, 2, 0], rotation[:, 2, 1], rotation[:, 2, 2]
    trace = m00 + m11 + m22

    case_a = trace > 0
    case_b = (~case_a) & (m00 >= m11) & (m00 >= m22)
    case_c = (~case_a) & (~case_b) & (m11 >= m22)
    case_d = (~case_a) & (~case_b) & (~case_c)

    eps = 1e-12
    s_a = torch.sqrt(torch.clamp(trace + 1.0, min=eps)) * 2.0
    qw_a, qx_a, qy_a, qz_a = 0.25 * s_a, (m21 - m12) / s_a, (m02 - m20) / s_a, (m10 - m01) / s_a

    s_b = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=eps)) * 2.0
    qw_b, qx_b, qy_b, qz_b = (m21 - m12) / s_b, 0.25 * s_b, (m01 + m10) / s_b, (m02 + m20) / s_b

    s_c = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=eps)) * 2.0
    qw_c, qx_c, qy_c, qz_c = (m02 - m20) / s_c, (m01 + m10) / s_c, 0.25 * s_c, (m12 + m21) / s_c

    s_d = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=eps)) * 2.0
    qw_d, qx_d, qy_d, qz_d = (m10 - m01) / s_d, (m02 + m20) / s_d, (m12 + m21) / s_d, 0.25 * s_d

    def _select(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return torch.where(case_a, a, torch.where(case_b, b, torch.where(case_c, c, d)))

    quaternion = torch.stack(
        (_select(qw_a, qw_b, qw_c, qw_d), _select(qx_a, qx_b, qx_c, qx_d), _select(qy_a, qy_b, qy_c, qy_d), _select(qz_a, qz_b, qz_c, qz_d)),
        dim=1,
    )
    return torch.nn.functional.normalize(quaternion, dim=1)


def _tangent_frame_quaternion(normals: torch.Tensor) -> torch.Tensor:
    """Deterministic wxyz frame whose local z axis follows the surface normal."""
    normals = torch.nn.functional.normalize(normals, dim=1)
    reference = torch.zeros_like(normals)
    reference[:, 2] = 1.0
    near_parallel = normals[:, 2].abs() > 0.9
    reference[near_parallel] = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    tangent_u = torch.nn.functional.normalize(torch.linalg.cross(reference, normals, dim=1), dim=1)
    tangent_v = torch.linalg.cross(normals, tangent_u, dim=1)
    rotation = torch.stack((tangent_u, tangent_v, normals), dim=2)
    return _quaternion_from_rotation_matrix(rotation)


def _baseline_like_surface_covariance(
    points: torch.Tensor, normals: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct locally scaled, tangent-aligned covariance from baseline 3DGS statistics.

    ``output/graphdeco_ab_3k`` has sampled major/minor anisotropy median 5.44,
    p25/p75 3.14/10.09, and 90.9% of Gaussians above 2x.  The absolute size is
    tied to local nearest-neighbor spacing so the law remains valid for every
    deterministic synthetic resolution.
    """
    count = int(points.shape[0])
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    spacing = distances.min(dim=1).values.clamp_min(1e-4)
    ratio = torch.exp(torch.log(torch.tensor(5.44)) + 0.92 * torch.randn(count, generator=generator))
    ratio = ratio.clamp(1.5, 32.0).to(dtype=points.dtype, device=points.device)
    tangent_major = (spacing * torch.exp(-0.65 + 0.32 * torch.randn(count, generator=generator))).clamp_min(2e-4)
    tangent_minor = (tangent_major * torch.exp(0.20 * torch.randn(count, generator=generator))).clamp_min(2e-4)
    normal = (torch.maximum(tangent_major, tangent_minor) / ratio).clamp_min(5e-5)
    scales = torch.stack((tangent_major, tangent_minor, normal), dim=1).to(dtype=points.dtype, device=points.device)
    return scales, _tangent_frame_quaternion(normals)


_SURFACE_ALIGNED_RATIO = 12.0  # near p75 (10.09) of the baseline anisotropy stats above.


def _surface_aligned_covariance(
    points: torch.Tensor, normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Idealized, uniformly flat tangent-plane-aligned covariance.

    Same tangent-frame rotation as ``_baseline_like_surface_covariance``
    (local z axis exactly the analytic surface normal), but with the
    per-point anisotropy-ratio and tangent-size log-normal noise removed:
    every Gaussian gets the same fixed flatness ratio, deterministically
    scaled only by local spacing. Every Gaussian's normal-direction extent
    is therefore a uniformly strong, noise-free signal of the true surface
    normal -- the opposite end of a benchmark axis from the noisy,
    baseline-realistic variant above.
    """
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    spacing = distances.min(dim=1).values.clamp_min(1e-4)
    tangent_major = spacing.clamp_min(2e-4)
    tangent_minor = tangent_major
    normal = (tangent_major / _SURFACE_ALIGNED_RATIO).clamp_min(5e-5)
    scales = torch.stack((tangent_major, tangent_minor, normal), dim=1).to(dtype=points.dtype, device=points.device)
    return scales, _tangent_frame_quaternion(normals)


def _make_covariance(
    covariance_mode: str, points: torch.Tensor, normals: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if covariance_mode == "surface_aligned":
        return _surface_aligned_covariance(points, normals)
    if covariance_mode == "baseline_noisy":
        return _baseline_like_surface_covariance(points, normals, generator)
    raise ValueError(f"Unknown covariance_mode: {covariance_mode}")

# --- Analytic height fields z = f(x, y) over the [-1, 1]^2 xy domain. ---

def _plane_height(xy: torch.Tensor) -> torch.Tensor:
    return torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)


def _sine_height(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0], xy[:, 1]
    return 0.20 * torch.sin(2.4 * x) * torch.cos(1.8 * y)


def _crease_height(xy: torch.Tensor) -> torch.Tensor:
    return 0.45 * xy[:, 0].abs()


def _mild_curved_height(xy: torch.Tensor) -> torch.Tensor:
    # Gentle paraboloid: curved everywhere but nowhere near a crease.
    return 0.12 * (xy[:, 0].square() + xy[:, 1].square())


_SHEET_GAP = 0.12  # close_parallel_sheets: z = +/- gap / 2


def _upper_sheet_height(xy: torch.Tensor) -> torch.Tensor:
    return torch.full((xy.shape[0],), _SHEET_GAP * 0.5, dtype=xy.dtype, device=xy.device)


def _lower_sheet_height(xy: torch.Tensor) -> torch.Tensor:
    return torch.full((xy.shape[0],), -_SHEET_GAP * 0.5, dtype=xy.dtype, device=xy.device)


# --- Ground-truth patch labels (topology). ---

def _single_patch_label(xy: torch.Tensor) -> torch.Tensor:
    return torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device)


def _crease_patch_label(xy: torch.Tensor) -> torch.Tensor:
    # Two planes meeting at the ridge x = 0.
    return (xy[:, 0] >= 0.0).long()


def _sheet_patch_label(points: torch.Tensor) -> torch.Tensor:
    # Label by sheet (z sign); expects (N, 3) points, unlike the xy-only labels.
    if points.shape[1] < 3:
        return torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    return (points[:, 2] >= 0.0).long()


def _plane_oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = points[:, 2]
    normals = torch.zeros_like(points)
    normals[:, 2] = 1.0
    return residual, normals


def _sine_oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = points[:, 0], points[:, 1]
    residual = points[:, 2] - 0.20 * torch.sin(2.4 * x) * torch.cos(1.8 * y)
    normals = torch.stack(
        [-0.48 * torch.cos(2.4 * x) * torch.cos(1.8 * y), 0.36 * torch.sin(2.4 * x) * torch.sin(1.8 * y), torch.ones_like(x)],
        dim=1,
    )
    return residual, torch.nn.functional.normalize(normals, dim=1)


def _crease_oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = points[:, 0]
    residual = points[:, 2] - 0.45 * x.abs()
    slope = 0.45 * torch.sign(x)
    normals = torch.stack([-slope, torch.zeros_like(x), torch.ones_like(x)], dim=1)
    return residual, torch.nn.functional.normalize(normals, dim=1)


def _mild_curved_oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = points[:, 0], points[:, 1]
    residual = points[:, 2] - 0.12 * (x.square() + y.square())
    normals = torch.stack([-0.24 * x, -0.24 * y, torch.ones_like(x)], dim=1)
    return residual, torch.nn.functional.normalize(normals, dim=1)


def _parallel_sheets_oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = points[:, 2]
    upper = z - _SHEET_GAP * 0.5
    lower = z + _SHEET_GAP * 0.5
    residual = torch.where(upper.abs() <= lower.abs(), upper, lower)
    normals = torch.zeros_like(points)
    normals[:, 2] = 1.0
    return residual, normals


def _density_gradient_xy(count: int, generator: torch.Generator) -> torch.Tensor:
    """Non-uniform point density: most samples cluster near the origin.

    Every other scene samples ``xy`` uniformly, which never stresses
    density-adaptive voxel subdivision (a coarse cell only splits when its
    weighted density clears a quantile threshold). Real COLMAP point clouds
    are highly non-uniform -- dense in well-textured regions, sparse
    elsewhere -- so this mimics that with a dense central cluster plus a
    sparse uniform background in the same ``[-1, 1]^2`` domain.
    """

    dense_count = max(1, int(round(count * 0.7)))
    sparse_count = max(0, int(count) - dense_count)
    dense = (torch.randn((dense_count, 2), generator=generator) * 0.18).clamp(-1.0, 1.0)
    sparse = torch.rand((sparse_count, 2), generator=generator) * 2.0 - 1.0
    return torch.cat([dense, sparse], dim=0)


def _annulus_density_gradient_xy(
    count: int, generator: torch.Generator, inner: float = 0.32, outer: float = 0.9, power: float = 2.5
) -> torch.Tensor:
    """Points restricted to the same ``annulus`` domain as ``planar_hole``, but
    with a radial density gradient (denser near the inner/hole boundary)
    instead of uniform. Unlike ``planar_hole``, per-O-grid-slice point counts
    are now systematically uneven -- exercises the Phase 4 hardening plan's
    Step 3 concern that gate thresholds must not be fit to a single,
    uniformly-sampled scene.
    """

    theta = torch.rand(count, generator=generator) * (2.0 * torch.pi)
    u = torch.rand(count, generator=generator)
    r = inner + (outer - inner) * u.pow(power)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)


def make_scene(
    name: str, count: int, seed: int = 0, noise_std: float = 0.0, covariance_mode: str = "baseline_noisy"
) -> SyntheticGaussianScene:
    """Create one named synthetic scene (see ``SCENE_NAMES``).

    ``covariance_mode`` selects between ``"baseline_noisy"`` (default: mimics
    real baseline 3DGS anisotropy-ratio statistics, log-normal noise on top of
    tangent-frame alignment) and ``"surface_aligned"`` (idealized, uniformly
    flat tangent-aligned disks, no ratio noise -- see ``_surface_aligned_covariance``).
    """

    if name not in ALL_SCENE_NAMES:
        raise ValueError(f"Unknown synthetic scene: {name}")
    count = max(4, int(count))
    if name == "box":
        scene = _make_box_scene(count, seed, covariance_mode=covariance_mode)
    elif name == "cylinder":
        scene = _make_cylinder_scene(count, seed, covariance_mode=covariance_mode)
    elif name == "sphere":
        scene = _make_sphere_scene(count, seed, covariance_mode=covariance_mode)
    else:
        scene = None
    if scene is not None:
        if noise_std > 0.0:
            generator = torch.Generator(device="cpu").manual_seed(seed + 1)
            noisy_points = scene.points + torch.randn(scene.points.shape, generator=generator) * float(noise_std)
            scene = _dataclass_replace(scene, points=noisy_points)
        return scene
    generator = torch.Generator(device="cpu").manual_seed(seed)
    support_predicate: SupportPredicate = full_square
    support_name = "square"
    if name == "density_gradient":
        xy = _density_gradient_xy(count, generator)
    elif name == "planar_hole_density_gradient":
        support_predicate, support_name = annulus, "annulus"
        xy = _annulus_density_gradient_xy(count, generator)
    elif name in {
        "triangle", "u_shape", "crescent", "planar_hole", "elongated_plane",
        "planar_hole_offcenter", "planar_hole_elliptical", "curved_annulus",
    }:
        support_predicate, support_name = {
            "triangle": (triangle, "triangle"), "u_shape": (u_shape, "u_shape"),
            "crescent": (crescent, "crescent"), "planar_hole": (annulus, "annulus"),
            "elongated_plane": (elongated_rect, "elongated_rect"),
            "planar_hole_offcenter": (annulus_off_center, "annulus_offcenter"),
            "planar_hole_elliptical": (annulus_elliptical, "annulus_elliptical"),
            "curved_annulus": (annulus, "annulus"),
        }[name]
        xy = sample_in_domain(support_predicate, count, generator)
    else:
        xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    x, y = xy[:, 0], xy[:, 1]
    gt_patch_count, gt_patch_label = 1, _single_patch_label
    sheet_fns: tuple[HeightFn, ...] | None = None
    z_override: torch.Tensor | None = None
    if name == "plane":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Legacy flat chart retained only for targeted compatibility tests."
    elif name == "sine":
        surface_fn, oracle = _sine_height, _sine_oracle
        description = "Smooth curved chart: LSQ and curvature fidelity."
    elif name == "density_gradient":
        surface_fn, oracle = _sine_height, _sine_oracle
        description = "Same smooth sheet as 'sine' but with a dense central cluster plus sparse background: stresses density-adaptive voxel subdivision (run with --adaptive-voxel to exercise it)."
    elif name == "triangle":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar triangular support: outer-boundary coverage and precision."
    elif name == "u_shape":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar U-shaped support: concavity and connected-support preservation."
    elif name == "crescent":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar crescent support: curved outer and inner boundaries."
    elif name == "planar_hole":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar annular support: hole preservation and Euler-equivalent topology."
    elif name == "planar_hole_offcenter":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar annular support, hole off-center: annulus O-grid with a non-origin-centered hole."
    elif name == "planar_hole_elliptical":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar annular support, elliptical inner/outer boundary: annulus O-grid on a non-circular ring."
    elif name == "planar_hole_density_gradient":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar annular support with radially non-uniform (inner-biased) point density: uneven per-slice point counts in the annulus O-grid."
    elif name == "curved_annulus":
        surface_fn, oracle = _sine_height, _sine_oracle
        description = "Curved (sine) annular support: annulus O-grid on a non-planar surface, where the true normal legitimately rotates around the ring."
    elif name == "elongated_plane":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Planar thin elongated support: anisotropic extent and aspect-ratio allocation."
    elif name == "mild_curved_sheet":
        surface_fn, oracle = _mild_curved_height, _mild_curved_oracle
        description = "Gently curved paraboloid sheet: curvature fidelity without creases."
    elif name == "close_parallel_sheets":
        surface_fn, oracle = _upper_sheet_height, _parallel_sheets_oracle
        sheet_fns = (_upper_sheet_height, _lower_sheet_height)
        gt_patch_count, gt_patch_label = 2, _sheet_patch_label
        # Alternate sheets by index so both sheets are spatially interleaved.
        sheet_pick = torch.arange(count) % 2 == 0
        z_override = torch.where(sheet_pick, _SHEET_GAP * 0.5, -_SHEET_GAP * 0.5)
        description = "Two close parallel planar sheets: layer separation vs. mid-plane merging."
    else:
        surface_fn, oracle = _crease_height, _crease_oracle
        gt_patch_count, gt_patch_label = 2, _crease_patch_label
        description = "Two planes with a sharp crease: voxel-boundary and multi-patch behavior."
    z = surface_fn(xy) if z_override is None else z_override
    points = torch.stack([x, y, z], dim=1)
    if noise_std > 0.0:
        points = points + torch.randn(points.shape, generator=generator) * float(noise_std)
    _, normals = oracle(points)
    covariance_scales, covariance_rotations = _make_covariance(covariance_mode, points, normals, generator)
    return SyntheticGaussianScene(
        name=name,
        points=points,
        colors=_colors(x, y),
        covariance_scales=covariance_scales,
        covariance_rotations=covariance_rotations,
        covariance_normals=normals,
        oracle=oracle,
        description=description,
        surface_fn=surface_fn,
        gt_patch_count=gt_patch_count,
        gt_patch_label=gt_patch_label,
        support_predicate=support_predicate,
        support_name=support_name,
        sheet_fns=sheet_fns,
    )


# --- Genuinely 3D volumetric solids (box/cylinder/sphere) --------------------
#
# These replace the former single-height-field default benchmark population.
# Every point is sampled directly on a real, multi-face 3D solid boundary; the
# ``oracle`` below is an exact analytic signed-distance function to that
# boundary (not a height-field residual), valid for ANY query point (not just
# points already on the surface) -- this is what ``runner.py`` needs to score
# reconstructed/generated points against ground truth.


def _circle_predicate(xy: torch.Tensor) -> torch.Tensor:
    """Unit-disk predicate in a face's own normalized local domain (radius 1)."""
    return xy.square().sum(dim=1).sqrt() <= 1.0


def _box_oracle(half_extent: tuple[float, float, float]) -> Oracle:
    h = torch.tensor(half_extent, dtype=torch.float32)

    def oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = points.abs() - h
        outside = q.clamp(min=0.0).norm(dim=1)
        inside = q.max(dim=1).values.clamp(max=0.0)
        residual = outside + inside  # signed distance; 0 exactly on the surface
        face_axis = q.argmax(dim=1)
        rows = torch.arange(points.shape[0])
        sign = torch.sign(points[rows, face_axis])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        normal = torch.zeros_like(points)
        normal[rows, face_axis] = sign
        return residual, normal

    return oracle


def _box_patch_label(half_extent: tuple[float, float, float]) -> LabelFn:
    h = torch.tensor(half_extent, dtype=torch.float32)
    # face ordering matches ``_box_faces`` below: 0=+x,1=-x,2=+y,3=-y,4=+z,5=-z
    def label(points: torch.Tensor) -> torch.Tensor:
        q = points[:, :3].abs() - h
        face_axis = q.argmax(dim=1)
        sign_positive = points[torch.arange(points.shape[0]), face_axis] >= 0
        return (face_axis * 2 + (~sign_positive).long()).long()

    return label


def _box_faces(half_extent: tuple[float, float, float]) -> tuple[GroundTruthFace, ...]:
    hx, hy, hz = half_extent

    def _face(axis: int, sign: float, hu: float, hv: float, extent_axes: tuple[int, int]) -> GroundTruthFace:
        au, av = extent_axes

        def to_world(uv: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((uv.shape[0], 3), dtype=uv.dtype, device=uv.device)
            out[:, au] = hu * uv[:, 0]
            out[:, av] = hv * uv[:, 1]
            out[:, axis] = sign * [hx, hy, hz][axis]
            return out

        def to_local(points: torch.Tensor) -> torch.Tensor:
            return torch.stack((points[:, au] / hu, points[:, av] / hv), dim=1)

        def normal_fn(uv: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((uv.shape[0], 3), dtype=uv.dtype, device=uv.device)
            out[:, axis] = sign
            return out

        return to_world, to_local, normal_fn

    faces = []
    specs = (
        (0, 1.0, hy, hz, (1, 2), "face_px"),
        (0, -1.0, hy, hz, (1, 2), "face_nx"),
        (1, 1.0, hx, hz, (0, 2), "face_py"),
        (1, -1.0, hx, hz, (0, 2), "face_ny"),
        (2, 1.0, hx, hy, (0, 1), "face_pz"),
        (2, -1.0, hx, hy, (0, 1), "face_nz"),
    )
    for face_id, (axis, sign, hu, hv, extent_axes, name) in enumerate(specs):
        to_world, to_local, normal_fn = _face(axis, sign, hu, hv, extent_axes)
        faces.append(GroundTruthFace(face_id, name, to_world, to_local, normal_fn, full_square, "square"))
    return tuple(faces)


def _grid_uv(count: int, generator: torch.Generator, jitter: float = 0.02, aspect_ratio: float = 1.0, periodic_u: bool = False) -> torch.Tensor:
    """A near-uniform-density regular grid of ``[-1, 1]^2`` local uv points
    (with a small position jitter), rather than pure random sampling.

    Random uniform sampling has much higher local density VARIANCE (Poisson-
    disk-like clumping and gaps) than a regular grid at the same point count;
    that variance alone was enough to push some regions of a face below the
    candidate-connectivity threshold, silently fragmenting a single flat face
    into dozens of spurious regions -- the same class of density-calibration
    issue found and fixed for other fixtures in worklog 115.

    ``aspect_ratio`` is the face's PHYSICAL world-space (u-extent / v-extent)
    -- a plain square grid on a face whose two local axes cover very different
    physical distances (e.g. a cylinder's circumference vs. its height) is
    anisotropically dense, which reproduces the same fragmentation from a
    different cause. The grid's cell COUNT per axis is scaled so world-space
    spacing stays equal in both directions.
    """
    count_u = max(1, int(round((count * aspect_ratio) ** 0.5)))
    count_v = max(1, int(round((count / aspect_ratio) ** 0.5)))
    if periodic_u:
        # u = -1 and u = +1 both map to the SAME physical angle (0 == 2*pi) --
        # including both would duplicate an entire ring of points, which then
        # show up as tiny isolated same-location pairs disconnected from
        # their true neighbors (a real bug found via worklog 127's cylinder
        # fixture: dozens of stray 2-point regions, all landing exactly on
        # the seam). Drop the duplicated endpoint.
        lin_u = torch.linspace(-1.0, 1.0, count_u + 1)[:-1]
    else:
        lin_u = torch.linspace(-1.0, 1.0, count_u)
    lin_v = torch.linspace(-1.0, 1.0, count_v)
    grid_u, grid_v = torch.meshgrid(lin_u, lin_v, indexing="ij")
    uv = torch.stack((grid_u.reshape(-1), grid_v.reshape(-1)), dim=1)
    uv = uv + jitter * (torch.rand(uv.shape, generator=generator) * 2.0 - 1.0)
    return uv.clamp(-1.0, 1.0)


def _make_box_scene(
    count: int, seed: int, half_extent: tuple[float, float, float] = (1.0, 1.0, 0.7), covariance_mode: str = "baseline_noisy"
) -> SyntheticGaussianScene:
    generator = torch.Generator().manual_seed(seed)
    faces = _box_faces(half_extent)
    per_face = max(1, count // len(faces))
    positions, normals, labels = [], [], []
    for face in faces:
        uv = _grid_uv(per_face, generator)
        positions.append(face.to_world(uv))
        normals.append(face.normal_fn(uv))
        labels.append(torch.full((uv.shape[0],), face.face_id, dtype=torch.long))
    points = torch.cat(positions, dim=0)
    normals = torch.nn.functional.normalize(torch.cat(normals, dim=0), dim=1)
    x, y = points[:, 0], points[:, 1]
    colors = _colors(x.clamp(-1, 1), y.clamp(-1, 1))
    covariance_scales, covariance_rotations = _make_covariance(covariance_mode, points, normals, generator)
    return SyntheticGaussianScene(
        name="box",
        points=points,
        colors=colors,
        covariance_scales=covariance_scales,
        covariance_rotations=covariance_rotations,
        covariance_normals=normals,
        oracle=_box_oracle(half_extent),
        description="Closed 3D box: 6 planar faces, 12 real edges, 8 corners -- a genuine volumetric solid, not a height field.",
        surface_fn=_plane_height,  # unused placeholder (legacy-only field); face-aware path uses `faces`.
        gt_patch_count=len(faces),
        gt_patch_label=_box_patch_label(half_extent),
        support_predicate=full_square,
        support_name="volumetric_box",
        sheet_fns=None,
        faces=faces,
    )


def _cylinder_oracle(radius: float, half_height: float) -> Oracle:
    def oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radial = torch.sqrt(x * x + y * y).clamp_min(1e-12)
        q_side = radial - radius
        q_cap = z.abs() - half_height
        stacked = torch.stack((q_side, q_cap), dim=1)
        outside = stacked.clamp(min=0.0).norm(dim=1)
        inside = stacked.max(dim=1).values.clamp(max=0.0)
        residual = outside + inside
        side_dominant = q_side >= q_cap
        radial_normal = torch.stack((x / radial, y / radial, torch.zeros_like(z)), dim=1)
        cap_normal = torch.zeros_like(points)
        cap_sign = torch.sign(z)
        cap_normal[:, 2] = torch.where(cap_sign == 0, torch.ones_like(cap_sign), cap_sign)
        normal = torch.where(side_dominant.unsqueeze(1), radial_normal, cap_normal)
        return residual, normal

    return oracle


def _cylinder_patch_label(radius: float, half_height: float) -> LabelFn:
    def label(points: torch.Tensor) -> torch.Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        radial = torch.sqrt(x * x + y * y)
        q_side = radial - radius
        q_cap = z.abs() - half_height
        side_dominant = q_side >= q_cap
        return torch.where(side_dominant, torch.zeros_like(z, dtype=torch.long), torch.where(z >= 0, torch.ones_like(z, dtype=torch.long), torch.full_like(z, 2, dtype=torch.long)))

    return label


def _cylinder_faces(radius: float, half_height: float) -> tuple[GroundTruthFace, ...]:
    def side_to_world(uv: torch.Tensor) -> torch.Tensor:
        theta = torch.pi * (uv[:, 0] + 1.0)
        z = half_height * uv[:, 1]
        return torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=1)

    def side_to_local(points: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(points[:, 1], points[:, 0])
        theta = torch.where(theta < 0, theta + 2.0 * torch.pi, theta)
        u = theta / torch.pi - 1.0
        v = points[:, 2] / half_height
        return torch.stack((u, v), dim=1)

    def side_normal_fn(uv: torch.Tensor) -> torch.Tensor:
        theta = torch.pi * (uv[:, 0] + 1.0)
        return torch.stack((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=1)

    def _cap(sign: float, name: str, face_id: int) -> GroundTruthFace:
        def to_world(uv: torch.Tensor) -> torch.Tensor:
            out = torch.stack((radius * uv[:, 0], radius * uv[:, 1], torch.full((uv.shape[0],), sign * half_height, dtype=uv.dtype)), dim=1)
            return out

        def to_local(points: torch.Tensor) -> torch.Tensor:
            return torch.stack((points[:, 0] / radius, points[:, 1] / radius), dim=1)

        def normal_fn(uv: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((uv.shape[0], 3), dtype=uv.dtype, device=uv.device)
            out[:, 2] = sign
            return out

        return GroundTruthFace(face_id, name, to_world, to_local, normal_fn, _circle_predicate, "circle")

    side = GroundTruthFace(0, "side", side_to_world, side_to_local, side_normal_fn, full_square, "square")
    return (side, _cap(1.0, "top_cap", 1), _cap(-1.0, "bottom_cap", 2))


def _make_cylinder_scene(
    count: int, seed: int, radius: float = 0.7, half_height: float = 1.0, covariance_mode: str = "baseline_noisy"
) -> SyntheticGaussianScene:
    generator = torch.Generator().manual_seed(seed)
    faces = _cylinder_faces(radius, half_height)
    per_face = max(1, count // len(faces))
    positions, normals = [], []
    side_aspect_ratio = (2.0 * torch.pi * radius) / (2.0 * half_height)
    for face in faces:
        if face.support_name == "circle":
            # Oversample the enclosing square grid, then keep only the disk --
            # still a regular (jittered) grid, not pure random sampling.
            uv = _grid_uv(int(per_face * 4.0 / torch.pi) + 8, generator)
            uv = uv[_circle_predicate(uv)]
        else:
            # The side's local u axis spans the circumference, v spans the
            # height -- a plain square grid would be anisotropically dense
            # whenever these two physical extents differ (worklog 127).
            uv = _grid_uv(per_face, generator, aspect_ratio=float(side_aspect_ratio), periodic_u=True)
        positions.append(face.to_world(uv))
        normals.append(face.normal_fn(uv))
    points = torch.cat(positions, dim=0)
    normals = torch.nn.functional.normalize(torch.cat(normals, dim=0), dim=1)
    x, y = points[:, 0], points[:, 1]
    colors = _colors(x.clamp(-1, 1), y.clamp(-1, 1))
    covariance_scales, covariance_rotations = _make_covariance(covariance_mode, points, normals, generator)
    return SyntheticGaussianScene(
        name="cylinder",
        points=points,
        colors=colors,
        covariance_scales=covariance_scales,
        covariance_rotations=covariance_rotations,
        covariance_normals=normals,
        oracle=_cylinder_oracle(radius, half_height),
        description="Closed cylinder: circumferentially-curved side wall plus two flat end caps meeting it at a circular crease.",
        surface_fn=_plane_height,
        gt_patch_count=len(faces),
        gt_patch_label=_cylinder_patch_label(radius, half_height),
        support_predicate=full_square,
        support_name="volumetric_cylinder",
        sheet_fns=None,
        faces=faces,
    )


def _sphere_oracle(radius: float) -> Oracle:
    def oracle(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = points.norm(dim=1).clamp_min(1e-12)
        residual = norm - radius
        normal = points / norm.unsqueeze(1)
        return residual, normal

    return oracle


def _sphere_patch_label(points: torch.Tensor) -> torch.Tensor:
    return torch.zeros(points.shape[0], dtype=torch.long, device=points.device)


def _sphere_faces(radius: float) -> tuple[GroundTruthFace, ...]:
    def to_world(uv: torch.Tensor) -> torch.Tensor:
        theta = torch.pi * (uv[:, 0] + 1.0)
        phi = (torch.pi / 2.0) * uv[:, 1]
        return radius * torch.stack(
            (torch.cos(phi) * torch.cos(theta), torch.cos(phi) * torch.sin(theta), torch.sin(phi)), dim=1
        )

    def to_local(points: torch.Tensor) -> torch.Tensor:
        norm = points.norm(dim=1).clamp_min(1e-12)
        phi = torch.asin((points[:, 2] / norm).clamp(-1.0, 1.0))
        theta = torch.atan2(points[:, 1], points[:, 0])
        theta = torch.where(theta < 0, theta + 2.0 * torch.pi, theta)
        u = theta / torch.pi - 1.0
        v = phi / (torch.pi / 2.0)
        return torch.stack((u, v), dim=1)

    def normal_fn(uv: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(to_world(uv), dim=1)

    return (GroundTruthFace(0, "sphere", to_world, to_local, normal_fn, full_square, "square"),)


def _make_sphere_scene(
    count: int, seed: int, radius: float = 0.9, covariance_mode: str = "baseline_noisy"
) -> SyntheticGaussianScene:
    generator = torch.Generator().manual_seed(seed)
    faces = _sphere_faces(radius)
    face = faces[0]
    # Fibonacci-sphere sampling directly in world space (near-uniform density,
    # no pole singularity artifacts), then converted back to local uv only
    # where a downstream consumer needs the chart parameterization.
    indices = torch.arange(count, dtype=torch.float32) + 0.5
    golden_angle = torch.pi * (3.0 - 5.0 ** 0.5)
    z_unit = 1.0 - 2.0 * indices / count
    radial = torch.sqrt(torch.clamp(1.0 - z_unit * z_unit, min=0.0))
    theta = golden_angle * indices
    unit = torch.stack((radial * torch.cos(theta), radial * torch.sin(theta), z_unit), dim=1)
    unit = torch.nn.functional.normalize(unit, dim=1)
    points = radius * unit
    normals = unit.clone()
    x, y = points[:, 0], points[:, 1]
    colors = _colors(x.clamp(-1, 1), y.clamp(-1, 1))
    covariance_scales, covariance_rotations = _make_covariance(covariance_mode, points, normals, generator)
    return SyntheticGaussianScene(
        name="sphere",
        points=points,
        colors=colors,
        covariance_scales=covariance_scales,
        covariance_rotations=covariance_rotations,
        covariance_normals=normals,
        oracle=_sphere_oracle(radius),
        description="Closed sphere: fully curved manifold with no flat region and no boundary anywhere.",
        surface_fn=_plane_height,
        gt_patch_count=1,
        gt_patch_label=_sphere_patch_label,
        support_predicate=full_square,
        support_name="volumetric_sphere",
        sheet_fns=None,
        faces=faces,
    )


# ``SCENE_NAMES`` is the benchmark dataset: genuine 3D volumetric solids.
# The former height-field default population (saddle_shell/spherical_cap/
# folded_roof/wave_annulus) has been discarded entirely (worklog 127). All
# other height-field scenes remain callable via ``LEGACY_SCENE_NAMES`` for the
# older component/decomposition regression suite, which tests a different
# (KDE/raster-based) code path unrelated to `osn-gs benchmark`.
SCENE_NAMES = ("box", "cylinder", "sphere")
LEGACY_SCENE_NAMES = (
    "plane", "sine", "crease", "density_gradient", "triangle", "u_shape", "crescent", "planar_hole",
    "elongated_plane", "mild_curved_sheet", "close_parallel_sheets",
    "planar_hole_offcenter", "planar_hole_elliptical", "planar_hole_density_gradient", "curved_annulus",
)
ALL_SCENE_NAMES = SCENE_NAMES + LEGACY_SCENE_NAMES
