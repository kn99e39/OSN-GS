"""Deterministic synthetic Gaussian scenes with analytic surface oracles."""

from __future__ import annotations

from dataclasses import dataclass
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


def _colors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5, 0.55 + 0.25 * x * y], dim=1).clamp(0.0, 1.0)


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


def make_scene(name: str, count: int, seed: int = 0, noise_std: float = 0.0) -> SyntheticGaussianScene:
    """Create one named synthetic scene (see ``SCENE_NAMES``)."""

    if name not in SCENE_NAMES:
        raise ValueError(f"Unknown synthetic scene: {name}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = max(4, int(count))
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
        description = "Flat chart: baseline fitting and normal stability."
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
    return SyntheticGaussianScene(
        name=name,
        points=points,
        colors=_colors(x, y),
        oracle=oracle,
        description=description,
        surface_fn=surface_fn,
        gt_patch_count=gt_patch_count,
        gt_patch_label=gt_patch_label,
        support_predicate=support_predicate,
        support_name=support_name,
        sheet_fns=sheet_fns,
    )


SCENE_NAMES = (
    "plane", "sine", "crease", "density_gradient", "triangle", "u_shape", "crescent", "planar_hole",
    "elongated_plane", "mild_curved_sheet", "close_parallel_sheets",
    "planar_hole_offcenter", "planar_hole_elliptical", "planar_hole_density_gradient", "curved_annulus",
)
