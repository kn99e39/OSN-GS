"""Deterministic synthetic Gaussian scenes with analytic surface oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


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


# --- Ground-truth patch labels (topology). ---

def _single_patch_label(xy: torch.Tensor) -> torch.Tensor:
    return torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device)


def _crease_patch_label(xy: torch.Tensor) -> torch.Tensor:
    # Two planes meeting at the ridge x = 0.
    return (xy[:, 0] >= 0.0).long()


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


def make_scene(name: str, count: int, seed: int = 0, noise_std: float = 0.0) -> SyntheticGaussianScene:
    """Create one named scene: ``plane``, ``sine``, ``crease``, or ``density_gradient``."""

    if name not in SCENE_NAMES:
        raise ValueError(f"Unknown synthetic scene: {name}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = max(4, int(count))
    if name == "density_gradient":
        xy = _density_gradient_xy(count, generator)
    else:
        xy = torch.rand((count, 2), generator=generator) * 2.0 - 1.0
    x, y = xy[:, 0], xy[:, 1]
    gt_patch_count, gt_patch_label = 1, _single_patch_label
    if name == "plane":
        surface_fn, oracle = _plane_height, _plane_oracle
        description = "Flat chart: baseline fitting and normal stability."
    elif name == "sine":
        surface_fn, oracle = _sine_height, _sine_oracle
        description = "Smooth curved chart: LSQ and curvature fidelity."
    elif name == "density_gradient":
        surface_fn, oracle = _sine_height, _sine_oracle
        description = "Same smooth sheet as 'sine' but with a dense central cluster plus sparse background: stresses density-adaptive voxel subdivision (run with --adaptive-voxel to exercise it)."
    else:
        surface_fn, oracle = _crease_height, _crease_oracle
        gt_patch_count, gt_patch_label = 2, _crease_patch_label
        description = "Two planes with a sharp crease: voxel-boundary and multi-patch behavior."
    z = surface_fn(xy)
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
    )


SCENE_NAMES = ("plane", "sine", "crease", "density_gradient")
