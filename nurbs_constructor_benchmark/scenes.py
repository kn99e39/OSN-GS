"""Deterministic synthetic Gaussian scenes with analytic surface oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Oracle = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class SyntheticGaussianScene:
    """Observed Gaussian centers plus an analytic reference surface.

    The production pipeline creates the actual ``TorchGaussianModel`` from these
    centers and colors.  The oracle is intentionally test-only: it supplies
    ground truth residuals and normals that a real COLMAP scene cannot provide.
    """

    name: str
    points: torch.Tensor
    colors: torch.Tensor
    oracle: Oracle
    description: str


def _colors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5, 0.55 + 0.25 * x * y], dim=1).clamp(0.0, 1.0)


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


def make_scene(name: str, count: int, seed: int = 0, noise_std: float = 0.0) -> SyntheticGaussianScene:
    """Create one named scene: ``plane``, ``sine``, or ``crease``."""

    if name not in {"plane", "sine", "crease"}:
        raise ValueError(f"Unknown synthetic scene: {name}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    xy = torch.rand((max(4, int(count)), 2), generator=generator) * 2.0 - 1.0
    x, y = xy[:, 0], xy[:, 1]
    if name == "plane":
        z, oracle, description = torch.zeros_like(x), _plane_oracle, "Flat chart: baseline fitting and normal stability."
    elif name == "sine":
        z, oracle, description = 0.20 * torch.sin(2.4 * x) * torch.cos(1.8 * y), _sine_oracle, "Smooth curved chart: LSQ and curvature fidelity."
    else:
        z, oracle, description = 0.45 * x.abs(), _crease_oracle, "Two planes with a sharp crease: voxel-boundary and multi-patch behavior."
    points = torch.stack([x, y, z], dim=1)
    if noise_std > 0.0:
        points = points + torch.randn(points.shape, generator=generator) * float(noise_std)
    return SyntheticGaussianScene(name, points, _colors(x, y), oracle, description)


SCENE_NAMES = ("plane", "sine", "crease")
