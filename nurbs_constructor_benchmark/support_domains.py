"""Analytic 2-D support domains used by synthetic NURBS benchmark scenes."""
from __future__ import annotations
from typing import Callable
import torch
SupportPredicate = Callable[[torch.Tensor], torch.Tensor]
def full_square(xy: torch.Tensor) -> torch.Tensor:
    return torch.ones(xy.shape[0], dtype=torch.bool, device=xy.device)
def triangle(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0], xy[:, 1]
    return (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= x)
def crescent(xy: torch.Tensor) -> torch.Tensor:
    r = xy.square().sum(1).sqrt()
    cutout = torch.tensor([0.28, 0.0], dtype=xy.dtype, device=xy.device)
    return (r <= 0.95) & ((xy - cutout).square().sum(1).sqrt() >= 0.48)
def annulus(xy: torch.Tensor) -> torch.Tensor:
    r = xy.square().sum(1).sqrt()
    return (r <= 0.9) & (r >= 0.32)
def u_shape(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0], xy[:, 1]
    return ((x.abs() >= 0.55) & (y >= -0.9) & (y <= 0.9)) | ((y <= -0.45) & (x.abs() <= 0.9))
def elongated_rect(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0], xy[:, 1]
    return (x.abs() <= 1.0) & (y.abs() <= 0.28)
def mask_on_grid(predicate: SupportPredicate, resolution: int = 128) -> torch.Tensor:
    lin = torch.linspace(-1.0, 1.0, int(resolution))
    x, y = torch.meshgrid(lin, lin, indexing="ij")
    return predicate(torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)).reshape(int(resolution), int(resolution))
def sample_in_domain(predicate: SupportPredicate, count: int, generator: torch.Generator) -> torch.Tensor:
    accepted, remaining = [], max(0, int(count))
    for _ in range(128):
        if remaining == 0: break
        candidates = torch.rand((max(64, remaining * 3), 2), generator=generator) * 2.0 - 1.0
        inside = candidates[predicate(candidates)]
        take = min(remaining, int(inside.shape[0]))
        if take:
            accepted.append(inside[:take])
            remaining -= take
    if remaining: raise RuntimeError("Support-domain sampler could not draw enough in-domain points.")
    return torch.cat(accepted, dim=0)
