from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.surface.base_curves import Curve
from osn_gs.utils.geometry import lerp


@dataclass
class NURBSSurface:
    control_grid: np.ndarray
    degree_u: int = 2
    degree_v: int = 1
    weights: np.ndarray | None = None
    observed_v_max: float = 0.5

    def __post_init__(self) -> None:
        self.control_grid = np.asarray(self.control_grid, dtype=np.float32)
        if self.control_grid.ndim != 3 or self.control_grid.shape[2] != 3:
            raise ValueError("control_grid must have shape (U, V, 3).")
        if self.weights is None:
            self.weights = np.ones(self.control_grid.shape[:2], dtype=np.float32)
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != self.control_grid.shape[:2]:
            raise ValueError("weights must match the first two control grid dimensions.")

    def evaluate(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim == 1:
            uv = uv[None, :]
        u = np.clip(uv[:, 0], 0.0, 1.0)
        v = np.clip(uv[:, 1], 0.0, 1.0)
        u_positions = u * (self.control_grid.shape[0] - 1)
        lo = np.floor(u_positions).astype(np.int32)
        hi = np.clip(lo + 1, 0, self.control_grid.shape[0] - 1)
        t = (u_positions - lo).astype(np.float32)[:, None]
        curve_lo = lerp(self.control_grid[lo, 0], self.control_grid[hi, 0], t)
        curve_hi = lerp(self.control_grid[lo, 1], self.control_grid[hi, 1], t)
        return lerp(curve_lo, curve_hi, v[:, None])

    def occluded_mask(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float32)
        return uv[:, 1] > self.observed_v_max


def build_surface_from_curves(base_curves: list[Curve], occlusion_curves: list[Curve]) -> NURBSSurface:
    if not base_curves or not occlusion_curves:
        raise ValueError("At least one base curve and one occlusion curve are required.")
    count = min(len(base_curves), len(occlusion_curves))
    control_grid = []
    for base, occluded in zip(base_curves[:count], occlusion_curves[:count]):
        control_grid.append([base.control_points.mean(axis=0), occluded.control_points.mean(axis=0)])
    return NURBSSurface(control_grid=np.asarray(control_grid, dtype=np.float32))
