from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.surface.point_cloud import ObservedPointCloud
from osn_gs.utils.geometry import principal_axis


@dataclass
class Curve:
    control_points: np.ndarray
    confidence: float = 1.0
    observed: bool = True

    def __post_init__(self) -> None:
        self.control_points = np.asarray(self.control_points, dtype=np.float32)
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 3:
            raise ValueError("control_points must have shape (N, 3).")


def fit_base_curves(point_cloud: ObservedPointCloud, curve_count: int = 4) -> list[Curve]:
    points = point_cloud.points
    if len(points) == 0:
        return []
    axis = principal_axis(points)
    ordering = np.argsort(points @ axis)
    sorted_points = points[ordering]
    curve_count = max(1, min(curve_count, len(sorted_points)))
    chunks = np.array_split(sorted_points, curve_count)
    curves: list[Curve] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        if len(chunk) == 1:
            control_points = np.repeat(chunk, 2, axis=0)
        else:
            control_points = np.vstack([chunk[0], chunk.mean(axis=0), chunk[-1]])
        curves.append(Curve(control_points=control_points, confidence=1.0, observed=True))
    return curves
