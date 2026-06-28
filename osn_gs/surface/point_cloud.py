from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.gaussian.certain_gaussians import CertainGaussianSet
from osn_gs.gaussian.projection import gaussian_centers_as_points
from osn_gs.utils.geometry import as_points


@dataclass
class ObservedPointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = as_points(self.points)
        n = len(self.points)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.float32)
            if self.colors.shape != (n, 3):
                raise ValueError("colors must have shape (N, 3).")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
            if self.confidence.shape != (n,):
                raise ValueError("confidence must have shape (N,).")


def from_certain_gaussians(gaussians: CertainGaussianSet) -> ObservedPointCloud:
    points = gaussian_centers_as_points(gaussians)
    return ObservedPointCloud(points=points)
