from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianPrimitiveSet:
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.colors = np.asarray(self.colors, dtype=np.float32)
        self.opacities = np.asarray(self.opacities, dtype=np.float32).reshape(-1)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.confidence is None:
            self.confidence = np.ones(len(self.positions), dtype=np.float32)
        else:
            self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self._validate()

    def _validate(self) -> None:
        n = len(self.positions)
        if self.positions.shape != (n, 3):
            raise ValueError("positions must have shape (N, 3).")
        if self.colors.shape != (n, 3):
            raise ValueError("colors must have shape (N, 3).")
        if self.opacities.shape != (n,):
            raise ValueError("opacities must have shape (N,).")
        if self.scales.shape != (n, 3):
            raise ValueError("scales must have shape (N, 3).")
        if self.confidence is None or self.confidence.shape != (n,):
            raise ValueError("confidence must have shape (N,).")

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def clone(self) -> "GaussianPrimitiveSet":
        return GaussianPrimitiveSet(
            positions=self.positions.copy(),
            colors=self.colors.copy(),
            opacities=self.opacities.copy(),
            scales=self.scales.copy(),
            confidence=self.confidence.copy() if self.confidence is not None else None,
        )

    def select(self, mask: np.ndarray) -> "GaussianPrimitiveSet":
        mask = np.asarray(mask, dtype=bool)
        return GaussianPrimitiveSet(
            positions=self.positions[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            scales=self.scales[mask],
            confidence=self.confidence[mask] if self.confidence is not None else None,
        )


class CertainGaussianSet(GaussianPrimitiveSet):
    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        opacity: float = 0.8,
        scale: float = 0.01,
    ) -> "CertainGaussianSet":
        points = np.asarray(points, dtype=np.float32)
        n = len(points)
        if colors is None:
            colors = np.full((n, 3), 0.5, dtype=np.float32)
        return cls(
            positions=points,
            colors=np.asarray(colors, dtype=np.float32),
            opacities=np.full(n, opacity, dtype=np.float32),
            scales=np.full((n, 3), scale, dtype=np.float32),
            confidence=np.ones(n, dtype=np.float32),
        )
