from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.gaussian.certain_gaussians import GaussianPrimitiveSet


@dataclass
class UncertainGaussianSet(GaussianPrimitiveSet):
    surface_uv: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        n = len(self.positions)
        if self.surface_uv is None:
            self.surface_uv = np.zeros((n, 2), dtype=np.float32)
        else:
            self.surface_uv = np.asarray(self.surface_uv, dtype=np.float32)
        if self.cluster_ids is None:
            self.cluster_ids = np.full(n, -1, dtype=np.int32)
        else:
            self.cluster_ids = np.asarray(self.cluster_ids, dtype=np.int32).reshape(-1)
        if self.surface_uv.shape != (n, 2):
            raise ValueError("surface_uv must have shape (N, 2).")
        if self.cluster_ids.shape != (n,):
            raise ValueError("cluster_ids must have shape (N,).")

    @classmethod
    def empty(cls) -> "UncertainGaussianSet":
        return cls(
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros((0,), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            surface_uv=np.zeros((0, 2), dtype=np.float32),
            cluster_ids=np.zeros((0,), dtype=np.int32),
        )

    def promote(self, threshold: float = 0.8) -> np.ndarray:
        if self.confidence is None:
            return np.zeros(len(self), dtype=bool)
        return self.confidence >= threshold
