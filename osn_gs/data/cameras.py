from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.extrinsic = np.asarray(self.extrinsic, dtype=np.float32)
        self.intrinsic = np.asarray(self.intrinsic, dtype=np.float32)
        if self.extrinsic.shape != (4, 4):
            raise ValueError("extrinsic must have shape (4, 4).")
        if self.intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3).")


def identity_camera(width: int = 64, height: int = 64) -> Camera:
    intrinsic = np.array(
        [[width, 0.0, width / 2.0], [0.0, height, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return Camera(extrinsic=np.eye(4, dtype=np.float32), intrinsic=intrinsic, width=width, height=height)
