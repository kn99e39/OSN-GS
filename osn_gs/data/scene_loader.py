from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.data.cameras import Camera
from osn_gs.gaussian.certain_gaussians import CertainGaussianSet


@dataclass
class ImageBatch:
    cameras: list[Camera]
    images: np.ndarray


@dataclass
class Scene:
    initial_gaussians: CertainGaussianSet
    cameras: list[Camera]
    images: np.ndarray

    def sample_views(self, count: int = 1) -> ImageBatch:
        count = max(1, min(count, len(self.cameras)))
        return ImageBatch(cameras=self.cameras[:count], images=self.images[:count])
