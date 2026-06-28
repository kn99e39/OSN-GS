from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.data.cameras import Camera, identity_camera
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


def make_synthetic_scene(point_count: int = 32, image_size: int = 32) -> Scene:
    x = np.linspace(-1.0, 1.0, point_count, dtype=np.float32)
    y = 0.15 * np.sin(np.pi * x)
    z = np.zeros_like(x)
    points = np.stack([x, y, z], axis=1)
    colors = np.stack([(x + 1.0) * 0.5, 0.5 + 0.25 * y, 1.0 - (x + 1.0) * 0.5], axis=1)
    gaussians = CertainGaussianSet.from_points(points, colors=colors, scale=0.03)
    camera = identity_camera(width=image_size, height=image_size)
    target_color = colors.mean(axis=0)
    images = np.broadcast_to(target_color, (1, image_size, image_size, 3)).astype(np.float32)
    return Scene(initial_gaussians=gaussians, cameras=[camera], images=images)
