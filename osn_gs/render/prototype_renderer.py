from __future__ import annotations

"""Lightweight renderer for the numpy prototype training path."""

import numpy as np

from osn_gs.data.cameras import Camera
from osn_gs.gaussian.certain_gaussians import CertainGaussianSet
from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet


class OSNPrototypeRenderer:
    """Render prototype Gaussian sets as simple mean-color frames."""

    def render(
        self,
        certain_gaussians: CertainGaussianSet,
        uncertain_gaussians: UncertainGaussianSet,
        cameras: list[Camera],
    ) -> np.ndarray:
        all_colors = [certain_gaussians.colors]
        all_opacities = [certain_gaussians.opacities]
        if len(uncertain_gaussians) > 0:
            all_colors.append(uncertain_gaussians.colors)
            all_opacities.append(uncertain_gaussians.opacities)
        colors = np.concatenate(all_colors, axis=0)
        opacities = np.concatenate(all_opacities, axis=0)
        weights = np.maximum(opacities[:, None], 1e-6)
        mean_color = (colors * weights).sum(axis=0) / weights.sum(axis=0)
        frames = []
        for camera in cameras:
            frames.append(np.broadcast_to(mean_color, (camera.height, camera.width, 3)))
        return np.asarray(frames, dtype=np.float32)
