from __future__ import annotations

import numpy as np

from osn_gs.gaussian.certain_gaussians import CertainGaussianSet


def observed_surface_mask(
    gaussians: CertainGaussianSet,
    min_opacity: float = 0.05,
    min_confidence: float = 0.1,
) -> np.ndarray:
    confidence = gaussians.confidence
    if confidence is None:
        confidence = np.ones(len(gaussians), dtype=np.float32)
    return (gaussians.opacities >= min_opacity) & (confidence >= min_confidence)


def gaussian_centers_as_points(
    gaussians: CertainGaussianSet,
    min_opacity: float = 0.05,
    min_confidence: float = 0.1,
) -> np.ndarray:
    return gaussians.positions[observed_surface_mask(gaussians, min_opacity, min_confidence)]
