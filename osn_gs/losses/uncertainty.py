from __future__ import annotations

import numpy as np

from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet


def uncertainty_confidence_loss(gaussians: UncertainGaussianSet) -> float:
    if gaussians.confidence is None or len(gaussians) == 0:
        return 0.0
    return float((1.0 - gaussians.confidence).mean())


def residual_weighted_uncertainty_loss(residual: float, gaussians: UncertainGaussianSet) -> float:
    if len(gaussians) == 0:
        return 0.0
    confidence = gaussians.confidence if gaussians.confidence is not None else np.zeros(len(gaussians))
    return float(residual * (1.0 - confidence).mean())
