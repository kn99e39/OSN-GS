from __future__ import annotations

from dataclasses import dataclass

from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet


@dataclass
class CurveUpdateDecision:
    should_rebuild_surface: bool
    reason: str


def decide_curve_update(
    residual: float,
    uncertain_gaussians: UncertainGaussianSet,
    residual_threshold: float = 0.1,
    confidence_threshold: float = 0.3,
) -> CurveUpdateDecision:
    if len(uncertain_gaussians) == 0:
        return CurveUpdateDecision(False, "no uncertain gaussians")
    mean_confidence = float(uncertain_gaussians.confidence.mean()) if uncertain_gaussians.confidence is not None else 0.0
    if residual > residual_threshold and mean_confidence < confidence_threshold:
        return CurveUpdateDecision(True, "high residual with low uncertain confidence")
    return CurveUpdateDecision(False, "surface hypothesis remains stable")
