from __future__ import annotations

from osn_gs.surface.base_curves import Curve
from osn_gs.surface.structural_prior import estimate_surface_offset


def predict_occlusion_curves(base_curves: list[Curve], offset_scale: float = 0.25) -> list[Curve]:
    offset = estimate_surface_offset(base_curves, scale=offset_scale)
    return [
        Curve(
            control_points=curve.control_points + offset,
            confidence=curve.confidence * 0.5,
            observed=False,
        )
        for curve in base_curves
    ]
