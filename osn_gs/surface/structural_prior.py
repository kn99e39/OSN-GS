from __future__ import annotations

import numpy as np

from osn_gs.surface.base_curves import Curve
from osn_gs.utils.geometry import normalize_vectors


def estimate_curve_direction(curve: Curve) -> np.ndarray:
    control = curve.control_points
    if len(control) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return normalize_vectors((control[-1] - control[0])[None, :])[0]


def estimate_surface_offset(curves: list[Curve], scale: float = 0.25) -> np.ndarray:
    if not curves:
        return np.array([0.0, 0.0, scale], dtype=np.float32)
    directions = np.asarray([estimate_curve_direction(curve) for curve in curves], dtype=np.float32)
    mean_direction = normalize_vectors(directions.mean(axis=0, keepdims=True))[0]
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normal = np.cross(mean_direction, reference)
    if np.linalg.norm(normal) < 1e-5:
        normal = reference
    normal = normalize_vectors(normal[None, :])[0]
    return normal * scale


def curve_smoothness(curves: list[Curve]) -> float:
    penalty = 0.0
    count = 0
    for curve in curves:
        points = curve.control_points
        if len(points) < 3:
            continue
        second = points[:-2] - 2.0 * points[1:-1] + points[2:]
        penalty += float(np.square(second).mean())
        count += 1
    return penalty / max(count, 1)
