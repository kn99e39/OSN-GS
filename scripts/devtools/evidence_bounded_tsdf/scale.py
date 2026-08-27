from __future__ import annotations

"""Worklog 127 -- CANONICAL SPATIAL SCALE (directive section 4).

ONE derived scale, no sweep. For every valid renderer median event

    footprint = stored_median_depth / sqrt(fx * fy)

using that camera's own focal lengths, and

    h  = GLOBAL MEDIAN of the valid positive footprints
    mu = 3 * h                       (ratio fixed a priori)

Nothing in this module may be re-derived after seeing a reconstruction. There
is no percentile alternative, no per-region scale and no resolution search.

`fx`/`fy` are reconstructed from the camera's own field of view exactly the way
the canonical rasterizer does it (`focal = S / (2 * tan(FoV/2))`, the inverse of
graphdeco's `focal2fov`), so this is the renderer's own sampling rate and not a
new calibration.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

# Fixed a priori by the directive. Never swept, never re-selected.
TRUNCATION_RATIO = 3.0


def camera_focal_lengths(camera: Any) -> tuple[float, float]:
    """`fx`, `fy` in pixels, from the camera's own FoV -- the exact inverse of
    graphdeco's `focal2fov`, which is how these cameras were built."""

    width, height = int(camera.image_width), int(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    return fx, fy


def view_footprints(camera: Any, median_depth_flat: torch.Tensor) -> torch.Tensor:
    """World-space pixel footprint of every VALID median event in one view."""

    fx, fy = camera_focal_lengths(camera)
    depth = median_depth_flat.reshape(-1)
    valid = depth > 0
    return depth[valid] / math.sqrt(fx * fy)


@dataclass
class CanonicalScale:
    h: float
    mu: float
    valid_event_count: int
    footprint_percentiles: dict[str, float]
    median_convention: str
    lower_median: float
    upper_median: float

    def as_report(self) -> dict[str, Any]:
        return {
            "valid_median_event_count": self.valid_event_count,
            "footprint_distribution": self.footprint_percentiles,
            "h_canonical_voxel_size": self.h,
            "mu_truncation": self.mu,
            "mu_over_h_FIXED_A_PRIORI": TRUNCATION_RATIO,
            "median_convention": self.median_convention,
            "lower_median": self.lower_median,
            "upper_median": self.upper_median,
            "swept": "NOTHING. h, mu and mu/h are each derived or fixed once and never re-selected.",
        }


def derive_canonical_scale(footprints: Sequence[torch.Tensor] | torch.Tensor) -> CanonicalScale:
    """`footprints` is the per-view output of `view_footprints`, or one tensor."""

    if isinstance(footprints, torch.Tensor):
        values = footprints.reshape(-1)
    else:
        values = torch.cat([f.reshape(-1) for f in footprints])
    values = values[torch.isfinite(values) & (values > 0)]
    count = int(values.numel())
    if count == 0:
        raise ValueError("no valid positive renderer median footprints -- cannot derive h")
    ordered = torch.sort(values).values
    lower = float(ordered[(count - 1) // 2].item())
    upper = float(ordered[count // 2].item())
    h = 0.5 * (lower + upper)

    def pct(fraction: float) -> float:
        position = min(count - 1, max(0, int(math.floor(fraction * (count - 1)))))
        return float(ordered[position].item())

    percentiles = {
        "min": pct(0.0), "p05": pct(0.05), "p25": pct(0.25), "median": h,
        "p75": pct(0.75), "p95": pct(0.95), "max": pct(1.0),
        "p01": pct(0.01), "p99": pct(0.99),
    }
    ratio = ordered / h
    for bound in (2.0, 3.0, 4.0, 5.0, 7.0, 9.0):
        percentiles[f"fraction_footprint_over_{bound:g}h"] = float((ratio > bound).to(torch.float64).mean().item())
    return CanonicalScale(
        h=h, mu=TRUNCATION_RATIO * h, valid_event_count=count,
        footprint_percentiles=percentiles,
        median_convention="mean of the two central order statistics of the ascending footprint order (n is even)",
        lower_median=lower, upper_median=upper,
    )
