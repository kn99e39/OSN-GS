from __future__ import annotations

import numpy as np

from osn_gs.surface.nurbs_surface import NURBSSurface


def sample_occluded_surface(
    surface: NURBSSurface,
    samples_u: int = 8,
    samples_v: int = 2,
    v_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    v_min = surface.observed_v_max if v_min is None else v_min
    u = np.linspace(0.0, 1.0, max(samples_u, 1), dtype=np.float32)
    v = np.linspace(v_min, 1.0, max(samples_v, 1), dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    return surface.evaluate(uv), uv
