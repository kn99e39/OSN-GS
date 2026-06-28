from __future__ import annotations

import numpy as np

from osn_gs.surface.nurbs_surface import NURBSSurface


def surface_smoothness_loss(surface: NURBSSurface) -> float:
    grid = surface.control_grid
    if grid.shape[0] < 3:
        return 0.0
    second = grid[:-2] - 2.0 * grid[1:-1] + grid[2:]
    return float(np.square(second).mean())
