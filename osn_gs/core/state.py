from __future__ import annotations

from dataclasses import dataclass

from osn_gs.gaussian.certain_gaussians import CertainGaussianSet
from osn_gs.gaussian.uncertain_gaussians import UncertainGaussianSet
from osn_gs.surface.base_curves import Curve
from osn_gs.surface.nurbs_surface import NURBSSurface


@dataclass
class OSNGSState:
    certain_gaussians: CertainGaussianSet
    uncertain_gaussians: UncertainGaussianSet
    base_curves: list[Curve]
    occlusion_curves: list[Curve]
    nurbs_surface: NURBSSurface | None
    iteration: int = 0
    last_loss: float = 0.0
