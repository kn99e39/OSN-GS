from __future__ import annotations

"""Boundary-first surface materialization without rectangle fallback."""
from dataclasses import dataclass
from typing import Any
from osn_gs.surface.torch_boundary_support_network import BoundarySupportCurveNetwork
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.utils.torch_ops import require_torch

@dataclass(frozen=True)
class BoundaryConstrainedSurfaceResult:
    state: str
    reason: str | None
    surfaces: tuple[TorchNURBSSurface, ...]
    network_payload: dict[str, Any]
    diagnostics: dict[str, Any]

def _surface_from_grid(grid: Any, *, degree_u: int = 1, degree_v: int = 1) -> TorchNURBSSurface:
    torch=require_torch()
    return TorchNURBSSurface(control_grid=grid.detach().clone(),weights=torch.ones(tuple(grid.shape[:2]),dtype=grid.dtype,device=grid.device),degree_u=degree_u,degree_v=degree_v,uv_support_mask=torch.ones((max(1,int(grid.shape[0])-1),max(1,int(grid.shape[1])-1)),dtype=torch.bool,device=grid.device))

def _closed_cubic_wedge(grid: Any, index: int) -> Any:
    """Cubic circumferential control rows, exact at both support seams."""
    torch=require_torch(); count=int(grid.shape[0]); previous=grid[(index-1)%count]; start=grid[index]; end=grid[(index+1)%count]; following=grid[(index+2)%count]
    return torch.stack((start,start+(end-previous)/6.0,end-(following-start)/6.0,end),dim=0)

def build_boundary_constrained_surface(network: BoundarySupportCurveNetwork) -> BoundaryConstrainedSurfaceResult:
    torch=require_torch(); grid=torch.as_tensor(network.support_curves)
    if grid.ndim!=3 or tuple(grid.shape[2:])!=(3,) or int(grid.shape[0])<2 or int(grid.shape[1])<2: raise ValueError('Support network must contain a (U>=2, V>=2, 3) curve grid.')
    if not bool(torch.isfinite(grid).all()): raise ValueError('Support network contains non-finite geometry.')
    if network.closed:
        if int(grid.shape[0])<3: raise ValueError('Closed support network requires at least three support curves.')
        surfaces=tuple(_surface_from_grid(_closed_cubic_wedge(grid,index),degree_u=3,degree_v=1) for index in range(int(grid.shape[0])))
        return BoundaryConstrainedSurfaceResult('constructed_multi_patch',None,surfaces,network.payload(),{'fallback_used':False,'layout':'closed_cubic_seam_multi_patch','seam_count':int(grid.shape[0]),'support_curve_count':int(grid.shape[0]),'circumferential_degree':3})
    return BoundaryConstrainedSurfaceResult('constructed',None,(_surface_from_grid(grid,degree_u=1,degree_v=1),),network.payload(),{'fallback_used':False,'layout':'open_quadratic_boundary_constrained','boundary_constraint':'clamped_observed_support','support_curve_count':int(grid.shape[0]),'samples_per_curve':int(grid.shape[1])})