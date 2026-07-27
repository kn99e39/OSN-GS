from __future__ import annotations

"""Observed Boundary-first support network with deterministic loop correspondence."""
from dataclasses import dataclass
from typing import Any
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment
from osn_gs.utils.torch_ops import require_torch

@dataclass(frozen=True)
class ObservedBoundaryCurve:
    boundary_id: str
    world: Any
    closed: bool
    source_kind: str
    provenance: dict[str, Any]

@dataclass(frozen=True)
class BoundarySupportCurveNetwork:
    boundary_a_id: str
    boundary_b_id: str
    closed: bool
    parameters: Any
    boundary_a_samples: Any
    boundary_b_samples: Any
    support_curves: Any
    correspondence: dict[str, Any]
    provenance: dict[str, Any]
    def payload(self) -> dict[str, Any]:
        return {"boundary_a_id":self.boundary_a_id,"boundary_b_id":self.boundary_b_id,"closed":bool(self.closed),"curve_count":int(self.parameters.numel()),"samples_per_curve":int(self.support_curves.shape[1]),"correspondence":dict(self.correspondence),"provenance":dict(self.provenance)}

def observed_boundary_curve_from_patch(boundary: PatchBoundarySegment) -> ObservedBoundaryCurve:
    return ObservedBoundaryCurve(boundary.boundary_id,boundary.world,bool(boundary.closed),boundary.source_kind,{"patch_id":int(boundary.patch_id),**dict(boundary.provenance)})

def _component_loop(boundary_result: Any, role: str, loop: Any) -> ObservedBoundaryCurve:
    points=getattr(loop,'ordered_boundary_world_points',None)
    if not points: raise ValueError('Observed component loop lacks a stitched closed contour; raw boundary cells cannot define correspondence.')
    component_id=int(getattr(boundary_result,'component_id'))
    return ObservedBoundaryCurve(f'component:{component_id}:{role}:{int(loop.label)}',points,True,f'component_{role}_loop',{'component_id':component_id,'loop_label':int(loop.label),'loop_area_world':float(loop.area_world),'ordering':'stitched_closed_contour'})

def observed_outer_boundary_curve_from_component(boundary_result: Any) -> ObservedBoundaryCurve:
    outer=list(getattr(boundary_result,'outer_loops',()))
    if len(outer)!=1: raise ValueError('Disk support construction requires exactly one outer loop.')
    return _component_loop(boundary_result,'outer',outer[0])

def observed_boundary_curves_from_annulus_component(boundary_result: Any) -> tuple[ObservedBoundaryCurve,ObservedBoundaryCurve]:
    outer=list(getattr(boundary_result,'outer_loops',())); holes=list(getattr(boundary_result,'hole_loops',()))
    if len(outer)!=1 or len(holes)!=1: raise ValueError('Annulus support construction requires exactly one outer loop and one hole loop.')
    return _component_loop(boundary_result,'outer',outer[0]),_component_loop(boundary_result,'inner',holes[0])

def _as_curve(value: ObservedBoundaryCurve|PatchBoundarySegment) -> ObservedBoundaryCurve:
    if isinstance(value,ObservedBoundaryCurve): return value
    if isinstance(value,PatchBoundarySegment): return observed_boundary_curve_from_patch(value)
    raise TypeError('Boundary support network requires ObservedBoundaryCurve or PatchBoundarySegment.')

def _unique_points(boundary: ObservedBoundaryCurve) -> Any:
    torch=require_torch(); points=torch.as_tensor(boundary.world)
    if points.ndim!=2 or int(points.shape[1])!=3: raise ValueError('Boundary world samples must have shape (N, 3).')
    if boundary.closed and int(points.shape[0])>1 and bool(torch.allclose(points[0],points[-1])): points=points[:-1]
    if int(points.shape[0])<(3 if boundary.closed else 2): raise ValueError('Boundary has too few unique samples for its topology.')
    if not bool(torch.isfinite(points).all()): raise ValueError('Boundary world samples must be finite.')
    return points

def _resample(points: Any,parameters: Any,*,closed: bool) -> Any:
    torch=require_torch(); ends=torch.cat((points[1:],points[:1]),dim=0) if closed else points[1:]; lengths=torch.linalg.vector_norm(ends-points[:int(ends.shape[0])],dim=1)
    if not bool(torch.isfinite(lengths).all()) or not bool((lengths>1e-12).all()): raise ValueError('Boundary contains a zero-length or non-finite segment.')
    cumulative=torch.cat((torch.zeros((1,),dtype=points.dtype,device=points.device),lengths.cumsum(0))); distances=(parameters.remainder(1.) if closed else parameters)*cumulative[-1]; indices=(torch.searchsorted(cumulative,distances,right=True)-1).clamp(0,int(lengths.numel())-1); fractions=((distances-cumulative[indices])/lengths[indices]).unsqueeze(1)
    return points[indices]+fractions*(ends[indices]-points[indices])

def _best_closed_correspondence(a: Any,b: Any) -> tuple[bool,int,float]:
    torch=require_torch(); count=int(a.shape[0]); best=None
    for reverse in (False,True):
        candidate=torch.flip(b,dims=(0,)) if reverse else b
        for shift in range(count):
            score=float((a-torch.roll(candidate,shifts=shift,dims=0)).square().sum(dim=1).mean()); choice=(reverse,shift,score)
            if best is None or score<best[2]-1e-12 or (abs(score-best[2])<=1e-12 and choice[:2]<best[:2]): best=choice
    return best

def build_boundary_support_curve_network(boundary_a: ObservedBoundaryCurve|PatchBoundarySegment,boundary_b: ObservedBoundaryCurve|PatchBoundarySegment,*,curve_count:int,samples_per_curve:int,reverse_boundary_b:bool|None=None,boundary_b_phase:float|None=None) -> BoundarySupportCurveNetwork:
    torch=require_torch(); a_curve,b_curve=_as_curve(boundary_a),_as_curve(boundary_b)
    if a_curve.boundary_id==b_curve.boundary_id: raise ValueError('Boundary support network requires two distinct boundaries.')
    if bool(a_curve.closed)!=bool(b_curve.closed): raise ValueError('Paired boundaries must both be open or both be closed.')
    if curve_count<(3 if a_curve.closed else 2) or samples_per_curve<2: raise ValueError('Support network sample count is too small.')
    a=_unique_points(a_curve); b=_unique_points(b_curve).to(dtype=a.dtype,device=a.device); closed=bool(a_curve.closed); parameters=torch.arange(curve_count,dtype=a.dtype,device=a.device)/float(curve_count) if closed else torch.linspace(0.,1.,curve_count,dtype=a.dtype,device=a.device); a_samples=_resample(a,parameters,closed=closed)
    if closed and (reverse_boundary_b is None or boundary_b_phase is None):
        selected_reverse,shift,score=_best_closed_correspondence(a_samples,_resample(b,parameters,closed=True)); selected_phase=float(shift)/float(curve_count); selection='observed_connector_length_minimization'
    else:
        selected_reverse=bool(reverse_boundary_b); selected_phase=0. if boundary_b_phase is None else float(boundary_b_phase); score=None; selection='explicit'
    b_parameters=((-parameters if closed else 1.-parameters) if selected_reverse else parameters)+selected_phase; b_samples=_resample(b,b_parameters,closed=closed); interpolation=torch.linspace(0.,1.,samples_per_curve,dtype=a.dtype,device=a.device); curves=a_samples[:,None,:]*(1.-interpolation[None,:,None])+b_samples[:,None,:]*interpolation[None,:,None]
    return BoundarySupportCurveNetwork(a_curve.boundary_id,b_curve.boundary_id,closed,parameters.detach(),a_samples.detach(),b_samples.detach(),curves.detach(),{'reverse_boundary_b':selected_reverse,'boundary_b_phase':selected_phase,'parameterization':'world_arclength','selection':selection,'connector_length_score':score},{'boundary_a':dict(a_curve.provenance),'boundary_b':dict(b_curve.provenance),'boundary_a_patch_id':a_curve.provenance.get('patch_id'),'boundary_b_patch_id':b_curve.provenance.get('patch_id'),'boundary_a_source_kind':a_curve.source_kind,'boundary_b_source_kind':b_curve.source_kind,'construction':'boundary_first_support_curve_network'})