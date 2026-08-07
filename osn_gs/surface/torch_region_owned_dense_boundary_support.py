"""Dense, region-owned observed boundary support (Worklog 72).

This is deliberately independent of sparse representative half-edge loops.
It finds local missing-support sectors in the full observed cloud, then joins
only mutually selected +/- tangent neighbours; it never builds a radius graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_region_owned_full_evidence_boundary_topology import evaluate_closed_loop_geometry
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8

@dataclass(frozen=True)
class DenseBoundarySupportCandidate:
    stable_id: Any; position: tuple[float,float,float]; normal: tuple[float,float,float]
    tangent: tuple[float,float,float]; boundary_reason: str; full_evidence_scale: float

@dataclass(frozen=True)
class DenseBoundaryAdjacencyRejection:
    source_id: Any; target_id: Any; reason: str

@dataclass(frozen=True)
class DenseBoundaryComponent:
    stable_ids: tuple[Any,...]; closed: bool; status: str; geometry: Any | None

@dataclass(frozen=True)
class DenseBoundarySupportResult:
    candidates: tuple[DenseBoundarySupportCandidate,...]; components: tuple[DenseBoundaryComponent,...]
    full_evidence_scale: float; representative_scale: float | None
    rejection_counts: dict[str,int]

def estimate_full_evidence_sampling_scale(points: Any) -> float:
    torch=require_torch(); n=int(points.shape[0])
    if n < 2: return 0.0
    d=torch.cdist(points, points); d.fill_diagonal_(float('inf'))
    return float(d.min(dim=1).values.median())

def extract_dense_boundary_support(points: Any, normals: Any, stable_ids: Sequence[Any], *, representative_scale: float|None=None, neighbors: int=12, missing_sector_radians: float=math.pi, boundary_support_spacing_mode: str|None=None) -> DenseBoundarySupportResult:
    """Classify boundary support from local full-cloud angular evidence only.

    A candidate is admitted only when its kNN tangent-plane directions contain
    an observed empty sector.  The reason is local `observed_support_termination`,
    not a reason inherited from a nearby sparse seed.

    Worklog 76: ``boundary_support_spacing_mode`` selects the SCALE DOMAIN of
    the connectivity certificate (see ``torch_boundary_support_spacing``).
    ``None`` keeps the pre-worklog-76 behaviour exactly (full-evidence
    spacing). Candidate ADMISSION above is never affected -- the mode only
    changes which spacing the already-admitted candidates are connected in.
    """
    torch=require_torch(); n=int(points.shape[0]); scale=estimate_full_evidence_sampling_scale(points)
    if n < 4 or scale <= 0: return DenseBoundarySupportResult((),(),scale,representative_scale,{"insufficient_local_evidence":n})
    d=torch.cdist(points,points); d.fill_diagonal_(float('inf')); k=min(neighbors,n-1)
    near=d.topk(k,largest=False).indices; out=[]
    for i in range(n):
        normal=normals[i]/normals[i].norm().clamp_min(_EPS); ref=points[near[i,0]]-points[i]; ref=ref-normal*(ref@normal)
        if float(ref.norm()) <= _EPS: continue
        ref=ref/ref.norm(); axis=torch.linalg.cross(normal,ref); axis=axis/axis.norm().clamp_min(_EPS)
        delta=points[near[i]]-points[i]; delta=delta-normal[None,:]*(delta@normal)[:,None]
        angles=torch.atan2(delta@axis,delta@ref).remainder(2*math.pi).sort().values
        gaps=torch.diff(torch.cat((angles,angles[:1]+2*math.pi))); gap,ix=gaps.max(dim=0)
        if float(gap) < missing_sector_radians: continue
        outward=torch.cos(angles[ix]+gap/2)*ref+torch.sin(angles[ix]+gap/2)*axis
        tangent=torch.linalg.cross(normal,outward); tangent=tangent/tangent.norm().clamp_min(_EPS)
        out.append(DenseBoundarySupportCandidate(stable_ids[i],tuple(float(x) for x in points[i]),tuple(float(x) for x in normal),tuple(float(x) for x in tangent),'observed_support_termination',scale))
    candidates=tuple(out)
    connectivity_scale=None
    if boundary_support_spacing_mode is not None and candidates:
        from osn_gs.surface.torch_boundary_support_spacing import resolve_boundary_support_spacing
        resolved=resolve_boundary_support_spacing(
            boundary_support_spacing_mode,
            torch.tensor([c.position for c in candidates],dtype=points.dtype,device=points.device),
            full_evidence_spacing=scale, representative_spacing=representative_scale,
        )
        connectivity_scale=resolved.per_candidate_scale
    return _connect(candidates, representative_scale, connectivity_scale)

def _connect(candidates: tuple[DenseBoundarySupportCandidate,...], representative_scale: float|None, connectivity_scale: Sequence[float]|None=None) -> DenseBoundarySupportResult:
    """Worklog 76: ``connectivity_scale`` is an OPTIONAL per-candidate scale for
    the distance gate and the ambiguity tolerance. ``None`` (the default)
    reproduces the pre-worklog-76 behaviour exactly -- every candidate uses the
    region's ``full_evidence_scale``. The certificate itself is unchanged: same
    stage order, same 2.5x distance multiplier, same 0.1x ambiguity tolerance,
    same reason/tangent/normal predicates, same mutuality requirement. Only the
    SCALE DOMAIN those multipliers are applied to can differ (see
    ``torch_boundary_support_spacing``)."""
    torch=require_torch(); n=len(candidates)
    if not n: return DenseBoundarySupportResult((),(),0.0,representative_scale,{"no_local_termination_support":1})
    p=torch.tensor([x.position for x in candidates]); t=torch.tensor([x.tangent for x in candidates]); z=torch.tensor([x.normal for x in candidates]); scale=candidates[0].full_evidence_scale
    scales=[float(scale)]*n if connectivity_scale is None else [float(s) for s in connectivity_scale]
    chosen={}; rejected={}
    for i in range(n):
        local_scale=scales[i]
        for sign in (-1,1):
            valid=[]
            for j in range(n):
                if i==j: continue
                delta=p[j]-p[i]; dist=float(delta.norm())
                if dist > 2.5*local_scale: rejected['distance_local_scale']=rejected.get('distance_local_scale',0)+1; continue
                if candidates[i].boundary_reason!=candidates[j].boundary_reason: rejected['reason_incompatibility']=rejected.get('reason_incompatibility',0)+1; continue
                if abs(float(t[i]@t[j]))<.5: rejected['tangent_mismatch']=rejected.get('tangent_mismatch',0)+1; continue
                if abs(float(z[i]@z[j]))<.8: rejected['normal_mismatch']=rejected.get('normal_mismatch',0)+1; continue
                if sign*float(delta@t[i]) <= 0: continue
                valid.append((dist,j))
            valid.sort()
            if len(valid)>1 and abs(valid[1][0]-valid[0][0]) <= .1*local_scale:
                rejected['ambiguity']=rejected.get('ambiguity',0)+1; continue
            if valid: chosen[(i,sign)]=valid[0][1]
    adj={i:set() for i in range(n)}
    for (i,s),j in chosen.items():
        if any(v==i for (k,_),v in chosen.items() if k==j): adj[i].add(j); adj[j].add(i)
    comps=[]; seen=set()
    for start in range(n):
        if start in seen: continue
        stack=[start]; members=[]
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); members.append(x); stack.extend(adj[x]-seen)
        deg=[len(adj[x]) for x in members]; closed=len(members)>=3 and all(v==2 for v in deg); status='closed_loop' if closed else ('branch_detected' if any(v>2 for v in deg) else 'open_or_ambiguous')
        geometry=None
        if closed:
            order=[members[0]]; prev=None
            while True:
                nxt=next(v for v in adj[order[-1]] if v!=prev)
                if nxt==order[0]: break
                order.append(nxt); prev=order[-2]
            geometry=evaluate_closed_loop_geometry([candidates[x].position for x in order])
        comps.append(DenseBoundaryComponent(tuple(candidates[x].stable_id for x in members),closed,status,geometry))
    return DenseBoundarySupportResult(candidates,tuple(comps),scale,representative_scale,rejected)
