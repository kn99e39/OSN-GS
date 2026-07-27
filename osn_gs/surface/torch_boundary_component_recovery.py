from __future__ import annotations

"""Conservative non-face smooth-continuation evidence for Boundary-first recovery."""
from dataclasses import dataclass
from typing import Any, Sequence
from osn_gs.surface.torch_surface_components import SurfaceComponent
from osn_gs.surface.torch_voxel_hierarchy import _fit_leaf_plane
from osn_gs.utils.torch_ops import require_torch

@dataclass(frozen=True)
class BoundaryFirstRecoveryEdge:
    component_id_a: int
    component_id_b: int
    minimum_world_distance: float
    local_spacing_a: float
    local_spacing_b: float
    normalized_distance: float
    normal_dot: float
    aabb_contact_dimension: int
    accepted: bool
    reasons: tuple[str,...]
    def payload(self) -> dict[str,Any]: return self.__dict__.copy()

def _median_nearest_spacing(points: Any) -> float:
    torch=require_torch(); n=int(points.shape[0])
    if n<2: return float('inf')
    distances=torch.cdist(points,points); distances.fill_diagonal_(float('inf'))
    return float(distances.min(1).values.median())

def _aabb_contact_dimension(a: Any,b: Any) -> int:
    torch=require_torch(); lower=torch.maximum(a.aabb_min,b.aabb_min); upper=torch.minimum(a.aabb_max,b.aabb_max); extent=upper-lower
    # positive overlap axes plus touching axes; negative means disjoint.
    if bool((extent < -1e-7).any()): return -1
    return int((extent > 1e-7).sum())

def propose_boundary_first_component_recovery(components: Sequence[Any], points: Any, *, max_normalized_distance: float=2.0, min_normal_dot: float=0.9, max_contact_dimension: int=1) -> list[BoundaryFirstRecoveryEdge]:
    """Return diagnostics-only candidates; never mutates/merges components.

    This is specifically for smooth regions split because face adjacency is too
    strict. A candidate requires a close raw-support witness relative to both
    components' local point spacings and normal agreement; close parallel sheets
    must fail the normalized-distance test rather than be globally merged.
    """
    torch=require_torch(); points=torch.as_tensor(points); output=[]
    for left in range(len(components)):
        for right in range(left+1,len(components)):
            a,b=components[left],components[right]; pa=points[a.gaussian_indices]; pb=points[b.gaussian_indices]
            distance=float(torch.cdist(pa,pb).min()); spacing_a=_median_nearest_spacing(pa); spacing_b=_median_nearest_spacing(pb); scale=max(min(spacing_a,spacing_b),1e-12); ratio=distance/scale; dot=float((a.normal*b.normal).sum()); contact=_aabb_contact_dimension(a,b)
            reasons=[]
            if ratio>max_normalized_distance: reasons.append('support_gap_too_large')
            if abs(dot)<min_normal_dot: reasons.append('normal_disagreement')
            if contact<0 or contact>max_contact_dimension: reasons.append('contact_dimension_not_nonface')
            output.append(BoundaryFirstRecoveryEdge(int(a.component_id),int(b.component_id),distance,spacing_a,spacing_b,ratio,dot,contact,not reasons,tuple(reasons)))
    return output
@dataclass(frozen=True)
class RecoveredBoundaryFirstRegion:
    component: SurfaceComponent
    source_component_ids: tuple[int,...]
    accepted_recovery_edges: tuple[BoundaryFirstRecoveryEdge,...]

def materialize_boundary_first_recovery_regions(components: Sequence[Any], points: Any, edges: Sequence[BoundaryFirstRecoveryEdge]) -> list[RecoveredBoundaryFirstRegion]:
    """Build immutable recovered regions from accepted evidence without mutating input components."""
    torch=require_torch(); by_id={int(component.component_id):component for component in components}; parent={key:key for key in by_id}
    def find(value: int) -> int:
        while parent[value]!=value:
            parent[value]=parent[parent[value]]; value=parent[value]
        return value
    for edge in edges:
        if edge.accepted:
            left,right=find(edge.component_id_a),find(edge.component_id_b)
            if left!=right: parent[right]=left
    groups: dict[int,list[int]]={}
    for identifier in sorted(by_id): groups.setdefault(find(identifier),[]).append(identifier)
    output=[]
    for root,identifiers in sorted(groups.items()):
        source=[by_id[identifier] for identifier in identifiers]
        indices=torch.unique(torch.cat([torch.as_tensor(component.gaussian_indices,dtype=torch.long,device=points.device) for component in source])).sort().values
        aabb_min=torch.stack([component.aabb_min for component in source]).min(0).values; aabb_max=torch.stack([component.aabb_max for component in source]).max(0).values
        plane=_fit_leaf_plane(points[indices])
        if plane is None: raise ValueError('Recovered region plane fit failed.')
        reference=torch.as_tensor(source[0].normal,dtype=plane.normal.dtype,device=plane.normal.device); normal=plane.normal if float((plane.normal*reference).sum())>=0 else -plane.normal
        member_leaf_ids=sorted({leaf for component in source for leaf in component.member_leaf_ids}); boundary_leaf_ids=sorted({leaf for component in source for leaf in component.boundary_leaf_ids})
        component=SurfaceComponent(min(identifiers),member_leaf_ids,indices,aabb_min,aabb_max,plane.centroid,normal,plane.tangent_u,plane.tangent_v,boundary_leaf_ids)
        accepted=tuple(edge for edge in edges if edge.accepted and edge.component_id_a in identifiers and edge.component_id_b in identifiers)
        output.append(RecoveredBoundaryFirstRegion(component,tuple(identifiers),accepted))
    return output