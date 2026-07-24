from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import hashlib, torch
from osn_gs.surface.torch_sampled_surface_geometry import sample_nurbs_triangle_mesh,build_triangle_aabb_pairs,triangle_triangle_intersection
@dataclass
class OccludedChartConflictEdge:
 chart_id_a:str; chart_id_b:str; reasons:list[str]; shared_candidate_ids:list[str]; shared_domain_ids:list[str]; shared_boundary_ids:list[str]; aabb_metrics:dict[str,Any]; distance_metrics:dict[str,Any]; intersection_metrics:dict[str,Any]; near_duplicate_metrics:dict[str,Any]; unresolved:bool=True; provenance:dict[str,Any]=field(default_factory=dict)
 def __post_init__(self):
  if not isinstance(self.unresolved,bool): raise ValueError("unresolved must be bool")
 @property
 def conflict_id(self): return 'chart-conflict:'+hashlib.sha256((self.chart_id_a+'|'+self.chart_id_b).encode()).hexdigest()[:16]
 def payload(self): return {**self.__dict__,"conflict_id":self.conflict_id}
def build_occluded_chart_conflicts(charts:Sequence[Any], *, sample_resolution_u:int=16,sample_resolution_v:int=16,world_tolerance:float=1e-8):
 items=sorted([c for c in charts if c.surface is not None],key=lambda c:c.chart_id); meshes={c.chart_id:sample_nurbs_triangle_mesh(c.surface,sample_resolution_u,sample_resolution_v) for c in items}; out=[]
 for i,a in enumerate(items):
  for b in items[i+1:]:
   ma,mb=meshes[a.chart_id],meshes[b.chart_id]; pairs=build_triangle_aabb_pairs(ma,mb,tolerance=world_tolerance); crossing=0; overlap=0
   for x,y in pairs:
    h=triangle_triangle_intersection(ma.triangle_points(x),mb.triangle_points(y),tolerance=world_tolerance); crossing+=h['kind']=='proper_interior_intersection'; overlap+=h['kind']=='coplanar_area_overlap'
   dist=float(torch.cdist(ma.vertices,mb.vertices).min());
   amin,amax=ma.vertices.min(0).values,ma.vertices.max(0).values; bmin,bmax=mb.vertices.min(0).values,mb.vertices.max(0).values
   aabb_overlap_min=torch.maximum(amin,bmin); aabb_overlap_max=torch.minimum(amax,bmax)
   aabb_overlap_extent=(aabb_overlap_max-aabb_overlap_min).clamp_min(0.)
   a_area=sum(float(torch.linalg.vector_norm(torch.cross(t[1]-t[0],t[2]-t[0],dim=0))/2.) for t in (ma.triangle_points(k) for k in range(len(ma.triangles))))
   b_area=sum(float(torch.linalg.vector_norm(torch.cross(t[1]-t[0],t[2]-t[0],dim=0))/2.) for t in (mb.triangle_points(k) for k in range(len(mb.triangles))))
   normals_a=torch.stack([torch.nn.functional.normalize(torch.cross(t[1]-t[0],t[2]-t[0],dim=0),dim=0) for t in (ma.triangle_points(k) for k in range(len(ma.triangles)))])
   normals_b=torch.stack([torch.nn.functional.normalize(torch.cross(t[1]-t[0],t[2]-t[0],dim=0),dim=0) for t in (mb.triangle_points(k) for k in range(len(mb.triangles)))])
   normal_agreement=float(torch.abs(normals_a.mean(0).dot(normals_b.mean(0))))
   shared_domains=sorted(set(a.supporting_domain_ids)&set(b.supporting_domain_ids)); shared_boundaries=sorted(set(a.supporting_boundary_ids)&set(b.supporting_boundary_ids)); reasons=[]
   if crossing:reasons.append('sampled_triangle_crossing')
   if overlap:reasons.append('sampled_coplanar_overlap')
   if shared_domains or shared_boundaries:reasons.append('same_source_competing_chart')
   if dist<=world_tolerance and not (crossing or overlap):reasons.append('near_duplicate_chart')
   if reasons:
    out.append(OccludedChartConflictEdge(
     chart_id_a=a.chart_id, chart_id_b=b.chart_id, reasons=reasons,
     shared_candidate_ids=sorted(set([a.source_candidate_id]) & set([b.source_candidate_id])),
     shared_domain_ids=shared_domains, shared_boundary_ids=shared_boundaries,
     aabb_metrics={"aabb_min_a":amin.tolist(),"aabb_max_a":amax.tolist(),"aabb_min_b":bmin.tolist(),"aabb_max_b":bmax.tolist(),"overlap_min":aabb_overlap_min.tolist(),"overlap_max":aabb_overlap_max.tolist(),"overlap_extent":aabb_overlap_extent.tolist(),"broad_phase_pair_count":len(pairs)},
     distance_metrics={"minimum_sampled_distance":dist,"normalized_minimum_distance":dist},
     intersection_metrics={"intersection_count":int(crossing+overlap),"crossing_count":int(crossing),"coplanar_overlap_count":int(overlap)},
     near_duplicate_metrics={"symmetric_sampled_distance":dist,"normal_agreement":normal_agreement,"area_ratio":min(a_area,b_area)/max(a_area,b_area,1e-12),"sample_coverage_ratio":min(len(pairs),len(ma.triangles),len(mb.triangles))/max(len(ma.triangles),len(mb.triangles),1)},
     unresolved=True,
     provenance={"method":"sampled_piecewise_linear_triangle_intersection","continuous_surface_guarantee":False},
    ))
 return out
def attach_conflict_edges(safety_results, edges):
    """Annotate mutable safety results; unresolved conflict blocks Phase G eligibility."""
    by_chart={r.chart_id:r for r in safety_results}
    for edge in edges:
        for chart_id in (edge.chart_id_a, edge.chart_id_b):
            result=by_chart.get(chart_id)
            if result is None: continue
            result.conflict_edge_ids.append(edge.conflict_id)
            if edge.unresolved:
                if "unresolved_chart_conflict" not in result.reasons: result.reasons.append("unresolved_chart_conflict")
                result.eligibility="ineligible"
    return safety_results