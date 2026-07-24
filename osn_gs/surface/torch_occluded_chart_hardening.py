from __future__ import annotations
"""Phase F.1 sampled safety checks; never mutates an OccludedChartResult."""
from dataclasses import dataclass, field
from typing import Any, Mapping
import torch
from osn_gs.surface.torch_sampled_surface_geometry import sample_nurbs_triangle_mesh, build_triangle_aabb_pairs, triangle_triangle_intersection

ELIGIBLE="eligible"; REVIEW_REQUIRED="review_required"; INELIGIBLE="ineligible"; UNSUPPORTED="unsupported"
@dataclass(frozen=True)
class OccludedChartHardeningConfig:
 sample_resolution_u:int=16; sample_resolution_v:int=16; world_tolerance:float=1e-8; allowed_contact_tolerance:float=1e-8
@dataclass
class OccludedChartSafetyResult:
 chart_id:str; source_candidate_id:str; self_intersection:dict[str,Any]; visible_surface_penetration:dict[str,Any]; attachment_and_coverage:dict[str,Any]; conflict_edge_ids:list[str]; eligibility:str; reasons:list[str]; uncertainty:dict[str,Any]; provenance:dict[str,Any]
 def payload(self): return {k:(list(v) if isinstance(v,list) else dict(v) if isinstance(v,dict) else v) for k,v in self.__dict__.items()}

def _mesh_meta(mesh, cfg, scale=1.):
 d=mesh.payload(); d.update({"world_tolerance":float(cfg.world_tolerance),"scale_normalized_tolerance":float(cfg.world_tolerance)/max(float(scale),1e-12),"method":"sampled_piecewise_linear_triangle_intersection","continuous_surface_guarantee":False}); return d

def _self(mesh,cfg):
 broad=build_triangle_aabb_pairs(mesh,tolerance=cfg.world_tolerance); hard=near=excluded=narrow=same_cell=coplanar=0; kinds=[]
 for i,j in broad:
  ci,cj=mesh.triangle_cells[i],mesh.triangle_cells[j]
  if bool((ci==cj).all()): same_cell+=1; continue
  if int(torch.abs(ci-cj).max())<=1: excluded+=1; continue
  narrow+=1; hit=triangle_triangle_intersection(mesh.triangle_points(i),mesh.triangle_points(j),tolerance=cfg.world_tolerance); k=hit['kind']
  if k=='coplanar_area_overlap': coplanar+=1
  if hit['intersects'] and k in ('proper_interior_intersection','coplanar_area_overlap'): hard+=1; kinds.append(k)
  elif float(torch.cdist(mesh.triangle_points(i),mesh.triangle_points(j)).min()) <= cfg.world_tolerance: near+=1
 return {**_mesh_meta(mesh,cfg),"checked":True,"hard_intersection_count":hard,"near_contact_count":near,"excluded_same_cell_pair_count":same_cell,"excluded_adjacent_pair_count":excluded,"coplanar_overlap_count":coplanar,"broad_phase_pair_count":len(broad),"narrow_phase_pair_count":narrow,"intersection_kinds":kinds}

def _point_to_polyline_distance(point, polyline):
    """Distance to a sampled source boundary; no implicit UV-boundary shortcut."""
    best=float("inf")
    for a,b in zip(polyline[:-1],polyline[1:]):
        ab=b-a; denom=float((ab*ab).sum())
        q=a if denom <= 1e-24 else a + max(0.,min(1.,float(((point-a)*ab).sum())/denom))*ab
        best=min(best,float(torch.linalg.vector_norm(point-q)))
    return best

def _source_provenance(candidate, chart, domains_by_id, boundaries_by_id):
    """Validate the Phase D/E source chain without reconstructing geometry."""
    if candidate is None or domains_by_id is None or boundaries_by_id is None:
        return False, "candidate_or_source_registry_missing", []
    if (list(getattr(candidate,"supporting_domain_ids",())) != list(chart.supporting_domain_ids)
        or list(getattr(candidate,"supporting_boundary_ids",())) != list(chart.supporting_boundary_ids)
        or list(getattr(candidate,"supporting_patch_ids",())) != list(chart.supporting_patch_ids)):
        return False, "chart_candidate_source_mismatch", []
    bindings=[]
    for did,bid,pid in zip(candidate.supporting_domain_ids,candidate.supporting_boundary_ids,candidate.supporting_patch_ids):
        domain=domains_by_id.get(did); boundary=boundaries_by_id.get(bid)
        if domain is None or boundary is None:
            return False, "source_domain_or_boundary_missing", []
        if domain.source_boundary_id != bid or int(domain.source_patch_id) != int(pid) or boundary.boundary_id != bid or int(boundary.patch_id) != int(pid):
            return False, "source_domain_boundary_provenance_mismatch", []
        bindings.append((domain,boundary,int(pid)))
    return True, "ok", bindings

def _explicit_source_boundary_correspondence(candidate, boundary_id):
    value=getattr(candidate,"provenance",{}).get("source_boundary_correspondence", {})
    return bool(value is True or (isinstance(value,dict) and value.get(boundary_id,False)))

def _is_allowed_contact(hit, pid, candidate, chart, bindings, cfg):
    """Allow only documented source-boundary contact, never a mesh-edge heuristic."""
    if candidate is None or pid not in chart.supporting_patch_ids or not hit.get("points"):
        return False
    for domain,boundary,source_pid in bindings:
        if source_pid != pid:
            continue
        edge_t = (lambda e: e.t_a if domain.domain_id == candidate.supporting_domain_ids[0] else e.t_b)
        selected_t_zero=any(float(domain.t_world[edge_t(e)]) <= cfg.allowed_contact_tolerance for e in candidate.correspondence_edges)
        explicit=_explicit_source_boundary_correspondence(candidate,boundary.boundary_id)
        if not (selected_t_zero or explicit):
            continue
        if all(_point_to_polyline_distance(torch.tensor(point,dtype=torch.float64),boundary.world.detach().cpu().to(torch.float64)) <= cfg.allowed_contact_tolerance for point in hit["points"]):
            return True
    return False

def _penetration(chart_mesh, chart, visible, cfg, *, candidate=None, bindings=()):
    tested=sorted(int(k) for k in visible); hard=allowed=near=0; minimum=float("inf"); broad=narrow=0
    kinds=[]
    for pid in tested:
      vm=sample_nurbs_triangle_mesh(visible[pid],cfg.sample_resolution_u,cfg.sample_resolution_v); pairs=build_triangle_aabb_pairs(chart_mesh,vm,tolerance=cfg.world_tolerance); broad+=len(pairs)
      for i,j in pairs:
       narrow+=1; hit=triangle_triangle_intersection(chart_mesh.triangle_points(i),vm.triangle_points(j),tolerance=cfg.world_tolerance)
       if hit["intersects"]:
        if _is_allowed_contact(hit,pid,candidate,chart,bindings,cfg): allowed+=1
        else: hard+=1; kinds.append(hit["kind"])
       elif float(torch.cdist(chart_mesh.triangle_points(i),vm.triangle_points(j)).min()) <= cfg.world_tolerance: near+=1
      d=float(torch.cdist(chart_mesh.vertices,vm.vertices).min()); minimum=min(minimum,d)
      if d<=cfg.world_tolerance and not pairs: near+=1
    return {"checked":True,"method":"sampled_triangle_intersection","signed_inside_outside_tested":False,"detects_surface_crossing":True,"detects_volumetric_penetration":False,"continuous_surface_guarantee":False,"tested_visible_patch_ids":tested,"hard_penetration_count":hard,"allowed_contact_count":allowed,"near_visible_contact_count":near,"boundary_exclusion_count":allowed,"minimum_sampled_distance":minimum,"normalized_minimum_distance":minimum,"broad_phase_pair_count":broad,"narrow_phase_pair_count":narrow,"intersection_kinds":kinds,**_mesh_meta(chart_mesh,cfg)}

def evaluate_occluded_chart_safety(chart, surfaces_by_patch_id:Mapping[int,Any]|None, *, config:OccludedChartHardeningConfig|None=None, candidate:Any|None=None, domains_by_id:Mapping[str,Any]|None=None, boundaries_by_id:Mapping[str,Any]|None=None)->OccludedChartSafetyResult:
 cfg=config or OccludedChartHardeningConfig(); reasons=[]; uncertainty={}; coverage={"coverage_scope":"central_bridge_only","transition_surface_modeled":False}
 if chart.state in ("rejected","unsupported"): return OccludedChartSafetyResult(chart.chart_id,chart.source_candidate_id,{"checked":False},{"checked":False},coverage,[],UNSUPPORTED,[f"chart_state:{chart.state}"],{}, {"config":cfg.__dict__})
 if chart.state!="validated" or chart.surface is None: return OccludedChartSafetyResult(chart.chart_id,chart.source_candidate_id,{"checked":False},{"checked":False},coverage,[],INELIGIBLE,["chart_not_validated"],{}, {"config":cfg.__dict__})
 mesh=sample_nurbs_triangle_mesh(chart.surface,cfg.sample_resolution_u,cfg.sample_resolution_v); self_result=_self(mesh,cfg)
 if self_result['hard_intersection_count']: reasons.append('sampled_self_intersection')
 if surfaces_by_patch_id is None: pen={"checked":False,"reason":"visible_surface_registry_missing"}; reasons.append('visible_surface_registry_missing')
 else:
  source_valid,source_reason,bindings=_source_provenance(candidate,chart,domains_by_id,boundaries_by_id)
  pen=_penetration(mesh,chart,surfaces_by_patch_id,cfg,candidate=candidate if source_valid else None,bindings=bindings)
 if pen.get('hard_penetration_count',0): reasons.append('visible_surface_penetration')
 if candidate is None or domains_by_id is None or boundaries_by_id is None:
  coverage.update({"valid":False,"reason":"candidate_or_source_registry_missing"}); reasons.append('support_coverage_unavailable')
 else:
  source_valid,source_reason,bindings=_source_provenance(candidate,chart,domains_by_id,boundaries_by_id)
  edges=getattr(candidate,'correspondence_edges',[]); vals=[]; ratios=[]; finite=True
  for e in edges:
   for did, ti in ((candidate.supporting_domain_ids[0], e.t_a), (candidate.supporting_domain_ids[1], e.t_b)):
    if did not in domains_by_id: finite=False; continue
    d=domains_by_id[did]; value=float(d.t_world[ti]); vals.append(value); ratios.append(value/max(float(d.continuation_extent),1e-12)); finite = finite and torch.isfinite(d.world).all().item()
  continuous=all(edges[i].s_a <= edges[i+1].s_a for i in range(max(0,len(edges)-1)))
  support_points=torch.cat([chart.support_samples_a.detach().cpu(),chart.support_samples_b.detach().cpu()]) if getattr(chart,'support_samples_a',None) is not None else None
  boundary_points=torch.cat([b.world.detach().cpu() for _,b,_ in bindings]) if bindings else None
  support_distance=None if support_points is None or boundary_points is None else float(torch.cdist(support_points,boundary_points).min())
  coverage.update({"valid":bool(vals) and finite and continuous and source_valid,"reason":source_reason,"selected_t_min":min(vals) if vals else None,"selected_t_max":max(vals) if vals else None,"selected_t_median":float(torch.tensor(vals).median()) if vals else None,"selected_t_over_extent_median":float(torch.tensor(ratios).median()) if ratios else None,"visible_boundary_to_chart_support_distance":support_distance})
  if not coverage['valid']: reasons.append('support_coverage_failed')
 if chart.evidence_consistency.get('candidate_hard_contradiction',False): reasons.append('full_known_free_contradiction')
 if chart.evidence_consistency.get('conflicting_evidence') or chart.evidence_consistency.get('free_space_contradiction',{}).get('known_free_section_count',0): uncertainty['evidence_requires_review']=True
 for name in ('curvature_uncertainty','area_uncertainty','continuation_extent_uncertainty'):
  if chart.evidence_consistency.get(name): uncertainty[name]=True
 # central_bridge_only is canonical provenance, never an eligibility penalty.
 if pen.get('near_visible_contact_count',0): uncertainty['near_visible_contact']=True
 if self_result.get('near_contact_count',0): uncertainty['near_self_contact']=True
 if reasons: eligibility=INELIGIBLE
 elif uncertainty: eligibility=REVIEW_REQUIRED
 else: eligibility=ELIGIBLE
 return OccludedChartSafetyResult(chart.chart_id,chart.source_candidate_id,self_result,pen,coverage,[],eligibility,reasons,uncertainty,{"config":cfg.__dict__,"mesh":mesh.payload()})