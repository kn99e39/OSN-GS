"""Read-only Worklog 73 diagnostics for dense boundary connectivity."""
from __future__ import annotations
from typing import Any, Sequence
from osn_gs.utils.torch_ops import require_torch

def diagnose_dense_boundary_connectivity(candidates: Sequence[Any], connectivity_scale: Sequence[float]|None=None) -> dict[str, Any]:
    """Worklog 76: ``connectivity_scale`` is an OPTIONAL per-candidate scale
    matching ``torch_region_owned_dense_boundary_support._connect``'s own.
    ``None`` (the default) reproduces the pre-worklog-76 reading exactly. The
    staged attribution order is unchanged."""
    torch=require_torch(); n=len(candidates)
    if not n: return {'half_line_denominator':0,'terminal_outcomes':{},'directional_coverage':{},'stages':{},'reciprocity_loss':0,'spatial_coverage':{'candidate_count':0,'measurement_only':True}}
    p=torch.tensor([x.position for x in candidates]); t=torch.tensor([x.tangent for x in candidates]); z=torch.tensor([x.normal for x in candidates]); scale=float(candidates[0].full_evidence_scale)
    scales=[scale]*n if connectivity_scale is None else [float(s) for s in connectivity_scale]
    chosen={}; outcomes={}; edge_stages={k:set() for k in ('distance_local_scale','reason','normal','tangent','ambiguity','mutuality')}
    for i in range(n):
      local_scale=scales[i]
      for sign in (-1,1):
        pool=[]
        for j in range(n):
          if i==j: continue
          delta=p[j]-p[i]
          if sign*float(delta@t[i])<=0: continue
          if float(delta.norm())<=2.5*local_scale: pool.append((float(delta.norm()),j))
        if not pool: outcomes['no_candidate_within_local_scale']=outcomes.get('no_candidate_within_local_scale',0)+1; continue
        edge_stages['distance_local_scale'].update((i,j) for _,j in pool)
        pool=[x for x in pool if candidates[i].boundary_reason==candidates[x[1]].boundary_reason]
        if not pool: outcomes['reason_incompatible']=outcomes.get('reason_incompatible',0)+1; continue
        edge_stages['reason'].update((i,j) for _,j in pool)
        pool=[x for x in pool if abs(float(z[i]@z[x[1]]))>=.8]
        if not pool: outcomes['normal_incompatible']=outcomes.get('normal_incompatible',0)+1; continue
        edge_stages['normal'].update((i,j) for _,j in pool)
        pool=[x for x in pool if abs(float(t[i]@t[x[1]]))>=.5]
        if not pool: outcomes['tangent_incompatible']=outcomes.get('tangent_incompatible',0)+1; continue
        edge_stages['tangent'].update((i,j) for _,j in pool); pool.sort()
        if len(pool)>1 and abs(pool[1][0]-pool[0][0])<=.1*local_scale: outcomes['ambiguous_competition']=outcomes.get('ambiguous_competition',0)+1; continue
        chosen[(i,sign)]=pool[0][1]; edge_stages['ambiguity'].add((i,pool[0][1]))
    reciprocal=set(); loss=0
    for key,j in chosen.items():
      i,_=key
      if any(v==i for (k,_),v in chosen.items() if k==j): outcomes['valid_reciprocal_neighbor']=outcomes.get('valid_reciprocal_neighbor',0)+1; reciprocal.add(tuple(sorted((i,j))))
      else: outcomes['valid_nonreciprocal_neighbor']=outcomes.get('valid_nonreciprocal_neighbor',0)+1; loss+=1
    edge_stages['mutuality']=reciprocal
    def snap(edges):
      a={i:set() for i in range(n)}
      for x,y in edges:a[x].add(y);a[y].add(x)
      seen=set(); cc=cy=0
      for i in a:
       if i not in seen:
        cc+=1; st=[i]; nodes=[]
        while st:
         q=st.pop()
         if q in seen:continue
         seen.add(q);nodes.append(q);st.extend(a[q]-seen)
        cy+=int(len(nodes)>=3 and all(len(a[q])==2 for q in nodes))
      return {'surviving_directional_proposals':len(edges),'degree_0':sum(len(v)==0 for v in a.values()),'degree_1':sum(len(v)==1 for v in a.values()),'degree_2':sum(len(v)==2 for v in a.values()),'connected_components':cc,'closed_cycles':cy}
    coverage={'both_directions_valid':0,'one_direction_valid':0,'neither_direction_valid':0}
    for i in range(n):coverage[('neither_direction_valid','one_direction_valid','both_directions_valid')[sum((i,s) in chosen for s in (-1,1))]]+=1
    return {'half_line_denominator':2*n,'terminal_outcomes':outcomes,'terminal_percentages':{k:100*v/(2*n) for k,v in outcomes.items()},'directional_coverage':coverage,'stages':{k:snap(v) for k,v in edge_stages.items()},'reciprocity_loss':loss,'spatial_coverage':{'candidate_count':n,'candidate_bbox_extent':(p.max(0).values-p.min(0).values).tolist(),'measurement_only':True}}
