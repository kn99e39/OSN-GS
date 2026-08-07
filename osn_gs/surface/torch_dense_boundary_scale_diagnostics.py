"""Read-only Worklog 74 scale-domain and cycle-edge diagnostics."""
from __future__ import annotations
from typing import Any, Sequence
from osn_gs.utils.torch_ops import require_torch

def diagnose_scale_domain(candidates: Sequence[Any], representative_spacing: float|None) -> dict[str,Any]:
 torch=require_torch(); n=len(candidates)
 if not n:return {'candidate_count':0}
 p=torch.tensor([c.position for c in candidates]); t=torch.tensor([c.tangent for c in candidates]); z=torch.tensor([c.normal for c in candidates]); s=float(candidates[0].full_evidence_scale)
 d=torch.cdist(p,p); d.fill_diagonal_(float('inf')); candidate_spacing=d.min(1).values
 ratios=[]; distance_edges=set(); normal_edges=set(); tangent_edges=set(); ambiguity_edges=set(); chosen={}; cycles=[]
 for i in range(n):
  for sign in (-1,1):
   direction=[j for j in range(n) if j!=i and sign*float((p[j]-p[i])@t[i])>0]
   within=[j for j in direction if float(d[i,j])<=2.5*s]
   if not within:
    if direction: ratios.append(float(d[i,min(direction,key=lambda j:float(d[i,j]))])/s)
    continue
   distance_edges.update((i,j) for j in within)
   normal=[j for j in within if abs(float(z[i]@z[j]))>=.8]
   normal_edges.update((i,j) for j in normal)
   tangent=[j for j in normal if abs(float(t[i]@t[j]))>=.5]
   tangent_edges.update((i,j) for j in tangent)
   tangent=sorted((float(d[i,j]),j) for j in tangent)
   if len(tangent)>1 and abs(tangent[1][0]-tangent[0][0])<=.1*s: continue
   if tangent: chosen[(i,sign)]=tangent[0][1]; ambiguity_edges.add((i,tangent[0][1]))
 # find distance undirected 2-regular components, trace removed edges
 adj={i:set() for i in range(n)}
 for i,j in distance_edges:adj[i].add(j);adj[j].add(i)
 seen=set()
 for start in range(n):
  if start in seen:continue
  st=[start]; nodes=[]
  while st:
   q=st.pop()
   if q in seen:continue
   seen.add(q);nodes.append(q);st.extend(adj[q]-seen)
  if len(nodes)<3 or not all(len(adj[q])==2 for q in nodes):continue
  for i in nodes:
   for j in adj[i]:
    if i>=j:continue
    na=abs(float(z[i]@z[j])); ta=abs(float(t[i]@t[j])); key=(i,j)
    reason='survives_mutuality' if tuple(sorted((i,j))) in {tuple(sorted((a,b))) for (a,_),b in chosen.items() if any(v==a for (k,_),v in chosen.items() if k==b)} else ('tangent_incompatible' if key not in normal_edges or key not in tangent_edges else 'ambiguity_or_nonreciprocal')
    cycles.append({'source_id':candidates[i].stable_id,'target_id':candidates[j].stable_id,'distance_ratio':float(d[i,j])/s,'normal_alignment':na,'normal_margin':na-.8,'tangent_alignment':ta,'tangent_margin':ta-.5,'removal_reason':reason})
 q=lambda x:{'median':float(torch.quantile(x,.5)),'p75':float(torch.quantile(x,.75)),'p90':float(torch.quantile(x,.9)),'p95':float(torch.quantile(x,.95))} if x.numel() else None
 # Projection only measures angular support; it does not construct a loop.
 centered=p-p.mean(0); _,_,v=torch.pca_lowrank(centered,q=2); uv=centered@v[:,:2]; angles=torch.atan2(uv[:,1],uv[:,0]).remainder(6.283185307).sort().values; gaps=torch.diff(torch.cat((angles,angles[:1]+6.283185307)))
 return {'candidate_count':n,'representative_spacing':representative_spacing,'full_evidence_spacing':s,'boundary_support_candidate_spacing':q(candidate_spacing),'candidate_to_full_spacing_ratio':q(candidate_spacing/s),'no_candidate_directional_nearest_ratio':q(torch.tensor(ratios)),'no_candidate_directional_nearest_ratio_buckets':{'<=2.5':sum(x<=2.5 for x in ratios),'2.5_5':sum(2.5<x<=5 for x in ratios),'5_10':sum(5<x<=10 for x in ratios),'>10':sum(x>10 for x in ratios)},'candidate_angular_largest_gap_degrees':float(gaps.max()*180/3.14159265),'distance_cycle_edge_traces':cycles,'orientation_source':'covariance eigenframe normal_candidate; tangent=cross(normal, local missing-sector outward direction)','measurement_only':True}
