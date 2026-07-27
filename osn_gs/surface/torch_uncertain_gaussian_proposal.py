from __future__ import annotations
"""Phase G proposal-only foundation. Never mutates charts or Gaussian models."""
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib, json, math, torch

ELIGIBLE="eligible"; REVIEW_REQUIRED="review_required"; INELIGIBLE="ineligible"; UNSUPPORTED="unsupported"
REJECT_NONFINITE=1; REJECT_DOMAIN=2; REJECT_FRAME=4; REJECT_SCALE=8; REJECT_DUPLICATE=16

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

@dataclass(frozen=True)
class UncertainGaussianProposalConfig:
    target_spacing: float
    metric_probe_count: int=17
    max_samples_u: int=64
    max_samples_v: int=64
    max_samples_per_chart: int=1024
    footprint_factor: float=.5
    normal_scale_ratio: float=.5
    duplicate_absolute_tolerance: float=1e-8
    duplicate_relative_tolerance: float=.1
    dtype: str="float32"
    device: str="cpu"
    schema_version: int=1
    def payload(self): return self.__dict__.copy()
    def digest(self): return _digest(self.payload())

@dataclass(frozen=True)
class ProposalEligibilityDecision:
    state: str; reason_codes: tuple[str,...]; source_chart_id: str; source_candidate_id: str
    supporting_patch_ids: tuple[int,...]; supporting_domain_ids: tuple[str,...]; supporting_boundary_ids: tuple[str,...]
    safety_digest: str; conflict_ids: tuple[str,...]; coverage_provenance: dict[str,Any]

@dataclass(frozen=True)
class UncertainGaussianProposalBatch:
    proposal_batch_id: str; schema_version: int; metadata: dict[str,Any]
    sample_ids: tuple[str,...]; sample_indices: Any; uv: Any; position: Any; normal: Any; local_frame: Any
    rotation_quaternion: Any; linear_scale: Any; valid_mask: Any; rejection_reason: Any
    appearance_state: str="unset"; opacity_state: str="unset"; append_state: str="not_appended"

def _coverage(safety): return dict(safety.attachment_and_coverage)
def decide_occluded_chart_proposal(chart, safety, conflict_edges: Sequence[Any]=()) -> ProposalEligibilityDecision:
    reasons=list(safety.reasons); conflict_ids=tuple(sorted(set(safety.conflict_edge_ids)))
    unresolved=any(bool(getattr(e,"unresolved",False)) and chart.chart_id in (e.chart_id_a,e.chart_id_b) for e in conflict_edges)
    required=(chart.chart_id,chart.source_candidate_id,chart.supporting_patch_ids,chart.supporting_domain_ids,chart.supporting_boundary_ids)
    if not all(x is not None and x != [] for x in required): reasons.append("proposal_provenance_missing")
    if chart.state in ("unsupported","rejected") or safety.eligibility==UNSUPPORTED: state=UNSUPPORTED
    elif chart.state!="validated" or safety.eligibility==INELIGIBLE or "proposal_provenance_missing" in reasons or "full_known_free_contradiction" in reasons or "visible_surface_penetration" in reasons: state=INELIGIBLE
    elif safety.eligibility==REVIEW_REQUIRED or unresolved or conflict_ids: state=REVIEW_REQUIRED
    else: state=ELIGIBLE
    if unresolved: reasons.append("unresolved_chart_conflict")
    return ProposalEligibilityDecision(state,tuple(sorted(set(reasons))),chart.chart_id,chart.source_candidate_id,tuple(chart.supporting_patch_ids),tuple(chart.supporting_domain_ids),tuple(chart.supporting_boundary_ids),_digest(safety.payload()),conflict_ids,_coverage(safety))

def default_target_spacing(domains_by_id: Mapping[str,Any], domain_ids: Sequence[str]) -> float:
    values=sorted(float(domains_by_id[x].local_surface_scale) for x in domain_ids if x in domains_by_id and math.isfinite(float(domains_by_id[x].local_surface_scale)) and float(domains_by_id[x].local_surface_scale)>0)
    if not values: raise ValueError("no finite positive supporting-domain local_surface_scale")
    mid=len(values)//2; return values[mid] if len(values)%2 else .5*(values[mid-1]+values[mid])

def _axis_lengths(surface, probes, device, dtype):
    t=torch.linspace(0.,1.,probes,device=device,dtype=dtype); uv_u=torch.stack([t,torch.full_like(t,.5)],1); uv_v=torch.stack([torch.full_like(t,.5),t],1)
    a=surface.evaluate(uv_u); b=surface.evaluate(uv_v)
    return float(torch.linalg.vector_norm(a[1:]-a[:-1],dim=1).sum()),float(torch.linalg.vector_norm(b[1:]-b[:-1],dim=1).sum())
def _counts(lu,lv,cfg):
    nu=max(1,min(cfg.max_samples_u,math.ceil(lu/cfg.target_spacing))); nv=max(1,min(cfg.max_samples_v,math.ceil(lv/cfg.target_spacing))); clamped=False
    while nu*nv>cfg.max_samples_per_chart:
        clamped=True
        if nu>=nv and nu>1: nu-=1
        elif nv>1: nv-=1
        else: break
    return nu,nv,clamped

def _duplicate_mask(position, tolerance):
    """Deterministic bounded-neighborhood spatial hash; avoids dense all-pairs."""
    buckets={}; duplicate=torch.zeros((len(position),),dtype=torch.bool,device=position.device)
    cpu=position.detach().cpu()
    for i,p in enumerate(cpu.tolist()):
        key=tuple(math.floor(x/tolerance) for x in p); found=False
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    for j in buckets.get((key[0]+dx,key[1]+dy,key[2]+dz),[]):
                        if float(torch.linalg.vector_norm(cpu[i]-cpu[j])) <= tolerance: found=True; break
                    if found: break
                if found: break
            if found: break
        if found: duplicate[i]=True
        else: buckets.setdefault(key,[]).append(i)
    return duplicate

def _quat_from_frame(frame):
    # columns X,Y,Z; deterministic w>=0 conversion
    m=frame; n=m.shape[0]; q=torch.empty((n,4),dtype=m.dtype,device=m.device)
    trace=m[:,0,0]+m[:,1,1]+m[:,2,2]; mask=trace>0
    s=torch.sqrt(torch.clamp(trace[mask]+1,min=1e-20))*2; q[mask,0]=.25*s; q[mask,1]=(m[mask,2,1]-m[mask,1,2])/s; q[mask,2]=(m[mask,0,2]-m[mask,2,0])/s; q[mask,3]=(m[mask,1,0]-m[mask,0,1])/s
    for idx in torch.nonzero(~mask,as_tuple=False).reshape(-1).tolist():
        r=m[idx]; k=int(torch.argmax(torch.diag(r)).item()); j=(k+1)%3; l=(k+2)%3; s=math.sqrt(max(1e-20,float(1+r[k,k]-r[j,j]-r[l,l])))*2; qq=torch.zeros(4,dtype=m.dtype,device=m.device); qq[k+1]=.25*s; qq[0]=(r[l,j]-r[j,l])/s; qq[j+1]=(r[j,k]+r[k,j])/s; qq[l+1]=(r[l,k]+r[k,l])/s; q[idx]=qq
    q=torch.nn.functional.normalize(q,dim=1); return torch.where(q[:,:1]<0,-q,q)

def generate_uncertain_gaussian_proposals(chart, safety, *, config: UncertainGaussianProposalConfig, conflict_edges: Sequence[Any]=()) -> UncertainGaussianProposalBatch:
    decision=decide_occluded_chart_proposal(chart,safety,conflict_edges); device=torch.device(config.device); dtype=getattr(torch,config.dtype)
    meta={"schema_version":config.schema_version,"source_chart_id":decision.source_chart_id,"source_candidate_id":decision.source_candidate_id,"source_patch_ids":list(decision.supporting_patch_ids),"supporting_domain_ids":list(decision.supporting_domain_ids),"supporting_boundary_ids":list(decision.supporting_boundary_ids),"eligibility":decision.state,"safety_reasons":list(decision.reason_codes),"uncertainty_reasons":sorted(safety.uncertainty),"coverage_provenance":decision.coverage_provenance,"conflict_ids":list(decision.conflict_ids),"sampler_config_digest":config.digest(),"appearance_state":"unset","opacity_state":"unset","append_state":"not_appended"}
    empty=lambda shape,dt=dtype: torch.empty(shape,dtype=dt,device=device)
    if decision.state!=ELIGIBLE:
        z=empty((0,)); return UncertainGaussianProposalBatch(_digest(meta),config.schema_version,meta,(),torch.empty((0,2),dtype=torch.long,device=device),empty((0,2)),empty((0,3)),empty((0,3)),empty((0,3,3)),empty((0,4)),empty((0,3)),torch.empty((0,),dtype=torch.bool,device=device),torch.empty((0,),dtype=torch.int64,device=device))
    lu,lv=_axis_lengths(chart.surface,config.metric_probe_count,device,dtype); nu,nv,clamped=_counts(lu,lv,config); meta["axis_counts"]=[nu,nv]; meta["budget_clamped"]=clamped
    iu,iv=torch.meshgrid(torch.arange(nu,device=device),torch.arange(nv,device=device),indexing="ij"); indices=torch.stack([iu.reshape(-1),iv.reshape(-1)],1); uv=(indices.to(dtype)+.5)/torch.tensor([nu,nv],dtype=dtype,device=device)
    pos,su,sv=chart.surface.evaluate_with_derivatives(uv); x=torch.nn.functional.normalize(su,dim=1); z=torch.nn.functional.normalize(torch.cross(su,sv,dim=1),dim=1); y=torch.nn.functional.normalize(torch.cross(z,x,dim=1),dim=1); frame=torch.stack([x,y,z],2); quat=_quat_from_frame(frame)
    du,dv=1./nu,1./nv; scale_u=torch.linalg.vector_norm(su,dim=1)*du*config.footprint_factor; scale_v=torch.linalg.vector_norm(sv,dim=1)*dv*config.footprint_factor; scale_n=config.normal_scale_ratio*torch.minimum(scale_u,scale_v); scale=torch.stack([scale_u,scale_v,scale_n],1)
    reason=torch.zeros((len(uv),),dtype=torch.int64,device=device); finite=torch.isfinite(torch.cat([uv,pos,z,quat,scale],1)).all(1); reason[~finite]|=REJECT_NONFINITE; reason[((uv<0)|(uv>1)).any(1)]|=REJECT_DOMAIN; reason[(torch.linalg.vector_norm(torch.cross(su,sv,dim=1),dim=1)<=1e-12)]|=REJECT_FRAME; reason[(scale<=0).any(1)]|=REJECT_SCALE
    duplicate=_duplicate_mask(pos,max(config.duplicate_absolute_tolerance,config.duplicate_relative_tolerance*config.target_spacing)); reason[duplicate]|=REJECT_DUPLICATE
    valid=reason==0; ids=tuple(_digest([config.schema_version,chart.chart_id,config.digest(),int(i),int(j)]) for i,j in indices.detach().cpu().tolist())
    return UncertainGaussianProposalBatch(_digest([config.schema_version,chart.chart_id,config.digest()]),config.schema_version,meta,ids,indices,uv,pos,z,frame,quat,scale,valid,reason)
