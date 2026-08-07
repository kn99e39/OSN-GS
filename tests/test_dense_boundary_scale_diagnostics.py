import torch
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_dense_boundary_scale_diagnostics import diagnose_scale_domain
def test_scales_remain_distinct_and_no_candidate_is_reported():
 r=extract_dense_boundary_support(torch.tensor([[x,y,0.] for x in range(5) for y in range(5)],dtype=torch.float32),torch.tensor([[0.,0.,1.]]*25),list(range(25)))
 d=diagnose_scale_domain(r.candidates,9.)
 assert d['representative_spacing']==9. and d['full_evidence_spacing']>0 and d['boundary_support_candidate_spacing']
