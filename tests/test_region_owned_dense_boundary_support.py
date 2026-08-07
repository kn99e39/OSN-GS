import torch
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support

def test_dense_square_boundary_is_local_and_has_scale():
    grid=torch.tensor([[x,y,0.] for x in range(5) for y in range(5)],dtype=torch.float32)
    r=extract_dense_boundary_support(grid,torch.tensor([[0.,0.,1.]]*25),list(range(25)),representative_scale=9.)
    assert r.full_evidence_scale < 2 and r.representative_scale == 9.
    assert len(r.candidates) >= 12
    assert 'distance_local_scale' in r.rejection_counts

def test_no_full_evidence_never_invents_boundary():
    r=extract_dense_boundary_support(torch.zeros((0,3)),torch.zeros((0,3)),[])
    assert not r.candidates and r.rejection_counts['insufficient_local_evidence']==0
