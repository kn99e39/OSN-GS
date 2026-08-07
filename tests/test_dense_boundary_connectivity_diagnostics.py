import torch
from osn_gs.surface.torch_region_owned_dense_boundary_support import extract_dense_boundary_support
from osn_gs.surface.torch_dense_boundary_connectivity_diagnostics import diagnose_dense_boundary_connectivity
def test_half_lines_have_exact_terminal_accounting():
 r=extract_dense_boundary_support(torch.tensor([[x,y,0.] for x in range(5) for y in range(5)],dtype=torch.float32),torch.tensor([[0.,0.,1.]]*25),list(range(25)))
 d=diagnose_dense_boundary_connectivity(r.candidates)
 assert sum(d['terminal_outcomes'].values())==d['half_line_denominator']
 assert d['stages']['mutuality']['closed_cycles']>=0
