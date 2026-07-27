from types import SimpleNamespace
import unittest, torch
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_occluded_chart_hardening import OccludedChartSafetyResult
from osn_gs.surface.torch_uncertain_gaussian_proposal import *
def surface():
 g=torch.tensor([[[0.,0.,0.],[0.,1.,0.]],[[1.,0.,0.],[1.,1.,0.]]]); return TorchNURBSSurface(g,torch.ones((2,2)),degree_u=1,degree_v=1)
def chart(): return SimpleNamespace(chart_id='c',source_candidate_id='k',supporting_patch_ids=[1,2],supporting_domain_ids=['a','b'],supporting_boundary_ids=['x','y'],state='validated',surface=surface())
def safety(state='eligible',reasons=(),uncertainty={}): return OccludedChartSafetyResult('c','k',{}, {},{'coverage_scope':'central_bridge_only','transition_surface_modeled':False},[],state,list(reasons),dict(uncertainty),{})
class ProposalTest(unittest.TestCase):
 def test_eligible_batch_is_deterministic_and_read_only(self):
  c=chart(); before=c.surface.control_grid.clone(); cfg=UncertainGaussianProposalConfig(target_spacing=.3)
  a=generate_uncertain_gaussian_proposals(c,safety(),config=cfg); b=generate_uncertain_gaussian_proposals(c,safety(),config=cfg)
  self.assertTrue(a.valid_mask.all()); self.assertEqual(a.proposal_batch_id,b.proposal_batch_id); self.assertEqual(a.sample_ids,b.sample_ids); self.assertEqual(a.metadata['append_state'],'not_appended'); torch.testing.assert_close(c.surface.control_grid,before)
  self.assertTrue(((a.uv>0)&(a.uv<1)).all()); self.assertTrue((a.linear_scale>0).all()); self.assertTrue((a.rotation_quaternion[:,0]>=0).all())
 def test_review_and_veto_generate_no_samples(self):
  cfg=UncertainGaussianProposalConfig(target_spacing=.3)
  self.assertEqual(generate_uncertain_gaussian_proposals(chart(),safety('review_required',uncertainty={'x':True}),config=cfg).uv.shape[0],0)
  self.assertEqual(decide_occluded_chart_proposal(chart(),safety('eligible'),[SimpleNamespace(unresolved=True,chart_id_a='c',chart_id_b='z')]).state,'review_required')
  self.assertEqual(decide_occluded_chart_proposal(chart(),safety('ineligible',['full_known_free_contradiction'])).state,'ineligible')
 def test_config_and_spacing_contract(self):
  cfg=UncertainGaussianProposalConfig(target_spacing=.25,max_samples_per_chart=4); b=generate_uncertain_gaussian_proposals(chart(),safety(),config=cfg)
  self.assertLessEqual(len(b.uv),4); self.assertEqual(b.metadata['coverage_provenance']['coverage_scope'],'central_bridge_only')
  domains={'a':SimpleNamespace(local_surface_scale=2.),'b':SimpleNamespace(local_surface_scale=4.)}; self.assertEqual(default_target_spacing(domains,['a','b']),3.)

class EligibilityContractTest(unittest.TestCase):
 def test_all_states_and_provenance_are_distinct(self):
  c=chart(); self.assertEqual(decide_occluded_chart_proposal(c,safety('eligible')).state,'eligible')
  self.assertEqual(decide_occluded_chart_proposal(c,safety('review_required')).state,'review_required')
  self.assertEqual(decide_occluded_chart_proposal(c,safety('ineligible')).state,'ineligible')
  self.assertEqual(decide_occluded_chart_proposal(c,safety('unsupported')).state,'unsupported')
  c.supporting_patch_ids=[]; d=decide_occluded_chart_proposal(c,safety('eligible')); self.assertEqual(d.state,'ineligible'); self.assertIn('proposal_provenance_missing',d.reason_codes)
 def test_known_free_and_ordering(self):
  d=decide_occluded_chart_proposal(chart(),safety('eligible',['z','full_known_free_contradiction','a']))
  self.assertEqual(d.state,'ineligible'); self.assertEqual(d.reason_codes,tuple(sorted(d.reason_codes)))
class SamplingContractTest(unittest.TestCase):
 def test_cell_center_formula_config_identity_and_payload_shapes(self):
  c=chart(); a=generate_uncertain_gaussian_proposals(c,safety(),config=UncertainGaussianProposalConfig(target_spacing=.6)); b=generate_uncertain_gaussian_proposals(c,safety(),config=UncertainGaussianProposalConfig(target_spacing=.3))
  nu,nv=a.metadata['axis_counts']; expected=(a.sample_indices.to(a.uv.dtype)+.5)/torch.tensor([nu,nv],dtype=a.uv.dtype)
  torch.testing.assert_close(a.uv,expected); self.assertNotEqual(a.proposal_batch_id,b.proposal_batch_id)
  self.assertEqual(tuple(a.local_frame.shape[1:]),(3,3)); self.assertEqual(a.rotation_quaternion.shape[1],4); self.assertEqual(a.rejection_reason.dtype,torch.int64)
 def test_spacing_helper_rejects_invalid_and_is_order_independent(self):
  d={'x':SimpleNamespace(local_surface_scale=float('nan')),'a':SimpleNamespace(local_surface_scale=4.),'b':SimpleNamespace(local_surface_scale=2.)}
  self.assertEqual(default_target_spacing(d,['a','b']),default_target_spacing(d,['b','a']))
  with self.assertRaises(ValueError): default_target_spacing(d,['x'])
 def test_frame_scale_and_read_only_extended(self):
  c=chart(); before=(c.state,c.chart_id,c.surface.control_grid.clone(),c.surface.weights.clone()); out=generate_uncertain_gaussian_proposals(c,safety(),config=UncertainGaussianProposalConfig(target_spacing=.3))
  f=out.local_frame; eye=torch.eye(3,dtype=f.dtype).expand_as(torch.bmm(f.transpose(1,2),f)); torch.testing.assert_close(torch.bmm(f.transpose(1,2),f),eye,atol=1e-5,rtol=1e-5)
  self.assertTrue((torch.det(f)>0).all()); self.assertTrue(torch.isfinite(out.linear_scale).all()); self.assertTrue((out.linear_scale>0).all()); self.assertEqual((c.state,c.chart_id),before[:2]); torch.testing.assert_close(c.surface.control_grid,before[2]); torch.testing.assert_close(c.surface.weights,before[3])

if __name__=='__main__': unittest.main()
