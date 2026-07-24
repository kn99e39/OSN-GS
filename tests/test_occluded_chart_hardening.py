from types import SimpleNamespace
import unittest
import torch
from osn_gs.surface.torch_nurbs import TorchNURBSSurface
from osn_gs.surface.torch_sampled_surface_geometry import SampledTriangleMesh, triangle_triangle_intersection, sample_nurbs_triangle_mesh, build_triangle_aabb_pairs
from osn_gs.surface.torch_occluded_chart_hardening import OccludedChartHardeningConfig, _self, evaluate_occluded_chart_safety
from osn_gs.surface.torch_chart_conflict import build_occluded_chart_conflicts, attach_conflict_edges, OccludedChartConflictEdge
from osn_gs.surface.torch_continuation_domain import ContinuationDomain
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment
from osn_gs.surface.torch_occluded_region_candidate import OccludedRegionCandidate, SupportChain, CorrespondenceEdge

def plane(z=0.):
 g=torch.tensor([[[0.,0.,z],[0.,1.,z]],[[1.,0.,z],[1.,1.,z]]],dtype=torch.float64); return TorchNURBSSurface(g,torch.ones((2,2),dtype=torch.float64),degree_u=1,degree_v=1)
def vertical_plane(y=0.):
 g=torch.tensor([[[0.,y,-1.],[0.,y,1.]],[[1.,y,-1.],[1.,y,1.]]],dtype=torch.float64); return TorchNURBSSurface(g,torch.ones((2,2),dtype=torch.float64),degree_u=1,degree_v=1)
def chart(i,s): return SimpleNamespace(chart_id=i,source_candidate_id=i,supporting_domain_ids=['d1','d2'],supporting_boundary_ids=['b1','b2'],supporting_patch_ids=[1,2],surface=s,state='validated',evidence_consistency={})
def source_fixture(*, t=0., provenance=True):
 dtype=torch.float64; world=torch.tensor([[[0.,0.,0.],[0.,t,0.]],[[1.,0.,0.],[1.,t,0.]]],dtype=dtype)
 def domain(did,bid,pid):
  return ContinuationDomain(did,pid,bid,False,2,2,torch.tensor([0.,1.],dtype=dtype),1.,torch.tensor([0.,t],dtype=dtype),world,torch.tensor([[[1.,0.,0.]]*2]*2,dtype=dtype),torch.tensor([[[0.,1.,0.]]*2]*2,dtype=dtype),torch.tensor([[[0.,0.,1.]]*2]*2,dtype=dtype),torch.tensor([[0.,1.,0.],[0.,1.,0.]],dtype=dtype),torch.ones((2,2),dtype=torch.bool),torch.ones((2,2),dtype=torch.bool),torch.ones((2,2),dtype=torch.bool),1.,1.,1.,world.reshape(-1,3).min(0).values,world.reshape(-1,3).max(0).values,'valid','ok',{}, {}, {})
 def boundary(bid,pid,y):
  w=torch.tensor([[0.,y,0.],[1.,y,0.]],dtype=dtype); z=torch.zeros((2,2),dtype=dtype); v=torch.tensor([[1.,0.,0.]]*2,dtype=dtype)
  return PatchBoundarySegment(bid,pid,'outer',z,w,z,w,v,torch.tensor([[0.,1.,0.]]*2,dtype=dtype),torch.tensor([[0.,0.,1.]]*2,dtype=dtype),False,'ccw')
 da,db=domain('d1','b1',1),domain('d2','b2',2); ba,bb=boundary('b1',1,0.),boundary('b2',2,0.)
 e=CorrespondenceEdge(0,0,0,0,0.,0.,True,0.,0.,0.,'endpoint')
 candidate=OccludedRegionCandidate('cand',['d1','d2'],['b1','b2'],[1,2],SupportChain('d1',[(0,0),(1,0)],world[:,0]),SupportChain('d2',[(0,0),(1,0)],world[:,0]),[e],world[:,0],world[:,0],torch.zeros((0,4,3),dtype=dtype),world.reshape(-1,3).min(0).values,world.reshape(-1,3).max(0).values,{},{},{},{},{},{},{},{},{},{},{},'candidate','ok',{'source_boundary_correspondence': {'b1': True, 'b2': True}} if provenance else {})
 c=chart('actual',plane(2.)); c.source_candidate_id='cand'; c.supporting_patch_ids=[1,2]; c.support_samples_a=ba.world.clone(); c.support_samples_b=bb.world.clone()
 return c,candidate,{'d1':da,'d2':db},{'b1':ba,'b2':bb}

class SampledGeometryTest(unittest.TestCase):
 def test_crossing_triangles(self):
  a=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64); b=torch.tensor([[.2,.2,-1.],[.2,.2,1.],[.8,.2,0.]],dtype=torch.float64)
  self.assertEqual(triangle_triangle_intersection(a,b)['kind'],'proper_interior_intersection')
 def test_mesh_and_pairs_deterministic(self):
  m=sample_nurbs_triangle_mesh(plane(),4,4); self.assertEqual(m.triangles.shape[0],18); self.assertEqual(build_triangle_aabb_pairs(m),build_triangle_aabb_pairs(m))
 def test_registry_missing_ineligible(self):
  result=evaluate_occluded_chart_safety(chart('a',plane()),None); self.assertEqual(result.eligibility,'ineligible'); self.assertIn('visible_surface_registry_missing',result.reasons)
 def test_crossing_charts_conflict(self):
  a=chart('a',plane()); b=chart('b',plane()); edges=build_occluded_chart_conflicts([a,b]); self.assertTrue(edges); r1=evaluate_occluded_chart_safety(a,{}); r2=evaluate_occluded_chart_safety(b,{}); attach_conflict_edges([r1,r2],edges); self.assertEqual(r1.eligibility,'ineligible'); self.assertTrue(r1.conflict_edge_ids)

class HardeningContractTest(unittest.TestCase):
 def test_self_payload_has_all_counts(self):
  r=evaluate_occluded_chart_safety(chart('x',plane()),{}); s=r.self_intersection
  for k in ('triangle_count','excluded_same_cell_pair_count','excluded_adjacent_pair_count','broad_phase_pair_count','narrow_phase_pair_count','hard_intersection_count','coplanar_overlap_count','near_contact_count'): self.assertIn(k,s)
 def test_central_bridge_not_review_without_other_risk(self):
  r=evaluate_occluded_chart_safety(chart('x',plane()),{}); self.assertNotIn('central_bridge_only',r.uncertainty)
 def test_visible_payload_semantics(self):
  r=evaluate_occluded_chart_safety(chart('x',plane()),{}); p=r.visible_surface_penetration
  self.assertTrue(p['detects_surface_crossing']); self.assertFalse(p['detects_volumetric_penetration']); self.assertFalse(p['signed_inside_outside_tested'])
 def test_multiple_visible_ids_preserved(self):
  r=evaluate_occluded_chart_safety(chart('x',plane()),{4:plane(2.),2:plane(3.)}); self.assertEqual(r.visible_surface_penetration['tested_visible_patch_ids'],[2,4])
 def test_full_known_free_is_ineligible(self):
  c=chart('x',plane()); c.evidence_consistency={'candidate_hard_contradiction':True}; r=evaluate_occluded_chart_safety(c,{}); self.assertEqual(r.eligibility,'ineligible')
 def test_partial_evidence_requires_coverage_provenance(self):
  c=chart('x',plane()); c.evidence_consistency={'free_space_contradiction':{'known_free_section_count':1}}; r=evaluate_occluded_chart_safety(c,{}); self.assertEqual(r.eligibility,'ineligible')
 def test_chart_read_only(self):
  c=chart('x',plane()); before=(c.state,c.surface.control_grid.clone(),c.surface.weights.clone()); evaluate_occluded_chart_safety(c,{})
  self.assertEqual(c.state,before[0]); torch.testing.assert_close(c.surface.control_grid,before[1]); torch.testing.assert_close(c.surface.weights,before[2])
 def test_conflict_order_deterministic(self):
  a,b=chart('a',plane()),chart('b',plane()); self.assertEqual([x.payload() for x in build_occluded_chart_conflicts([a,b])],[x.payload() for x in build_occluded_chart_conflicts([b,a])])
 def test_separated_charts_no_conflict(self):
  a,b=chart('a',plane()),chart('b',plane(10.)); b.supporting_domain_ids=['x','y']; b.supporting_boundary_ids=['q','r']; self.assertFalse(build_occluded_chart_conflicts([a,b]))

class ProvenanceContractTest(unittest.TestCase):
 def test_actual_candidate_preserves_selected_t_and_source_distance(self):
  c,candidate,domains,boundaries=source_fixture(t=.5)
  r=evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual(r.eligibility,'eligible'); self.assertEqual(r.attachment_and_coverage['selected_t_min'],0.)
  self.assertEqual(r.attachment_and_coverage['selected_t_max'],0.); self.assertEqual(r.attachment_and_coverage['selected_t_over_extent_median'],0.)
  self.assertEqual(r.attachment_and_coverage['visible_boundary_to_chart_support_distance'],0.)
 def test_missing_source_provenance_is_ineligible(self):
  c,candidate,domains,boundaries=source_fixture(); del domains['d2']
  r=evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual(r.eligibility,'ineligible'); self.assertIn('support_coverage_failed',r.reasons)
 def test_partial_or_conflicting_evidence_is_review_required(self):
  c,candidate,domains,boundaries=source_fixture(); c.evidence_consistency={'conflicting_evidence':{'partial':True}}
  r=evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual(r.eligibility,'review_required')
 def test_t_positive_visible_intersection_is_hard_crossing(self):
  c,candidate,domains,boundaries=source_fixture(t=.5); candidate.correspondence_edges[0].t_a=1; candidate.correspondence_edges[0].t_b=1
  c.surface=plane(); r=evaluate_occluded_chart_safety(c,{1:plane()},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertGreater(r.visible_surface_penetration['hard_penetration_count'],0)
 def test_conflict_payload_preserves_raw_metrics_and_bool_contract(self):
  edge=build_occluded_chart_conflicts([chart('a',plane()),chart('b',plane())])[0]
  for key in ('aabb_metrics','distance_metrics','intersection_metrics','near_duplicate_metrics','reasons','unresolved'): self.assertIn(key,edge.payload())
  self.assertIn('intersection_count',edge.intersection_metrics); self.assertIs(edge.unresolved,True)
  with self.assertRaises(ValueError): OccludedChartConflictEdge('a','b',[],[],[],[],{},{},{},{},unresolved=1)


class FixtureCompletionTest(unittest.TestCase):
 def test_full_evidence_with_valid_coverage_is_eligible(self):
  c,candidate,domains,boundaries=source_fixture(); c.evidence_consistency={}
  self.assertEqual(evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries).eligibility,'eligible')
 def test_independent_uncertainty_reasons_require_review(self):
  for name in ('curvature_uncertainty','area_uncertainty','continuation_extent_uncertainty'):
   c,candidate,domains,boundaries=source_fixture(); c.evidence_consistency={name:True}
   r=evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
   self.assertEqual(r.eligibility,'review_required'); self.assertEqual(r.uncertainty,{name:True})
 def test_transition_scope_is_provenance_not_blocker(self):
  c,candidate,domains,boundaries=source_fixture(); r=evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual((r.attachment_and_coverage['coverage_scope'],r.attachment_and_coverage['transition_surface_modeled']),('central_bridge_only',False)); self.assertEqual(r.eligibility,'eligible')
 def test_self_pair_accounting_for_same_adjacent_and_nonadjacent(self):
  v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,0.,0.],[3.,0.,0.],[2.,1.,0.],[.2,.2,-1.],[.2,.2,1.],[.8,.2,0.]],dtype=torch.float64)
  mesh=SampledTriangleMesh(v,torch.tensor([[0,1,2],[0,1,2],[1,3,4],[6,7,8]]),torch.tensor([[0,0],[0,0],[1,0],[3,3]]),4,4)
  result=_self(mesh,OccludedChartHardeningConfig())
  self.assertGreaterEqual(result['excluded_same_cell_pair_count'],1); self.assertGreaterEqual(result['excluded_adjacent_pair_count'],1); self.assertGreaterEqual(result['hard_intersection_count'],1)
  self.assertEqual(result,_self(mesh,OccludedChartHardeningConfig()))
 def test_coplanar_and_near_contact_are_not_hard_crossing(self):
  a=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
  b=torch.tensor([[.2,.2,0.],[.8,.2,0.],[.2,.8,0.]],dtype=torch.float64)
  self.assertEqual(triangle_triangle_intersection(a,b)['kind'],'coplanar_area_overlap')
  c=chart('near',plane(0.)); r=evaluate_occluded_chart_safety(c,{7:plane(1e-9)},config=OccludedChartHardeningConfig(world_tolerance=1e-8))
  self.assertEqual(r.visible_surface_penetration['hard_penetration_count'],0); self.assertGreater(r.visible_surface_penetration['near_visible_contact_count'],0)
 def test_aabb_only_visible_overlap_is_not_surface_crossing(self):
  c=chart('aabb',plane()); r=evaluate_occluded_chart_safety(c,{7:plane(1e-6)},config=OccludedChartHardeningConfig(world_tolerance=1e-5))
  self.assertEqual(r.visible_surface_penetration['hard_penetration_count'],0)
 def test_explicit_source_boundary_contact_is_allowed(self):
  c,candidate,domains,boundaries=source_fixture(); c.surface=plane()
  r=evaluate_occluded_chart_safety(c,{1:vertical_plane(0.)},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual(r.visible_surface_penetration['hard_penetration_count'],0); self.assertGreater(r.visible_surface_penetration['allowed_contact_count'],0)
 def test_non_source_boundary_contact_is_hard_crossing(self):
  c,candidate,domains,boundaries=source_fixture(); c.surface=plane()
  r=evaluate_occluded_chart_safety(c,{1:vertical_plane(.5)},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertGreater(r.visible_surface_penetration['hard_penetration_count'],0); self.assertEqual(r.visible_surface_penetration['allowed_contact_count'],0)
 def test_conflict_reasons_are_separate(self):
  near_a,near_b=chart('a',plane()),chart('b',plane(5e-9)); near_b.supporting_domain_ids=['n1','n2']; near_b.supporting_boundary_ids=['nb1','nb2']; near_b.supporting_patch_ids=[7,8]
  near=build_occluded_chart_conflicts([near_a,near_b])[0]
  self.assertEqual(near.reasons,['near_duplicate_chart'])
  same_a,same_b=chart('c',plane()),chart('d',plane(10.)); same_b.supporting_domain_ids=['d1','x']
  same=build_occluded_chart_conflicts([same_a,same_b])[0]
  self.assertEqual(same.reasons,['same_source_competing_chart'])
 def test_separated_source_conflict_free_uses_all_source_ids(self):
  a,b=chart('a',plane()),chart('b',plane(10.)); b.supporting_domain_ids=['d3','d4']; b.supporting_boundary_ids=['b3','b4']; b.supporting_patch_ids=[3,4]
  self.assertFalse(build_occluded_chart_conflicts([a,b]))
 def test_chart_read_only_preserves_identity_and_payload_fields(self):
  c,candidate,domains,boundaries=source_fixture(); c.provenance={'phase_f':'preserved'}; c.weights=c.surface.weights
  before=(c.chart_id,c.state,c.surface.control_grid.clone(),c.surface.weights.clone(),dict(c.provenance),candidate.payload(),domains['d1'].payload(),boundaries['b1'].payload())
  evaluate_occluded_chart_safety(c,{},candidate=candidate,domains_by_id=domains,boundaries_by_id=boundaries)
  self.assertEqual((c.chart_id,c.state,dict(c.provenance),candidate.payload(),domains['d1'].payload(),boundaries['b1'].payload()),(before[0],before[1],before[4],before[5],before[6],before[7]))
  torch.testing.assert_close(c.surface.control_grid,before[2]); torch.testing.assert_close(c.surface.weights,before[3])

if __name__=='__main__': unittest.main()