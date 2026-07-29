import unittest
from nurbs_constructor_benchmark.scenes import make_scene
from nurbs_constructor_benchmark.boundary_first_support import construct_boundary_first_support

class BoundaryFirstSupportPipelineTest(unittest.TestCase):
 def test_curved_annulus_uses_recovery_then_boundary_pair_multi_patch(self):
  result=construct_boundary_first_support(make_scene('curved_annulus',600,seed=0),curve_count=8,samples_per_curve=8)
  self.assertEqual(result.raw_component_count,2); self.assertEqual(len(result.recovered_regions),1); self.assertEqual(len(result.visible_results),1)
  visible=result.visible_results[0]
  self.assertEqual(visible.state,'constructed'); self.assertEqual(visible.topology,'boundary_role_network'); self.assertEqual(visible.provenance['boundary_roles'], ['outer_boundary', 'interior_boundary']); self.assertEqual(visible.surface_result.state,'constructed_multi_patch'); self.assertEqual(len(visible.surface_result.surfaces),8); self.assertFalse(visible.surface_result.diagnostics['fallback_used'])

 def test_plane_and_parallel_sheets_use_the_same_boundary_role_contract(self):
  for name in ('plane','close_parallel_sheets'):
   result=construct_boundary_first_support(make_scene(name,600,seed=0))
   self.assertTrue(all(visible.topology=='boundary_role_network' for visible in result.visible_results),name)
   self.assertTrue(all(visible.state in ('constructed','unsupported') for visible in result.visible_results),name)
   self.assertTrue(all(visible.provenance.get('boundary_roles', ['outer_boundary'])[0]=='outer_boundary' for visible in result.visible_results),name)
 def test_resolution96_raster_tolerance_sweep_keeps_positive_and_concave_controls_separate(self):
  # sine/seed=2 (both point counts) is a newly-discovered genuine marginal
  # case since the star-shape validation landed (worklog 112): its observed
  # boundary's monotonicity_ratio around the selected anchor is ~0.79-0.82,
  # below the 0.85 star-shape threshold that comfortably separates every
  # other genuinely star-shaped scene (>=0.85) from the known-concave
  # u_shape (~0.64). This is the gate correctly declining to force an
  # angle-based fan through weaker evidence, not a regression to relax away.
  known_marginal = {("sine", 400, 2), ("sine", 600, 2)}
  for name, expected in (("sine", "constructed"), ("curved_annulus", "constructed"), ("u_shape", "unsupported")):
   for count in (400, 600):
    for seed in (0, 1, 2):
     result = construct_boundary_first_support(make_scene(name, count, seed=seed), boundary_resolution=96)
     if (name, count, seed) in known_marginal:
      self.assertTrue(all(visible.state == 'unsupported' and visible.reason == 'insufficient_observed_interior_support' for visible in result.visible_results), (name, count, seed))
      continue
     self.assertTrue(all(visible.state == expected for visible in result.visible_results), (name, count, seed))
if __name__=='__main__': unittest.main()