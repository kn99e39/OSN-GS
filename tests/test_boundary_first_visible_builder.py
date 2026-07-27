from types import SimpleNamespace
import math
import unittest
from osn_gs.surface.torch_boundary_first_visible_builder import build_boundary_first_visible_surface

def circle(radius):
 return [[radius*math.cos(2*math.pi*k/8),radius*math.sin(2*math.pi*k/8),0.] for k in range(9)]
def loop(label,area,points,ordered=True): return SimpleNamespace(label=label,area_world=area,boundary_world_points=points,ordered_boundary_world_points=points if ordered else [])

class BoundaryFirstVisibleBuilderTest(unittest.TestCase):
 def test_annulus_is_built_from_pre_surface_outer_inner_loops(self):
  boundary=SimpleNamespace(component_id=5,outer_loops=[loop(1,12.,circle(2.))],hole_loops=[loop(2,3.,circle(1.))])
  result=build_boundary_first_visible_surface(boundary,curve_count=8,samples_per_curve=4)
  self.assertEqual(result.state,'constructed'); self.assertEqual(result.topology,'boundary_role_network'); self.assertEqual(result.provenance['boundary_roles'], ['outer_boundary', 'interior_boundary']); self.assertEqual(result.surface_result.state,'constructed_multi_patch'); self.assertEqual(len(result.surface_result.surfaces),8); self.assertFalse(result.surface_result.diagnostics['fallback_used'])
  outer_entity=result.provenance['observed_outer_boundary']; inner_entity=result.provenance['observed_inner_boundary']
  self.assertEqual(outer_entity['representation_kind'],'observed_evidence_points'); self.assertEqual(inner_entity['representation_kind'],'observed_evidence_points')
  self.assertEqual(len(outer_entity['points']),9); self.assertEqual(len(inner_entity['points']),9)

 def test_unordered_pair_is_explicitly_unsupported(self):
  boundary=SimpleNamespace(component_id=7,outer_loops=[loop(1,12.,circle(2.),ordered=False)],hole_loops=[loop(2,3.,circle(1.),ordered=False)])
  result=build_boundary_first_visible_surface(boundary)
  self.assertEqual(result.state,'unsupported'); self.assertEqual(result.reason,'ordered_boundary_required')
 def test_unpaired_topology_is_explicitly_unsupported_not_rectangle(self):
  boundary=SimpleNamespace(component_id=6,outer_loops=[loop(1,12.,circle(2.))],hole_loops=[])
  result=build_boundary_first_visible_surface(boundary)
  self.assertEqual(result.state,'unsupported'); self.assertEqual(result.reason,'interior_support_network_required'); self.assertIsNone(result.surface_result)

if __name__=='__main__': unittest.main()