from types import SimpleNamespace
import unittest
import torch
from osn_gs.surface.torch_boundary_component_recovery import propose_boundary_first_component_recovery, materialize_boundary_first_recovery_regions

class BoundaryComponentRecoveryTest(unittest.TestCase):
 def component(self,identifier,indices,normal,aabb_min,aabb_max): return SimpleNamespace(component_id=identifier,gaussian_indices=torch.tensor(indices),normal=torch.tensor(normal,dtype=torch.float32),aabb_min=torch.tensor(aabb_min,dtype=torch.float32),aabb_max=torch.tensor(aabb_max,dtype=torch.float32))
 def test_smooth_edge_contact_is_recovery_candidate(self):
  points=torch.tensor([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[.2,0.,0.],[.3,0.,0.],[.4,0.,0.]])
  a=self.component(0,[0,1,2],[0.,0.,1.],[0.,0.,0.],[.2,0.,0.]); b=self.component(1,[3,4,5],[0.,0.,1.],[.2,0.,0.],[.4,0.,0.])
  edge=propose_boundary_first_component_recovery([a,b],points,max_normalized_distance=2.)[0]
  self.assertTrue(edge.accepted); self.assertLess(edge.normalized_distance,2.)
 def test_curved_annulus_face_split_is_detected_without_mutating_components(self):
  from nurbs_constructor_benchmark.scenes import make_scene
  from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy
  from osn_gs.surface.torch_surface_components import build_surface_components
  scene=make_scene('curved_annulus',600,seed=0)
  components=build_surface_components(build_voxel_gaussian_hierarchy(scene.points),scene.points).components
  self.assertEqual(len(components),2)
  edges=propose_boundary_first_component_recovery(components,scene.points,max_normalized_distance=2.)
  self.assertEqual(len(edges),1); self.assertTrue(edges[0].accepted); self.assertGreater(edges[0].aabb_contact_dimension,-1)
 def test_curved_annulus_recovered_region_restores_annulus_boundary(self):
  from nurbs_constructor_benchmark.scenes import make_scene
  from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy
  from osn_gs.surface.torch_surface_components import build_surface_components
  from osn_gs.surface.torch_component_boundary import extract_component_boundary
  from osn_gs.surface.torch_chart_topology import classify_boundary_result
  scene=make_scene('curved_annulus',600,seed=0); hierarchy=build_voxel_gaussian_hierarchy(scene.points); components=build_surface_components(hierarchy,scene.points).components
  regions=materialize_boundary_first_recovery_regions(components,scene.points,propose_boundary_first_component_recovery(components,scene.points,max_normalized_distance=2.))
  self.assertEqual(len(regions),1); self.assertEqual(regions[0].source_component_ids,(0,1))
  self.assertEqual(classify_boundary_result(extract_component_boundary(regions[0].component,hierarchy,scene.points)),'annulus')
 def test_curved_annulus_recovery_reaches_boundary_first_multi_patch(self):
  from nurbs_constructor_benchmark.scenes import make_scene
  from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy
  from osn_gs.surface.torch_surface_components import build_surface_components
  from osn_gs.surface.torch_component_boundary import extract_component_boundary
  from osn_gs.surface.torch_boundary_first_visible_builder import build_boundary_first_visible_surface
  scene=make_scene('curved_annulus',600,seed=0); hierarchy=build_voxel_gaussian_hierarchy(scene.points); components=build_surface_components(hierarchy,scene.points).components
  region=materialize_boundary_first_recovery_regions(components,scene.points,propose_boundary_first_component_recovery(components,scene.points,max_normalized_distance=2.))[0]
  result=build_boundary_first_visible_surface(extract_component_boundary(region.component,hierarchy,scene.points),curve_count=8,samples_per_curve=8)
  self.assertEqual(result.state,'constructed'); self.assertEqual(result.surface_result.state,'constructed_multi_patch'); self.assertEqual(len(result.surface_result.surfaces),8); self.assertFalse(result.surface_result.diagnostics['fallback_used'])
 def test_seed_and_density_sweep_keeps_only_nonface_curved_annulus_recovery(self):
  from nurbs_constructor_benchmark.scenes import make_scene
  from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy
  from osn_gs.surface.torch_surface_components import build_surface_components
  for name,expected in (('curved_annulus',True),('close_parallel_sheets',False)):
   for count in (400,600):
    for seed in (0,1,2):
     scene=make_scene(name,count,seed=seed); components=build_surface_components(build_voxel_gaussian_hierarchy(scene.points),scene.points).components
     edge=propose_boundary_first_component_recovery(components,scene.points,max_normalized_distance=2.)[0]
     self.assertEqual(edge.accepted,expected,(name,count,seed,edge.payload()))
 def test_parallel_separated_sheets_are_not_recovery_candidate(self):
  points=torch.tensor([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[0.,0.,.5],[.1,0.,.5],[.2,0.,.5]])
  a=self.component(0,[0,1,2],[0.,0.,1.],[0.,0.,0.],[.2,0.,0.]); b=self.component(1,[3,4,5],[0.,0.,1.],[0.,0.,.5],[.2,0.,.5])
  edge=propose_boundary_first_component_recovery([a,b],points,max_normalized_distance=2.)[0]
  self.assertFalse(edge.accepted); self.assertIn('support_gap_too_large',edge.reasons)

if __name__=='__main__': unittest.main()