import math
import unittest
import torch
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment
from osn_gs.surface.torch_boundary_support_network import build_boundary_support_curve_network, observed_boundary_curves_from_annulus_component


def boundary(identifier, points, *, closed, patch_id):
    world=torch.tensor(points,dtype=torch.float32)
    count=len(points)
    uv=torch.zeros((count,2),dtype=torch.float32)
    vectors=torch.zeros_like(world)
    return PatchBoundarySegment(identifier,patch_id,'observed_support',uv,world,uv,world,vectors,vectors,vectors,closed,'ccw' if closed else 'open')


def circle(radius):
    values=[]
    for k in range(9):
        angle=2*math.pi*k/8
        values.append([radius*math.cos(angle),radius*math.sin(angle),0.])
    return values


class BoundarySupportNetworkTest(unittest.TestCase):
    def test_closed_inner_outer_loops_produce_radial_support_family(self):
        inner=boundary('inner',circle(1.),closed=True,patch_id=3)
        outer=boundary('outer',circle(2.),closed=True,patch_id=4)
        first=build_boundary_support_curve_network(inner,outer,curve_count=8,samples_per_curve=5)
        second=build_boundary_support_curve_network(inner,outer,curve_count=8,samples_per_curve=5)
        self.assertTrue(first.closed); self.assertEqual(tuple(first.support_curves.shape),(8,5,3))
        torch.testing.assert_close(first.support_curves,second.support_curves)
        torch.testing.assert_close(torch.linalg.vector_norm(first.support_curves[:,0],dim=1),torch.ones(8))
        torch.testing.assert_close(torch.linalg.vector_norm(first.support_curves[:,-1],dim=1),torch.full((8,),2.))
        self.assertEqual(first.provenance['boundary_a_patch_id'],3); self.assertEqual(first.payload()['curve_count'],8)

    def test_open_pair_preserves_explicit_reversed_correspondence(self):
        a=boundary('a',[[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]],closed=False,patch_id=1)
        b=boundary('b',[[1.,1.,0.],[.5,1.,0.],[0.,1.,0.]],closed=False,patch_id=2)
        network=build_boundary_support_curve_network(a,b,curve_count=3,samples_per_curve=3,reverse_boundary_b=True)
        torch.testing.assert_close(network.boundary_a_samples[:,0],network.boundary_b_samples[:,0])
        torch.testing.assert_close(network.support_curves[:,0],network.boundary_a_samples)
        torch.testing.assert_close(network.support_curves[:,-1],network.boundary_b_samples)
        self.assertTrue(network.correspondence['reverse_boundary_b'])

    def test_component_annulus_loops_are_valid_pre_surface_inputs(self):
        from types import SimpleNamespace
        outer=SimpleNamespace(label=4,area_world=12.,boundary_world_points=circle(2.),ordered_boundary_world_points=circle(2.))
        hole=SimpleNamespace(label=9,area_world=3.,boundary_world_points=circle(1.),ordered_boundary_world_points=circle(1.))
        result=SimpleNamespace(component_id=12,outer_loops=[outer],hole_loops=[hole])
        outer_curve,inner_curve=observed_boundary_curves_from_annulus_component(result)
        network=build_boundary_support_curve_network(inner_curve,outer_curve,curve_count=8,samples_per_curve=3)
        self.assertEqual(network.provenance['boundary_a']['component_id'],12)
        self.assertEqual(network.boundary_b_id,'component:12:outer:4')
    def test_rejects_silent_topology_or_identity_fallback(self):
        open_boundary=boundary('open',[[0.,0.,0.],[1.,0.,0.]],closed=False,patch_id=1)
        loop=boundary('loop',circle(1.),closed=True,patch_id=2)
        with self.assertRaises(ValueError): build_boundary_support_curve_network(open_boundary,loop,curve_count=3,samples_per_curve=2)
        with self.assertRaises(ValueError): build_boundary_support_curve_network(open_boundary,open_boundary,curve_count=2,samples_per_curve=2)

if __name__=='__main__': unittest.main()