import unittest
import torch
from tests.test_boundary_support_network import boundary, circle
from osn_gs.surface.torch_boundary_support_network import build_boundary_support_curve_network
from osn_gs.surface.torch_boundary_constrained_surface import build_boundary_constrained_surface

class BoundaryConstrainedSurfaceTest(unittest.TestCase):
    def test_open_network_becomes_exact_boundary_constrained_surface(self):
        a=boundary('a',[[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]],closed=False,patch_id=1)
        b=boundary('b',[[0.,1.,0.],[.5,1.,.2],[1.,1.,0.]],closed=False,patch_id=2)
        network=build_boundary_support_curve_network(a,b,curve_count=3,samples_per_curve=4)
        result=build_boundary_constrained_surface(network)
        self.assertEqual(result.state,'constructed'); self.assertFalse(result.diagnostics['fallback_used'])
        uv=torch.stack((network.parameters,torch.zeros_like(network.parameters)),dim=1)
        torch.testing.assert_close(result.surfaces[0].evaluate(uv),network.boundary_a_samples,atol=1e-5,rtol=1e-5)
        uv[:,1]=1.
        torch.testing.assert_close(result.surfaces[0].evaluate(uv),network.boundary_b_samples,atol=1e-5,rtol=1e-5)

    def test_closed_network_becomes_explicit_seam_multi_patch_not_rectangle(self):
        network=build_boundary_support_curve_network(boundary('i',circle(1.),closed=True,patch_id=1),boundary('o',circle(2.),closed=True,patch_id=2),curve_count=8,samples_per_curve=4)
        result=build_boundary_constrained_surface(network)
        self.assertEqual(result.state,'constructed_multi_patch'); self.assertEqual(len(result.surfaces),8); self.assertFalse(result.diagnostics['fallback_used'])
        for index,surface in enumerate(result.surfaces):
            uv=torch.stack((torch.zeros(4),torch.linspace(0.,1.,4)),dim=1)
            torch.testing.assert_close(surface.evaluate(uv),network.support_curves[index])
            uv[:,0]=1.
            torch.testing.assert_close(surface.evaluate(uv),network.support_curves[(index+1)%8],atol=1e-5,rtol=1e-5)

    def test_closed_network_uses_cubic_circumferential_control_without_moving_seams(self):
        network = build_boundary_support_curve_network(boundary('i',circle(1.),closed=True,patch_id=1),boundary('o',circle(2.),closed=True,patch_id=2),curve_count=8,samples_per_curve=5)
        result = build_boundary_constrained_surface(network)
        self.assertTrue(all(surface.degree_u == 3 for surface in result.surfaces))
        self.assertTrue(all(surface.degree_v == 1 for surface in result.surfaces))
        for index, surface in enumerate(result.surfaces):
            uv = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
            expected = torch.stack((network.support_curves[index, 0], network.support_curves[index, -1], network.support_curves[(index + 1) % 8, 0], network.support_curves[(index + 1) % 8, -1]))
            torch.testing.assert_close(surface.evaluate(uv), expected, atol=1e-5, rtol=1e-5)
if __name__=='__main__': unittest.main()