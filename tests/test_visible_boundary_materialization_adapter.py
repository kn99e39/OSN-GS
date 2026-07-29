from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent
from osn_gs.surface.torch_visible_boundary_materialization_adapter import materialize_visible_boundary_component


class VisibleBoundaryMaterializationAdapterTest(unittest.TestCase):
    def test_admissible_outer_loop_materializes_evaluable_surface(self):
        component = OrderedBoundaryComponent(
            "region:0:component:test", 0, ("a", "b", "c", "d"), (0, 1, 2, 3),
            "ordered_closed_loop", True, (), {"observed_support_termination": 4}, 0.9,
            "outer_boundary_candidate", "reliable_core_only", False, (),
        )
        boundary = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
        interior = torch.tensor([[0., 0., 0.], [0.3, 0.2, 0.]])
        result = materialize_visible_boundary_component(component, boundary, interior, boundary_ids=(0, 1, 2, 3), interior_ids=(4, 5))
        self.assertEqual(result.state, "materialized")
        self.assertIsNotNone(result.surface)
        self.assertTrue(torch.isfinite(result.surface.evaluate(torch.tensor([[0.5, 0.5]]))).all())

    def test_open_component_never_gets_synthetic_closure(self):
        component = OrderedBoundaryComponent("region:0:component:open", 0, ("a",), (0,), "ordered_open_chain", False, (), {}, 0.5, "open_boundary_candidate", "reliable_core_only", False, ())
        point = torch.tensor([[0., 0., 0.]])
        result = materialize_visible_boundary_component(component, point, point, boundary_ids=(0,), interior_ids=(0,))
        self.assertEqual(result.state, "unsupported_topology")
        self.assertIsNone(result.surface)


if __name__ == "__main__":
    unittest.main()
