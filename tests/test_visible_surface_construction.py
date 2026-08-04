from __future__ import annotations

import unittest

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_curvature_sweep_scene, make_gaussian_reliability_scene
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


class VisibleSurfaceConstructionTest(unittest.TestCase):
    def test_box_face_runs_from_gaussian_input_and_retains_all_canonical_stages(self):
        scene = make_gaussian_reliability_scene("box_face")
        result = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))))
        self.assertEqual(result.diagnostic_summary["input_gaussian_count"], len(scene.positions))
        self.assertEqual(result.coverage_semantics, "reliable_core_only")
        self.assertIn(result.construction_state, {"constructed", "partially_constructed", "review_required", "boundary_recovery_failed", "no_admissible_region"})
        self.assertIsNotNone(result.covariance_frame)
        self.assertIsNotNone(result.manifold_affinity)
        self.assertIsNotNone(result.surface_regions)

    def test_reliability_failure_stage_reflects_intrinsic_vs_contextual_collapse(self):
        """Worklog 135: a healthy scene must classify as 'not_failed', and the
        new field must never contradict reliable_count/intrinsic_reliable_count."""
        scene = make_gaussian_reliability_scene("box_face")
        result = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances)
        summary = result.diagnostic_summary
        self.assertIn("reliability_failure_stage", summary)
        self.assertIn("intrinsic_reliable_count", summary)
        if summary["intrinsic_reliable_count"] == 0:
            self.assertEqual(summary["reliability_failure_stage"], "intrinsic_reliability_collapse")
        elif summary["reliable_count"] == 0:
            self.assertEqual(summary["reliability_failure_stage"], "contextual_reliability_collapse")
        elif summary["reliable_count"] < summary["intrinsic_reliable_count"]:
            self.assertEqual(summary["reliability_failure_stage"], "partial_contextual_reliability_collapse")
        else:
            self.assertEqual(summary["reliability_failure_stage"], "not_failed")

    def test_box_face_and_curved_patch_materialize_with_shared_methodology(self):
        import torch
        box_face = make_gaussian_reliability_scene("box_face")
        curved = make_curvature_sweep_scene(10.0)
        box_face_result = construct_visible_nurbs_from_gaussians(box_face.positions, covariance=box_face.covariances)
        curved_result = construct_visible_nurbs_from_gaussians(curved.positions, covariance=curved.covariances)
        self.assertEqual(box_face_result.construction_state, "constructed")
        self.assertEqual(curved_result.construction_state, "constructed")
        self.assertEqual(len(box_face_result.materialized_visible_nurbs_surfaces), 1)
        self.assertEqual(len(curved_result.materialized_visible_nurbs_surfaces), 1)
        uv = torch.tensor([[0., 0.], [.5, .5], [1., 1.]])
        samples = curved_result.materialized_visible_nurbs_surfaces[0].surface.evaluate(uv)
        self.assertGreater(float(samples[:, 2].std()), 1e-4)
    def test_end_to_end_invariance_for_box_face_and_curved_patch(self):
        import torch
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for scene in (make_gaussian_reliability_scene("box_face"), make_curvature_sweep_scene(10.0)):
            ids = tuple(range(len(scene.positions)))
            baseline = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=ids)
            variants = (
                (scene.positions + torch.tensor([1.7, -0.8, 0.4]), scene.covariances),
                (scene.positions @ rotation.T, rotation @ scene.covariances @ rotation.T),
                (scene.positions * 1.7, scene.covariances * (1.7 ** 2)),
                (scene.positions.flip(0), scene.covariances.flip(0), tuple(reversed(ids))),
            )
            base_members = tuple(sorted((frozenset(region.member_ids) for region in baseline.surface_regions.regions), key=lambda item: tuple(sorted(item))))
            base_boundary = tuple(sorted((frozenset(component.ordered_source_ids) for component in baseline.ordered_boundary_components), key=lambda item: tuple(sorted(item))))
            for variant in variants:
                positions, covariance = variant[0], variant[1]
                variant_ids = variant[2] if len(variant) == 3 else ids
                result = construct_visible_nurbs_from_gaussians(positions, covariance=covariance, stable_ids=variant_ids)
                self.assertEqual(result.construction_state, baseline.construction_state)
                self.assertEqual(len(result.materialized_visible_nurbs_surfaces), len(baseline.materialized_visible_nurbs_surfaces))
                self.assertEqual(tuple(sorted((frozenset(region.member_ids) for region in result.surface_regions.regions), key=lambda item: tuple(sorted(item)))), base_members)
                self.assertEqual(tuple(sorted((frozenset(component.ordered_source_ids) for component in result.ordered_boundary_components), key=lambda item: tuple(sorted(item)))), base_boundary)
    def test_separated_surface_controls_do_not_create_a_bridge_surface(self):
        for name in ("thin_slab", "box"):
            scene = make_gaussian_reliability_scene(name)
            result = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances)
            materialized_regions = {item.input.source_region_id for item in result.materialized_visible_nurbs_surfaces}
            self.assertTrue(materialized_regions.issubset({region.region_id for region in result.surface_regions.regions}))
            self.assertLessEqual(len(materialized_regions), result.diagnostic_summary["region_count"])
            self.assertNotEqual(result.construction_state, "constructed")
    def test_open_or_unresolved_topology_never_creates_placeholder_surface(self):
        scene = make_gaussian_reliability_scene("box_isolated_floater")
        result = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances)
        self.assertFalse(any(item.surface is not None and item.state != "materialized" for item in result.review_results))


if __name__ == "__main__":
    unittest.main()

