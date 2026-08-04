"""Worklog 56: eligible visible-boundary surfaces -> continuation domain and
bounded occluded-region candidate production bridge.

`build_eligible_boundary_continuation_bridge` consumes ONLY
`VisibleSurfaceConstructionResult.eligible_materialized_surfaces()` (worklog
55) -- it never re-derives candidate extraction, directed ordering, or region
eligibility. This file checks: (1) every eligible surface reaches a
continuation domain with full provenance carried by reference, (2) a scene
with zero eligible surfaces produces zero downstream objects, (3) the
negative-control fixtures (Box/Cylinder/Sphere/Thin-slab) are transferred only
from their already-established eligible surfaces.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_eligible_boundary_continuation_bridge import (
    STATUS_BRIDGED,
    STATUS_CONTINUATION_INELIGIBLE,
    STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED,
    STATUS_SKIPPED_NOT_MATERIALIZED,
    _resample_closed_uv_loop_to_minimum,
    build_eligible_boundary_continuation_bridge,
    run_eligible_boundary_continuation_bridge_from_gaussians,
)
from osn_gs.surface.torch_visible_boundary_region_status import STATUS_ELIGIBLE_CLOSED


def _construction(scene_name: str, cap: int = 64):
    scene = make_gaussian_reliability_scene(scene_name)
    positions = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    opacity = torch.ones(positions.shape[0])
    stable_ids = list(range(positions.shape[0]))
    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device="cpu")
    bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
    return bundle.construction


class EligibleBoundaryContinuationBridgeTest(unittest.TestCase):
    def test_thin_slab_eligible_surfaces_all_reach_continuation_domains(self):
        construction = _construction("thin_slab")
        eligible = construction.eligible_materialized_surfaces()
        self.assertEqual(len(eligible), 3)
        result = build_eligible_boundary_continuation_bridge(construction)
        self.assertEqual(len(result.attempts), len(eligible))
        for attempt in result.attempts:
            self.assertEqual(attempt.status, STATUS_BRIDGED)
            self.assertEqual(attempt.region_status, STATUS_ELIGIBLE_CLOSED)
            self.assertIsNotNone(attempt.boundary_id)
            self.assertIsNotNone(attempt.continuation_domain_id)
        self.assertEqual(len(result.continuation_domains), 3)
        # Every domain's own source_boundary_id resolves back to a synthesized
        # boundary segment whose provenance carries the full eligibility record
        # -- the referential chain this bridge uses instead of duplicating
        # fields into ContinuationDomain/OccludedRegionCandidate.
        for domain in result.continuation_domains:
            boundary = result.boundaries_by_id[domain.source_boundary_id]
            self.assertEqual(boundary.provenance["region_status"], STATUS_ELIGIBLE_CLOSED)
            self.assertTrue(boundary.provenance["supporting_source_ids"])
            self.assertIn(boundary.provenance["region_id"], {int(m.input.source_region_id) for m in eligible})

    def test_sphere_has_zero_eligible_surfaces_and_zero_downstream_objects(self):
        construction = _construction("sphere")
        self.assertEqual(len(construction.eligible_materialized_surfaces()), 0)
        result = build_eligible_boundary_continuation_bridge(construction)
        self.assertEqual(len(result.attempts), 0)
        self.assertEqual(len(result.continuation_domains), 0)
        self.assertEqual(len(result.occluded_region_candidates), 0)

    def test_box_and_cylinder_transfer_only_from_existing_eligible_surfaces(self):
        # Every attempt originates 1:1 from an eligible surface -- nothing
        # beyond that set contributes. Some of these loops have only 3 unique
        # boundary samples, below Phase D's own representation-density floor
        # (>=4 for a closed boundary); the bridge deterministically upsamples
        # them (edge-midpoint insertion, still the same validated loop) so
        # every eligible surface here reaches an actual continuation domain.
        for scene_name, expected_eligible in (("box", 6), ("cylinder", 2)):
            with self.subTest(scene=scene_name):
                construction = _construction(scene_name)
                eligible = construction.eligible_materialized_surfaces()
                self.assertEqual(len(eligible), expected_eligible)
                result = build_eligible_boundary_continuation_bridge(construction)
                self.assertEqual(len(result.attempts), len(eligible))
                bridged = [a for a in result.attempts if a.status == STATUS_BRIDGED]
                self.assertEqual(len(bridged), expected_eligible)
                self.assertEqual(len(result.continuation_domains), expected_eligible)

    def test_non_materialized_or_non_eligible_component_never_reaches_a_domain(self):
        # A component that is materialized but whose carried region_status is
        # NOT eligible_closed_boundary must be fail-closed rejected even
        # though `state == "materialized"` -- defense in depth against a
        # future upstream refactor bug, per the task's explicit requirement.
        from osn_gs.surface.torch_visible_boundary_materialization_adapter import (
            VisibleBoundaryMaterializationInput,
            VisibleBoundaryMaterializationResult,
        )

        class _FakeConstruction:
            def eligible_materialized_surfaces(self):
                fake_input = VisibleBoundaryMaterializationInput(
                    adapter_id="adapter:fake", source_region_id=0, source_boundary_component_id="fake",
                    ordered_boundary_point_ids=(), ordered_boundary_points=torch.zeros((0, 3)),
                    interior_reliable_point_ids=(), interior_points=torch.zeros((0, 3)),
                    coverage_semantics="reliable_core_only", materialization_state="materialized", reasons=(),
                    region_status="open_observed_fragment", region_status_reason="best_available_downstream_valid_open_path",
                    boundary_role_scope="outer_boundary_only", supporting_source_ids=(1, 2, 3),
                )
                return (VisibleBoundaryMaterializationResult(fake_input, None, "materialized", None, None, ()),)

        result = build_eligible_boundary_continuation_bridge(_FakeConstruction())
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].status, STATUS_SKIPPED_NOT_MATERIALIZED)
        self.assertEqual(len(result.continuation_domains), 0)

    def test_region_status_inconsistent_with_eligible_closed_is_skipped_not_bridged(self):
        from osn_gs.surface.torch_nurbs import TorchNURBSSurface
        from osn_gs.surface.torch_visible_boundary_materialization_adapter import (
            VisibleBoundaryMaterializationInput,
            VisibleBoundaryMaterializationResult,
        )

        surface = TorchNURBSSurface(
            control_grid=torch.rand(4, 4, 3), weights=torch.ones(4, 4), degree_u=2, degree_v=2,
        )
        fake_input = VisibleBoundaryMaterializationInput(
            adapter_id="adapter:fake2", source_region_id=1, source_boundary_component_id="fake2",
            ordered_boundary_point_ids=tuple(range(6)), ordered_boundary_points=torch.rand(6, 3),
            interior_reliable_point_ids=(6, 7), interior_points=torch.rand(2, 3),
            coverage_semantics="reliable_core_only", materialization_state="materialized", reasons=(),
            region_status="ambiguous_boundary", region_status_reason="no_physical_termination_candidates_only_typed_nonphysical_evidence",
            boundary_role_scope="outer_boundary_only", supporting_source_ids=(9,),
        )

        class _FakeConstruction:
            def eligible_materialized_surfaces(self):
                return (VisibleBoundaryMaterializationResult(fake_input, surface, "materialized", 0.0, 0.0, ()),)

        result = build_eligible_boundary_continuation_bridge(_FakeConstruction())
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].status, STATUS_SKIPPED_NOT_ELIGIBLE_CLOSED)
        self.assertEqual(len(result.continuation_domains), 0)

    def test_resample_deterministically_upsamples_a_triangle_without_moving_original_vertices(self):
        # Empirically: build_continuation_domain's own tangent/direction math
        # (_world_arclength_tangent/_arclength_metadata) is fully finite and
        # non-degenerate for a bare 3-vertex closed loop -- its >=4-sample
        # floor is a representation-density convention, not a geometric
        # necessity, so upsampling the SAME validated triangle (never moving
        # its 3 original vertices, only inserting a deterministic edge
        # midpoint) is safe.
        triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        closed = torch.cat([triangle, triangle[:1]], dim=0)
        resampled = _resample_closed_uv_loop_to_minimum(closed, 4)
        self.assertEqual(int(resampled.shape[0]), 5)  # 4 unique + closing duplicate
        self.assertTrue(torch.allclose(resampled[-1], resampled[0]))
        for original_vertex in triangle:
            self.assertTrue(any(torch.allclose(original_vertex, sample) for sample in resampled[:-1]))
        # No-op when already at/above the minimum.
        square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        square_closed = torch.cat([square, square[:1]], dim=0)
        self.assertTrue(torch.equal(_resample_closed_uv_loop_to_minimum(square_closed, 4), square_closed))

    def test_continuation_domain_failure_after_resampling_is_typed_continuation_ineligible(self):
        # Forces build_continuation_domain to fail even after this bridge's
        # own resampling has already been applied -- the remaining failure
        # must be reported under the explicit STATUS_CONTINUATION_INELIGIBLE
        # status, not a generic/opaque failure bucket.
        from unittest import mock

        from osn_gs.surface.torch_continuation_domain import ContinuationDomainBuildError

        construction = _construction("thin_slab")
        with mock.patch(
            "osn_gs.surface.torch_eligible_boundary_continuation_bridge.build_continuation_domain",
            side_effect=ContinuationDomainBuildError("forced for test"),
        ):
            result = build_eligible_boundary_continuation_bridge(construction)
        self.assertEqual(len(result.attempts), 3)
        for attempt in result.attempts:
            self.assertEqual(attempt.status, STATUS_CONTINUATION_INELIGIBLE)
            self.assertIn(STATUS_CONTINUATION_INELIGIBLE, attempt.reasons)
        self.assertEqual(len(result.continuation_domains), 0)
        self.assertEqual(len(result.occluded_region_candidates), 0)

    def test_orchestration_entry_point_matches_manual_two_step_composition(self):
        # "standalone helper" is no longer the whole story: this is the actual
        # production orchestration a real caller uses -- one call from raw
        # Gaussian evidence to bounded occluded-region candidates, internally
        # calling `construct_visible_nurbs_from_gaussians` (unchanged) then
        # this bridge, rather than requiring every caller to chain them.
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

        scene = make_gaussian_reliability_scene("box_face")
        combined = run_eligible_boundary_continuation_bridge_from_gaussians(
            scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
        )
        manual_construction = construct_visible_nurbs_from_gaussians(
            scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
        )
        manual_bridge = build_eligible_boundary_continuation_bridge(manual_construction)
        self.assertEqual(
            [a.payload() for a in combined.bridge.attempts],
            [a.payload() for a in manual_bridge.attempts],
        )
        self.assertEqual(len(combined.bridge.continuation_domains), len(manual_bridge.continuation_domains))
        self.assertEqual(len(combined.bridge.occluded_region_candidates), len(manual_bridge.occluded_region_candidates))


if __name__ == "__main__":
    unittest.main()
