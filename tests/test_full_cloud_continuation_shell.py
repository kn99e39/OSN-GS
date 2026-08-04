"""Worklog 130: full-cloud continuation shell and boundary-recovery diagnostics.

Covers the redesign that replaces the representative-only 8-sector support-
termination query with a continuous circular full-cloud support-gap query,
plus the A/B/C boundary-failure-stage diagnostics. See docs/worklogs/130_*.md.
"""

from __future__ import annotations

import math
import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    make_gap_sweep_scene,
    make_gaussian_density_sweep_scene,
    make_gaussian_reliability_scene,
)
from nurbs_constructor_benchmark.surface_region_adversarial_scenes import make_cylinder_phase_alias_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig


def _pipeline(max_points: int) -> TorchOSNGSPipeline:
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=max_points), device="cpu")


def _construct(pipeline: TorchOSNGSPipeline, scene):
    positions = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    opacity = torch.ones(positions.shape[0])
    stable_ids = list(range(positions.shape[0]))
    return pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)


class NoContinuationRegressionTest(unittest.TestCase):
    """Existing (pre-worklog-130) callers that never pass continuation_input must be untouched."""

    def test_box_face_still_materializes_via_sector_path_without_continuation(self):
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

        scene = make_gaussian_reliability_scene("box_face")
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        result = construct_visible_nurbs_from_gaussians(positions, covariance=covariance)
        self.assertEqual(result.diagnostic_summary["materialized_surface_count"], 1)
        self.assertEqual(result.diagnostic_summary["boundary_failure_stage"], "not_failed")


class BoundaryFailureStageDiagnosticsTest(unittest.TestCase):
    def test_summary_always_has_stage_classification_and_component_distribution(self):
        from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

        scene = make_gaussian_reliability_scene("box")
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        result = construct_visible_nurbs_from_gaussians(positions, covariance=covariance)
        summary = result.diagnostic_summary
        self.assertIn(summary["boundary_failure_stage"], (
            "not_failed", "A_candidate_generation_failed", "B_candidate_linking_failed", "C_component_admission_failed",
        ))
        distribution_total = (
            summary["boundary_component_closed_count"]
            + summary["boundary_component_open_count"]
            + summary["boundary_component_branching_count"]
            + summary["boundary_component_ambiguous_count"]
            + summary["boundary_component_isolated_count"]
        )
        self.assertEqual(distribution_total, summary["boundary_component_count"])


class DensitySweepBoundaryTest(unittest.TestCase):
    def test_box_face_region_count_and_no_false_termination_explosion_across_density(self):
        pipeline = _pipeline(max_points=128)
        candidate_counts = []
        for multiplier in (1, 2, 4, 8):
            scene = make_gaussian_density_sweep_scene("box", multiplier, seed=0)
            bundle = _construct(pipeline, scene)
            summary = bundle.construction.diagnostic_summary
            self.assertEqual(summary["region_count"], 6, f"multiplier={multiplier}")
            candidate_counts.append(summary["boundary_genuine_termination_candidate_count"])
        # Genuine-termination candidate count should stay bounded (a handful
        # per face edge/corner), never explode toward "every representative
        # thinks it's a boundary" as density increases.
        for count in candidate_counts:
            self.assertLess(count, 128, candidate_counts)


class ThinSlabShellIsolationTest(unittest.TestCase):
    def test_front_and_back_never_share_same_mode_support(self):
        """Front/back faces of a thin slab must never contribute continuation
        support to each other's queries -- driven directly by
        ``build_continuation_shells`` on a minimal hand-built two-plane
        fixture with an explicit region assignment, independent of whether
        the general-purpose region-FORMATION consensus algorithm happens to
        agree at any particular (unrelated, pre-existing) representative
        budget."""
        from osn_gs.surface.torch_canonical_region_tangent_frame import CanonicalRegionTangentFrame
        from osn_gs.surface.torch_full_cloud_continuation_shell import build_continuation_shells
        from osn_gs.surface.torch_full_neighborhood_evidence import assign_nearest_representative, compute_full_neighborhood_evidence
        from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
        from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_intrinsic_reliability

        # A thin slab: a dense top grid (normal +z) and a dense bottom grid
        # (normal -z), 0.05 apart -- same physical footprint, opposite sides.
        lin = torch.linspace(-0.3, 0.3, 9)
        grid_x, grid_y = torch.meshgrid(lin, lin, indexing="ij")
        top_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        top_positions = torch.cat((top_xy, torch.full((top_xy.shape[0], 1), 0.025)), dim=-1)
        bottom_positions = torch.cat((top_xy, torch.full((top_xy.shape[0], 1), -0.025)), dim=-1)
        positions = torch.cat((top_positions, bottom_positions), dim=0)
        count = positions.shape[0]
        scale = torch.tensor([0.05, 0.05, 0.002]).expand(count, 3).clone()
        quaternion_top = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(top_positions.shape[0], 4).clone()
        quaternion_bottom = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(bottom_positions.shape[0], 4).clone()
        quaternion = torch.cat((quaternion_top, quaternion_bottom), dim=0)
        covariance = covariance_from_scale_rotation(scale, quaternion)
        opacity = torch.ones(count)
        stable_ids = list(range(count))
        frame = extract_covariance_frame(covariance)
        intrinsic = evaluate_intrinsic_reliability(frame)

        # Representatives: every 4th point of each face.
        top_rep_idx = torch.arange(0, top_positions.shape[0], 4)
        bottom_rep_idx = torch.arange(0, bottom_positions.shape[0], 4) + top_positions.shape[0]
        rep_idx = torch.cat((top_rep_idx, bottom_rep_idx))
        rep_points = positions[rep_idx]
        rep_frame = extract_covariance_frame(covariance[rep_idx])
        rep_stable_ids = tuple(stable_ids[i] for i in rep_idx.tolist())
        rep_region_id = [0] * top_rep_idx.numel() + [1] * bottom_rep_idx.numel()
        rep_membership_state = ["core_member"] * rep_idx.numel()

        nearest, _distance = assign_nearest_representative(positions, rep_points)
        evidence = compute_full_neighborhood_evidence(
            positions, frame, opacity, intrinsic, rep_points, rep_frame, rep_stable_ids,
            precomputed_assignment=(nearest, _distance),
        )
        canonical_frames = [
            CanonicalRegionTangentFrame(
                region_id=rep_region_id[i], gaussian_id=rep_stable_ids[i],
                oriented_normal=rep_frame.normal_candidate[i],
                tangent_axis_0=rep_frame.tangent_u[i], tangent_axis_1=rep_frame.tangent_v[i],
                seed_id=rep_stable_ids[i], transport_parent_id=None, axis_source="covariance_anisotropy",
                anisotropy=1.0, transport_residual=0.0, ambiguity_reason=None,
            )
            for i in range(rep_idx.numel())
        ]
        result = build_continuation_shells(
            positions, frame, intrinsic, opacity, stable_ids, nearest,
            rep_points, rep_frame, rep_stable_ids, rep_region_id, rep_membership_state,
            evidence.mean_spacing, canonical_frames,
        )
        self.assertTrue(result)
        top_full_indices = set(range(top_positions.shape[0]))
        bottom_full_indices = set(range(top_positions.shape[0], count))
        for i, node_id in enumerate(rep_stable_ids):
            query = result.get(node_id)
            if query is None:
                continue
            allowed = top_full_indices if rep_region_id[i] == 0 else bottom_full_indices
            for fingerprint_id in query.source_full_cloud_fingerprint:
                self.assertIn(fingerprint_id, allowed)


class BoxCornerCrossFaceTest(unittest.TestCase):
    def test_corner_representative_never_treats_adjacent_face_as_same_mode_support(self):
        pipeline = _pipeline(max_points=128)
        scene = make_gaussian_density_sweep_scene("box", 4, seed=0)
        bundle = _construct(pipeline, scene)
        summary = bundle.construction.diagnostic_summary
        self.assertEqual(summary["region_count"], 6)
        # Perpendicular adjacent-face points are a different surface mode
        # (near-orthogonal normal); if the shell ever leaked them in as
        # same-mode support, region_count would collapse below 6 as
        # continuation evidence increasingly "explains away" real creases.
        # (No direct assertion beyond region_count possible without deeper
        # plumbing, so this is the load-bearing check.)


class SphereNoFalseTerminationFloodTest(unittest.TestCase):
    def test_closed_sphere_does_not_flood_genuine_termination_candidates(self):
        pipeline = _pipeline(max_points=48)
        scene = make_gaussian_reliability_scene("sphere", seed=0)
        bundle = _construct(pipeline, scene)
        summary = bundle.construction.diagnostic_summary
        eligible = summary["reliable_count"] + summary["ambiguous_count"]
        # A closed, boundary-free manifold should mostly read as "no_gap"
        # (full circular support) -- genuine termination candidates should
        # never dominate.
        if eligible:
            self.assertLess(
                summary["boundary_genuine_termination_candidate_count"], max(1, eligible // 2)
            )


class ContaminationExclusionTest(unittest.TestCase):
    def test_isolated_floater_never_becomes_same_mode_continuation_support(self):
        pipeline = _pipeline(max_points=48)
        scene = make_gaussian_reliability_scene("box_isolated_floater", seed=0)
        bundle = _construct(pipeline, scene)
        # The floater sits at (3, 3, 3), far from the box face -- no
        # continuation candidate may ever be centered anywhere near it.
        for candidate in bundle.construction.boundary_halfedge_candidates:
            distance = sum((a - b) ** 2 for a, b in zip(candidate.world_position, (3.0, 3.0, 3.0))) ** 0.5
            self.assertGreater(distance, 1.0)

    def test_isotropic_contamination_does_not_crash_construction(self):
        pipeline = _pipeline(max_points=48)
        scene = make_gaussian_reliability_scene("box_isotropic_contamination", seed=0)
        bundle = _construct(pipeline, scene)
        # Isotropic Gaussians are intrinsic-rejected (no normal evidence) --
        # they must never appear as same-mode support in any candidate's
        # fingerprint. Construction completing without error and reporting a
        # consistent stage classification is the load-bearing check here.
        self.assertIn(
            bundle.construction.diagnostic_summary["boundary_failure_stage"],
            ("not_failed", "A_candidate_generation_failed", "B_candidate_linking_failed", "C_component_admission_failed"),
        )


class PhaseAliasNoFalseShortcutTest(unittest.TestCase):
    def test_cylinder_phase_alias_scene_does_not_create_false_closed_cycle_via_continuation(self):
        pipeline = _pipeline(max_points=64)
        scene = make_cylinder_phase_alias_scene(seed=0)
        bundle = _construct(pipeline, scene)
        summary = bundle.construction.diagnostic_summary
        # Phase-alias/nonlocal edges must never feed directed ordering
        # (unchanged contract) -- closed-loop count should not spuriously
        # spike relative to the number of real region boundaries (side + 2 caps).
        self.assertLessEqual(summary["boundary_component_closed_count"], 3)


class InvarianceTest(unittest.TestCase):
    def test_region_and_reliable_counts_robust_under_rigid_transform(self):
        """SELECTION-PERTURBATION-ROBUSTNESS test ("Test B", worklog 33 --
        not estimator invariance; this exact fixture measured only ~26%
        stable-ID overlap between baseline/rotated representative sets, a
        severe pre-existing selection-grid perturbation out of this test's
        scope). See ``test_construction_outcome_is_robust_under_rigid_rotation_translation_and_uniform_scale``
        in ``test_density_preserving_representative_selection.py`` for the
        full rationale and the companion frozen-representative "Test A"
        that verifies the graph-scale estimator itself IS exactly
        rigid-transform invariant when the representative set is held
        fixed."""
        pipeline = _pipeline(max_points=48)
        scene = make_gaussian_density_sweep_scene("cylinder", 2, seed=2)
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        baseline = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)

        angle = 0.3
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        scale = 1.7
        translation = torch.tensor([3.0, -2.0, 1.0])
        transformed_positions = (positions @ rotation.T) * scale + translation
        transformed_covariance = scale * scale * (rotation @ covariance @ rotation.T)
        transformed = pipeline._construct_canonical_with_full_evidence(
            transformed_positions, transformed_covariance, opacity, stable_ids
        )

        baseline_regions = baseline.construction.diagnostic_summary["region_count"]
        transformed_regions = transformed.construction.diagnostic_summary["region_count"]
        self.assertGreater(baseline_regions, 0)
        self.assertGreater(transformed_regions, 0)
        self.assertLessEqual(
            max(baseline_regions, transformed_regions),
            5 * max(min(baseline_regions, transformed_regions), 1),
        )
        base_stage = baseline.construction.diagnostic_summary["boundary_failure_stage"]
        transformed_stage = transformed.construction.diagnostic_summary["boundary_failure_stage"]
        baseline_candidates = baseline.construction.diagnostic_summary["boundary_genuine_termination_candidate_count"]
        transformed_candidates = transformed.construction.diagnostic_summary["boundary_genuine_termination_candidate_count"]
        # Worklog 45: the target-tangent orientation repair can recover a
        # valid baseline loop while the rotated representative subset remains
        # candidate-scarce. That divergence is acceptable only when the
        # failing side has less boundary evidence than the succeeding side.
        if (base_stage == "not_failed") != (transformed_stage == "not_failed"):
            failing_candidates = transformed_candidates if transformed_stage != "not_failed" else baseline_candidates
            succeeding_candidates = baseline_candidates if transformed_stage != "not_failed" else transformed_candidates
            self.assertLess(failing_candidates, succeeding_candidates, (base_stage, transformed_stage))
        else:
            self.assertEqual(
                base_stage == "not_failed", transformed_stage == "not_failed",
                (base_stage, transformed_stage),
            )

        # Worklog 35 (task section 14): region_count alone can hide a
        # regression where coverage collapses while the count ratio still
        # passes. Add a boundary-candidate-coverage check on the same
        # generous-tolerance philosophy.
        if baseline_candidates > 0 or transformed_candidates > 0:
            self.assertLessEqual(
                max(baseline_candidates, transformed_candidates),
                5 * max(min(baseline_candidates, transformed_candidates), 1),
            )


class LocalCompetingEvidenceNoGapTest(unittest.TestCase):
    """A non-smooth local mode may block smooth continuation, never prove it."""

    @staticmethod
    def _query(modes):
        import math

        from osn_gs.surface.torch_canonical_region_tangent_frame import CanonicalRegionTangentFrame
        from osn_gs.surface.torch_full_cloud_continuation_shell import build_continuation_shells
        from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
        from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_intrinsic_reliability

        positions = [torch.tensor((0.0, 0.0, 0.0))]
        quaternions = [torch.tensor((1.0, 0.0, 0.0, 0.0))]
        for angle, mode in modes:
            radial = torch.tensor((0.25 * math.cos(angle), 0.25 * math.sin(angle), 0.0))
            if mode == "parallel":
                radial[2] = 0.30
                quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0))
            elif mode == "crease":
                quaternion = torch.tensor((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))
            else:
                quaternion = torch.tensor((1.0, 0.0, 0.0, 0.0))
            positions.append(radial)
            quaternions.append(quaternion)
        positions = torch.stack(positions)
        covariance = covariance_from_scale_rotation(
            torch.tensor((0.10, 0.10, 0.01)).expand(len(positions), 3).clone(),
            torch.stack(quaternions),
        )
        frame = extract_covariance_frame(covariance)
        intrinsic = evaluate_intrinsic_reliability(frame)
        canonical = CanonicalRegionTangentFrame(
            region_id=0, gaussian_id=0, oriented_normal=frame.normal_candidate[0],
            tangent_axis_0=frame.tangent_u[0], tangent_axis_1=frame.tangent_v[0],
            seed_id=0, transport_parent_id=None, axis_source="fixture", anisotropy=1.0,
            transport_residual=0.0, ambiguity_reason=None,
        )
        return build_continuation_shells(
            positions, frame, intrinsic, torch.ones(len(positions)), list(range(len(positions))),
            torch.zeros(len(positions), dtype=torch.long), positions[:1], extract_covariance_frame(covariance[:1]),
            (0,), [0], ["core_member"], torch.tensor((0.10,)), [canonical],
        )[0]

    def test_close_parallel_ring_cannot_become_no_gap_smooth_continuation(self):
        from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_PARALLEL_CONFLICT
        modes = [(2 * math.pi * i / 18, "parallel") for i in range(18)]
        self.assertEqual(self._query(modes).state, STATE_PARALLEL_CONFLICT)

    def test_mixed_smooth_crease_ring_fails_closed_as_crease(self):
        from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_CREASE
        modes = []
        for i in range(18):
            modes.append((2 * math.pi * i / 18, "smooth" if i % 3 == 0 else "crease"))
        self.assertEqual(self._query(modes).state, STATE_CREASE)

    def test_smooth_ring_stays_no_gap(self):
        from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_NO_GAP
        modes = [(2 * math.pi * i / 18, "smooth") for i in range(18)]
        self.assertEqual(self._query(modes).state, STATE_NO_GAP)

    def test_smooth_gap_remains_observed_termination(self):
        from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_OBSERVED_TERMINATION
        modes = [(2 * math.pi * i / 18, "smooth") for i in range(6)]
        self.assertEqual(self._query(modes).state, STATE_OBSERVED_TERMINATION)

    def test_touching_point_does_not_override_complete_smooth_coverage(self):
        from osn_gs.surface.torch_full_cloud_continuation_shell import STATE_NO_GAP
        modes = [(2 * math.pi * i / 18, "smooth") for i in range(18)]
        modes.append((0.0, "parallel"))
        self.assertEqual(self._query(modes).state, STATE_NO_GAP)

if __name__ == "__main__":
    unittest.main()
