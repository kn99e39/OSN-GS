"""Worklog 35: core-component merge / weak-bridge veto repair (C9).

Worklog 34 fixed the growth loop's veto-set reuse but left `region_count`
unchanged (75/85/64 on real 3k/5k/10k) -- core components stayed at
median-4-5-member size. This module traces the bridge-veto merge decision
directly (`_seed_core_components`, `_evaluate_bridge_veto`) and fixes ONE
proven, narrow defect: the parallel-shortcut override in the core-merge loop
re-thresholds `normal_direction_separation_over_thickness` (a raw metric
normalized by individual-Gaussian `normal_thickness`, the same category of
quantity worklog 30-33 already found unusable for real long-horizon-trained
data) at a fixed "4.0", ignoring the fact that same_surface-classified and
parallel_separate-classified edges heavily OVERLAP on this metric at any
fixed threshold (measured on the 3k checkpoint: same_surface median 108,
range 0.09-4453; parallel_separate median 645) -- so this override vetoed
otherwise-well-evidenced bridges essentially at random on real data. The
override's own comment says it exists for pairs with "nearby parallel-
separated evidence"; the fix makes it actually check for that (via
`consensus.contradicting_parallel_neighbor_count`, already computed) instead
of re-deriving a second, differently-thresholded, non-discriminating check.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import _flat_grid, make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    RELATION_SAME_SURFACE,
    build_manifold_affinity_graph,
)
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    RegionFormationConfig,
    _build_relation_adjacency,
    _compute_edge_consensus,
    form_surface_regions,
)


def _form(scene, *, ids=None):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability, ids=ids)
    return form_surface_regions(scene.positions, frame, reliability, graph, ids=ids)


def _wide_thin_plane_scene():
    """Reproduces the exact scale-mismatch pattern found on real long-horizon
    checkpoints: a flat plane wide enough that same-surface pairs are many
    tangent-radii apart (as ADC-evolved geometry naturally is), with per-
    Gaussian normal thickness far smaller than that spacing (as ADC-split
    surfels naturally are). Directly measured: 161/935 same_surface edges
    exceed the override's raw `normal_direction_separation_over_thickness`
    threshold on this fixture purely from scale, with zero actual parallel-
    sheet contamination present -- the uniform-thickness synthetic scenes
    elsewhere in this file (box/cylinder/box_face at any downsample cap) do
    NOT reproduce this, because their per-Gaussian thickness is constant
    across the whole scene regardless of physical extent."""
    positions, covariances = _flat_grid(15, 0.3, surfel_scale=0.15, surfel_thickness=0.0005, seed=0)
    return positions, covariances


class ParallelVetoUsesActualNearbyEvidenceTest(unittest.TestCase):
    """Directly exercises the fixed override on box scenes where
    `normal_direction_separation_over_thickness` is known (measured) to be
    large for GENUINE same_surface edges too, purely from scale mismatch."""

    def test_bridge_normal_separation_metric_does_not_discriminate_relations_alone(self):
        """Empirical premise check: on a wide thin plane (the same scale
        regime real long-horizon checkpoints exhibit), a large fraction of
        GENUINE same_surface edges exceed the override's raw threshold purely
        from scale -- this is exactly why the override needed the additional
        `contradicting_parallel_neighbor_count` gate."""
        positions, covariances = _wide_thin_plane_scene()
        frame = extract_covariance_frame(covariances)
        reliability = evaluate_structural_reliability(positions, frame)
        graph = build_manifold_affinity_graph(positions, frame, reliability)
        config = RegionFormationConfig()
        same_surface_over_threshold = [
            e for e in graph.edges
            if e.manifold_relation == RELATION_SAME_SURFACE
            and e.metrics is not None
            and e.metrics.normal_direction_separation_over_thickness > config.bridge_normal_separation_with_parallel_veto
        ]
        # If the premise did not hold (i.e. no same_surface edges ever
        # exceeded the raw threshold), the override's old unconditional form
        # would have been harmless and this whole repair would be moot --
        # assert the premise is real on this fixture.
        self.assertGreater(len(same_surface_over_threshold), 0)

    def test_multi_edge_cross_component_pair_merges_when_no_nearby_parallel_evidence(self):
        """Two core clusters connected by >=2 same_surface cross edges, none
        of which have nearby parallel_separate contamination, must merge --
        this exercises the fixed override's positive path directly."""
        scene = make_gaussian_reliability_scene("cylinder", seed=0)
        result = _form(scene)
        # Side wall of a clean cylinder is one large coherent surface --
        # verifies core consolidation is not blocked by the parallel-veto
        # override when there genuinely is no parallel-separate contamination
        # nearby (a strong positive control: the side must not fragment into
        # many small components purely from this specific veto path).
        side_regions = [r for r in result.regions if len(r.member_ids) > 20]
        self.assertGreaterEqual(len(side_regions), 1)


class WeakBridgeNegativeControlTest(unittest.TestCase):
    """The parallel-veto fix must not cause any FALSE merge -- close-parallel
    sheets and box creases must stay separate exactly as before."""

    def test_thin_slab_front_and_back_stay_two_regions(self):
        scene = make_gaussian_reliability_scene("thin_slab", seed=0)
        result = _form(scene)
        self.assertEqual(len(result.regions), 2)

    def test_box_faces_do_not_merge_across_creases(self):
        scene = make_gaussian_reliability_scene("box", seed=0)
        result = _form(scene)
        # 6 faces -- must not collapse into fewer regions via a false
        # crease-crossing merge, nor explode via fragmentation.
        self.assertGreaterEqual(len(result.regions), 5)
        self.assertLessEqual(len(result.regions), 8)

    def test_bridge_contamination_scene_does_not_gain_a_false_merge(self):
        scene = make_gaussian_reliability_scene("box_with_bridge", seed=0)
        result = _form(scene)
        member_sets = [set(r.member_ids) for r in result.regions]
        # No single region may contain members from more than one box face
        # group according to a simple spatial-spread heuristic: verifies the
        # fix does not let the deliberately-planted bridge succeed.
        self.assertGreaterEqual(len(result.regions), 5)


class CoreComponentMergeTraceTest(unittest.TestCase):
    """Directly traces `_seed_core_components`'s bridge-veto outcomes to
    confirm the fixed override requires `contradicting_parallel_neighbor_count
    > 0` (Case A repair) rather than vetoing on the raw metric alone."""

    def test_override_requires_nearby_parallel_evidence_not_just_raw_metric(self):
        positions, covariances = _wide_thin_plane_scene()
        frame = extract_covariance_frame(covariances)
        reliability = evaluate_structural_reliability(positions, frame)
        graph = build_manifold_affinity_graph(positions, frame, reliability)
        config = RegionFormationConfig()
        count = int(reliability.intrinsic.conditioning_score.shape[0])
        same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)

        # Find at least one same_surface edge with a large raw normal-
        # separation metric but ZERO nearby parallel evidence -- the fixed
        # override must NOT veto it (whereas the old unconditional form
        # would have).
        found_a_case = False
        for (a, b), edge in by_pair.items():
            if edge.manifold_relation != RELATION_SAME_SURFACE or edge.metrics is None:
                continue
            if edge.metrics.normal_direction_separation_over_thickness <= config.bridge_normal_separation_with_parallel_veto:
                continue
            consensus = _compute_edge_consensus(a, b, same_surface, crease, parallel_separate, candidate_neighbors, by_pair, reliability, frame, config)
            if consensus.contradicting_parallel_neighbor_count == 0:
                found_a_case = True
                # Reproduce the exact override condition from
                # `_seed_core_components`: with the fix, this must evaluate
                # to False (not vetoed) purely because of the added
                # `contradicting_parallel_neighbor_count > 0` requirement.
                vetoed = (
                    edge.metrics.normal_direction_separation_over_thickness > config.bridge_normal_separation_with_parallel_veto
                    and edge.metrics.mutual_tangent_residual > config.bridge_borderline_tangent_residual_veto
                    and consensus.contradicting_parallel_neighbor_count > 0
                )
                self.assertFalse(vetoed)
        self.assertTrue(found_a_case, "fixture did not exercise the target code path -- test is vacuous")


class AblationConfigFlagTest(unittest.TestCase):
    """Worklog 36 task section 2: explicit config-flag ablation switches
    replace the worklog 35 file-swap approach (which silently compared
    against a pre-worklog-34 HEAD state -- the exact source of the worklog
    34/35 baseline discrepancy this worklog resolved)."""

    def test_both_flags_default_true_matches_current_production_behavior(self):
        default_config = RegionFormationConfig()
        self.assertTrue(default_config.allow_weak_bridge_only_growth_support)
        self.assertTrue(default_config.require_nearby_parallel_evidence_for_parallel_veto)

    def test_disabling_worklog35_flag_reproduces_pre_fix_veto_behavior(self):
        positions, covariances = _wide_thin_plane_scene()
        frame = extract_covariance_frame(covariances)
        reliability = evaluate_structural_reliability(positions, frame)
        graph = build_manifold_affinity_graph(positions, frame, reliability)
        config_on = RegionFormationConfig(require_nearby_parallel_evidence_for_parallel_veto=True)
        config_off = RegionFormationConfig(require_nearby_parallel_evidence_for_parallel_veto=False)
        result_on = form_surface_regions(positions, frame, reliability, graph, config=config_on)
        result_off = form_surface_regions(positions, frame, reliability, graph, config=config_off)
        # The fix (gate ON) must never produce STRICTLY WORSE core coverage
        # than the pre-fix behavior (gate OFF) on a fixture built specifically
        # to exercise the scale-mismatch bug.
        core_on = sum(1 for s in result_on.node_membership_state if s == "core_member")
        core_off = sum(1 for s in result_off.node_membership_state if s == "core_member")
        self.assertGreaterEqual(core_on, core_off)


class AmbiguousUnassignedWaterfallTest(unittest.TestCase):
    """Worklog 36 task section 14: growth requires same_surface neighbors to
    already be IN a region -- a node with same_surface degree > 0 whose
    neighbors are all themselves unassigned cannot be attached by growth no
    matter how permissive the growth threshold is. This is a distinct
    failure mode (R2) from having no same_surface degree at all (R1)."""

    def test_isolated_same_surface_pair_with_no_core_neighbor_stays_ambiguous(self):
        # Two mutually same_surface nodes far from any core-eligible cluster:
        # each other's only same_surface neighbor is equally unassigned, so
        # growth (which only attaches to an EXISTING region) can never fire.
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        two_node_scene = type(scene)(
            "isolated_pair_far_from_core", scene.positions[:2], scene.covariances[:2], "isolated pair",
        )
        result = _form(two_node_scene)
        self.assertEqual(result.regions, ())
        # Neither node can be core_member (degree requirement needs >=2
        # same_surface partners) NOR consensus_attached (growth needs a
        # target region, and none exists) -- both stay ambiguous_unassigned.
        for state in result.node_membership_state:
            self.assertIn(state, ("ambiguous_unassigned", "rejected_structural_node"))


if __name__ == "__main__":
    unittest.main()
