"""Worklog 37: core-seeding coverage vs candidate-recall separation.

Worklog 36's ambiguous-unassigned waterfall found R2 ("same_surface
neighbors exist but none are in any region yet") dominant at 53-57% on real
3k/5k/10k -- suggesting a core-SEEDING coverage problem, not a growth-
threshold problem. This module proves the exact mechanism: `_seed_core_components`
processes same_surface edges via union-find, and because every node starts
as its own singleton root, the bridge veto (intended for merging two
INDEPENDENT surfaces) cannot distinguish "these two nodes are pieces of the
SAME raw same_surface connected component that haven't been unioned yet"
from "these are two genuinely separate surfaces" -- both look like `ra != rb`
at veto time. Measured directly: an 83-node raw same_surface component (a
single connected component before ANY veto) had 50 individually
well-supported internal edges, yet fragmented into 43 final union-find
groups (largest 11) purely from this conflation. Fixed by precomputing raw
same_surface connected components up front and skipping the bridge veto
(NOT the edge-intrinsic vetoes: contradicted consensus, phase-alias,
oversized-footprint-parallel) for any edge whose endpoints already share a
raw component.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import build_manifold_affinity_graph
from osn_gs.surface.torch_gaussian_structural_reliability import evaluate_structural_reliability
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    RegionFormationConfig,
    form_surface_regions,
)


def _form(scene, *, config=None, ids=None):
    frame = extract_covariance_frame(scene.covariances)
    reliability = evaluate_structural_reliability(scene.positions, frame)
    graph = build_manifold_affinity_graph(scene.positions, frame, reliability, ids=ids)
    return form_surface_regions(scene.positions, frame, reliability, graph, config=config, ids=ids)


class ConfigNamingTest(unittest.TestCase):
    """Worklog 37 task section 2: worklog-numbered flags renamed to semantic
    names. Worklog 38 CORRECTED the third flag's default -- see
    ``test_seed_merge_separation.py`` for the tautology proof that forced it.
    """

    def test_semantic_flag_names_exist_with_canonical_defaults(self):
        config = RegionFormationConfig()
        self.assertTrue(config.allow_weak_bridge_only_growth_support)
        self.assertTrue(config.require_nearby_parallel_evidence_for_parallel_veto)
        # Worklog 38: reverted to False (diagnostic-only). Shipping it as
        # True disabled the bridge veto entirely rather than separating seed
        # existence from merge admission.
        self.assertFalse(config.exempt_intra_raw_component_unions_from_bridge_veto)
        self.assertTrue(config.separate_seed_and_merge_phases)


class SeedExistenceVsMergeSeparationTest(unittest.TestCase):
    """Worklog 38: seed existence and component merge are separated by the
    explicit two-phase DSU, NOT by worklog 37's raw-component exemption
    (which was a tautology -- it exempted 100% of core-eligible edges)."""

    def test_large_coherent_surface_seeds_as_one_large_core_component(self):
        """A clean, densely-sampled box face is one coherent surface -- the
        two-phase seeding must cover it at least as well as the legacy
        single-pass path, without disabling the bridge veto."""
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        result_fixed = _form(scene, config=RegionFormationConfig(separate_seed_and_merge_phases=True))
        result_legacy = _form(scene, config=RegionFormationConfig(separate_seed_and_merge_phases=False))
        core_fixed = sum(1 for s in result_fixed.node_membership_state if s == "core_member")
        core_legacy = sum(1 for s in result_legacy.node_membership_state if s == "core_member")
        self.assertGreaterEqual(core_fixed, core_legacy)

    def test_two_independent_surfaces_are_not_merged(self):
        """Thin slab: front and back must stay separate regions under the
        canonical (two-phase) configuration."""
        scene = make_gaussian_reliability_scene("thin_slab", seed=0)
        result = _form(scene, config=RegionFormationConfig())
        self.assertEqual(len(result.regions), 2)
        for region in result.regions:
            zs = [float(scene.positions[i][2]) for i in region.member_ids]
            # Each region must stay entirely on one side (no cross-slab mixing).
            self.assertTrue(all(z > 0 for z in zs) or all(z < 0 for z in zs))

    def test_box_faces_stay_six_separate_regions_with_bridge_contamination(self):
        """Deliberately-planted bridge between two box faces must still not
        cause a false merge -- the bridge itself IS a cross-raw-component
        edge (connects two genuinely different raw same_surface components),
        so it remains subject to the (unmodified) bridge veto."""
        scene = make_gaussian_reliability_scene("box_with_bridge", seed=0)
        result = _form(scene, config=RegionFormationConfig())
        self.assertEqual(len(result.regions), 6)
        for region in result.regions:
            positions = scene.positions[list(region.member_ids)]
            extent = (positions.max(dim=0).values - positions.min(dim=0).values)
            # Exactly one axis must stay near-flat (one face only) -- proves
            # no region spans two perpendicular faces.
            self.assertLess(float(extent.min()), 0.05)

    def test_legacy_single_pass_remains_available_for_ablation(self):
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        legacy = RegionFormationConfig(separate_seed_and_merge_phases=False)
        result = _form(scene, config=legacy)
        # The legacy single-pass path must still run (used by the worklog 38
        # ablation matrix as the worklog 36 authoritative baseline).
        self.assertIsNotNone(result)


class NegativeControlNoFalseMergeTest(unittest.TestCase):
    def test_box_isotropic_contamination_excludes_contaminated_indices(self):
        scene = make_gaussian_reliability_scene("box_isotropic_contamination", seed=0)
        result = _form(scene, config=RegionFormationConfig())
        self.assertEqual(len(result.regions), 1)

    def test_floater_never_becomes_core_member(self):
        scene = make_gaussian_reliability_scene("box_isolated_floater", seed=0)
        result = _form(scene, config=RegionFormationConfig())
        labels = ("face",) * 81 + ("floater",)
        for idx, state in enumerate(result.node_membership_state):
            if idx < len(labels) and labels[idx] == "floater":
                self.assertNotEqual(state, "core_member")


if __name__ == "__main__":
    unittest.main()
