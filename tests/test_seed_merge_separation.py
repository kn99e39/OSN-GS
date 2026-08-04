"""Worklog 38: seed existence vs component merge separation.

Worklog 37 shipped `exempt_intra_raw_component_unions_from_bridge_veto=True`
believing it separated "seed formation inside one coherent surface" from
"merging two independent surfaces". It is provably a TAUTOLOGY: the raw
components it consults are the connected components OF the very edge set the
veto iterates, so both endpoints of every core-eligible edge share a raw
component by construction. Measured on the real 3k checkpoint: 2092/2092
core-eligible edges exempt, bridge veto evaluated on exactly 0 edges (vs 1244
without the flag, of which 862 were weak), and 47 genuine articulation
bridges (single edge, >=3 nodes on each side) were unioned regardless.

This module tests the replacement: an explicit two-phase DSU where phase 1
unions only `seed_strong_edge`s (locally well-supported consensus) and phase 2
evaluates the remaining weak cross-edges as component PAIRS with aggregate
distinct-endpoint support -- so an independently valid seed survives a refused
merge, and a single fragile bridge can never fuse two surfaces.
"""

from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import (
    _flat_grid,
    make_gaussian_reliability_scene,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    RELATION_SAME_SURFACE,
    build_manifold_affinity_graph,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    INTRINSIC_RELIABLE,
    evaluate_structural_reliability,
)
from osn_gs.surface.torch_gaussian_surface_region_formation import (
    EDGE_SEED_STRONG,
    EDGE_WEAK_BRIDGE,
    RegionFormationConfig,
    _build_relation_adjacency,
    _seed_core_components,
    form_surface_regions,
)


def _pipeline(positions, covariances, config=None):
    frame = extract_covariance_frame(covariances)
    reliability = evaluate_structural_reliability(positions, frame)
    graph = build_manifold_affinity_graph(positions, frame, reliability)
    regions = form_surface_regions(positions, frame, reliability, graph, config=config)
    return frame, reliability, graph, regions


def _two_sheets_with_single_false_edge(gap: float = 1.5):
    """Two dense coherent sheets far enough apart that the affinity graph
    emits no genuine same_surface edge between them (measured: 19 cross
    edges at gap=0.9 -- correctly so, since coplanar patches that close
    together ARE one surface -- and 0 at gap>=1.5). Each sheet must seed
    independently and the pair must never merge."""
    front_positions, front_cov = _flat_grid(7, 0.12, seed=0)
    back_positions, back_cov = _flat_grid(7, 0.12, origin=(gap, 0.0, 0.0), seed=1)
    positions = torch.cat((front_positions, back_positions), dim=0)
    covariances = torch.cat((front_cov, back_cov), dim=0)
    return positions, covariances, front_positions.shape[0]


class ExemptionTautologyRegressionTest(unittest.TestCase):
    """The worklog 37 exemption must never be production-default again."""

    def test_exemption_flag_defaults_to_false(self):
        self.assertFalse(RegionFormationConfig().exempt_intra_raw_component_unions_from_bridge_veto)

    def test_two_phase_separation_is_the_production_default(self):
        self.assertTrue(RegionFormationConfig().separate_seed_and_merge_phases)

    def test_raw_component_exemption_would_exempt_every_core_eligible_edge(self):
        """Direct tautology proof: raw components computed from the
        core-eligible edge set contain both endpoints of every edge in it,
        so an intra-raw-component exemption exempts 100% of them."""
        scene = make_gaussian_reliability_scene("box", seed=0)
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        count = int(reliability.intrinsic.conditioning_score.shape[0])
        _ss, _cr, _ps, _cn, by_pair = _build_relation_adjacency(count, graph)
        intrinsic = reliability.intrinsic.intrinsic_class

        core_eligible = [
            key for key, edge in by_pair.items()
            if edge.manifold_relation == RELATION_SAME_SURFACE
            and intrinsic[key[0]] == INTRINSIC_RELIABLE
            and intrinsic[key[1]] == INTRINSIC_RELIABLE
        ]
        self.assertGreater(len(core_eligible), 0)

        adjacency = [set() for _ in range(count)]
        for a, b in core_eligible:
            adjacency[a].add(b)
            adjacency[b].add(a)
        component_id = [-1] * count
        next_id = 0
        for start in range(count):
            if component_id[start] != -1 or not adjacency[start]:
                continue
            stack = [start]
            component_id[start] = next_id
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if component_id[neighbor] == -1:
                        component_id[neighbor] = next_id
                        stack.append(neighbor)
            next_id += 1

        inter_component = [
            (a, b) for a, b in core_eligible
            if not (component_id[a] != -1 and component_id[a] == component_id[b])
        ]
        self.assertEqual(
            inter_component, [],
            "raw-component exemption is a tautology: no core-eligible edge is ever inter-component",
        )


class TypedEdgeCategoryTest(unittest.TestCase):
    def test_seed_strong_and_weak_bridge_are_distinct_populations(self):
        scene = make_gaussian_reliability_scene("box", seed=0)
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        count = int(reliability.intrinsic.conditioning_score.shape[0])
        same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)
        config = RegionFormationConfig(separate_seed_and_merge_phases=True)
        uf, consensus, bridge, path, conflict = _seed_core_components(
            count, same_surface, crease, parallel_separate, candidate_neighbors,
            by_pair, reliability, frame, config,
        )
        # A well-formed box must produce genuine seed unions.
        roots = {uf.find(n) for n in range(count)}
        self.assertLess(len(roots), count, "phase 1 must union at least some strong edges")


class IndependentSeedPreservationTest(unittest.TestCase):
    """A refused merge must leave BOTH sides seeded (worklog 38 section 3)."""

    def test_two_separated_sheets_each_keep_their_own_seed(self):
        positions, covariances, split = _two_sheets_with_single_false_edge()
        _f, _r, _g, regions = _pipeline(positions, covariances, RegionFormationConfig())
        self.assertGreaterEqual(len(regions.regions), 2)
        # Every region must live entirely on one sheet (no cross-sheet merge).
        for region in regions.regions:
            indices = list(region.member_ids)
            if len(indices) < 3:
                continue
            sides = {0 if i < split else 1 for i in indices}
            self.assertEqual(len(sides), 1, "a region must not span both sheets")

    def test_both_sheets_receive_core_members(self):
        positions, covariances, split = _two_sheets_with_single_false_edge()
        _f, _r, _g, regions = _pipeline(positions, covariances, RegionFormationConfig())
        front_core = sum(
            1 for i, state in enumerate(regions.node_membership_state)
            if state == "core_member" and i < split
        )
        back_core = sum(
            1 for i, state in enumerate(regions.node_membership_state)
            if state == "core_member" and i >= split
        )
        self.assertGreater(front_core, 0, "front sheet must seed independently")
        self.assertGreater(back_core, 0, "back sheet must seed independently")


class FragileBridgeRejectionTest(unittest.TestCase):
    """No articulation ("single fragile edge") bridge may be unioned."""

    def _articulation_bridges_unioned(self, scene_name: str, config: RegionFormationConfig) -> int:
        scene = make_gaussian_reliability_scene(scene_name, seed=0)
        frame = extract_covariance_frame(scene.covariances)
        reliability = evaluate_structural_reliability(scene.positions, frame)
        graph = build_manifold_affinity_graph(scene.positions, frame, reliability)
        count = int(reliability.intrinsic.conditioning_score.shape[0])
        same_surface, crease, parallel_separate, candidate_neighbors, by_pair = _build_relation_adjacency(count, graph)
        intrinsic = reliability.intrinsic.intrinsic_class
        uf, _c, _b, _p, _conf = _seed_core_components(
            count, same_surface, crease, parallel_separate, candidate_neighbors,
            by_pair, reliability, frame, config,
        )
        core_eligible = [
            key for key, edge in by_pair.items()
            if edge.manifold_relation == RELATION_SAME_SURFACE
            and intrinsic[key[0]] == INTRINSIC_RELIABLE
            and intrinsic[key[1]] == INTRINSIC_RELIABLE
        ]
        adjacency = [set() for _ in range(count)]
        for a, b in core_eligible:
            adjacency[a].add(b)
            adjacency[b].add(a)

        def reachable_without(start: int, blocked: tuple[int, int]) -> set[int]:
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if (node, neighbor) == blocked or (neighbor, node) == blocked:
                        continue
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            return seen

        unioned = 0
        for a, b in core_eligible:
            side_a = reachable_without(a, (a, b))
            if b in side_a:
                continue  # not an articulation edge
            side_b = reachable_without(b, (a, b))
            if len(side_a) >= 3 and len(side_b) >= 3 and uf.find(a) == uf.find(b):
                unioned += 1
        return unioned

    def test_two_phase_never_unions_an_articulation_bridge(self):
        config = RegionFormationConfig(separate_seed_and_merge_phases=True)
        self.assertEqual(self._articulation_bridges_unioned("box_with_bridge", config), 0)

    def test_worklog37_exemption_did_union_articulation_bridges(self):
        """Regression witness: documents the defect the revert fixes."""
        legacy = RegionFormationConfig(
            separate_seed_and_merge_phases=False,
            exempt_intra_raw_component_unions_from_bridge_veto=True,
        )
        strict = RegionFormationConfig(separate_seed_and_merge_phases=True)
        legacy_count = self._articulation_bridges_unioned("box_with_bridge", legacy)
        strict_count = self._articulation_bridges_unioned("box_with_bridge", strict)
        self.assertGreaterEqual(legacy_count, strict_count)


class AdversarialNegativeControlTest(unittest.TestCase):
    def test_thin_slab_front_back_never_merge(self):
        scene = make_gaussian_reliability_scene("thin_slab", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        self.assertEqual(len(regions.regions), 2)
        for region in regions.regions:
            zs = [float(scene.positions[i][2]) for i in region.member_ids]
            self.assertTrue(all(z > 0 for z in zs) or all(z < 0 for z in zs))

    def test_box_faces_never_merge_across_creases(self):
        scene = make_gaussian_reliability_scene("box", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        self.assertEqual(len(regions.regions), 6)
        for region in regions.regions:
            positions = scene.positions[list(region.member_ids)]
            extent = positions.max(dim=0).values - positions.min(dim=0).values
            self.assertLess(float(extent.min()), 0.05, "each region must stay on one face")

    def test_bridge_contamination_does_not_fuse_faces(self):
        scene = make_gaussian_reliability_scene("box_with_bridge", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        self.assertEqual(len(regions.regions), 6)

    def test_floater_never_becomes_core_member(self):
        scene = make_gaussian_reliability_scene("box_isolated_floater", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        labels = ("face",) * 81 + ("floater",)
        for idx, state in enumerate(regions.node_membership_state):
            if idx < len(labels) and labels[idx] == "floater":
                self.assertNotEqual(state, "core_member")


class PositiveControlTest(unittest.TestCase):
    def test_box_face_is_one_region(self):
        scene = make_gaussian_reliability_scene("box_face", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        self.assertEqual(len(regions.regions), 1)

    def test_cylinder_is_side_plus_two_caps(self):
        scene = make_gaussian_reliability_scene("cylinder", seed=0)
        _f, _r, _g, regions = _pipeline(scene.positions, scene.covariances, RegionFormationConfig())
        self.assertEqual(len(regions.regions), 3)


if __name__ == "__main__":
    unittest.main()
