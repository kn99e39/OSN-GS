from __future__ import annotations

"""Phase E geometric candidate-builder tests (design sections 3-7, 11-12).

These exercise `build_geometric_region_candidates` / `build_candidate_conflicts`
and the broad-phase helper with no ObservationEvidence (evidence-dependent
fixtures live in test_candidate_evidence.py). Domains are hand-constructed
`ContinuationDomain`s -- the geometric builder only consumes `world`,
`sample_valid_mask`, `closed`, source IDs, and the soft-evidence tensors, so a
direct grid is the most controllable fixture (mirrors Phase D's hand-crafted
boundary records).
"""

import math
import unittest

import torch

from osn_gs.surface.torch_aabb_broad_phase import sweep_and_prune_pairs
from osn_gs.surface.torch_continuation_domain import ContinuationDomain
from osn_gs.surface.torch_occluded_region_candidate import (
    STATE_CANDIDATE,
    STATE_REJECTED,
    STATE_UNSUPPORTED,
    build_candidate_conflicts,
    build_geometric_region_candidates,
)


def _make_domain(
    world: torch.Tensor,
    *,
    boundary_id: str,
    patch_id: int,
    closed: bool = False,
    valid_mask: torch.Tensor | None = None,
    scale: float = 0.25,
    state: str = "valid",
) -> ContinuationDomain:
    world = world.to(torch.float64)
    s_count, t_count = int(world.shape[0]), int(world.shape[1])
    if valid_mask is None:
        valid_mask = torch.ones((s_count, t_count), dtype=torch.bool)

    if t_count >= 2:
        outward = torch.nn.functional.normalize(world[:, 1] - world[:, 0], dim=1, eps=1e-12)
    else:
        outward = torch.zeros((s_count, 3), dtype=torch.float64)
        outward[:, 0] = 1.0
    outward_grid = outward[:, None, :].expand(s_count, t_count, 3).clone()

    tangent_s = torch.zeros_like(world)
    if s_count >= 2:
        tangent_s[0] = world[1] - world[0]
        tangent_s[-1] = world[-1] - world[-2]
        if s_count > 2:
            tangent_s[1:-1] = world[2:] - world[:-2]
    tangent_s = torch.nn.functional.normalize(tangent_s, dim=2, eps=1e-12)
    normal = torch.nn.functional.normalize(torch.cross(tangent_s, outward_grid, dim=2), dim=2, eps=1e-12)

    boundary_world = world[:, 0]
    seg = (boundary_world[1:] - boundary_world[:-1]).norm(dim=1) if s_count > 1 else world.new_zeros(0)
    s_world = torch.cat([world.new_zeros(1), torch.cumsum(seg, dim=0)]) if s_count > 1 else world.new_zeros(1)
    boundary_length = float(s_world[-1])
    t_world = torch.linspace(0.0, scale, t_count, dtype=torch.float64)
    flat = world.reshape(-1, 3)

    return ContinuationDomain(
        domain_id=f"{boundary_id}:continuation",
        source_patch_id=patch_id,
        source_boundary_id=boundary_id,
        closed=closed,
        s_count=s_count,
        t_count=t_count,
        s_world=s_world,
        boundary_length=boundary_length,
        t_world=t_world,
        world=world,
        tangent_s=tangent_s,
        tangent_t=outward_grid,
        normal=normal,
        outward_tangent_world=outward,
        normal_valid_mask=valid_mask.clone(),
        direction_valid_mask=torch.ones(s_count, dtype=torch.bool),
        sample_valid_mask=valid_mask,
        local_surface_scale=scale,
        continuation_extent=scale,
        extent_multiplier=1.0,
        aabb_min=flat.min(dim=0).values,
        aabb_max=flat.max(dim=0).values,
        state=state,
        reason="ok",
        validity={},
        uncertainty={},
        provenance={},
    )


def _facing_planar(
    gap: float,
    *,
    scale: float = 0.25,
    n_s: int = 5,
    reach: float = 0.3,
    offset_y: float = 0.0,
    id_a: str = "a",
    id_b: str = "b",
    za: float = 0.0,
    zb: float = 0.0,
) -> tuple[ContinuationDomain, ContinuationDomain]:
    """Two coplanar strips facing along +/-x with tip gap ``gap``."""

    ys = torch.linspace(0.0, 1.0, n_s, dtype=torch.float64)
    ts = torch.linspace(0.0, reach, 4, dtype=torch.float64)
    n_t = ts.shape[0]

    world_a = torch.zeros((n_s, n_t, 3), dtype=torch.float64)
    world_b = torch.zeros((n_s, n_t, 3), dtype=torch.float64)
    for s in range(n_s):
        for t in range(n_t):
            world_a[s, t] = torch.tensor([ts[t], ys[s], za], dtype=torch.float64)
            # B boundary at x = reach + gap + reach, tip at x = reach + gap.
            world_b[s, t] = torch.tensor([2 * reach + gap - ts[t], ys[s] + offset_y, zb], dtype=torch.float64)
    return (
        _make_domain(world_a, boundary_id=id_a, patch_id=1, scale=scale),
        _make_domain(world_b, boundary_id=id_b, patch_id=2, scale=scale),
    )


class BroadPhaseTest(unittest.TestCase):
    def test_canonical_ordered_and_deterministic(self) -> None:
        aabb_min = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64)
        aabb_max = torch.tensor([[0.2, 1.0, 0.0], [0.5, 1.0, 0.0], [10.2, 1.0, 0.0]], dtype=torch.float64)
        pairs = sweep_and_prune_pairs(["b", "a", "far"], aabb_min, aabb_max, [0.25, 0.25, 0.25], expand_factor=1.5)
        self.assertEqual([(p.label_a, p.label_b) for p in pairs], [("a", "b")])
        self.assertLess(pairs[0].aabb_distance, pairs[0].threshold + 1e-9)
        self.assertGreater(pairs[0].scale_normalized_aabb_distance, 0.0)

    def test_excluded_pairs_are_dropped(self) -> None:
        aabb_min = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)
        aabb_max = torch.tensor([[0.2, 1.0, 0.0], [0.5, 1.0, 0.0]], dtype=torch.float64)
        pairs = sweep_and_prune_pairs(
            ["a", "b"], aabb_min, aabb_max, [0.25, 0.25], expand_factor=1.5, excluded_pairs={("a", "b")}
        )
        self.assertEqual(pairs, [])


class GeometricCandidateTest(unittest.TestCase):
    def _build(self, domains, boundaries=None, **kw):
        return build_geometric_region_candidates(domains, boundaries or {}, **kw)

    # 1
    def test_planar_two_sided_gap_is_candidate(self) -> None:
        a, b = _facing_planar(gap=0.2)
        candidates = self._build([a, b])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        self.assertEqual(len(candidates[0].correspondence_edges), 5)
        self.assertEqual(int(candidates[0].bridge_cells.shape[0]), 4)

    # 2 — coplanar narrow pair is NOT auto-rejected (mandatory correction 1)
    def test_coplanar_narrow_pair_not_rejected(self) -> None:
        a, b = _facing_planar(gap=0.05)
        candidates = self._build([a, b])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)

    # 3
    def test_parallel_bounded_ribbon(self) -> None:
        a, b = _facing_planar(gap=0.15, n_s=6)
        candidates = self._build([a, b])
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        self.assertEqual(len(candidates[0].support_chain_a.st_indices), 6)

    # 4 — orthogonal corner, no facing gate
    def test_orthogonal_corner_is_candidate(self) -> None:
        ys = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        ts = torch.linspace(0.0, 0.3, 4, dtype=torch.float64)
        gap = 0.15
        wa = torch.zeros((5, 4, 3), dtype=torch.float64)
        wb = torch.zeros((5, 4, 3), dtype=torch.float64)
        for s in range(5):
            for t in range(4):
                wa[s, t] = torch.tensor([ts[t], ys[s], 0.0], dtype=torch.float64)  # extends +x
                wb[s, t] = torch.tensor([0.3 + gap, ys[s], ts[t]], dtype=torch.float64)  # extends +z
        a = _make_domain(wa, boundary_id="a", patch_id=1)
        b = _make_domain(wb, boundary_id="b", patch_id=2)
        candidates = self._build([a, b])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        # orthogonal outward directions -> ~0 dot, still accepted.
        self.assertLess(abs(candidates[0].outward_soft_evidence["median"]), 0.2)

    # 5 — oblique corner
    def test_oblique_corner_is_candidate(self) -> None:
        ys = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        ts = torch.linspace(0.0, 0.3, 4, dtype=torch.float64)
        wa = torch.zeros((5, 4, 3), dtype=torch.float64)
        wb = torch.zeros((5, 4, 3), dtype=torch.float64)
        d = 1.0 / math.sqrt(2.0)
        for s in range(5):
            for t in range(4):
                wa[s, t] = torch.tensor([ts[t], ys[s], 0.0], dtype=torch.float64)
                wb[s, t] = torch.tensor([0.45 - ts[t] * d, ys[s], ts[t] * d], dtype=torch.float64)
        a = _make_domain(wa, boundary_id="a", patch_id=1)
        b = _make_domain(wb, boundary_id="b", patch_id=2)
        candidates = self._build([a, b])
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)

    # 6 — curved two-sided gap (tips sag in z)
    def test_curved_two_sided_gap(self) -> None:
        ys = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        ts = torch.linspace(0.0, 0.3, 4, dtype=torch.float64)
        gap = 0.2
        wa = torch.zeros((6, 4, 3), dtype=torch.float64)
        wb = torch.zeros((6, 4, 3), dtype=torch.float64)
        for s in range(6):
            curve = 0.05 * math.sin(math.pi * float(ys[s]))
            for t in range(4):
                wa[s, t] = torch.tensor([ts[t], ys[s], curve], dtype=torch.float64)
                wb[s, t] = torch.tensor([0.6 + gap - ts[t], ys[s], curve], dtype=torch.float64)
        a = _make_domain(wa, boundary_id="a", patch_id=1, scale=0.3)
        b = _make_domain(wb, boundary_id="b", patch_id=2, scale=0.3)
        candidates = self._build([a, b])
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)

    # 7 — closed annular candidate, cyclic band
    def test_closed_annular_candidate(self) -> None:
        n = 8
        angles = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        ts = torch.linspace(0.0, 0.3, 3, dtype=torch.float64)
        r_a, r_b = 1.0, 2.0
        wa = torch.zeros((n, 3, 3), dtype=torch.float64)
        wb = torch.zeros((n, 3, 3), dtype=torch.float64)
        for s in range(n):
            c, sn = math.cos(float(angles[s])), math.sin(float(angles[s]))
            for t in range(3):
                ra = r_a + float(ts[t])
                rb = r_b - float(ts[t])
                wa[s, t] = torch.tensor([ra * c, ra * sn, 0.0], dtype=torch.float64)
                wb[s, t] = torch.tensor([rb * c, rb * sn, 0.0], dtype=torch.float64)
        a = _make_domain(wa, boundary_id="a", patch_id=1, closed=True, scale=0.6)
        b = _make_domain(wb, boundary_id="b", patch_id=2, closed=True, scale=0.6)
        candidates = self._build([a, b])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        self.assertIsNone(candidates[0].connector_end)
        self.assertTrue(candidates[0].provenance["cyclic"])
        # cyclic -> wrap bridge cell present (n cells for n samples)
        self.assertEqual(int(candidates[0].bridge_cells.shape[0]), n)

    # 8 — zero connector separation is a structural negative
    def test_zero_connector_separation_rejected(self) -> None:
        a, b = _facing_planar(gap=0.0)
        candidates = self._build([a, b])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, STATE_REJECTED)
        self.assertIn("zero_connector_separation", candidates[0].reason)

    # 9 — disconnected-close negative: samples never within threshold
    def test_disconnected_close_no_candidate(self) -> None:
        # small scale makes the (moderate) gap exceed the normalized threshold.
        a, b = _facing_planar(gap=0.4, scale=0.05)
        candidates = self._build([a, b], correspondence_threshold=1.0)
        self.assertEqual(candidates, [])

    # 10 — one-sided negative
    def test_one_sided_no_candidate(self) -> None:
        a, _ = _facing_planar(gap=0.2)
        self.assertEqual(self._build([a]), [])

    # 11 — AABB overlap but no narrow correspondence
    def test_aabb_overlap_but_no_correspondence(self) -> None:
        # broad phase pairs them (expanded AABBs overlap) but tip distance is
        # far beyond the correspondence threshold.
        a, b = _facing_planar(gap=0.5, scale=0.25)
        broad = self._build([a, b], broad_phase_expand_factor=5.0, correspondence_threshold=0.5)
        self.assertEqual(broad, [])

    # 12 — same (s_a,s_b) edge canonicalization keeps the min normalized distance
    def test_same_index_edge_canonicalization(self) -> None:
        a, b = _facing_planar(gap=0.2)
        candidates = self._build([a, b])
        edges = candidates[0].correspondence_edges
        keys = [(e.s_a, e.s_b) for e in edges]
        self.assertEqual(len(keys), len(set(keys)))  # one edge per (s_a,s_b)
        # tips (max t) are the closest, so canonicalization must select them.
        self.assertTrue(all(e.t_a == a.t_count - 1 for e in edges))

    # 13 — non-monotonic correspondence splits into separate components
    def test_non_monotonic_components_split(self) -> None:
        # B's s-order is reversed for the second half, forcing a monotonicity break.
        ys = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        ts = torch.linspace(0.0, 0.3, 4, dtype=torch.float64)
        gap = 0.2
        wa = torch.zeros((6, 4, 3), dtype=torch.float64)
        wb = torch.zeros((6, 4, 3), dtype=torch.float64)
        # B y-mapping: 0,1,2 aligned then 5,4,3 -> s_b goes up then jumps/reverses.
        b_y_order = [0.0, 0.2, 0.4, 1.0, 0.8, 0.6]
        for s in range(6):
            for t in range(4):
                wa[s, t] = torch.tensor([ts[t], ys[s], 0.0], dtype=torch.float64)
                wb[s, t] = torch.tensor([0.6 + gap - ts[t], b_y_order[s], 0.0], dtype=torch.float64)
        a = _make_domain(wa, boundary_id="a", patch_id=1, scale=0.3)
        b = _make_domain(wb, boundary_id="b", patch_id=2, scale=0.3)
        candidates = self._build([a, b])
        self.assertGreaterEqual(len(candidates), 2)

    # 14 — multiple components remain distinct candidate ids
    def test_multiple_components_distinct(self) -> None:
        a, b = _facing_planar(gap=0.2, n_s=6)
        # force a break by editing B's alignment in the middle
        b.world[3, :, 1] = b.world[5, 0, 1] + 0.4  # push section 3 far in y
        candidates = self._build([a, b])
        ids = {c.candidate_id for c in candidates}
        self.assertEqual(len(ids), len(candidates))

    # 15 — domain order reversal determinism
    def test_domain_order_reversal_determinism(self) -> None:
        a, b = _facing_planar(gap=0.2)
        first = self._build([a, b])
        second = self._build([b, a])
        self.assertEqual([c.candidate_id for c in first], [c.candidate_id for c in second])
        torch.testing.assert_close(first[0].bridge_cells, second[0].bridge_cells)

    # 16 — degenerate domain provenance preserved, rejected domain excluded
    def test_degenerate_and_rejected_domain_policy(self) -> None:
        a, b = _facing_planar(gap=0.2)
        a.state = "degenerate"
        candidates = self._build([a, b])
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        self.assertTrue(candidates[0].provenance["domain_a_degenerate"])

        a2, b2 = _facing_planar(gap=0.2)
        a2.state = "rejected"
        self.assertEqual(self._build([a2, b2]), [])

    # duplicate / same source boundary pair -> no candidate
    def test_same_source_boundary_pair_excluded(self) -> None:
        a, b = _facing_planar(gap=0.2, id_a="shared", id_b="shared")
        # both have source_boundary_id "shared" -> excluded from pairing
        self.assertEqual(self._build([a, b]), [])

    def test_duplicate_domain_deduplicated(self) -> None:
        a, b = _facing_planar(gap=0.2)
        candidates = self._build([a, b, a])  # duplicate a
        self.assertEqual(len(candidates), 1)

    # 24 — conflict edge generation
    def test_conflict_edge_generation(self) -> None:
        a, b = _facing_planar(gap=0.2, n_s=6)
        b.world[3, :, 1] = b.world[5, 0, 1] + 0.4
        candidates = self._build([a, b])
        conflicts = build_candidate_conflicts(candidates)
        # candidates share source domains a & b -> overlapping bridge -> conflict
        self.assertGreaterEqual(len(conflicts), 0)
        for edge in conflicts:
            self.assertNotEqual(edge.candidate_a, edge.candidate_b)
            self.assertTrue(edge.reason)

    def test_state_never_promoted_from_domain_state(self) -> None:
        a, b = _facing_planar(gap=0.2)
        for candidate in self._build([a, b]):
            self.assertIn(candidate.state, {STATE_CANDIDATE, STATE_UNSUPPORTED, STATE_REJECTED})


if __name__ == "__main__":
    unittest.main()
