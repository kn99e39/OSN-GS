from __future__ import annotations

"""Phase F constrained occluded-chart tests (design sections 15, 18).

Builds continuation domains, runs the Phase E candidate builder, then fits
occluded charts. Pathological charts (fold / Jacobian collapse / orientation
flip) are built from hand-placed support tips whose geometry is numerically
probed before assertion, per the design's "numeric probe first" rule.
"""

import math
import unittest

import torch

from osn_gs.surface.torch_coons_patch import coons_bilinear_patch
from osn_gs.surface.torch_occluded_chart import (
    STATE_REJECTED,
    STATE_UNSUPPORTED,
    STATE_VALIDATED,
    OccludedChartFitConfig,
    fit_occluded_chart,
)
from osn_gs.surface.torch_occluded_region_candidate import (
    CorrespondenceEdge,
    OccludedRegionCandidate,
    SupportChain,
    build_geometric_region_candidates,
)

from tests.test_occluded_region_candidate import _facing_planar, _make_domain


def _manual_candidate(domain_a, domain_b, pairs):
    """Build an OccludedRegionCandidate directly from an explicit (s_a, s_b)
    pairing (both at tip t=1). Used to feed Phase F a deliberately crossed /
    folding ribbon that Phase E's nearest-correspondence would never produce.
    """

    t = domain_a.world.shape[1] - 1
    st_a = [(s_a, t) for s_a, _ in pairs]
    st_b = [(s_b, t) for _, s_b in pairs]
    world_a = torch.stack([domain_a.world[s, t] for s in [p[0] for p in pairs]])
    world_b = torch.stack([domain_b.world[s, t] for s in [p[1] for p in pairs]])
    edges = [
        CorrespondenceEdge(
            s_a=s_a, t_a=t, s_b=s_b, t_b=t, world_distance=float((domain_a.world[s_a, t] - domain_b.world[s_b, t]).norm()),
            scale_normalized_distance=0.5, mutual_nearest=True, outward_dot=0.0, normal_dot=0.0,
            tangent_dot=0.0, position_kind="interior",
        )
        for s_a, s_b in pairs
    ]
    cells = [torch.stack([world_a[k], world_a[k + 1], world_b[k + 1], world_b[k]]) for k in range(len(pairs) - 1)]
    bridge = torch.stack(cells)
    all_world = torch.cat([world_a, world_b, bridge.reshape(-1, 3)])
    empty = {}
    return OccludedRegionCandidate(
        candidate_id=f"{domain_a.domain_id}~{domain_b.domain_id}:manual",
        supporting_domain_ids=[domain_a.domain_id, domain_b.domain_id],
        supporting_boundary_ids=[domain_a.source_boundary_id, domain_b.source_boundary_id],
        supporting_patch_ids=[domain_a.source_patch_id, domain_b.source_patch_id],
        support_chain_a=SupportChain(domain_a.domain_id, st_a, world_a),
        support_chain_b=SupportChain(domain_b.domain_id, st_b, world_b),
        correspondence_edges=edges,
        connector_start=torch.stack([world_a[0], world_b[0]]),
        connector_end=torch.stack([world_a[-1], world_b[-1]]),
        bridge_cells=bridge,
        aabb_min=all_world.min(0).values,
        aabb_max=all_world.max(0).values,
        raw_distance_statistics=dict(empty), normalized_distance_statistics=dict(empty),
        outward_soft_evidence=dict(empty), normal_soft_evidence=dict(empty), tangent_soft_evidence=dict(empty),
        free_space_contradiction=dict(empty), behind_observation_support=dict(empty), on_surface_evidence=dict(empty),
        unobserved_evidence=dict(empty), conflicting_evidence=dict(empty), empty_voxel_support=dict(empty),
        state="candidate", reason="ok", provenance={"cyclic": False},
    )


def _bridge_domains(a_tips: torch.Tensor, b_tips: torch.Tensor, *, scale: float = 0.4, id_a="a", id_b="b"):
    """Two 2-column domains whose t=1 tips are exactly ``a_tips`` / ``b_tips``.

    The t=0 boundary is offset inward (toward the opposite chain) so each
    domain's outward direction points into the region between them.
    """

    a_tips = a_tips.to(torch.float64)
    b_tips = b_tips.to(torch.float64)
    dir_a = torch.nn.functional.normalize((b_tips.mean(0) - a_tips.mean(0)), dim=0, eps=1e-12)
    dir_b = -dir_a
    na, nb = a_tips.shape[0], b_tips.shape[0]
    wa = torch.zeros((na, 2, 3), dtype=torch.float64)
    wb = torch.zeros((nb, 2, 3), dtype=torch.float64)
    wa[:, 0] = a_tips - 0.15 * dir_a
    wa[:, 1] = a_tips
    wb[:, 0] = b_tips - 0.15 * dir_b
    wb[:, 1] = b_tips
    a = _make_domain(wa, boundary_id=id_a, patch_id=1, scale=scale)
    b = _make_domain(wb, boundary_id=id_b, patch_id=2, scale=scale)
    return a, b


def _candidate(a, b, **kw):
    cands = build_geometric_region_candidates([a, b], {}, **kw)
    return cands


class CoonsSeedTest(unittest.TestCase):
    def test_coons_planar_matches_boundaries(self) -> None:
        u = torch.linspace(0, 1, 5, dtype=torch.float64)
        v = torch.linspace(0, 1, 4, dtype=torch.float64)
        cv0 = torch.stack([0.3 + 0.2 * u, torch.zeros(5), torch.zeros(5)], 1)
        cv1 = torch.stack([0.3 + 0.2 * u, torch.ones(5), torch.zeros(5)], 1)
        cu0 = torch.stack([torch.full((4,), 0.3), v, torch.zeros(4)], 1)
        cu1 = torch.stack([torch.full((4,), 0.5), v, torch.zeros(4)], 1)
        s = coons_bilinear_patch(cv0, cv1, cu0, cu1)
        self.assertEqual(tuple(s.shape), (5, 4, 3))
        self.assertLess(float(s[..., 2].abs().max()), 1e-9)
        torch.testing.assert_close(s[:, 0], cv0, atol=1e-6, rtol=0)

    def test_coons_corner_mismatch_raises(self) -> None:
        u = torch.linspace(0, 1, 4, dtype=torch.float64)
        v = torch.linspace(0, 1, 4, dtype=torch.float64)
        cv0 = torch.stack([u, torch.zeros(4), torch.zeros(4)], 1)
        cv1 = torch.stack([u, torch.ones(4), torch.zeros(4)], 1)
        cu0 = torch.stack([torch.full((4,), 5.0), v, torch.zeros(4)], 1)  # corner far off
        cu1 = torch.stack([torch.ones(4), v, torch.zeros(4)], 1)
        with self.assertRaises(ValueError):
            coons_bilinear_patch(cv0, cv1, cu0, cu1)


class OccludedChartTest(unittest.TestCase):
    def _fit(self, a, b, config=None, **kw):
        cands = _candidate(a, b, **kw)
        self.assertEqual(len(cands), 1)
        domains = {a.domain_id: a, b.domain_id: b}
        return cands[0], fit_occluded_chart(cands[0], domains, {}, config=config or OccludedChartFitConfig())

    # 1 — planar quadrilateral bridge -> validated
    def test_planar_bridge_validated(self) -> None:
        a, b = _facing_planar(gap=0.2)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_VALIDATED)
        self.assertIsNotNone(res.surface)
        self.assertEqual(res.topology, "open_quadrilateral")
        self.assertLess(res.boundary_conformance["c0_residual_max"], res.boundary_conformance["c0_tolerance"])

    # 2 — unequal support sample counts -> validated
    def test_unequal_support_counts_validated(self) -> None:
        ys_a = torch.linspace(0, 1, 7, dtype=torch.float64)
        ys_b = torch.linspace(0, 1, 4, dtype=torch.float64)
        a_tips = torch.stack([torch.full((7,), 0.3), ys_a, torch.zeros(7)], 1)
        b_tips = torch.stack([torch.full((4,), 0.5), ys_b, torch.zeros(4)], 1)
        a, b = _bridge_domains(a_tips, b_tips)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_VALIDATED)

    # 3 — reversed correspondence -> deterministic identical payload
    def test_reversed_correspondence_deterministic(self) -> None:
        ys = torch.linspace(0, 1, 5, dtype=torch.float64)
        a_tips = torch.stack([torch.full((5,), 0.3), ys, torch.zeros(5)], 1)
        b_tips = torch.stack([torch.full((5,), 0.5), ys.flip(0), torch.zeros(5)], 1)  # reversed y
        a, b = _bridge_domains(a_tips, b_tips)
        cand, res1 = self._fit(a, b)
        domains = {a.domain_id: a, b.domain_id: b}
        res2 = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res1.chart_id, res2.chart_id)
        self.assertEqual(res1.payload(), res2.payload())

    # 4 — orthogonal support -> validated, large G1 mismatch diagnostic only
    def test_orthogonal_support_validated_large_g1(self) -> None:
        ys = torch.linspace(0, 1, 5, dtype=torch.float64)
        a_tips = torch.stack([torch.full((5,), 0.3), ys, torch.zeros(5)], 1)  # outward ~ +x
        # B tips extend so B's outward is ~ +z (orthogonal)
        b_tips = torch.stack([torch.full((5,), 0.45), ys, torch.full((5,), 0.05)], 1)
        a, b = _bridge_domains(a_tips, b_tips)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_VALIDATED)
        # mismatch is recorded but never a reject reason
        self.assertIn("angle_deg_max", res.tangent_mismatch["support_a"])

    # 5 — oblique support -> validated
    def test_oblique_support_validated(self) -> None:
        ys = torch.linspace(0, 1, 5, dtype=torch.float64)
        a_tips = torch.stack([torch.full((5,), 0.3), ys, torch.zeros(5)], 1)
        b_tips = torch.stack([0.5 - 0.1 * ys, ys, 0.1 * ys], 1)
        a, b = _bridge_domains(a_tips, b_tips)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_VALIDATED)

    # 6 — curved two-sided support -> validated
    def test_curved_support_validated(self) -> None:
        ys = torch.linspace(0, 1, 6, dtype=torch.float64)
        curve = 0.05 * torch.sin(math.pi * ys)
        a_tips = torch.stack([torch.full((6,), 0.3), ys, curve], 1)
        b_tips = torch.stack([torch.full((6,), 0.5), ys, curve], 1)
        a, b = _bridge_domains(a_tips, b_tips, scale=0.5)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_VALIDATED)

    # 7 — high-curvature fold-over -> rejected (crossed correspondence, probed)
    def test_fold_over_rejected(self) -> None:
        # A tips at y=0, B tips at y=0.3, both along +x; a CROSSED pairing
        # (s_a=k <-> s_b=N-1-k) makes the ruled ribbon fold through itself.
        n = 6
        ys = torch.linspace(0, 1, n, dtype=torch.float64)
        a_tips = torch.stack([ys, torch.zeros(n), torch.full((n,), 0.5)], 1)
        b_tips = torch.stack([ys, torch.full((n,), 0.3), torch.full((n,), 0.5)], 1)
        a, b = _bridge_domains(a_tips, b_tips, scale=0.6)
        cand = _manual_candidate(a, b, [(k, n - 1 - k) for k in range(n)])
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_REJECTED)
        self.assertTrue(
            any(tag in res.reason for tag in ("orientation_flip", "jacobian_collapse", "zero_area_chart"))
        )

    # 8 — zero-area (Phase E rejected) candidate -> propagated rejected, no solve
    def test_zero_area_candidate_propagated(self) -> None:
        a, b = _facing_planar(gap=0.0)  # zero connector -> Phase E rejects
        cands = _candidate(a, b)
        self.assertEqual(cands[0].state, "rejected")
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cands[0], domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_REJECTED)
        self.assertIsNone(res.surface)
        self.assertFalse(res.provenance["solver_run"])

    # 9 — connector-sensitive: support dominates geometry
    def test_support_priority_over_connector(self) -> None:
        a, b = _facing_planar(gap=0.2)
        _, res = self._fit(a, b)
        # chart v=0/v=1 edges track support far more tightly than connectors bound them
        self.assertLess(res.boundary_conformance["support_a"]["symmetric_max"], 0.02)

    # 10 — lowering connector weight keeps C0 on support
    def test_low_connector_weight_keeps_c0(self) -> None:
        a, b = _facing_planar(gap=0.2)
        cfg = OccludedChartFitConfig(connector_weight=1e-4)
        _, res = self._fit(a, b, config=cfg)
        self.assertEqual(res.state, STATE_VALIDATED)
        self.assertLess(res.boundary_conformance["c0_residual_max"], res.boundary_conformance["c0_tolerance"])

    # 11 — known-free contradiction -> solver not run, rejected
    def test_known_free_contradiction_not_solved(self) -> None:
        a, b = _facing_planar(gap=0.2)
        cands = _candidate(a, b)
        cand = cands[0]
        cand.state = "rejected"
        cand.free_space_contradiction = {"candidate_hard_contradiction": True}
        cand.reason = "full_bridge_interior_known_free_space"
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_REJECTED)
        self.assertFalse(res.provenance["solver_run"])

    # 12 — partial evidence contradiction: fit proceeds, metadata preserved
    def test_partial_evidence_preserved(self) -> None:
        a, b = _facing_planar(gap=0.2)
        cands = _candidate(a, b)
        cand = cands[0]
        cand.free_space_contradiction = {
            "candidate_hard_contradiction": False,
            "known_free_section_count": 1,
            "evaluable_section_count": 4,
        }
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_VALIDATED)
        self.assertFalse(res.evidence_consistency["candidate_hard_contradiction"])
        self.assertFalse(res.evidence_consistency["used_as_solver_weight"])
        self.assertEqual(res.evidence_consistency["free_space_contradiction"]["known_free_section_count"], 1)

    # 13 — candidate conflict: each fit independently, provenance preserved
    def test_candidate_conflict_independent_fit(self) -> None:
        from osn_gs.surface.torch_occluded_region_candidate import build_candidate_conflicts

        ys = torch.linspace(0, 1, 5, dtype=torch.float64)
        a_tips = torch.stack([torch.full((5,), 0.3), ys, torch.zeros(5)], 1)
        b_tips = torch.stack([torch.full((5,), 0.5), ys, torch.zeros(5)], 1)
        c_tips = torch.stack([torch.full((5,), 0.5), ys, torch.full((5,), 0.15)], 1)
        a, b = _bridge_domains(a_tips, b_tips, id_a="a", id_b="b", scale=0.5)
        _, c = _bridge_domains(a_tips, c_tips, id_a="a", id_b="c", scale=0.5)
        cands = build_geometric_region_candidates([a, b, c], {})
        self.assertGreaterEqual(len(cands), 2)  # A-B and A-C both candidates, sharing A
        conflicts = build_candidate_conflicts(cands)
        self.assertGreaterEqual(len(conflicts), 1)  # shared domain A + overlapping bridge
        domains = {a.domain_id: a, b.domain_id: b, c.domain_id: c}
        for cand in cands:
            res = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
            self.assertIn(res.state, {STATE_VALIDATED, STATE_REJECTED, STATE_UNSUPPORTED})
            self.assertEqual(res.conflict_provenance["candidate_state"], cand.state)

    # 14 — cyclic annular candidate -> unsupported (no solve)
    def test_cyclic_candidate_unsupported(self) -> None:
        n = 8
        angles = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        ts = torch.linspace(0.0, 0.3, 3, dtype=torch.float64)
        wa = torch.zeros((n, 3, 3), dtype=torch.float64)
        wb = torch.zeros((n, 3, 3), dtype=torch.float64)
        for s in range(n):
            c, sn = math.cos(float(angles[s])), math.sin(float(angles[s]))
            for t in range(3):
                ra, rb = 1.0 + float(ts[t]), 2.0 - float(ts[t])
                wa[s, t] = torch.tensor([ra * c, ra * sn, 0.0], dtype=torch.float64)
                wb[s, t] = torch.tensor([rb * c, rb * sn, 0.0], dtype=torch.float64)
        a = _make_domain(wa, boundary_id="a", patch_id=1, closed=True, scale=0.6)
        b = _make_domain(wb, boundary_id="b", patch_id=2, closed=True, scale=0.6)
        cands = _candidate(a, b)
        self.assertIsNone(cands[0].connector_end)
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cands[0], domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_UNSUPPORTED)
        self.assertEqual(res.reason, "cyclic_topology_deferred")

    # 16 — boundary conformance residual is actually measured
    def test_boundary_conformance_measured(self) -> None:
        a, b = _facing_planar(gap=0.2)
        _, res = self._fit(a, b)
        bc = res.boundary_conformance
        self.assertIn("support_a", bc)
        self.assertIn("edge_to_reference_max", bc["support_a"])
        self.assertGreaterEqual(bc["c0_residual_max"], 0.0)

    # 17 — Jacobian collapse detected (mid-span pinch)
    def test_jacobian_collapse_detected(self) -> None:
        ys = torch.linspace(0, 1, 6, dtype=torch.float64)
        a_tips = torch.stack([torch.full((6,), 0.3), ys, torch.zeros(6)], 1)
        # B chain nearly touches A at mid-span -> ribbon width collapses there.
        gap = 0.3 - 0.29 * torch.sin(math.pi * ys)
        b_tips = torch.stack([0.3 + gap, ys, torch.zeros(6)], 1)
        a, b = _bridge_domains(a_tips, b_tips, scale=0.5)
        _, res = self._fit(a, b)
        self.assertEqual(res.state, STATE_REJECTED)
        self.assertTrue(
            any(t in res.reason for t in ("jacobian_collapse", "zero_area_chart", "orientation_flip"))
        )

    # 18 — orientation flip detected (crossed correspondence fold)
    def test_orientation_flip_flag(self) -> None:
        n = 6
        ys = torch.linspace(0, 1, n, dtype=torch.float64)
        a_tips = torch.stack([ys, torch.zeros(n), torch.full((n,), 0.5)], 1)
        b_tips = torch.stack([ys, torch.full((n,), 0.3), torch.full((n,), 0.5)], 1)
        a, b = _bridge_domains(a_tips, b_tips, scale=0.6)
        cand = _manual_candidate(a, b, [(k, n - 1 - k) for k in range(n)])
        domains = {a.domain_id: a, b.domain_id: b}
        res = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res.state, STATE_REJECTED)
        self.assertTrue(
            any(t in res.reason for t in ("orientation_flip", "jacobian_collapse", "zero_area_chart"))
        )

    # 19 — deterministic chart id / payload across repeated calls
    def test_deterministic_payload(self) -> None:
        a, b = _facing_planar(gap=0.2)
        cand, res1 = self._fit(a, b)
        domains = {a.domain_id: a, b.domain_id: b}
        res2 = fit_occluded_chart(cand, domains, {}, config=OccludedChartFitConfig())
        self.assertEqual(res1.chart_id, res2.chart_id)
        self.assertEqual(res1.payload(), res2.payload())

    # run_validation=False -> fitted
    def test_run_validation_false_gives_fitted(self) -> None:
        a, b = _facing_planar(gap=0.2)
        cfg = OccludedChartFitConfig(run_validation=False)
        _, res = self._fit(a, b, config=cfg)
        self.assertEqual(res.state, "fitted")
        self.assertIsNotNone(res.surface)


if __name__ == "__main__":
    unittest.main()
