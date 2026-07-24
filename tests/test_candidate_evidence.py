from __future__ import annotations

"""Phase E evidence-validation tests (design sections 8-10).

Uses a minimal hand-built `ObservationEvidence`: a down-+z camera whose
world_view_transform is identity (view depth == world z) and whose depth image
is a constant plane, so a bridge sample's classification is fully determined by
its world z relative to the image depth. This gives exact control over which
bridge sections/samples read as free / behind / on-surface / conflicting.
"""

import types
import unittest

import torch

from osn_gs.surface.torch_candidate_evidence import validate_candidate_observation_evidence
from osn_gs.surface.torch_observation_evidence import CameraViewEvidence, ObservationEvidence
from osn_gs.surface.torch_occluded_region_candidate import (
    STATE_CANDIDATE,
    STATE_REJECTED,
    build_geometric_region_candidates,
)

from tests.test_occluded_region_candidate import _make_domain


def _camera(observed_depth: float, *, image: int = 8, a: float = 0.5, index: int = 0) -> CameraViewEvidence:
    wvt = torch.eye(4, dtype=torch.float64)
    fpt = torch.tensor(
        [[a, 0.0, 0.0, 0.0], [0.0, a, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    view_depth = torch.full((image, image), float(observed_depth), dtype=torch.float64)
    valid = torch.ones((image, image), dtype=torch.bool)
    return CameraViewEvidence(
        camera_index=index,
        image_height=image,
        image_width=image,
        world_view_transform=wvt,
        full_proj_transform=fpt,
        view_depth=view_depth,
        valid_depth_mask=valid,
        coverage_alpha=None,
        backend_source="fallback",
        coverage_kind="alpha_fraction",
        depth_kind="direct_linear",
        depth_is_approximate=False,
    )


def _evidence(cameras: list[CameraViewEvidence], depth_epsilon: float = 1e-2) -> ObservationEvidence:
    return ObservationEvidence(
        views=cameras,
        near=1e-3,
        far=1e6,
        depth_epsilon=depth_epsilon,
        topology_version="t",
        camera_set_version="c",
    )


def _bridge_pair(za, zb, *, gap: float = 0.2, scale: float = 0.3, x_a: float = 0.4):
    """Two 2-column domains whose matched tips are at (x_a, y, za[s]) and
    (x_a+gap, y, zb[s]); the candidate's cross-sections run between them."""

    n_s = len(za)
    ys = torch.linspace(0.0, 1.0, n_s, dtype=torch.float64)
    x_b = x_a + gap
    wa = torch.zeros((n_s, 2, 3), dtype=torch.float64)
    wb = torch.zeros((n_s, 2, 3), dtype=torch.float64)
    for s in range(n_s):
        wa[s, 0] = torch.tensor([x_a - 0.1, ys[s], za[s]], dtype=torch.float64)
        wa[s, 1] = torch.tensor([x_a, ys[s], za[s]], dtype=torch.float64)
        wb[s, 0] = torch.tensor([x_b + 0.1, ys[s], zb[s]], dtype=torch.float64)
        wb[s, 1] = torch.tensor([x_b, ys[s], zb[s]], dtype=torch.float64)
    a = _make_domain(wa, boundary_id="a", patch_id=1, scale=scale)
    b = _make_domain(wb, boundary_id="b", patch_id=2, scale=scale)
    return a, b


class CandidateEvidenceTest(unittest.TestCase):
    def _candidate(self, za, zb, **kw):
        a, b = _bridge_pair(za, zb, **kw)
        candidates = build_geometric_region_candidates([a, b], {})
        self.assertEqual(len(candidates), 1, "fixture must build exactly one geometric candidate")
        self.assertEqual(candidates[0].state, STATE_CANDIDATE)
        return candidates

    # None evidence leaves state and evidence-evaluated flag untouched.
    def test_none_evidence_is_noop_on_state(self) -> None:
        candidates = self._candidate([0.5] * 4, [0.5] * 4)
        out = validate_candidate_observation_evidence(candidates, None)
        self.assertEqual(out[0].state, STATE_CANDIDATE)
        self.assertFalse(out[0].free_space_contradiction["evaluated"])

    # 17 — full bridge-interior known-free contradiction -> rejected
    def test_full_free_contradiction_rejects(self) -> None:
        candidates = self._candidate([0.5] * 4, [0.5] * 4)
        ev = _evidence([_camera(observed_depth=1.0)])
        out = validate_candidate_observation_evidence(candidates, ev)
        self.assertEqual(out[0].state, STATE_REJECTED)
        self.assertTrue(out[0].free_space_contradiction["candidate_hard_contradiction"])
        self.assertIn("full_bridge_interior_known_free_space", out[0].reason)

    # 18 — partial free contradiction preserved
    def test_partial_free_contradiction_preserved(self) -> None:
        # sections 0,1 free (z=0.5 < 1.0), sections 2,3 behind (z=1.5 > 1.0)
        candidates = self._candidate([0.5, 0.5, 1.5, 1.5], [0.5, 0.5, 1.5, 1.5], scale=0.8)
        ev = _evidence([_camera(observed_depth=1.0)])
        out = validate_candidate_observation_evidence(candidates, ev)
        self.assertEqual(out[0].state, STATE_CANDIDATE)
        fc = out[0].free_space_contradiction
        self.assertFalse(fc["candidate_hard_contradiction"])
        self.assertGreater(fc["known_free_section_count"], 0)
        self.assertLess(fc["known_free_section_count"], fc["evaluable_section_count"])
        self.assertGreater(out[0].behind_observation_support["behind_support_count"], 0)

    # 19 — support endpoint on-surface must not block interior reject
    def test_endpoint_on_surface_does_not_block_reject(self) -> None:
        # endpoint A z=1.0 == observed depth (on-surface); interior all < 1.0 (free)
        candidates = self._candidate([1.0] * 4, [0.9] * 4)
        ev = _evidence([_camera(observed_depth=1.0)], depth_epsilon=1e-2)
        out = validate_candidate_observation_evidence(candidates, ev)
        self.assertEqual(out[0].state, STATE_REJECTED)
        self.assertGreater(out[0].on_surface_evidence["endpoint_on_surface_count"], 0)
        self.assertEqual(out[0].on_surface_evidence["interior_on_surface_count"], 0)

    # 20 — no interior evidence is insufficient, not a reject
    def test_no_interior_evidence_is_insufficient(self) -> None:
        candidates = self._candidate([0.5] * 4, [0.5] * 4)
        ev = _evidence([_camera(observed_depth=1.0, a=10.0)])  # everything projects out of frame
        out = validate_candidate_observation_evidence(candidates, ev)
        self.assertEqual(out[0].state, STATE_CANDIDATE)
        fc = out[0].free_space_contradiction
        self.assertGreater(fc["insufficient_section_count"], 0)
        self.assertFalse(fc["candidate_hard_contradiction"])

    # 21 — conflicting evidence preserved, not used to approve/reject
    def test_conflicting_evidence_preserved(self) -> None:
        candidates = self._candidate([0.5] * 4, [0.5] * 4)
        # cam0 sees free (obs 2.0 > 0.5), cam1 sees behind (obs 0.2 < 0.5) -> conflict
        ev = _evidence([_camera(observed_depth=2.0, index=0), _camera(observed_depth=0.2, index=1)])
        out = validate_candidate_observation_evidence(candidates, ev)
        self.assertEqual(out[0].state, STATE_CANDIDATE)
        self.assertGreater(out[0].conflicting_evidence["interior_conflicting_count"], 0)
        self.assertFalse(out[0].conflicting_evidence["used_as_evidence"])

    # 22 — empty voxel no_observed_support does not promote/reject
    def test_empty_voxel_non_promotion(self) -> None:
        candidates = self._candidate([1.5] * 4, [1.5] * 4)  # behind -> not a free contradiction
        ev = _evidence([_camera(observed_depth=1.0)])

        def _query(_min, _max):
            return types.SimpleNamespace(support="no_observed_support", overlapping_empty_leaf_ids=["r0", "r3"])

        out = validate_candidate_observation_evidence(candidates, ev, empty_voxel_query=_query)
        self.assertEqual(out[0].state, STATE_CANDIDATE)
        self.assertEqual(out[0].empty_voxel_support["support"], "no_observed_support")
        self.assertFalse(out[0].empty_voxel_support["used_as_evidence"])
        self.assertEqual(out[0].empty_voxel_support["overlapping_empty_leaf_ids"], ["r0", "r3"])

    # validate does not duplicate candidates
    def test_validation_preserves_candidate_count(self) -> None:
        candidates = self._candidate([0.5] * 4, [0.5] * 4)
        out = validate_candidate_observation_evidence(candidates, _evidence([_camera(observed_depth=1.0)]))
        self.assertEqual(len(out), len(candidates))


if __name__ == "__main__":
    unittest.main()
