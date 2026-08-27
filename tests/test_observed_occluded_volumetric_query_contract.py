from __future__ import annotations

"""Worklog 123 -- volumetric frontier query contract closure.

Focused tests for the query-representation layer sitting above the FROZEN
candidate B: historical B invariance, exhaustive event-provenance identity,
the provenance-removal control, the float32-vs-float64 reference comparison,
cross-view aggregation, true-fragmentation endpoint identity, and the
count-vs-weight accounting audit that closes worklog 122's ambiguity.
"""

import ast
import io
import json
import sys
import tokenize
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS_DIR = REPO_ROOT / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import volumetric_query  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    STATE_NON_RELEVANT, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED,
    ViewGeometry, aggregate_global,
)
from observed_occluded.volumetric_query import (  # noqa: E402
    IDENTITY_NOT_APPLIED, IDENTITY_ON_FRONTIER, IDENTITY_REJECTED_STALE,
    VolumetricQueryBank, apply_event_identity, project_queries_float64, reference_side,
)

CUDA_AVAILABLE = torch.cuda.is_available()
QDEPTH_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        from osn_gs.render.torch_surfel_query_depth_diagnostics import get_qdepth_extension
        get_qdepth_extension()
        QDEPTH_AVAILABLE = True
    except Exception:  # pragma: no cover - environment dependent
        QDEPTH_AVAILABLE = False

requires_qdepth = unittest.skipUnless(
    CUDA_AVAILABLE and QDEPTH_AVAILABLE, "CUDA and the diagnostic qdepth build are required"
)

WL122_REPORT = REPO_ROOT / "output/confirmed/122_osn_gs_median_frontier_validation/median_frontier_validation_report.json"
requires_wl122 = unittest.skipUnless(WL122_REPORT.exists(), f"worklog 122 report not present at {WL122_REPORT}")
WL123_REPORT = REPO_ROOT / "output/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract_report.json"
WL123_NPZ = REPO_ROOT / "output/123_osn_gs_volumetric_frontier_query_contract/volumetric_query_contract.npz"
requires_wl123 = unittest.skipUnless(WL123_REPORT.exists(), f"worklog 123 report not present at {WL123_REPORT}")


def _geometry(depth, pixel_index=None, relevant=None) -> ViewGeometry:
    depth = torch.as_tensor(depth, dtype=torch.float32)
    count = int(depth.shape[0])
    relevant = torch.ones(count, dtype=torch.bool) if relevant is None else torch.as_tensor(relevant, dtype=torch.bool)
    index = torch.arange(count, dtype=torch.int64) if pixel_index is None else torch.as_tensor(pixel_index, dtype=torch.int64)
    index = torch.where(relevant, index, torch.full_like(index, -1))
    return ViewGeometry(
        pixel_x=torch.zeros(count), pixel_y=torch.zeros(count),
        pixel_col=index.clone(), pixel_row=torch.zeros(count, dtype=torch.int64),
        pixel_index=index, depth=depth, relevant=relevant,
        relevance_code=torch.where(relevant, torch.zeros(count, dtype=torch.int8), torch.full((count,), 3, dtype=torch.int8)),
    )


def _bank(positions, camera=None, pixel=None, median=None, representative=None, kind=None) -> VolumetricQueryBank:
    positions = torch.as_tensor(positions, dtype=torch.float32)
    count = int(positions.shape[0])
    return VolumetricQueryBank(
        world_position=positions,
        kind=kind or ["test"] * count,
        provenance_camera=np.asarray(camera if camera is not None else [-1] * count, dtype=np.int64),
        provenance_pixel=np.asarray(pixel if pixel is not None else [-1] * count, dtype=np.int64),
        provenance_median_depth=np.asarray(median if median is not None else [np.nan] * count, dtype=np.float32),
        provenance_representative=np.asarray(
            representative if representative is not None else [-1] * count, dtype=np.int64
        ),
    )


# ==========================================================================
# Historical candidate B invariance
# ==========================================================================
class TestHistoricalBInvariance(unittest.TestCase):
    def test_decision_rule_unchanged(self):
        geometry = _geometry([3.0, 4.0, 5.0, 4.0])
        median = torch.tensor([4.0, 4.0, 4.0, 0.0], dtype=torch.float32)
        self.assertEqual(
            [int(v) for v in candidate_b.classify_view(geometry, median)["states"]],
            [STATE_OBSERVED, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED],
        )

    def test_candidate_b_still_declares_no_epsilon(self):
        source = (DEVTOOLS_DIR / "observed_occluded" / "candidate_b_median_depth.py").read_text(encoding="utf-8")
        constants = [
            node.value for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        self.assertEqual([value for value in constants if value != 0.0], [])

    def test_query_layer_introduces_no_tolerance(self):
        """No epsilon, ULP band, nextafter correction or percentage threshold
        may appear in the query-contract module."""

        source = (DEVTOOLS_DIR / "observed_occluded" / "volumetric_query.py").read_text(encoding="utf-8")
        # The module intentionally documents the forbidden mechanisms in its
        # docstring. Remove Python comments/string tokens rather than using a
        # line heuristic that mistakes that documentation for executable code.
        body = "".join(
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )
        for forbidden in ("nextafter", "epsilon", "EPSILON", "tolerance"):
            self.assertNotIn(forbidden, body, f"query layer mentions {forbidden} in code")

    def test_query_layer_calls_the_frozen_classifier(self):
        source = (DEVTOOLS_DIR / "observed_occluded" / "query_contract_synthetics.py").read_text(encoding="utf-8")
        self.assertIn("candidate_b.classify_view", source)


# ==========================================================================
# Event-identity provenance contract
# ==========================================================================
class TestEventIdentityContract(unittest.TestCase):
    def test_provenance_forces_on_frontier_for_its_own_view(self):
        geometry = _geometry([4.0000005])          # a hair behind -> frozen B says OCCLUDED
        median = torch.tensor([4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        self.assertEqual(int(base[0]), STATE_OCCLUDED)
        bank = _bank([[0.0, 0.0, 4.0]], camera=[3], pixel=[0], median=[4.0])
        result = apply_event_identity(3, bank, geometry, median, base)
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)
        self.assertEqual(int(result["identity"][0]), IDENTITY_ON_FRONTIER)
        self.assertEqual(result["applied"], 1)

    def test_provenance_never_touches_another_view(self):
        geometry = _geometry([4.0000005])
        median = torch.tensor([4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 4.0]], camera=[3], pixel=[0], median=[4.0])
        result = apply_event_identity(7, bank, geometry, median, base)
        self.assertEqual(int(result["states"][0]), STATE_OCCLUDED)
        self.assertEqual(int(result["identity"][0]), IDENTITY_NOT_APPLIED)
        self.assertEqual(result["applied"], 0)

    def test_stale_provenance_is_rejected_not_trusted(self):
        geometry = _geometry([4.0000005])
        median = torch.tensor([4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 4.0]], camera=[0], pixel=[0], median=[3.5])   # does not match the renderer
        result = apply_event_identity(0, bank, geometry, median, base)
        self.assertEqual(int(result["states"][0]), STATE_OCCLUDED)
        self.assertEqual(int(result["identity"][0]), IDENTITY_REJECTED_STALE)
        self.assertEqual(result["rejected"], 1)

    def test_provenance_match_is_bitwise_not_approximate(self):
        median_value = np.float32(4.0)
        one_ulp_off = float(np.nextafter(median_value, np.float32(1e9)))
        geometry = _geometry([4.0])
        median = torch.tensor([float(median_value)], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 4.0]], camera=[0], pixel=[0], median=[one_ulp_off])
        result = apply_event_identity(0, bank, geometry, median, base)
        self.assertEqual(int(result["identity"][0]), IDENTITY_REJECTED_STALE)

    def test_provenance_does_not_change_the_world_position(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        bank = _bank(positions, camera=[0], pixel=[0], median=[4.0])
        self.assertTrue(torch.equal(bank.world_position, positions))
        self.assertTrue(torch.equal(bank.without_provenance().world_position, positions))

    def test_provenance_removal_control_falls_back_to_frozen_b(self):
        geometry = _geometry([4.0000005])
        median = torch.tensor([4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 4.0]], camera=[0], pixel=[0], median=[4.0])
        stripped = bank.without_provenance()
        result = apply_event_identity(0, stripped, geometry, median, base)
        self.assertTrue(torch.equal(result["states"], base))
        self.assertEqual(result["applied"], 0)

    def test_provenance_cannot_produce_occluded_or_unresolved(self):
        """The layer may only settle ON_FRONTIER; it can never flip a query to
        OCCLUDED or UNRESOLVED, which would make it a visibility shortcut."""

        geometry = _geometry([1.0, 9.0])
        median = torch.tensor([4.0, 4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0]], camera=[0, 0], pixel=[0, 1], median=[4.0, 4.0])
        result = apply_event_identity(0, bank, geometry, median, base)
        self.assertTrue(bool((result["states"] == STATE_OBSERVED).all()))


# ==========================================================================
# Float64 reference arm
# ==========================================================================
@requires_qdepth
class TestReferenceArm(unittest.TestCase):
    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera

        self.camera = front_camera("cuda")

    def test_reference_projection_matches_float32_on_well_separated_points(self):
        from observed_occluded.shared import project_queries

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, -0.2, 1.0], [-0.5, 0.4, 2.0]], dtype=torch.float32, device="cuda"
        )
        float32 = project_queries(self.camera, positions)
        reference = project_queries_float64(self.camera, positions)
        self.assertTrue(torch.equal(float32.pixel_row, reference["pixel_row"]))
        self.assertTrue(torch.equal(float32.pixel_col, reference["pixel_col"]))
        self.assertTrue(torch.allclose(float32.depth.to(torch.float64), reference["depth"], rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.equal(float32.relevance_code, reference["relevance_code"]))

    def test_reference_side_reproduces_candidate_b_on_identical_inputs(self):
        """The diagnostic recomputation is candidate B's own rule -- fed the
        float32 arm's numbers it must agree exactly."""

        from observed_occluded.shared import project_queries

        positions = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"
        )
        float32 = project_queries(self.camera, positions)
        median = torch.full((64 * 64,), 4.0, dtype=torch.float32, device="cuda")
        frozen = candidate_b.classify_view(float32, median)["states"]
        mirrored = reference_side(
            {
                "depth": float32.depth.to(torch.float64), "relevant": float32.relevant,
                "pixel_index": float32.pixel_index, "pixel_row": float32.pixel_row,
                "pixel_col": float32.pixel_col, "relevance_code": float32.relevance_code,
            },
            median,
        )
        self.assertTrue(torch.equal(frozen, mirrored))

    def test_reference_side_honours_the_sentinel_and_non_relevance(self):
        reference = {
            "depth": torch.tensor([4.0, 4.0], dtype=torch.float64),
            "relevant": torch.tensor([True, False]),
            "pixel_index": torch.tensor([0, -1]),
            "pixel_row": torch.zeros(2, dtype=torch.int64),
            "pixel_col": torch.zeros(2, dtype=torch.int64),
            "relevance_code": torch.tensor([0, 3], dtype=torch.int8),
        }
        states = reference_side(reference, torch.tensor([0.0, 0.0], dtype=torch.float32))
        self.assertEqual(int(states[0]), STATE_UNRESOLVED)
        self.assertEqual(int(states[1]), STATE_NON_RELEVANT)


# ==========================================================================
# Cross-view aggregation stays frozen
# ==========================================================================
class TestCrossViewAggregation(unittest.TestCase):
    def test_identity_in_source_view_yields_global_observed(self):
        states = np.array([[STATE_OCCLUDED, STATE_OBSERVED, STATE_OCCLUDED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)

    def test_no_view_count_rule(self):
        states = np.full((1, 161), STATE_OCCLUDED, dtype=np.int8)
        states[0, 80] = STATE_OBSERVED
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)

    def test_identity_only_changes_the_source_view_column(self):
        geometry = _geometry([4.0000005, 4.0000005])
        median = torch.tensor([4.0, 4.0], dtype=torch.float32)
        base = candidate_b.classify_view(geometry, median)["states"]
        bank = _bank([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]], camera=[0, 5], pixel=[0, 1], median=[4.0, 4.0])
        result = apply_event_identity(0, bank, geometry, median, base)
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)
        self.assertEqual(int(result["states"][1]), int(base[1]))


# ==========================================================================
# Synthetic query contracts Q1-Q5
# ==========================================================================
@requires_qdepth
class TestQueryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from observed_occluded.query_contract_synthetics import run_query_contracts

        cls.results = run_query_contracts(device="cuda")

    def test_q1_exact_event_with_provenance_is_on_frontier(self):
        entry = self.results["Q1_event_with_provenance"]
        self.assertTrue(entry["pass"])
        self.assertEqual(entry["identity_outcome"], "ON_FRONTIER_BY_EVENT_IDENTITY")
        self.assertEqual(entry["source_view_state"], "OBSERVED")

    def test_q2_same_coordinate_without_provenance_is_reported_not_forced(self):
        entry = self.results["Q2_same_coordinate_provenance_removed"]
        self.assertTrue(entry["world_position_identical_to_Q1"])
        self.assertEqual(entry["identity_outcome"], "NO_PROVENANCE_FOR_THIS_VIEW")
        self.assertIn("source_view_state", entry)
        self.assertNotIn("pass", entry)   # deliberately not a pass/fail contract

    def test_q3_q4_provenance_is_irrelevant_far_from_the_frontier(self):
        for key in ("Q3_clearly_camera_side", "Q4_clearly_behind"):
            entry = self.results[key]
            self.assertTrue(entry["pass"])
            self.assertTrue(entry["base_equals_layered"])

    def test_q5_source_identity_preserves_global_observed(self):
        entry = self.results["Q5_event_occluded_elsewhere"]
        self.assertTrue(entry["pass"])
        self.assertIn("OCCLUDED", entry["per_view_layered"])
        self.assertEqual(entry["layered_global"], "OBSERVED")

    def test_contracts_are_deterministic(self):
        from observed_occluded.query_contract_synthetics import run_query_contracts

        again = run_query_contracts(device="cuda")
        self.assertEqual(again["Q1_event_with_provenance"]["pass"], self.results["Q1_event_with_provenance"]["pass"])
        self.assertEqual(
            again["Q2_same_coordinate_provenance_removed"]["signed_margin"],
            self.results["Q2_same_coordinate_provenance_removed"]["signed_margin"],
        )


# ==========================================================================
# Worklog 122 count-vs-weight accounting audit
# ==========================================================================
@requires_wl122
class TestPostMedianCountVsWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.post = json.loads(WL122_REPORT.read_text(encoding="utf-8"))["post_median_accounting"]

    def test_count_and_weight_fractions_are_genuinely_different(self):
        counts = self.post["counts_by_category"]
        weights = self.post["contribution_mass_by_category"]
        count_front = counts["depth_in_front_of_median"] / counts["all"]
        weight_front = weights["depth_in_front_of_median"] / weights["all"]
        self.assertNotAlmostEqual(count_front, weight_front, places=3)

    def test_worklog_122_quoted_27_65_percent_is_the_weight_fraction(self):
        weights = self.post["contribution_mass_by_category"]
        weight_front = weights["depth_in_front_of_median"] / weights["all"]
        self.assertAlmostEqual(weight_front, 0.27646, places=4)

    def test_historical_28_26_percent_chain_is_mathematically_valid(self):
        """Both factors are contribution-WEIGHT fractions, so the product is the
        weight fraction of total contribution behind the median. The direct
        recomputation must reproduce it exactly."""

        weights = self.post["contribution_mass_by_category"]
        total = self.post["total_accepted_contribution_mass"]
        post_fraction = self.post["post_median_contribution_mass"] / total
        behind_fraction = weights["depth_at_or_behind_median"] / weights["all"]
        direct = weights["depth_at_or_behind_median"] / total
        self.assertAlmostEqual(post_fraction * behind_fraction, direct, places=12)
        self.assertAlmostEqual(direct, 0.28258, places=4)

    def test_category_totals_are_internally_consistent(self):
        counts = self.post["counts_by_category"]
        self.assertEqual(counts["all"], counts["depth_in_front_of_median"] + counts["depth_at_or_behind_median"])
        self.assertEqual(counts["all"], counts["same_component"] + counts["cross_component"] + counts["unresolved_component"])


# ==========================================================================
# Completed Worklog 123 artifact regression
# ==========================================================================
@requires_wl123
class TestGeneratedWorklog123Artifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(WL123_REPORT.read_text(encoding="utf-8"))

    def test_exhaustive_event_identity_and_historical_replay(self):
        result = self.report["exact_event_identity"]
        self.assertEqual(result["total_source_median_events"], 43817760)
        self.assertEqual(result["historical_float32_source_contradiction"], 8157322)
        self.assertEqual(result["provenance_preserved_contradiction"], 0)
        self.assertEqual(result["provenance_applied"], 43817760)
        self.assertTrue(result["worklog_122_reference"]["reproduces_worklog_122_corpus"])
        self.assertTrue(result["worklog_122_reference"]["reproduces_worklog_122_contradiction_count"])

    def test_generic_reference_audit_excludes_exact_event_arm(self):
        result = self.report["arbitrary_3d_stability"]["generic_arbitrary_3D_queries"]
        self.assertEqual(result["query_view_pairs"], 2938572)
        self.assertEqual(result["relevant_pairs"], 1590240)
        self.assertEqual(result["float32_vs_reference_state_disagreement"], 1118)
        self.assertEqual(result["OBSERVED_OCCLUDED_disagreement"], 1118)
        self.assertNotIn("P1_RENDERER_EVENT_ANCHOR", result["per_kind_relevant_pairs"])

    def test_generic_disagreement_attribution_is_recorded(self):
        result = self.report["arbitrary_3d_stability"]["disagreement_attribution"]
        self.assertEqual(result["relevant_projected_pixel_changes"], 40)
        self.assertEqual(result["state_disagreements_same_projected_pixel"], 1112)
        self.assertEqual(result["state_disagreements_with_projected_pixel_change"], 6)

    def test_cross_view_identity_rescues_exact_historical_global_contradictions(self):
        result = self.report["cross_view_replay"]
        self.assertEqual(result["anchors"], 3400)
        self.assertEqual(result["global_OCCLUDED_without_provenance"], 19)
        self.assertEqual(result["global_OCCLUDED_with_provenance"], 0)
        self.assertEqual(result["anchors_rescued_by_event_identity"], 19)
        self.assertEqual(result["identity_applied_on_source_view"], 3400)

    def test_true_fragmentation_endpoint_identity_and_midpoint_control(self):
        result = self.report["true_fragmentation_replay"]["by_query_kind"]
        for key in ("T1_TOPOLOGY_GAP_ENDPOINT_A", "T1_TOPOLOGY_GAP_ENDPOINT_B"):
            self.assertEqual(result[key]["with_provenance"]["counts"]["OCCLUDED"], 0)
            self.assertEqual(result[key]["with_provenance"]["counts"]["OBSERVED"], 300)
            self.assertTrue(result[key]["without_provenance_matches_worklog_121"])
        midpoint = result["T1_TOPOLOGY_GAP_MIDPOINT"]
        self.assertEqual(midpoint["without_provenance"], midpoint["with_provenance"])
        self.assertEqual(midpoint["with_provenance"]["counts"]["OBSERVED"], 300)

    def test_pairwise_generic_reference_fields_cover_all_generic_pairs(self):
        self.assertTrue(WL123_NPZ.exists())
        with np.load(WL123_NPZ, allow_pickle=True) as payload:
            self.assertEqual(payload["generic_query_indices"].shape, (18252,))
            self.assertEqual(payload["generic_float32_pixel_row"].shape, (18252, 161))
            self.assertEqual(payload["generic_reference_query_depth"].shape, (18252, 161))
            self.assertEqual(payload["generic_float32_stored_median_depth"].shape, (18252, 161))


if __name__ == "__main__":
    unittest.main()