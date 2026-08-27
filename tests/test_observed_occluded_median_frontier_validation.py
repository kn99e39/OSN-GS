from __future__ import annotations

"""Worklog 122 -- renderer-defined median surface frontier validation.

Focused tests for the candidate B frontier diagnostics: historical B decision
invariance, frontier self-closure on known geometry, ULP/round-trip attribution,
exhaustive post-median contributor accounting (including completeness against the
uncapped accepted-contributor count), same/different component attribution,
cross-view disocclusion, the S1-S5 known-geometry contracts, deterministic
worklog 121 true-fragmentation replay, and mandatory canonical-output
exact-equivalence for the worklog 122 additive sibling outputs.
"""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS_DIR = REPO_ROOT / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import frontier_validation  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    STATE_NON_RELEVANT, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED,
    ViewGeometry, aggregate_global,
)

CUDA_AVAILABLE = torch.cuda.is_available()
QDEPTH_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        from osn_gs.render.torch_surfel_query_depth_diagnostics import (
            MAX_QUERY_SLOTS, get_qdepth_extension, render_with_query_depth_probe,
        )
        get_qdepth_extension()
        QDEPTH_AVAILABLE = True
    except Exception:  # pragma: no cover - environment dependent
        QDEPTH_AVAILABLE = False

requires_qdepth = unittest.skipUnless(
    CUDA_AVAILABLE and QDEPTH_AVAILABLE,
    "CUDA and the diagnostic diff_surfel_rasterization_qdepth build are required",
)

WL121_NPZ = REPO_ROOT / "output/confirmed/121_osn_gs_observed_occluded_value_space/value_space_supplemental_bank.npz"
requires_wl121 = unittest.skipUnless(WL121_NPZ.exists(), f"worklog 121 artifact not present at {WL121_NPZ}")


def _geometry(depth, relevant=None) -> ViewGeometry:
    depth = torch.as_tensor(depth, dtype=torch.float32)
    count = int(depth.shape[0])
    relevant = torch.ones(count, dtype=torch.bool) if relevant is None else torch.as_tensor(relevant, dtype=torch.bool)
    index = torch.where(relevant, torch.arange(count, dtype=torch.int64), torch.full((count,), -1, dtype=torch.int64))
    return ViewGeometry(
        pixel_x=torch.zeros(count), pixel_y=torch.zeros(count),
        pixel_col=index.clone(), pixel_row=torch.zeros(count, dtype=torch.int64),
        pixel_index=index, depth=depth, relevant=relevant,
        relevance_code=torch.where(relevant, torch.zeros(count, dtype=torch.int8), torch.full((count,), 3, dtype=torch.int8)),
    )


# ==========================================================================
# Historical candidate B invariance
# ==========================================================================
class TestHistoricalBInvariance(unittest.TestCase):
    def test_decision_rule_is_unchanged(self):
        geometry = _geometry([3.0, 4.0, 5.0, 4.0])
        median = torch.tensor([4.0, 4.0, 4.0, 0.0], dtype=torch.float32)
        states = candidate_b.classify_view(geometry, median)["states"]
        self.assertEqual(
            [int(v) for v in states],
            [STATE_OBSERVED, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED],
        )

    def test_no_tolerance_exists_anywhere_in_candidate_b(self):
        """A single ULP above the median must still be OCCLUDED -- worklog 122
        must not have smuggled in an epsilon."""

        median_value = np.float32(4.0)
        just_above = np.nextafter(median_value, np.float32(1e9))
        geometry = _geometry([float(just_above)])
        states = candidate_b.classify_view(geometry, torch.tensor([float(median_value)], dtype=torch.float32))["states"]
        self.assertEqual(int(states[0]), STATE_OCCLUDED)
        self.assertEqual(int(frontier_validation.float32_ulp_distance(np.float32([just_above]), np.float32([median_value]))[0]), 1)

    def test_candidate_b_source_declares_no_epsilon(self):
        source = (DEVTOOLS_DIR / "observed_occluded" / "candidate_b_median_depth.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        numeric_constants = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        self.assertEqual([value for value in numeric_constants if value not in (0.0,)], [])

    def test_frontier_module_never_assigns_a_state(self):
        source = (DEVTOOLS_DIR / "observed_occluded" / "frontier_validation.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                dumped = ast.dump(node)
                self.assertNotIn("STATE_OCCLUDED", dumped.split("value=")[0])
        self.assertIn("candidate_b.classify_view", source)


# ==========================================================================
# ULP / round-trip attribution
# ==========================================================================
class TestUlpAttribution(unittest.TestCase):
    def test_ulp_distance_is_exact_for_adjacent_floats(self):
        base = np.float32(7.25)
        for steps in (0, 1, 2, 5):
            other = base
            for _ in range(steps):
                other = np.nextafter(other, np.float32(1e9))
            distance = frontier_validation.float32_ulp_distance(np.float32([other]), np.float32([base]))
            self.assertEqual(int(distance[0]), steps)

    def test_ulp_distance_is_symmetric(self):
        a = np.float32([1.0, 100.0, 0.001])
        b = np.float32([np.nextafter(np.float32(1.0), np.float32(2.0)), 100.0, 0.001])
        self.assertTrue(np.array_equal(
            frontier_validation.float32_ulp_distance(a, b),
            frontier_validation.float32_ulp_distance(b, a),
        ))

    def test_cause_codes_cover_every_outcome(self):
        self.assertEqual(
            set(frontier_validation.CAUSE_NAMES),
            {
                frontier_validation.CAUSE_CLOSED,
                frontier_validation.CAUSE_PIXEL_REASSIGNMENT,
                frontier_validation.CAUSE_ROUNDTRIP_1ULP,
                frontier_validation.CAUSE_ROUNDTRIP_FEW_ULP,
                frontier_validation.CAUSE_ROUNDTRIP_LARGE,
                frontier_validation.CAUSE_NO_VALID_MEDIAN_AT_REPROJECTION,
                frontier_validation.CAUSE_NON_RELEVANT,
            },
        )


# ==========================================================================
# Frontier self-closure on known geometry
# ==========================================================================
@requires_qdepth
class TestFrontierSelfClosure(unittest.TestCase):
    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.0, 0.02, 0.04, 0.06], device="cuda")
        self.camera = front_camera("cuda")
        with torch.no_grad():
            rotation = self.model.get_rotation_matrix.detach()
            scaling = self.model.get_scaling.detach()
        self.geometry_args = (
            self.model.get_xyz.detach(), rotation[:, :, 0].contiguous(), rotation[:, :, 1].contiguous(),
            scaling[:, 0].contiguous(), scaling[:, 1].contiguous(),
        )

    def test_every_median_event_closes_on_a_flat_opaque_surface(self):
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=None)
        accumulator = frontier_validation.ClosureAccumulator()
        frontier_validation.evaluate_frontier_closure_for_view(
            0, self.camera, package, *self.geometry_args, accumulator,
        )
        summary = accumulator.summary()
        self.assertGreater(summary["total_median_events_tested"], 0)
        self.assertEqual(summary["source_pixel_preserved"], summary["total_median_events_tested"])
        self.assertEqual(summary["cause_counts"]["RASTER_PIXEL_REASSIGNMENT"], 0)
        self.assertEqual(summary["cause_counts"]["ROUND_TRIP_ABOVE_8_ULP"], 0)

    def test_exact_identity_representation_never_contradicts(self):
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=None)
        accumulator = frontier_validation.ClosureAccumulator()
        frontier_validation.evaluate_frontier_closure_for_view(
            0, self.camera, package, *self.geometry_args, accumulator,
        )
        identity = accumulator.summary()["exact_identity_representation"]
        self.assertGreater(identity["events_tested"], 0)
        self.assertEqual(identity["contradictions"], 0)
        self.assertEqual(identity["OBSERVED"], identity["events_tested"])

    def test_any_contradiction_is_a_round_trip_not_a_reassignment(self):
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=None)
        accumulator = frontier_validation.ClosureAccumulator()
        frontier_validation.evaluate_frontier_closure_for_view(
            0, self.camera, package, *self.geometry_args, accumulator,
        )
        causes = accumulator.summary()["cause_counts"]
        contradictions = accumulator.summary()["closure_contradiction_count"]
        round_trip = causes["ROUND_TRIP_1_ULP"] + causes["ROUND_TRIP_2_TO_8_ULP"] + causes["ROUND_TRIP_ABOVE_8_ULP"]
        self.assertEqual(round_trip, contradictions)


# ==========================================================================
# Post-median contributor accounting
# ==========================================================================
@requires_qdepth
class TestPostMedianAccounting(unittest.TestCase):
    """The 30-layer alpha-0.3 fixture has an exactly known split: T = 0.7^n, so
    T > 0.5 holds only for the first two acceptances and every later accepted
    contributor is post-median."""

    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.05 * i for i in range(30)], opacity=0.3, device="cuda")
        self.camera = front_camera("cuda")
        self.centre = 32

    def _package(self, component=None, representative=None):
        return render_with_query_depth_probe(
            self.camera, self.model, query_depths=None,
            primitive_component=component, primitive_representative_class=representative,
        )

    def test_accounting_is_complete_against_the_uncapped_accepted_count(self):
        package = self._package()
        accepted = int(package["contrib_count"][self.centre, self.centre])
        post = int(package["post_median_counts"][self.centre, self.centre, 0])
        # Exactly two acceptances have T > 0.5 (T = 1.0 then 0.7).
        self.assertEqual(accepted - post, 2)

    def test_category_totals_are_internally_consistent(self):
        component = torch.arange(len(self.model), dtype=torch.int32, device="cuda") // 10
        representative = torch.ones((len(self.model),), dtype=torch.int32, device="cuda")
        package = self._package(component, representative)
        counts = package["post_median_counts"][self.centre, self.centre].tolist()
        self.assertEqual(counts[0], counts[1] + counts[2] + counts[3])
        self.assertEqual(counts[0], counts[4] + counts[5] + counts[6])

    def test_same_and_different_component_attribution(self):
        # Components: surfels 0-9 -> 0, 10-19 -> 1, 20-29 -> 2. The median
        # representative is surfel 1 (second acceptance), so surfels 2..9 are
        # same-component and everything from 10 up is cross-component.
        component = torch.arange(len(self.model), dtype=torch.int32, device="cuda") // 10
        package = self._package(component)
        counts = package["post_median_counts"][self.centre, self.centre].tolist()
        self.assertEqual(int(package["representative_id"][self.centre, self.centre]), 1)
        self.assertEqual(counts[1], 8)
        self.assertEqual(counts[2], counts[0] - 8)

    def test_unresolved_component_is_counted_separately(self):
        component = torch.full((len(self.model),), -1, dtype=torch.int32, device="cuda")
        package = self._package(component)
        counts = package["post_median_counts"][self.centre, self.centre].tolist()
        self.assertEqual(counts[3], counts[0])
        self.assertEqual(counts[1], 0)
        self.assertEqual(counts[2], 0)

    def test_representative_class_attribution(self):
        representative = torch.zeros((len(self.model),), dtype=torch.int32, device="cuda")
        representative[5] = 2
        representative[6] = 1
        package = self._package(representative=representative)
        counts = package["post_median_counts"][self.centre, self.centre].tolist()
        self.assertEqual(counts[4], 1)
        self.assertEqual(counts[5], 1)
        self.assertEqual(counts[6], counts[0] - 2)

    def test_contribution_mass_is_bounded_by_the_total(self):
        package = self._package()
        post_mass = float(package["post_median_weights"][self.centre, self.centre, 0])
        total = float(package["total_accepted_weight"][self.centre, self.centre])
        self.assertGreater(total, 0.0)
        self.assertLessEqual(post_mass, total + 1e-6)
        # `total_accepted_weight` is the canonical accumulated alpha, `1 - T`,
        # which the kernel already writes to ALPHA_OFFSET = 1 of `out_others`
        # (see the vendored auxiliary.h). This pins the new aggregate to a
        # quantity the canonical renderer computes independently.
        alpha = float(package["out_others"][1][self.centre, self.centre])
        self.assertAlmostEqual(total, alpha, places=4)

    def test_depth_offsets_are_behind_the_median_on_a_monotone_stack(self):
        package = self._package()
        stats = package["post_median_depth_stats"][self.centre, self.centre].tolist()
        self.assertGreater(stats[1], 0.0)   # min offset already behind the median
        self.assertGreaterEqual(stats[2], stats[1])

    def test_accumulator_matches_a_hand_computed_single_pixel(self):
        component = torch.zeros((len(self.model),), dtype=torch.int32, device="cuda")
        package = self._package(component)
        accumulator = frontier_validation.PostMedianAccumulator()
        accumulator.accumulate(package, package["representative_id"].reshape(-1).to(torch.int64), None, 0)
        summary = accumulator.summary()
        self.assertEqual(summary["counts_by_category"]["all"], int(package["post_median_counts"][..., 0].sum()))
        self.assertAlmostEqual(
            summary["post_median_contribution_mass"],
            float(package["post_median_weights"][..., 0].sum()), places=2,
        )


# ==========================================================================
# Mandatory canonical-output equivalence for the additive sibling outputs
# ==========================================================================
@requires_qdepth
class TestCanonicalEquivalenceUnderWorklog122Additions(unittest.TestCase):
    CANONICAL_KEYS = (
        "render", "out_others", "radii", "representative_id", "forward_accepted",
        "contrib_ids", "contrib_post_median", "contrib_count",
        "median_rho3d", "median_rho2d", "median_s_u", "median_s_v",
    )
    WORKLOG_121_KEYS = (
        "query_T", "query_terminated", "query_reached", "query_prefix_count",
        "query_resolution_depth", "query_termination_alpha", "query_late_front_count",
        "pixel_inversion_count", "pixel_max_backward_jump",
    )

    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.05 * i for i in range(30)], opacity=0.3, device="cuda")
        self.camera = front_camera("cuda")

    def test_canonical_outputs_match_the_worklog_107_build(self):
        from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative

        reference = render_with_pixel_representative(self.camera, self.model)
        component = torch.arange(len(self.model), dtype=torch.int32, device="cuda") // 7
        representative = torch.ones((len(self.model),), dtype=torch.int32, device="cuda")
        for kwargs in (
            {},
            {"primitive_component": component},
            {"primitive_component": component, "primitive_representative_class": representative},
        ):
            package = render_with_query_depth_probe(self.camera, self.model, query_depths=None, **kwargs)
            for key in self.CANONICAL_KEYS:
                self.assertTrue(torch.equal(reference[key], package[key]), f"{key} changed with {sorted(kwargs)}")

    def test_worklog_121_probe_outputs_are_unchanged_by_the_worklog_122_inputs(self):
        query = torch.zeros((64, 64, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[32, 32, 0] = 4.60
        query[32, 32, 1] = 6.00
        component = torch.arange(len(self.model), dtype=torch.int32, device="cuda") // 7
        without = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        with_provenance = render_with_query_depth_probe(
            self.camera, self.model, query_depths=query, primitive_component=component,
        )
        for key in self.WORKLOG_121_KEYS:
            self.assertTrue(torch.equal(without[key], with_provenance[key]), f"{key} perturbed")

    def test_disabled_provenance_leaves_those_categories_empty(self):
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=None)
        counts = package["post_median_counts"][32, 32].tolist()
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[1] + counts[2] + counts[3], 0)
        self.assertEqual(counts[4] + counts[5] + counts[6], 0)


# ==========================================================================
# Cross-view disocclusion
# ==========================================================================
class TestCrossViewDisocclusion(unittest.TestCase):
    def test_hidden_in_one_view_but_observed_in_another_stays_globally_observed(self):
        states = np.array([[STATE_OCCLUDED, STATE_OBSERVED, STATE_OCCLUDED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)

    def test_hidden_in_every_relevant_view_is_globally_occluded(self):
        states = np.array([[STATE_OCCLUDED, STATE_NON_RELEVANT, STATE_OCCLUDED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OCCLUDED)

    def test_no_view_count_threshold(self):
        states = np.full((1, 200), STATE_OCCLUDED, dtype=np.int8)
        states[0, 137] = STATE_OBSERVED
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)


@requires_qdepth
class TestSyntheticFrontierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from observed_occluded.frontier_synthetic_contracts import run_frontier_contracts

        cls.results = run_frontier_contracts(device="cuda")

    def test_s1_single_exposed_surface(self):
        self.assertTrue(self.results["S1"]["pass"])
        states = [q["actual_global"] for q in self.results["S1"]["queries"]]
        self.assertEqual(states, ["OBSERVED", "OBSERVED", "OBSERVED", "OCCLUDED", "OCCLUDED"])

    def test_s2_fully_hidden_rear_surface(self):
        self.assertTrue(self.results["S2"]["pass"])
        for query in self.results["S2"]["queries"]:
            self.assertEqual(query["actual_global"], "OCCLUDED")

    def test_s3_cross_view_disocclusion(self):
        entry = self.results["S3"]
        self.assertTrue(entry["pass"])
        self.assertIn("OCCLUDED", entry["per_view"])
        self.assertIn("OBSERVED", entry["per_view"])
        self.assertEqual(entry["actual_global"], "OBSERVED")

    def test_s4_frontier_stays_inside_the_physical_surface(self):
        entry = self.results["S4"]
        self.assertTrue(entry["frontier_inside_physical_surface_span"])
        self.assertEqual(entry["post_median_cross_component_share"], 0.0)
        self.assertAlmostEqual(entry["post_median_same_component_share"], 1.0, places=6)

    def test_s4_category_widths_match_the_cuda_layout(self):
        """Regression guard: the synthetic modules once reshaped the (H, W, 10)
        post-median aggregate to width 8, which silently misaligned every
        category. Every reported category must be present and consistent."""

        entry = self.results["S4"]
        counts = entry["post_median_counts"]
        self.assertEqual(len(counts), len(frontier_validation.POST_MEDIAN_CATEGORIES))
        self.assertEqual(counts["all"], counts["same_component"] + counts["cross_component"] + counts["unresolved_component"])
        self.assertEqual(counts["all"], counts["depth_in_front_of_median"] + counts["depth_at_or_behind_median"])
        # 12 splats at alpha ~= 0.25: T = 0.75^n stays above 0.5 for n = 0, 1, 2,
        # so exactly three acceptances are pre-median and nine are post-median.
        self.assertEqual(entry["accepted_contributors"], 12)
        self.assertEqual(counts["all"], 9)

    def test_s5_places_the_frontier_on_the_near_visible_layer(self):
        entry = self.results["S5"]
        self.assertTrue(entry["median_on_near_layer"])
        probes = {p["label"]: p["global"] for p in entry["probes"]}
        self.assertEqual(probes["in_front_of_near_layer"], "OBSERVED")
        self.assertEqual(probes["behind_rear_layer"], "OCCLUDED")

    def test_translucent_fixture_is_labelled_out_of_scope(self):
        entry = self.results["OUT_OF_SCOPE_translucent"]
        self.assertIn("OUT-OF-SCOPE", entry["status"])

    def test_contracts_are_deterministic(self):
        from observed_occluded.frontier_synthetic_contracts import run_frontier_contracts

        again = run_frontier_contracts(device="cuda")
        self.assertEqual(again["S1"]["pass"], self.results["S1"]["pass"])
        self.assertEqual(again["S5"]["median_depth"], self.results["S5"]["median_depth"])


@requires_wl121
class TestWorklog121FragmentationArtifact(unittest.TestCase):
    def test_stored_contexts_have_the_historical_shape_and_gating(self):
        stored = np.load(WL121_NPZ, allow_pickle=True)
        gating = stored["context_gating_reason"]
        self.assertEqual(int(gating.shape[0]), 300)
        self.assertEqual(int((gating == 0).sum()), 288)
        self.assertEqual(int((gating == 1).sum()), 12)
        self.assertEqual(int((gating == 2).sum()), 0)

    def test_stored_endpoint_states_match_the_worklog_121_report(self):
        stored = np.load(WL121_NPZ, allow_pickle=True)
        kind = stored["kind"]
        global_b = stored["global_B"]
        endpoint_a = global_b[kind == "T1_TOPOLOGY_GAP_ENDPOINT_A"]
        endpoint_b = global_b[kind == "T1_TOPOLOGY_GAP_ENDPOINT_B"]
        midpoint = global_b[kind == "T1_TOPOLOGY_GAP_MIDPOINT"]
        self.assertEqual((int((endpoint_a == STATE_OBSERVED).sum()), int((endpoint_a == STATE_OCCLUDED).sum())), (290, 10))
        self.assertEqual((int((endpoint_b == STATE_OBSERVED).sum()), int((endpoint_b == STATE_OCCLUDED).sum())), (296, 4))
        self.assertEqual(int((midpoint == STATE_OBSERVED).sum()), 300)


if __name__ == "__main__":
    unittest.main()
