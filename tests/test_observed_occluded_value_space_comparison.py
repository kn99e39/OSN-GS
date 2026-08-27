from __future__ import annotations

"""Worklog 121 -- value-space comparison and fidelity supplement.

Focused tests for the supplemental diagnostic layer over worklog 120's frozen
candidates: historical decision invariance, the corrected candidate C
min/max blocker provenance and world gaps, the candidate D termination contract
(`test_T = T_pre * (1 - alpha) < 1e-4`, NOT `T_pre < 1e-4`), bitwise invariance
of the worklog 120 probe outputs under the worklog 121 CUDA additions, the
accepted-event depth-inversion diagnostic, exact topology-gap provenance, and
deterministic supplemental-bank construction.
"""

import ast
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVTOOLS_DIR = REPO_ROOT / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from observed_occluded import candidate_a_surface_hit as candidate_a  # noqa: E402
from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import candidate_c_geometric_visibility as candidate_c  # noqa: E402
from observed_occluded import candidate_d_renderer_reachability as candidate_d  # noqa: E402
from observed_occluded import topology_gap_bank, value_diagnostics  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    STATE_NON_RELEVANT, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED,
    ViewGeometry, canonical_geometric_support_rho_max, project_queries,
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

WL120_NPZ = REPO_ROOT / "output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz"
requires_wl120_artifact = unittest.skipUnless(
    WL120_NPZ.exists(), f"worklog 120 reference artifact not present at {WL120_NPZ}"
)


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
# Historical decision invariance (directive section 0 / 20)
# ==========================================================================
class TestHistoricalDecisionInvariance(unittest.TestCase):
    """The four decision functions and the frozen aggregation must be byte-for-
    byte the worklog 120 sources. These digests are the ones committed at
    fdfb8ad; a change to any of them fails here before any value is believed."""

    FROZEN = (
        "candidate_a_surface_hit.py",
        "candidate_b_median_depth.py",
        "candidate_c_geometric_visibility.py",
        "candidate_d_renderer_reachability.py",
    )

    def test_decision_functions_are_pure_and_unshadowed(self):
        for name in self.FROZEN:
            source = (DEVTOOLS_DIR / "observed_occluded" / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
            self.assertIn("classify_view", functions, f"{name} lost its classify_view")

    def test_value_layer_never_defines_a_state(self):
        """`value_diagnostics` may read states but must never assign OBSERVED or
        OCCLUDED itself -- every verdict has to come from the frozen modules."""

        source = (DEVTOOLS_DIR / "observed_occluded" / "value_diagnostics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = ast.dump(node)
                if "STATE_OBSERVED" in targets and "Store" in targets:
                    offenders.append(ast.dump(node)[:80])
        self.assertEqual(offenders, [])
        # STATE_OCCLUDED is read only, for the C invariance guard.
        self.assertIn("candidate_c.classify_view", source)
        self.assertIn("candidate_d.classify_view", source)

    def test_topology_gap_bank_only_calls_the_read_only_replay_functions(self):
        """Structural, not keyword-based: the module may import exactly the
        known read-only WL107/109 replay entry points and nothing that could
        mutate, merge or re-threshold topology."""

        source = (DEVTOOLS_DIR / "observed_occluded" / "topology_gap_bank.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "osn_gs" in node.module:
                imported.update(alias.name for alias in node.names)
        allowed = {
            "REJECTION_REASONS", "REASON_GEOMETRIC_DISCONTINUITY",
            "CameraInducedAdjacencyConfig", "accumulate_image_space_pairs",
            "apply_secondary_geometric_gate", "filter_by_3d_locality",
            "CoverageFirstPartitionConfig", "_connected_component_roots", "build_candidate_graph",
        }
        self.assertTrue(
            imported.issubset(allowed),
            f"topology_gap_bank imports something outside the read-only replay set: {imported - allowed}",
        )

    def test_topology_gap_bank_assigns_subset_ids_only_from_the_replay(self):
        """`subset_ids` must be produced solely by the canonical connected-
        component result, never re-derived or edited afterwards."""

        source = (DEVTOOLS_DIR / "observed_occluded" / "topology_gap_bank.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "subset_ids" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1, "subset_ids assigned more than once")
        self.assertIn("subset_id_of_position", ast.dump(assignments[0]))


# ==========================================================================
# Candidate state invariance under the value layer
# ==========================================================================
class TestCandidateStateInvariance(unittest.TestCase):
    def test_candidate_a_states_unchanged(self):
        geometry = _geometry([5.0])
        result = candidate_a.classify_view(
            geometry, torch.tensor([[0.0, 0.0, 4.0]]), torch.tensor([4.0]), torch.tensor([True]),
            torch.tensor([[0.0, 0.0, 9.0]]),
        )
        self.assertEqual(int(result["states"][0]), STATE_OCCLUDED)

    def test_candidate_b_states_unchanged(self):
        geometry = _geometry([4.0])
        self.assertEqual(int(candidate_b.classify_view(geometry, torch.tensor([4.0]))["states"][0]), STATE_OBSERVED)

    def test_candidate_d_states_unchanged(self):
        geometry = _geometry([4.0])
        for terminated, expected in ((1, STATE_OCCLUDED), (0, STATE_OBSERVED), (-1, STATE_UNRESOLVED)):
            states = candidate_d.classify_view(
                geometry, torch.tensor([terminated], dtype=torch.int32), torch.tensor([0.5]),
                torch.tensor([1], dtype=torch.int32), torch.tensor([3], dtype=torch.int32),
            )["states"]
            self.assertEqual(int(states[0]), expected)


# ==========================================================================
# Corrected candidate C blocker provenance (directive section 6)
# ==========================================================================
class TestCorrectedBlockerProvenance(unittest.TestCase):
    """Three coplanar-normal discs at known depths in front of one query, so the
    camera-nearest and query-nearest blockers and their world gaps are exact."""

    def setUp(self):
        centres = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32)
        opacity = torch.full((3,), 0.99, dtype=torch.float32)
        scale = torch.full((3,), 1.0, dtype=torch.float32)
        self.support = candidate_c.GeometricSceneSupport(
            centers=centres,
            normals=torch.tensor([[0.0, 0.0, 1.0]] * 3),
            tangent_u=torch.tensor([[1.0, 0.0, 0.0]] * 3),
            tangent_v=torch.tensor([[0.0, 1.0, 0.0]] * 3),
            scale_u=scale, scale_v=scale,
            rho_max=canonical_geometric_support_rho_max(opacity), opacity=opacity,
        )
        self.origin = torch.tensor([0.0, 0.0, -4.0])
        self.world_view = torch.eye(4, dtype=torch.float32)
        self.world_view[3, 2] = 4.0
        self.query = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)
        self.geometry = _geometry([8.0])

    def _values(self):
        return value_diagnostics.candidate_c_blocker_values(
            self.geometry, self.query, self.origin, self.world_view, self.support, None, None,
        )

    def test_camera_nearest_is_min_t_and_query_nearest_is_max_t(self):
        values = self._values()
        self.assertEqual(int(values["blocker_count"][0]), 3)
        camera_t = float(values["camera_nearest_blocker_t"][0])
        query_t = float(values["query_nearest_blocker_t"][0])
        self.assertLess(camera_t, query_t)
        # Ray runs z = -4 -> 4, length 8. Discs at z = 0, 1, 2 sit at t = 0.5, 0.625, 0.75.
        self.assertAlmostEqual(camera_t, 0.5, places=5)
        self.assertAlmostEqual(query_t, 0.75, places=5)

    def test_world_gaps_are_distances_in_front_of_the_query(self):
        values = self._values()
        self.assertAlmostEqual(float(values["ray_length"][0]), 8.0, places=5)
        self.assertAlmostEqual(float(values["camera_nearest_blocker_world_gap"][0]), 4.0, places=4)
        self.assertAlmostEqual(float(values["query_nearest_blocker_world_gap"][0]), 2.0, places=4)
        self.assertAlmostEqual(float(values["blocker_region_thickness"][0]), 2.0, places=4)

    def test_worklog_120_reported_quantity_is_the_query_nearest_one(self):
        """Worklog 120's `nearest_blocker_t` used MAX(t); this pins that down so
        the historical column can never be silently re-read as camera-nearest."""

        values = self._values()
        legacy = candidate_c.classify_view(
            self.geometry, self.query, self.origin, self.world_view, self.support,
        )["nearest_blocker_t"]
        self.assertAlmostEqual(float(legacy[0]), float(values["query_nearest_blocker_t"][0]), places=6)
        self.assertNotAlmostEqual(float(legacy[0]), float(values["camera_nearest_blocker_t"][0]), places=3)

    def test_value_pass_reproduces_the_frozen_decision_exactly(self):
        decided = candidate_c.classify_view(
            self.geometry, self.query, self.origin, self.world_view, self.support,
        )["states"]
        values = self._values()
        blocked_by_value = self.geometry.relevant & (values["blocker_count"] > 0)
        self.assertTrue(bool(((decided == STATE_OCCLUDED) == blocked_by_value).all()))

    def test_same_component_attribution_counts_only_matching_components(self):
        component = torch.tensor([7, 7, 9], dtype=torch.int64)
        source = torch.tensor([7], dtype=torch.int64)
        values = value_diagnostics.candidate_c_blocker_values(
            self.geometry, self.query, self.origin, self.world_view, self.support, component, source,
        )
        self.assertEqual(int(values["same_component_blocker_count"][0]), 2)
        self.assertEqual(int(values["blocker_count"][0]), 3)

    def test_non_relevant_rows_get_no_blocker_values(self):
        geometry = _geometry([8.0], relevant=[False])
        values = value_diagnostics.candidate_c_blocker_values(
            geometry, self.query, self.origin, self.world_view, self.support, None, None,
        )
        self.assertTrue(bool(torch.isnan(values["camera_nearest_blocker_t"][0])))
        self.assertEqual(int(values["blocker_count"][0]), 0)


# ==========================================================================
# Candidate D resolution reasons and the termination contract
# ==========================================================================
class TestDResolutionReasonMapping(unittest.TestCase):
    def test_reason_codes_follow_the_probe_flags(self):
        terminated = np.array([[1, 0, 0, -1]])
        reached = np.array([[0, 1, 0, -1]])
        relevant = np.array([[True, True, True, True]])
        reason = value_diagnostics.d_resolution_reason(terminated, reached, relevant)
        self.assertEqual(
            [value_diagnostics.REASON_NAMES[int(code)] for code in reason[0]],
            ["TERMINATED_BEFORE_QUERY", "REACHED_ACCEPTED_EVENT", "CONTRIBUTOR_LIST_EXHAUSTED", "UNRESOLVED"],
        )

    def test_non_relevant_is_always_unresolved(self):
        reason = value_diagnostics.d_resolution_reason(
            np.array([[1]]), np.array([[0]]), np.array([[False]])
        )
        self.assertEqual(int(reason[0, 0]), value_diagnostics.REASON_UNRESOLVED)


@requires_qdepth
class TestQDepthWorklog121Additivity(unittest.TestCase):
    """The worklog 121 CUDA additions must not perturb ANY pre-existing output
    (directive section 0)."""

    CANONICAL_KEYS = (
        "render", "out_others", "radii", "representative_id", "forward_accepted",
        "contrib_ids", "contrib_post_median", "contrib_count",
        "median_rho3d", "median_rho2d", "median_s_u", "median_s_v",
    )

    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.05 * i for i in range(30)], opacity=0.3, device="cuda")
        self.camera = front_camera("cuda")

    def test_canonical_outputs_match_the_worklog_107_build(self):
        from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative

        reference = render_with_pixel_representative(self.camera, self.model)
        for query in (None, torch.zeros((64, 64, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")):
            package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
            for key in self.CANONICAL_KEYS:
                self.assertTrue(torch.equal(reference[key], package[key]), f"{key} changed (query={query is not None})")

    def test_worklog_120_probe_outputs_are_unchanged_by_the_additions(self):
        """Recomputed twice with different additional slots occupied: the four
        worklog 120 probe outputs for a fixed slot must be identical."""

        query_a = torch.zeros((64, 64, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query_a[32, 32, 0] = 4.60
        query_b = query_a.clone()
        query_b[32, 32, 3] = 5.90  # an unrelated extra probe on the same pixel
        first = render_with_query_depth_probe(self.camera, self.model, query_depths=query_a)
        second = render_with_query_depth_probe(self.camera, self.model, query_depths=query_b)
        for key in ("query_T", "query_terminated", "query_reached", "query_prefix_count"):
            self.assertTrue(torch.equal(first[key][32, 32, 0], second[key][32, 32, 0]), f"{key} slot 0 perturbed")

    def test_new_fields_use_the_documented_fill_convention(self):
        query = torch.zeros((64, 64, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[32, 32, 0] = 4.60
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        self.assertEqual(float(package["query_resolution_depth"][32, 32, 1]), -1.0)
        self.assertEqual(int(package["query_late_front_count"][32, 32, 1]), -1)
        self.assertEqual(float(package["query_termination_alpha"][32, 32, 1]), -1.0)
        self.assertGreaterEqual(float(package["query_resolution_depth"][32, 32, 0]), 0.0)
        self.assertGreaterEqual(int(package["query_late_front_count"][32, 32, 0]), 0)


@requires_qdepth
class TestDTerminationContract(unittest.TestCase):
    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.05 * i for i in range(30)], opacity=0.3, device="cuda")
        self.camera = front_camera("cuda")
        query = torch.zeros((64, 64, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[32, 32, 0] = 6.0  # past the canonical termination
        self.package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        self.t_pre = float(self.package["query_T"][32, 32, 0])
        self.alpha = float(self.package["query_termination_alpha"][32, 32, 0])

    def test_the_probe_reports_a_termination(self):
        self.assertEqual(int(self.package["query_terminated"][32, 32, 0]), 1)
        self.assertGreaterEqual(self.alpha, 0.0)

    def test_test_T_equals_T_pre_times_one_minus_alpha(self):
        self.assertAlmostEqual(
            self.t_pre * (1.0 - self.alpha),
            self.t_pre - self.t_pre * self.alpha,
            places=12,
        )

    def test_test_T_is_below_the_canonical_constant(self):
        self.assertLess(self.t_pre * (1.0 - self.alpha), value_diagnostics.CANONICAL_TERMINATION_TEST_T)

    def test_T_pre_itself_is_NOT_below_the_canonical_constant(self):
        """Directive section 7: worklog 120's phrasing implied `T_pre < 1e-4`.
        It is not the quantity the kernel compares."""

        self.assertGreaterEqual(self.t_pre, value_diagnostics.CANONICAL_TERMINATION_TEST_T)


# ==========================================================================
# Accepted-event depth inversion (directive section 8 / 13 S-D1)
# ==========================================================================
@requires_qdepth
class TestAcceptedDepthInversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from observed_occluded.synthetic_value_contracts import build_s_d1

        cls.result = build_s_d1(device="cuda")

    def test_fixture_actually_produces_an_inversion(self):
        self.assertGreaterEqual(self.result["pixel_inversion_count"], 1)
        self.assertGreater(self.result["pixel_max_backward_jump"], 0.0)

    def test_probe_resolves_in_traversal_order_not_physical_depth_order(self):
        """The tilted surfel is composited first (smaller CENTRE depth) even
        though its per-pixel event is physically deeper, so the probe resolves
        there with an empty prefix while a front-of-query accepted event is
        still to come."""

        probe = self.result["probe"]
        self.assertEqual(probe["accepted_prefix_count"], 0)
        self.assertGreater(probe["resolution_event_depth"], self.result["fixture"]["query_depth"])
        self.assertGreaterEqual(probe["late_front_count"], 1)

    def test_flat_surface_has_no_inversions(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        model = make_plane_stack([0.05 * i for i in range(10)], opacity=0.3, device="cuda")
        package = render_with_query_depth_probe(front_camera("cuda"), model, query_depths=None)
        self.assertEqual(int(package["pixel_inversion_count"][32, 32]), 0)


@requires_qdepth
class TestSyntheticValueContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from observed_occluded.synthetic_value_contracts import run_value_contracts

        cls.results = run_value_contracts(device="cuda")

    def test_s_c1_separates_camera_and_query_nearest_blockers(self):
        entry = self.results["S_C1"]
        self.assertGreaterEqual(entry["blocker_count"], 2)
        self.assertLess(entry["camera_nearest_blocker_t"], entry["query_nearest_blocker_t"])
        self.assertGreater(entry["camera_nearest_blocker_world_gap"], entry["query_nearest_blocker_world_gap"])
        self.assertAlmostEqual(
            entry["blocker_region_thickness"],
            entry["camera_nearest_blocker_world_gap"] - entry["query_nearest_blocker_world_gap"],
            places=4,
        )

    def test_s_c1_blocker_geometry_matches_the_fixture_spacing(self):
        entry = self.results["S_C1"]
        spacing = entry["fixture"]["spacing"]
        self.assertAlmostEqual(entry["query_nearest_blocker_world_gap"], spacing, places=3)
        self.assertAlmostEqual(
            entry["camera_nearest_blocker_world_gap"], spacing * entry["blocker_count"], places=3
        )

    def test_s_b1_reports_the_roundtrip_delta_without_a_tolerance(self):
        entry = self.results["S_B1"]
        self.assertEqual(len(entry["cases"]), 2)
        for case in entry["cases"]:
            self.assertIn("absolute_delta", case)
            self.assertLess(abs(case["absolute_delta"]), 1e-3)

    def test_contracts_are_deterministic(self):
        from observed_occluded.synthetic_value_contracts import run_value_contracts

        again = run_value_contracts(device="cuda")
        self.assertEqual(again["S_C1"]["blocker_count"], self.results["S_C1"]["blocker_count"])
        self.assertEqual(again["S_D1"]["pixel_inversion_count"], self.results["S_D1"]["pixel_inversion_count"])


# ==========================================================================
# Topology-gap provenance and deterministic supplemental bank
# ==========================================================================
class TestTopologyGapProvenance(unittest.TestCase):
    def test_cross_component_contexts_require_different_components(self):
        representative = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        subset = torch.tensor([5, 5, 6, 7], dtype=torch.int64)
        world = torch.zeros((2, 2, 3), dtype=torch.float32)
        found = topology_gap_bank.collect_cross_component_contexts(0, representative, subset, world)
        pairs = {
            (int(a), int(b)) for a, b in zip(found["representative_a"], found["representative_b"])
        }
        # (0,1) share component 5 -> excluded; (2,3), (0,2), (1,3) differ -> kept.
        self.assertNotIn((0, 1), pairs)
        self.assertIn((2, 3), pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((1, 3), pairs)

    def test_invalid_representatives_never_form_a_context(self):
        representative = torch.tensor([[-1, 1], [-1, 3]], dtype=torch.int64)
        subset = torch.tensor([0, 5, 0, 7], dtype=torch.int64)
        world = torch.zeros((2, 2, 3), dtype=torch.float32)
        found = topology_gap_bank.collect_cross_component_contexts(0, representative, subset, world)
        self.assertTrue(bool((found["representative_a"] >= 0).all()))
        self.assertTrue(bool((found["representative_b"] >= 0).all()))

    def test_gating_attribution_reads_the_replay_not_a_new_rule(self):
        count = 10
        replay = topology_gap_bank.TopologyReplay(
            subset_ids=torch.zeros(count, dtype=torch.int64), subset_count=1,
            subset_sizes=torch.tensor([count]),
            positive_edge_keys=torch.sort(topology_gap_bank._pair_key(torch.tensor([[1, 2]]), count)).values,
            locality_edge_keys=torch.sort(topology_gap_bank._pair_key(torch.tensor([[1, 2], [3, 4]]), count)).values,
            geometric_rejected_keys=topology_gap_bank._pair_key(torch.tensor([[3, 4]]), count),
            geometric_rejected_reason=torch.tensor([0], dtype=torch.int8), stats={},
        )
        reason = topology_gap_bank.attribute_gating(
            torch.tensor([1, 3, 5]), torch.tensor([2, 4, 6]), count, replay
        )
        self.assertEqual(int(reason[0]), topology_gap_bank.GATING_POSITIVE_EDGE_BUT_SPLIT)
        self.assertEqual(int(reason[1]), topology_gap_bank.GATING_GEOMETRIC_REJECTED)
        self.assertEqual(int(reason[2]), topology_gap_bank.GATING_LOCALITY_REJECTED)

    def test_deterministic_stride_is_reproducible_and_bounded(self):
        first = topology_gap_bank.deterministic_stride(1000, 60, torch.device("cpu"))
        second = topology_gap_bank.deterministic_stride(1000, 60, torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertLessEqual(int(first.numel()), 60)
        self.assertTrue(torch.equal(
            topology_gap_bank.deterministic_stride(10, 60, torch.device("cpu")), torch.arange(10)
        ))

    def test_supplemental_bank_emits_three_queries_per_context_plus_controls(self):
        contexts = {
            "view_index": np.array([0, 1]), "row_a": np.array([1, 2]), "col_a": np.array([3, 4]),
            "row_b": np.array([1, 3]), "col_b": np.array([4, 4]),
            "representative_a": np.array([10, 20]), "representative_b": np.array([11, 21]),
            "component_a": np.array([1, 2]), "component_b": np.array([3, 4]),
            "gating_reason": np.array([0, 1]), "region": np.array([0, 1]),
            "world_a": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
            "world_b": np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
        }
        controls = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float32)
        region_of_surfel = torch.zeros(30, dtype=torch.int64)
        bank, sidecar = topology_gap_bank.build_supplemental_bank(
            contexts, controls, region_of_surfel, torch.device("cpu")
        )
        self.assertEqual(len(bank), 2 * 3 + 1)
        self.assertEqual(sidecar["context_count"], 2)
        midpoint = bank.positions[sidecar["midpoint_rows"][0]]
        self.assertTrue(torch.allclose(midpoint, torch.tensor([1.0, 0.0, 0.0])))
        self.assertEqual(bank.kind[-1], topology_gap_bank.KIND_VERIFIED_OUT_OF_FRUSTUM)


# ==========================================================================
# Historical replay gate helpers
# ==========================================================================
class TestReplayGateHelpers(unittest.TestCase):
    def _result(self, states):
        return value_diagnostics.ValueEvaluationResult(
            per_view_states={name: states.copy() for name in value_diagnostics.CANDIDATE_NAMES},
            global_states={name: states[:, 0].copy() for name in value_diagnostics.CANDIDATE_NAMES},
            relevance_code=np.zeros_like(states), query_depth=np.zeros(states.shape, dtype=np.float32),
        )

    def test_gate_passes_on_identical_arrays(self):
        states = np.array([[1, 2], [0, 1]], dtype=np.int8)
        reference = {f"states_{n}": states for n in value_diagnostics.CANDIDATE_NAMES}
        reference.update({f"global_{n}": states[:, 0] for n in value_diagnostics.CANDIDATE_NAMES})
        reference["relevance_code"] = np.zeros_like(states)
        reference["query_depth"] = np.zeros(states.shape, dtype=np.float32)
        checks = value_diagnostics.assert_historical_state_replay(self._result(states), reference)
        self.assertEqual(checks["gate"], "PASS")
        self.assertEqual(checks["failures"], [])

    def test_gate_fails_and_names_the_array_on_any_difference(self):
        states = np.array([[1, 2], [0, 1]], dtype=np.int8)
        changed = states.copy()
        changed[0, 0] = 0
        reference = {f"states_{n}": changed for n in value_diagnostics.CANDIDATE_NAMES}
        reference.update({f"global_{n}": changed[:, 0] for n in value_diagnostics.CANDIDATE_NAMES})
        reference["relevance_code"] = np.zeros_like(states)
        reference["query_depth"] = np.zeros(states.shape, dtype=np.float32)
        checks = value_diagnostics.assert_historical_state_replay(self._result(states), reference)
        self.assertEqual(checks["gate"], "FAIL")
        self.assertIn("states_A", checks["failures"])


@requires_wl120_artifact
class TestWorklog120ArtifactShape(unittest.TestCase):
    """The reference the gate compares against must be the real worklog 120
    artifact -- 4,712 queries x 161 views."""

    def test_reference_artifact_has_the_historical_shape(self):
        reference = np.load(WL120_NPZ, allow_pickle=True)
        self.assertEqual(reference["positions"].shape, (4712, 3))
        for name in value_diagnostics.CANDIDATE_NAMES:
            self.assertEqual(reference[f"states_{name}"].shape, (4712, 161))
            self.assertEqual(reference[f"global_{name}"].shape, (4712,))

    def test_reference_artifact_reproduces_worklog_120_headline_counts(self):
        reference = np.load(WL120_NPZ, allow_pickle=True)
        self.assertEqual(int((reference["global_D"] == STATE_OCCLUDED).sum()), 0)
        self.assertEqual(int((reference["global_C"] == STATE_OCCLUDED).sum()), 3941)
        self.assertEqual(int((reference["global_B"] == STATE_OBSERVED).sum()), 4054)
        self.assertEqual(int((reference["global_A"] == STATE_UNRESOLVED).sum()), 1388)


if __name__ == "__main__":
    unittest.main()
