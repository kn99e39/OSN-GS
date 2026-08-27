from __future__ import annotations

"""Worklog 120 -- Observed / Occluded volumetric operationalization audit.

Focused tests for the shared contract (query representation, relevant-view
detection, frozen global aggregation, deterministic bank construction), for each
candidate's decision function in isolation, for the diagnostic-only
`diff_surfel_rasterization_qdepth` CUDA sibling against canonical traversal, and
for the S1-S7 synthetic contracts.

Pure-logic tests run without CUDA. Everything needing a real forward pass is
skipped unless CUDA and the diagnostic build are both available.
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

from observed_occluded import candidate_a_surface_hit as candidate_a  # noqa: E402
from observed_occluded import candidate_b_median_depth as candidate_b  # noqa: E402
from observed_occluded import candidate_c_geometric_visibility as candidate_c  # noqa: E402
from observed_occluded import candidate_d_renderer_reachability as candidate_d  # noqa: E402
from observed_occluded.shared import (  # noqa: E402
    CANONICAL_MIN_ALPHA, CANONICAL_NEAR_N, RELEVANCE_DEPTH_BELOW_NEAR,
    RELEVANCE_INVALID_PROJECTION, RELEVANCE_OK, RELEVANCE_OUTSIDE_IMAGE,
    STATE_NON_RELEVANT, STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED,
    ViewGeometry, aggregate_global, assign_query_depth_slots,
    canonical_constants_from_source, canonical_geometric_support_rho_max,
    project_queries,
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


def _geometry(depth, relevant=None, pixel_index=None) -> ViewGeometry:
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


# ==========================================================================
# SHARED: frozen global aggregation (directive section 2)
# ==========================================================================
class TestGlobalAggregation(unittest.TestCase):
    def test_any_observed_wins(self):
        states = np.full((1, 100), STATE_OCCLUDED, dtype=np.int8)
        states[0, 57] = STATE_OBSERVED
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)

    def test_all_relevant_occluded_is_occluded(self):
        states = np.array([[STATE_OCCLUDED, STATE_NON_RELEVANT, STATE_OCCLUDED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OCCLUDED)

    def test_one_unresolved_blocks_global_occluded(self):
        states = np.array([[STATE_OCCLUDED, STATE_UNRESOLVED, STATE_OCCLUDED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_UNRESOLVED)

    def test_no_relevant_view_is_unresolved_not_occluded(self):
        states = np.full((1, 8), STATE_NON_RELEVANT, dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_UNRESOLVED)

    def test_no_majority_vote(self):
        """99 OCCLUDED plus 1 OBSERVED must be OBSERVED, not a vote."""

        states = np.full((1, 100), STATE_OCCLUDED, dtype=np.int8)
        states[0, 0] = STATE_OBSERVED
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)
        states = np.full((1, 100), STATE_OBSERVED, dtype=np.int8)
        states[0, 0] = STATE_OCCLUDED
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)

    def test_unresolved_plus_observed_is_observed(self):
        states = np.array([[STATE_UNRESOLVED, STATE_OBSERVED]], dtype=np.int8)
        self.assertEqual(int(aggregate_global(states)[0]), STATE_OBSERVED)


# ==========================================================================
# SHARED: relevant-view contract (directive section 3) + canonical constants
# ==========================================================================
class TestCanonicalConstants(unittest.TestCase):
    def test_constants_still_match_the_production_renderer(self):
        found = canonical_constants_from_source(REPO_ROOT)
        self.assertAlmostEqual(found["near_n"], CANONICAL_NEAR_N, places=9)
        self.assertTrue(found["min_alpha_source_present"])
        self.assertTrue(found["max_alpha_source_present"])
        self.assertTrue(found["termination_source_present"])

    def test_support_formula_matches_the_alpha_cutoff(self):
        opacity = torch.tensor([0.99, 0.5, 0.1, CANONICAL_MIN_ALPHA, 1e-6])
        rho_max = canonical_geometric_support_rho_max(opacity)
        for index in range(3):
            alpha = float(opacity[index]) * float(np.exp(-0.5 * float(rho_max[index])))
            self.assertAlmostEqual(alpha, CANONICAL_MIN_ALPHA, places=6)
        self.assertAlmostEqual(float(rho_max[3]), 0.0, places=6)
        self.assertEqual(float(rho_max[4]), 0.0)


@requires_qdepth
class TestRelevantViewContract(unittest.TestCase):
    def setUp(self):
        from observed_occluded.synthetic_contracts import IMAGE, front_camera

        self.camera = front_camera("cuda")
        self.image = IMAGE

    def test_in_frustum_point_is_relevant(self):
        geometry = project_queries(self.camera, torch.zeros((1, 3), device="cuda"))
        self.assertEqual(int(geometry.relevance_code[0]), RELEVANCE_OK)
        self.assertTrue(bool(geometry.relevant[0]))
        self.assertAlmostEqual(float(geometry.depth[0]), 4.0, places=5)

    def test_far_off_axis_is_outside_image_not_occluded(self):
        point = torch.tensor([[100.0, 100.0, 0.0]], device="cuda")
        geometry = project_queries(self.camera, point)
        self.assertEqual(int(geometry.relevance_code[0]), RELEVANCE_OUTSIDE_IMAGE)
        self.assertEqual(int(geometry.pixel_index[0]), -1)

    def test_behind_camera_is_non_relevant(self):
        point = torch.tensor([[0.0, 0.0, -100.0]], device="cuda")
        geometry = project_queries(self.camera, point)
        self.assertIn(int(geometry.relevance_code[0]), (RELEVANCE_INVALID_PROJECTION, RELEVANCE_DEPTH_BELOW_NEAR))

    def test_projection_uses_the_rasterizer_pixel_convention(self):
        """A pixel's own median surface event must project back onto that pixel."""

        from observed_occluded.synthetic_contracts import make_plane_stack, surface_event_world_point

        model = make_plane_stack([0.0], device="cuda")
        row = col = self.image // 2
        event, _ = surface_event_world_point(model, self.camera, row, col)
        geometry = project_queries(self.camera, event.reshape(1, 3))
        self.assertEqual(int(geometry.pixel_row[0]), row)
        self.assertEqual(int(geometry.pixel_col[0]), col)


class TestQueryDepthSlotAssignment(unittest.TestCase):
    def test_ranks_are_per_pixel_and_deterministic(self):
        pixels = np.array([5, 5, 7, -1, 5, 7], dtype=np.int64)
        ranks = assign_query_depth_slots(pixels, 8)
        self.assertEqual(ranks.tolist(), [0, 1, 0, -1, 2, 1])
        self.assertTrue(np.array_equal(ranks, assign_query_depth_slots(pixels, 8)))

    def test_no_query_is_dropped_when_a_pixel_overflows(self):
        pixels = np.full(20, 3, dtype=np.int64)
        ranks = assign_query_depth_slots(pixels, 8)
        self.assertEqual(sorted(ranks.tolist()), list(range(20)))


# ==========================================================================
# CANDIDATE A -- isolated
# ==========================================================================
class TestCandidateA(unittest.TestCase):
    def _run(self, query, event, event_depth, valid, query_depth):
        geometry = _geometry([query_depth])
        return candidate_a.classify_view(
            geometry,
            torch.as_tensor(event, dtype=torch.float32).reshape(1, 3),
            torch.tensor([event_depth], dtype=torch.float32),
            torch.tensor([valid]),
            torch.as_tensor(query, dtype=torch.float32).reshape(1, 3),
        )["states"]

    def test_query_that_is_the_event_is_observed(self):
        point = [1.0, 2.0, 3.0]
        self.assertEqual(int(self._run(point, point, 5.0, True, 5.0)[0]), STATE_OBSERVED)

    def test_event_in_front_occludes(self):
        self.assertEqual(int(self._run([0.0, 0.0, 9.0], [0.0, 0.0, 4.0], 4.0, True, 9.0)[0]), STATE_OCCLUDED)

    def test_free_space_in_front_of_the_event_is_unresolved_not_observed(self):
        """A's defining limitation: it cannot call exposed free space observed."""

        self.assertEqual(int(self._run([0.0, 0.0, 2.0], [0.0, 0.0, 4.0], 4.0, True, 2.0)[0]), STATE_UNRESOLVED)

    def test_no_event_at_the_pixel_is_unresolved(self):
        self.assertEqual(int(self._run([0.0, 0.0, 9.0], [0.0, 0.0, 4.0], 4.0, False, 9.0)[0]), STATE_UNRESOLVED)

    def test_non_relevant_view_never_produces_a_state(self):
        geometry = _geometry([5.0], relevant=[False])
        states = candidate_a.classify_view(
            geometry, torch.zeros((1, 3)), torch.tensor([1.0]), torch.tensor([True]), torch.zeros((1, 3)),
        )["states"]
        self.assertEqual(int(states[0]), STATE_NON_RELEVANT)

    def test_identity_epsilon_is_a_float_rule_not_a_coverage_radius(self):
        self.assertLessEqual(candidate_a.FLOAT32_IDENTITY_RELATIVE_EPSILON, 1e-5)
        far = [0.0, 0.0, 4.0 + 1e-3]
        self.assertEqual(int(self._run(far, [0.0, 0.0, 4.0], 4.0, True, 4.0 + 1e-3)[0]), STATE_OCCLUDED)


# ==========================================================================
# CANDIDATE B -- isolated
# ==========================================================================
class TestCandidateB(unittest.TestCase):
    def _run(self, query_depth, median):
        geometry = _geometry([query_depth])
        return int(candidate_b.classify_view(geometry, torch.tensor([median], dtype=torch.float32))["states"][0])

    def test_in_front_of_median_is_observed(self):
        self.assertEqual(self._run(3.0, 4.0), STATE_OBSERVED)

    def test_behind_median_is_occluded(self):
        self.assertEqual(self._run(5.0, 4.0), STATE_OCCLUDED)

    def test_exactly_at_median_is_observed(self):
        self.assertEqual(self._run(4.0, 4.0), STATE_OBSERVED)

    def test_sentinel_zero_median_is_unresolved(self):
        self.assertEqual(self._run(4.0, 0.0), STATE_UNRESOLVED)

    def test_median_offset_matches_the_canonical_kernel(self):
        forward = (REPO_ROOT / "osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/forward.cu").read_text(encoding="utf-8")
        self.assertIn("MIDDEPTH_OFFSET", forward)
        auxiliary = (REPO_ROOT / "osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/auxiliary.h").read_text(encoding="utf-8")
        self.assertIn(f"#define MIDDEPTH_OFFSET {candidate_b.MIDDEPTH_OFFSET}", auxiliary)


# ==========================================================================
# CANDIDATE C -- isolated, analytic ray/disc geometry
# ==========================================================================
class TestCandidateC(unittest.TestCase):
    def _support(self, centre, opacity=0.99, scale=1.0):
        centre = torch.as_tensor(centre, dtype=torch.float32).reshape(1, 3)
        opacity_t = torch.tensor([opacity], dtype=torch.float32)
        scale_t = torch.tensor([scale], dtype=torch.float32)
        return candidate_c.GeometricSceneSupport(
            centers=centre,
            normals=torch.tensor([[0.0, 0.0, 1.0]]),
            tangent_u=torch.tensor([[1.0, 0.0, 0.0]]),
            tangent_v=torch.tensor([[0.0, 1.0, 0.0]]),
            scale_u=scale_t, scale_v=scale_t,
            rho_max=canonical_geometric_support_rho_max(opacity_t),
            opacity=opacity_t,
        )

    def _classify(self, support, query, camera_at=(0.0, 0.0, -4.0)):
        query = torch.as_tensor(query, dtype=torch.float32).reshape(1, 3)
        origin = torch.as_tensor(camera_at, dtype=torch.float32)
        world_view = torch.eye(4, dtype=torch.float32)
        world_view[3, 2] = -float(origin[2])
        geometry = _geometry([float(query[0, 2] - origin[2])])
        return candidate_c.classify_view(geometry, query, origin, world_view, support)

    def test_ray_through_the_support_disc_is_occluded(self):
        result = self._classify(self._support([0.0, 0.0, 0.0]), [0.0, 0.0, 4.0])
        self.assertEqual(int(result["states"][0]), STATE_OCCLUDED)
        self.assertEqual(int(result["blocker_count"][0]), 1)

    def test_ray_outside_the_support_disc_is_observed(self):
        # rho_max for opacity 0.99 is 2*ln(252.45) ~= 11.06, radius ~3.33 * scale.
        result = self._classify(self._support([50.0, 0.0, 0.0]), [0.0, 0.0, 4.0])
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)
        self.assertEqual(int(result["blocker_count"][0]), 0)

    def test_surfel_at_the_query_itself_does_not_occlude_it(self):
        result = self._classify(self._support([0.0, 0.0, 4.0]), [0.0, 0.0, 4.0])
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)

    def test_surfel_behind_the_query_does_not_occlude_it(self):
        result = self._classify(self._support([0.0, 0.0, 6.0]), [0.0, 0.0, 4.0])
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)

    def test_opacity_below_the_canonical_cutoff_has_empty_support(self):
        result = self._classify(self._support([0.0, 0.0, 0.0], opacity=1.0 / 300.0), [0.0, 0.0, 4.0])
        self.assertEqual(int(result["states"][0]), STATE_OBSERVED)

    def test_support_extent_is_exactly_the_alpha_cutoff_boundary(self):
        support = self._support([0.0, 0.0, 0.0], opacity=0.99, scale=1.0)
        radius = float(torch.sqrt(support.rho_max[0]))
        origin = (0.0, 0.0, -4.0)
        # A ray aimed just inside the support boundary blocks; just outside does not.
        inside = self._classify(support, [radius * 0.999 * 2.0, 0.0, 4.0], camera_at=origin)
        outside = self._classify(support, [radius * 1.001 * 2.0, 0.0, 4.0], camera_at=origin)
        self.assertEqual(int(inside["states"][0]), STATE_OCCLUDED)
        self.assertEqual(int(outside["states"][0]), STATE_OBSERVED)

    def test_non_relevant_view_never_produces_a_state(self):
        support = self._support([0.0, 0.0, 0.0])
        query = torch.tensor([[0.0, 0.0, 4.0]])
        world_view = torch.eye(4, dtype=torch.float32)
        world_view[3, 2] = 4.0
        geometry = _geometry([8.0], relevant=[False])
        states = candidate_c.classify_view(geometry, query, torch.tensor([0.0, 0.0, -4.0]), world_view, support)["states"]
        self.assertEqual(int(states[0]), STATE_NON_RELEVANT)


# ==========================================================================
# CANDIDATE D -- isolated
# ==========================================================================
class TestCandidateD(unittest.TestCase):
    def _run(self, terminated, relevant=True):
        geometry = _geometry([4.0], relevant=[relevant])
        return int(candidate_d.classify_view(
            geometry, torch.tensor([terminated], dtype=torch.int32), torch.tensor([0.5]),
            torch.tensor([1], dtype=torch.int32), torch.tensor([3], dtype=torch.int32),
        )["states"][0])

    def test_terminated_before_query_depth_is_occluded(self):
        self.assertEqual(self._run(1), STATE_OCCLUDED)

    def test_not_terminated_is_observed(self):
        self.assertEqual(self._run(0), STATE_OBSERVED)

    def test_unwritten_probe_fails_closed_to_unresolved(self):
        self.assertEqual(self._run(-1), STATE_UNRESOLVED)

    def test_non_relevant_view_never_produces_a_state(self):
        self.assertEqual(self._run(1, relevant=False), STATE_NON_RELEVANT)


# ==========================================================================
# Candidate isolation (directive section 4 / 15)
# ==========================================================================
class TestCandidateIsolation(unittest.TestCase):
    PACKAGE = DEVTOOLS_DIR / "observed_occluded"

    def _imports(self, name: str) -> set[str]:
        tree = ast.parse((self.PACKAGE / name).read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        return modules

    def test_candidate_modules_never_import_each_other(self):
        names = [
            "candidate_a_surface_hit.py", "candidate_b_median_depth.py",
            "candidate_c_geometric_visibility.py", "candidate_d_renderer_reachability.py",
        ]
        for name in names:
            for other in names:
                if other == name:
                    continue
                self.assertNotIn(other[:-3], " ".join(self._imports(name)), f"{name} imports {other}")

    def test_shared_never_imports_a_candidate(self):
        for module in self._imports("shared.py"):
            self.assertFalse(module.startswith("candidate"), f"shared.py imports {module}")

    def test_engine_makes_no_observed_occluded_decision(self):
        source = (self.PACKAGE / "engine.py").read_text(encoding="utf-8")
        self.assertNotIn("STATE_OBSERVED", source)
        self.assertNotIn("STATE_OCCLUDED", source)

    def test_query_bank_makes_no_observed_occluded_decision(self):
        source = (self.PACKAGE / "query_bank.py").read_text(encoding="utf-8")
        self.assertNotIn("STATE_OBSERVED", source)
        self.assertNotIn("STATE_OCCLUDED", source)

    def test_shared_only_decides_states_inside_the_frozen_aggregation(self):
        tree = ast.parse((self.PACKAGE / "shared.py").read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names = {
                    sub.id for sub in ast.walk(node)
                    if isinstance(sub, ast.Name) and sub.id in {"STATE_OBSERVED", "STATE_OCCLUDED"}
                }
                if names and node.name not in {"aggregate_global", "agreement", "state_fractions"}:
                    offenders.append(node.name)
        self.assertEqual(offenders, [], f"shared.py functions deciding states: {offenders}")

    def test_candidate_b_never_reads_transmittance_or_geometry_support(self):
        source = (self.PACKAGE / "candidate_b_median_depth.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        body = "\n".join(
            ast.get_source_segment(source, node) or ""
            for node in tree.body if not isinstance(node, (ast.Expr, ast.ImportFrom, ast.Import))
        )
        for forbidden in ("query_T", "query_terminated", "rho_max", "representative_id", "blocker"):
            self.assertNotIn(forbidden, body, f"candidate_b code references {forbidden}")

    def test_candidate_d_never_reads_median_depth_or_geometry_support(self):
        source = (self.PACKAGE / "candidate_d_renderer_reachability.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        body = "\n".join(
            ast.get_source_segment(source, node) or ""
            for node in tree.body if not isinstance(node, (ast.Expr, ast.ImportFrom, ast.Import))
        )
        for forbidden in ("median", "rho_max", "blocker"):
            self.assertNotIn(forbidden, body, f"candidate_d code references {forbidden}")


# ==========================================================================
# Diagnostic CUDA sibling vs canonical traversal (directive section 8 / 17)
# ==========================================================================
@requires_qdepth
class TestQDepthCanonicalEquivalence(unittest.TestCase):
    CANONICAL_KEYS = (
        "render", "out_others", "radii", "representative_id", "forward_accepted",
        "contrib_ids", "contrib_post_median", "contrib_count",
        "median_rho3d", "median_rho2d", "median_s_u", "median_s_v",
    )

    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        self.model = make_plane_stack([0.0, 0.02, 0.04, 0.06], device="cuda")
        self.camera = front_camera("cuda")
        from osn_gs.render.torch_surfel_representative_diagnostics import render_with_pixel_representative

        self.reference = render_with_pixel_representative(self.camera, self.model)

    def test_probe_disabled_reproduces_the_worklog_107_build_exactly(self):
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=None)
        for key in self.CANONICAL_KEYS:
            self.assertTrue(torch.equal(self.reference[key], package[key]), f"{key} differs with probe disabled")

    def test_probe_enabled_changes_nothing_canonical(self):
        height, width = self.camera.image_height, self.camera.image_width
        query = torch.zeros((height, width, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[:, :, 0] = 4.5
        query[:, :, 1] = 3.5
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        for key in self.CANONICAL_KEYS:
            self.assertTrue(torch.equal(self.reference[key], package[key]), f"{key} differs with probe enabled")

    def test_unused_slots_stay_at_their_fill_value(self):
        height, width = self.camera.image_height, self.camera.image_width
        query = torch.zeros((height, width, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[32, 32, 0] = 4.5
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        self.assertEqual(int(package["query_terminated"][32, 32, 1]), -1)
        self.assertEqual(float(package["query_T"][32, 32, 1]), -1.0)


@requires_qdepth
class TestQDepthProbeAgainstCanonicalTraversal(unittest.TestCase):
    """Hand-computed traversal state on a controlled stack, compared against
    what the probe reports."""

    def setUp(self):
        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        # 30 layers, alpha = 0.3 at the ray centre. T after n accepted layers is
        # 0.7^n; `T * (1 - alpha) < 0.0001` first fires at layer 26.
        self.model = make_plane_stack([0.05 * i for i in range(30)], opacity=0.3, device="cuda")
        self.camera = front_camera("cuda")

    def _probe(self, depths):
        height, width = self.camera.image_height, self.camera.image_width
        query = torch.zeros((height, width, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        for slot, depth in enumerate(depths):
            query[32, 32, slot] = depth
        package = render_with_query_depth_probe(self.camera, self.model, query_depths=query)
        return [
            (
                float(package["query_T"][32, 32, slot]),
                int(package["query_terminated"][32, 32, slot]),
                int(package["query_reached"][32, 32, slot]),
                int(package["query_prefix_count"][32, 32, slot]),
            )
            for slot in range(len(depths))
        ]

    def test_transmittance_prefix_matches_the_hand_computed_product(self):
        # Query just past layer k resolves at layer k+1 with T = prod(1 - alpha_i).
        results = self._probe([4.0, 4.06, 4.16, 4.26])
        prefixes = [record[3] for record in results]
        self.assertEqual(prefixes, [0, 2, 4, 6])
        for transmittance, _terminated, reached, prefix in results:
            self.assertEqual(reached, 1)
            self.assertAlmostEqual(transmittance, 0.7 ** prefix, places=2)

    def test_termination_fires_exactly_at_the_canonical_condition(self):
        results = self._probe([5.20, 5.30, 6.00])
        # Layer 25 sits at depth 5.25; the canonical `T * (1 - alpha) < 0.0001`
        # fires there, so 5.20 is still reached and 5.30 / 6.00 are terminated.
        self.assertEqual(results[0][1], 0)
        self.assertEqual(results[1][1], 1)
        self.assertEqual(results[2][1], 1)
        for transmittance, terminated, _reached, _prefix in results[1:]:
            self.assertEqual(terminated, 1)
            self.assertLess(transmittance, 1e-3)

    def test_query_in_front_of_all_geometry_is_never_terminated(self):
        (transmittance, terminated, reached, prefix), = self._probe([3.0])
        self.assertEqual(terminated, 0)
        self.assertEqual(reached, 1)
        self.assertEqual(prefix, 0)
        self.assertAlmostEqual(transmittance, 1.0, places=6)

    def test_probe_resolves_only_at_accepted_contributors(self):
        """Regression guard for the corrected resolution site: a probe deep
        behind an opaque stack must NOT come back with T = 1 and prefix 0."""

        from observed_occluded.synthetic_contracts import front_camera, make_plane_stack

        model = make_plane_stack([0.0, 0.02, 0.04, 0.06], device="cuda")
        camera = front_camera("cuda")
        height, width = camera.image_height, camera.image_width
        query = torch.zeros((height, width, MAX_QUERY_SLOTS), dtype=torch.float32, device="cuda")
        query[32, 32, 0] = 4.5
        package = render_with_query_depth_probe(camera, model, query_depths=query)
        self.assertEqual(int(package["query_terminated"][32, 32, 0]), 1)
        self.assertGreater(int(package["query_prefix_count"][32, 32, 0]), 0)
        self.assertLess(float(package["query_T"][32, 32, 0]), 1e-3)


# ==========================================================================
# Synthetic contracts S1-S7 (directive section 9A)
# ==========================================================================
@requires_qdepth
class TestSyntheticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from observed_occluded.synthetic_contracts import contract_summary, run_contracts

        cls.results = run_contracts(device="cuda")
        cls.summary = contract_summary(cls.results)

    def test_every_candidate_matches_its_own_written_contract(self):
        failures = [
            (name, candidate) for name, entry in self.summary.items()
            for candidate, record in entry.items() if not record["implementation_fidelity_pass"]
        ]
        self.assertEqual(failures, [], f"implementation-fidelity failures: {failures}")

    def test_S1_all_candidates_observe_a_directly_exposed_surface(self):
        for candidate in "ABCD":
            self.assertEqual(self.summary["S1_directly_exposed_surface"][candidate]["states"], ["OBSERVED"])

    def test_S2_surface_only_mechanism_cannot_represent_observed_free_space(self):
        entry = self.summary["S2_exposed_free_space_before_surface"]
        self.assertEqual(entry["A"]["states"], ["UNRESOLVED"])
        self.assertFalse(entry["A"]["directive_semantics_pass"])
        for candidate in "BCD":
            self.assertEqual(entry[candidate]["states"], ["OBSERVED"])

    def test_S3_opaque_stack_occludes_every_candidate(self):
        for candidate in "ABCD":
            self.assertEqual(self.summary["S3a_behind_canonically_opaque_blocker"][candidate]["states"], ["OCCLUDED"])

    def test_S3b_single_semi_transparent_blocker_does_not_terminate_traversal(self):
        entry = self.summary["S3b_behind_single_primitive_blocker"]
        self.assertEqual(entry["D"]["states"], ["OBSERVED"])
        for candidate in "ABC":
            self.assertEqual(entry[candidate]["states"], ["OCCLUDED"])

    def test_S4_cross_view_disocclusion_is_globally_observed(self):
        for candidate in "ABCD":
            self.assertEqual(self.summary["S4_cross_view_disocclusion"][candidate]["states"], ["OBSERVED"])

    def test_S5_no_relevant_view_is_unresolved_never_occluded(self):
        for candidate in "ABCD":
            self.assertEqual(
                self.summary["S5_outside_camera_support"][candidate]["states"], ["UNRESOLVED", "UNRESOLVED"]
            )

    def test_S6_separates_median_from_renderer_termination(self):
        entry = self.summary["S6_layered_soft_compositing"]
        self.assertEqual(entry["B"]["states"][1], "OCCLUDED")
        self.assertEqual(entry["D"]["states"][1], "OBSERVED")
        record = self.results["S6_layered_soft_compositing"]["queries"][1]
        self.assertGreater(record["B_median_depth"][0], 0.0)
        self.assertLess(record["B_median_depth"][0], record["query_depth_per_view"][0])

    def test_S7_separates_low_pass_from_true_footprint_events(self):
        self.assertEqual(self.results["S7_rho3d_true_footprint"]["provenance"]["event"]["branch"], "rho3d")
        self.assertEqual(self.results["S7_rho2d_low_pass"]["provenance"]["event"]["branch"], "rho2d")
        # A low-pass-only event has no geometric support on the ray at all.
        self.assertEqual(self.summary["S7_rho2d_low_pass"]["C"]["states"][1], "OBSERVED")
        self.assertEqual(self.summary["S7_rho3d_true_footprint"]["C"]["states"][1], "OCCLUDED")

    def test_contracts_are_deterministic(self):
        from observed_occluded.synthetic_contracts import contract_summary, run_contracts

        again = contract_summary(run_contracts(device="cuda"))
        self.assertEqual(again, self.summary)


if __name__ == "__main__":
    unittest.main()
