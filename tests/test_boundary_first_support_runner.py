import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nurbs_constructor_benchmark.boundary_first_support_runner import main


class BoundaryFirstSupportRunnerTest(unittest.TestCase):
    def test_curved_annulus_exports_reviewable_renderer_pair_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            self.assertEqual(main(["--output", str(output), "--scenes", "curved_annulus", "plane"]), 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertEqual(result["state"], "constructed")
            self.assertEqual(result["patch_count"], 8)
            self.assertTrue(result["quality"]["finite"])
            self.assertFalse(result["fidelity_gate"]["has_invalid_support_crossing"])
            generated = output / "NURBS_output" / "curved_annulus"
            truth = output / "NURBS_output" / "curved_annulus_gt"
            self.assertTrue((generated / "point_cloud.ply").is_file())
            self.assertTrue((generated / "nurbs_surface.json").is_file())
            self.assertTrue((generated / "boundary_first_support_status.json").is_file())
            self.assertTrue((truth / "point_cloud.ply").is_file())
            self.assertTrue((truth / "nurbs_surface.json").is_file())
            plane = report["results"][1]
            self.assertEqual(
                plane["visible_results"][0]["provenance"]["boundary_roles"],
                ["outer_boundary", "interior_anchor"],
            )
            # Canonical materialization vs quality separation: a materialized
            # surface is never reported the same way as one that never got
            # built, even though neither reaches "eligible" yet (crossing is
            # only a representative sampled diagnostic; bidirectional fidelity
            # is not implemented).
            self.assertEqual(plane["materialization_state"], "materialized")
            self.assertEqual(plane["quality_state"], "review_required")
            self.assertEqual(plane["quality_reason"], "diagnostic_gates_incomplete_no_eligible_path_yet")
            self.assertNotEqual(plane["quality_state"], "eligible")  # never reachable yet, see worklog 110
            # Root-cause fixed (worklog 112): equal-angle, star-shape-validated
            # boundary correspondence resolves the crossing entirely.
            self.assertFalse(plane["fidelity_gate"]["has_invalid_support_crossing"])
            self.assertEqual(plane["visible_results"][0]["provenance"]["boundary_correspondence"], "equal_angle_star_shaped")
            self.assertTrue(plane["visible_results"][0]["provenance"]["star_shape_validation"]["is_valid"])
            self.assertTrue((output / "NURBS_output" / "plane" / "boundary_first_support_status.json").is_file())
            self.assertTrue((output / "NURBS_output" / "plane" / "nurbs_surface.json").is_file())

    def test_curved_annulus_review_layers_expose_observed_and_reconstructed_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            self.assertEqual(main(["--output", str(output), "--scenes", "curved_annulus"]), 0)
            status = json.loads((output / "NURBS_output" / "curved_annulus" / "boundary_first_support_status.json").read_text(encoding="utf-8"))
            review = status["visible_results"][0]["review"]
            patch_count = status["patch_count"]
            self.assertEqual(review["schema_version"], "boundary_first_review/2")
            self.assertEqual(review["boundary_roles"], ["outer_boundary", "interior_boundary"])
            self.assertIsNotNone(review["correspondence"])

            observed_outer = review["observed_outer_boundary"]
            observed_inner = review["observed_inner_boundary"]
            self.assertEqual(observed_outer["representation_kind"], "observed_evidence_points")
            self.assertEqual(observed_inner["representation_kind"], "observed_evidence_points")
            self.assertTrue(len(observed_outer["points"]) > 0)
            self.assertTrue(len(observed_inner["points"]) > 0)
            for point in observed_outer["points"]:
                self.assertEqual(len(point), 3)

            # Control polygon and evaluated curve are exported as SEPARATE
            # representations -- never conflated, per the corrected semantics.
            self.assertEqual(len(review["support_control_polygons"]), patch_count)
            self.assertEqual(len(review["evaluated_support_curves"]), patch_count)
            for entity in review["support_control_polygons"]:
                self.assertEqual(entity["representation_kind"], "control_polygon")
            for entity in review["evaluated_support_curves"]:
                self.assertEqual(entity["representation_kind"], "evaluated_curve")
            self.assertEqual(len(review["outer_boundary_control_polygons"]), patch_count)
            self.assertEqual(len(review["inner_boundary_control_polygons"]), patch_count)

            reconstructed_outer = review["reconstructed_outer_boundary"]
            reconstructed_inner = review["reconstructed_inner_boundary"]
            self.assertEqual(reconstructed_outer["representation_kind"], "evaluated_curve")
            self.assertEqual(reconstructed_inner["representation_kind"], "evaluated_curve")
            self.assertTrue(reconstructed_outer["closed"])
            self.assertTrue(reconstructed_inner["closed"])
            self.assertEqual(reconstructed_outer["patch_ids"], list(range(patch_count)))
            # 5 evaluated samples per patch, closed-loop shared junctions deduped.
            self.assertEqual(len(reconstructed_outer["points"]), patch_count * 4)
            self.assertEqual(len(reconstructed_inner["points"]), patch_count * 4)

            self.assertEqual(review["support_correspondence_chords"], [])
            self.assertIsNone(review["pole_metadata"])
            crossing = review["support_crossing"]
            self.assertEqual(crossing["state"], "checked")
            self.assertEqual(crossing["scope"], "representative_support_curve_bundle_crossing")
            self.assertFalse(crossing["has_invalid_crossing"])
            self.assertEqual(len(crossing["pairs"]), patch_count * (patch_count - 1) // 2)

            component = status["visible_results"][0]
            self.assertEqual(component["materialization_state"], "materialized")
            self.assertEqual(component["quality_state"], "review_required")
            self.assertNotEqual(component["quality_state"], "eligible")

            surface = json.loads((output / "NURBS_output" / "curved_annulus" / "nurbs_surface.json").read_text(encoding="utf-8"))
            self.assertEqual(len(surface["boundary_first_review"]), len(status["visible_results"]))
            self.assertTrue(len(surface["patch_boundaries"]) > 0)
            self.assertEqual(surface["metadata"]["patch_boundary_count"], len(surface["patch_boundaries"]))
            for boundary in surface["patch_boundaries"]:
                self.assertEqual(boundary["source_kind"], "boundary_first_support_seam")
                self.assertEqual(boundary["representation_kind"], "evaluated_curve")
                self.assertIn(boundary["patch_id"], range(patch_count))
                self.assertIn(boundary["adjacent_patch_id"], range(patch_count))

    def test_plane_anchor_cap_review_layers_expose_spokes_not_a_support_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            self.assertEqual(main(["--output", str(output), "--scenes", "plane"]), 0)
            status = json.loads((output / "NURBS_output" / "plane" / "boundary_first_support_status.json").read_text(encoding="utf-8"))
            review = status["visible_results"][0]["review"]
            patch_count = status["patch_count"]
            self.assertEqual(review["boundary_roles"], ["outer_boundary", "interior_anchor"])
            self.assertIsNone(review["observed_inner_boundary"])
            anchor_entity = review["observed_interior_anchor"]
            self.assertEqual(anchor_entity["representation_kind"], "observed_evidence_points")
            self.assertEqual(len(anchor_entity["points"]), 1)
            self.assertEqual(len(anchor_entity["points"][0]), 3)
            self.assertIsNone(review["reconstructed_inner_boundary"])

            reconstructed_outer = review["reconstructed_outer_boundary"]
            self.assertEqual(reconstructed_outer["representation_kind"], "evaluated_curve")
            self.assertEqual(len(reconstructed_outer["points"]), patch_count * 4)

            # The pole-to-corner probe is a diagnostic correspondence chord, not
            # "the" evaluated support curve -- and it must NOT be reused as the
            # seam either (see the seam entity check on the exported
            # patch_boundaries below): chord and seam stay separate entities
            # even though both happen to be the same straight line.
            chords = review["support_correspondence_chords"]
            self.assertEqual(len(chords), patch_count)
            self.assertTrue(all(entity["representation_kind"] == "correspondence_chord" for entity in chords))
            self.assertTrue(all(len(entity["points"]) == 2 for entity in chords))

            # The actual interior support curve is a representative BUNDLE (one
            # per configured interior-u fraction, default 3) per patch -- not
            # just a single midpoint sample, and distinct from the corner chord.
            curves = review["evaluated_support_curves"]
            self.assertEqual(len(curves), patch_count * 3)
            self.assertTrue(all(entity["representation_kind"] == "evaluated_curve" for entity in curves))
            self.assertTrue(all(len(entity["points"]) == 5 for entity in curves))
            self.assertEqual(review["support_control_polygons"], [])

            pole = review["pole_metadata"]
            self.assertIsNotNone(pole)
            self.assertTrue(pole["has_central_pole"])
            self.assertEqual(pole["singularity_kind"], "shared_observed_anchor_pole")
            self.assertEqual(pole["interior_support_curve_fractions"], [0.25, 0.5, 0.75])

            # This scene was the original crossing-gate discovery (patch 3 and
            # patch 7's u=0.5 interior curves nearly coincided away from the
            # shared pole). Root-caused in worklog 112 to coarse EQUAL-ARCLENGTH
            # boundary resampling producing wildly uneven angular coverage
            # around the anchor (confirmed star-shaped at raw resolution).
            # Equal-angle resampling (validated star-shaped first) now spreads
            # all 8 fan segments evenly around the anchor, so every pair only
            # shares the pole -- fully resolved, not just downgraded to ambiguous.
            crossing = review["support_crossing"]
            self.assertEqual(crossing["state"], "checked")
            self.assertEqual(crossing["scope"], "representative_support_curve_bundle_crossing")
            self.assertFalse(crossing["has_invalid_crossing"])
            self.assertEqual({p["classification"] for p in crossing["pairs"]}, {"valid_shared_pole"})

            # The seam entity backing patch_boundaries is a properly evaluated
            # curve, never the 2-point correspondence chord.
            surface = json.loads((output / "NURBS_output" / "plane" / "nurbs_surface.json").read_text(encoding="utf-8"))
            for boundary in surface["patch_boundaries"]:
                self.assertEqual(boundary["representation_kind"], "evaluated_curve")
                self.assertEqual(boundary["edge_a"], "u1")
                self.assertEqual(boundary["edge_b"], "u0")
                self.assertTrue(boundary["same_orientation"])

    def test_materialized_but_invalid_crossing_is_ineligible_not_unsupported(self):
        # After worklog 112's root-cause fix (equal-angle, star-shape-validated
        # boundary correspondence), no current benchmark scene actually
        # reaches quality_state="ineligible" anymore (that IS the fix working
        # -- see test_plane_anchor_cap_review_layers_expose_spokes_not_a_support_grid
        # and test_root_cause_fixed_scenes_have_no_invalid_crossing below). The
        # materialized+ineligible code path itself still needs its own direct
        # regression coverage so a future change can't silently reintroduce
        # collapsing "materialized but quality-failed" into "not_materialized".
        from nurbs_constructor_benchmark.boundary_first_support_runner import _component_quality_state, _scene_quality_projection

        materialized_item = SimpleNamespace(materialization_state="materialized", reason=None)
        layers_with_invalid_crossing = {"support_crossing": {"state": "checked", "has_invalid_crossing": True}}
        state, reason = _component_quality_state(materialized_item, layers_with_invalid_crossing)
        self.assertEqual(state, "ineligible")
        self.assertEqual(reason, "invalid_support_crossing")

        not_materialized_item = SimpleNamespace(materialization_state="not_materialized", reason="outer_boundary_ambiguous")
        state, reason = _component_quality_state(not_materialized_item, {"support_crossing": {"state": "not_checked"}})
        self.assertEqual(state, "unsupported")
        self.assertEqual(reason, "outer_boundary_ambiguous")
        # "materialized but ineligible" and "never materialized" must never
        # collapse into the same quality_state, even though both are
        # non-"eligible".
        self.assertNotEqual(state, "ineligible")

        scene_state, scene_reason = _scene_quality_projection([["ineligible", "invalid_support_crossing"], ["review_required", None]])
        self.assertEqual(scene_state, "ineligible")
        self.assertEqual(scene_reason, "invalid_support_crossing")

    def test_root_cause_fixed_scenes_have_no_invalid_crossing(self):
        # worklog 110 first found invalid crossings in 6 scenes; worklog 112's
        # equal-angle star-shaped boundary correspondence fix resolves all of
        # them (u_shape stays correctly rejected, but for an earlier, separate
        # anchor-ray-coverage reason -- concavity, not this crossing gate).
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            fixed_scenes = ["plane", "sine", "crease", "triangle", "elongated_plane", "close_parallel_sheets"]
            self.assertEqual(main(["--output", str(output), "--scenes", *fixed_scenes]), 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            for result in report["results"]:
                self.assertFalse(result["fidelity_gate"]["has_invalid_support_crossing"], result["scene"])
                self.assertNotEqual(result["quality_state"], "ineligible", result["scene"])

    def test_unsupported_scene_still_preserves_observed_evidence_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            self.assertEqual(main(["--output", str(output), "--scenes", "u_shape"]), 0)
            status = json.loads((output / "NURBS_output" / "u_shape" / "boundary_first_support_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "unsupported")
            self.assertEqual(status["materialization_state"], "not_materialized")
            self.assertEqual(status["quality_state"], "unsupported")
            component = status["visible_results"][0]
            self.assertEqual(component["materialization_state"], "not_materialized")
            self.assertEqual(component["quality_state"], "unsupported")
            self.assertEqual(component["quality_reason"], component["reason"])
            review = component["review"]
            self.assertEqual(review["observed_outer_boundary"]["representation_kind"], "observed_evidence_points")
            self.assertTrue(len(review["observed_outer_boundary"]["points"]) > 0)
            self.assertEqual(review["support_crossing"]["state"], "not_checked")
            self.assertEqual(review["support_crossing"]["scope"], "representative_support_curve_bundle_crossing")
            self.assertFalse((output / "NURBS_output" / "u_shape" / "nurbs_surface.json").is_file())


if __name__ == "__main__":
    unittest.main()
