import json
import tempfile
import unittest
from pathlib import Path

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
            # The plane scene's observed-anchor fan (segment_count=8, arclength-
            # resampled boundary around an off-center anchor) has two mid-patch
            # support spokes (index 3 and 7) that nearly coincide away from the
            # shared pole -- a real, previously-undetected defect that the new
            # crossing gate now surfaces instead of silently exporting it as
            # "constructed". See worklog 110.
            self.assertTrue(plane["fidelity_gate"]["has_invalid_support_crossing"])
            self.assertEqual(plane["state"], "review_required")
            self.assertTrue((output / "NURBS_output" / "plane" / "boundary_first_support_status.json").is_file())
            # Still materialized/exported (only flagged, not suppressed) --
            # geometry stays inspectable even when review-gated.
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
            self.assertFalse(crossing["has_invalid_crossing"])
            self.assertEqual(len(crossing["pairs"]), patch_count * (patch_count - 1) // 2)

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

            # The pole-to-corner spoke is a diagnostic correspondence chord, not
            # "the" evaluated support curve.
            chords = review["support_correspondence_chords"]
            self.assertEqual(len(chords), patch_count)
            self.assertTrue(all(entity["representation_kind"] == "correspondence_chord" for entity in chords))
            self.assertTrue(all(len(entity["points"]) == 2 for entity in chords))

            # The actual interior support curve is a separately-evaluated
            # radial iso-curve, distinct from the corner chord.
            curves = review["evaluated_support_curves"]
            self.assertEqual(len(curves), patch_count)
            self.assertTrue(all(entity["representation_kind"] == "evaluated_curve" for entity in curves))
            self.assertTrue(all(len(entity["points"]) == 5 for entity in curves))
            self.assertEqual(review["support_control_polygons"], [])

            pole = review["pole_metadata"]
            self.assertIsNotNone(pole)
            self.assertTrue(pole["has_central_pole"])
            self.assertEqual(pole["singularity_kind"], "shared_observed_anchor_pole")

            # This scene is exactly the known crossing-gate discovery (segments
            # 3 and 7 nearly coincide away from the shared pole).
            crossing = review["support_crossing"]
            self.assertEqual(crossing["state"], "checked")
            self.assertTrue(crossing["has_invalid_crossing"])
            invalid_pairs = [p for p in crossing["pairs"] if p["classification"] == "invalid_interior_crossing"]
            self.assertEqual({tuple(sorted((p["curve_a"], p["curve_b"]))) for p in invalid_pairs}, {(3, 7)})
            valid_pole_pairs = [p for p in crossing["pairs"] if p["classification"] == "valid_shared_pole"]
            self.assertEqual(len(valid_pole_pairs), patch_count * (patch_count - 1) // 2 - 1)

    def test_unsupported_scene_still_preserves_observed_evidence_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            self.assertEqual(main(["--output", str(output), "--scenes", "u_shape"]), 0)
            status = json.loads((output / "NURBS_output" / "u_shape" / "boundary_first_support_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "unsupported")
            review = status["visible_results"][0]["review"]
            self.assertEqual(review["observed_outer_boundary"]["representation_kind"], "observed_evidence_points")
            self.assertTrue(len(review["observed_outer_boundary"]["points"]) > 0)
            self.assertEqual(review["support_crossing"]["state"], "not_checked")
            self.assertFalse((output / "NURBS_output" / "u_shape" / "nurbs_surface.json").is_file())


if __name__ == "__main__":
    unittest.main()
