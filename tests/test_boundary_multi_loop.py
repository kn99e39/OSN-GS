from types import SimpleNamespace
import unittest

from osn_gs.surface.torch_boundary_multi_loop import assess_multi_loop_correspondence


def loop(label, nested, ordered=True):
    return SimpleNamespace(
        label=label,
        nested_in_outer_label=nested,
        ordered_boundary_world_points=[[0.0, 0.0, 0.0]] if ordered else [],
    )


class BoundaryMultiLoopTest(unittest.TestCase):
    def test_nested_ordered_holes_preserve_roles_but_require_partition(self):
        result = assess_multi_loop_correspondence(SimpleNamespace(
            outer_loops=[loop(10, None)],
            hole_loops=[loop(4, 10), loop(7, 10)],
        ))
        self.assertEqual(result.state, "review_required")
        self.assertEqual(result.reason, "planar_domain_decomposition_required")
        self.assertEqual(result.boundary_roles, ("outer_boundary", "interior_boundary", "interior_boundary"))
        self.assertEqual(result.provenance["hole_labels"], (4, 7))
        self.assertEqual(result.provenance["missing_materialization_evidence"], "non_overlapping_planar_domain_partition")
        self.assertEqual(result.provenance["overlap_prevention"], "outer_boundary_must_not_be_duplicated_per_hole")

    def test_incomplete_nesting_is_unsupported_with_role_evidence(self):
        result = assess_multi_loop_correspondence(SimpleNamespace(
            outer_loops=[loop(10, None)],
            hole_loops=[loop(4, 10), loop(7, None)],
        ))
        self.assertEqual(result.state, "unsupported")
        self.assertEqual(result.reason, "hole_nesting_evidence_incomplete")
        self.assertEqual(result.boundary_roles, ("outer_boundary", "interior_boundary"))
        self.assertIn("loop_evidence", result.provenance)


if __name__ == "__main__":
    unittest.main()