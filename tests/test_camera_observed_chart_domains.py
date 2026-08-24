from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_camera_observed_chart_domains import (
    build_view_chart_candidates,
    label_same_component_blobs,
    valid_chart_mask,
)


class LabelSameComponentBlobsTest(unittest.TestCase):
    def test_single_component_is_one_blob(self):
        comp = torch.zeros((4, 4), dtype=torch.int64)
        labels = label_same_component_blobs(comp)
        self.assertTrue(bool((labels == 0).all()))

    def test_two_side_by_side_components_never_merge(self):
        comp = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        labels = label_same_component_blobs(comp)
        left_labels = torch.unique(labels[:, :2])
        right_labels = torch.unique(labels[:, 2:])
        self.assertEqual(int(left_labels.numel()), 1)
        self.assertEqual(int(right_labels.numel()), 1)
        self.assertNotEqual(int(left_labels[0]), int(right_labels[0]))

    def test_invalid_pixels_stay_invalid_and_never_bridge(self):
        # Same component id on both sides of a -1 gap must NOT be joined.
        comp = torch.tensor([[0, 0, -1, 0, 0]], dtype=torch.int64)
        labels = label_same_component_blobs(comp)
        self.assertEqual(int(labels[0, 2].item()), -1)
        self.assertNotEqual(int(labels[0, 0].item()), int(labels[0, 4].item()))

    def test_diagonal_adjacency_does_not_connect(self):
        # 4-connectivity only: a diagonal-only same-component pair stays split.
        comp = torch.tensor([[0, -1], [-1, 0]], dtype=torch.int64)
        labels = label_same_component_blobs(comp)
        self.assertNotEqual(int(labels[0, 0].item()), int(labels[1, 1].item()))


class BuildViewChartCandidatesTest(unittest.TestCase):
    def test_two_components_never_share_a_chart(self):
        """Directive section 5: two different canonical components adjacent
        in image space must never be fitted into the same chart -- verified
        here as a structural property of blob construction, not a filter."""

        comp = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        rep = torch.tensor([[10, 10, 20, 20], [11, 11, 21, 21]], dtype=torch.int64)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertEqual(vc.blob_count, 2)
        component_by_blob = {int(b): int(c) for b, c in zip(range(vc.blob_count), vc.blob_component_id.tolist())}
        for blob_id, rep_id in zip(vc.blob_of_member.tolist(), vc.member_representative_id.tolist()):
            expected_component = 0 if rep_id < 20 else 1
            self.assertEqual(component_by_blob[blob_id], expected_component)

    def test_member_uv_normalized_into_unit_box_per_blob(self):
        comp = torch.zeros((3, 5), dtype=torch.int64)
        rep = torch.tensor([[1, 1, 1, 1, 1]] * 3, dtype=torch.int64)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertTrue(bool((vc.member_uv >= 0.0).all()))
        self.assertTrue(bool((vc.member_uv <= 1.0).all()))

    def test_empty_view_yields_no_candidates(self):
        comp = torch.full((3, 3), -1, dtype=torch.int64)
        rep = torch.full((3, 3), -1, dtype=torch.int64)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertEqual(vc.blob_count, 0)
        self.assertEqual(int(vc.member_representative_id.numel()), 0)

    def test_member_pixel_count_sums_to_blob_pixel_total(self):
        comp = torch.zeros((4, 4), dtype=torch.int64)
        rep = torch.tensor([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ], dtype=torch.int64)
        vc = build_view_chart_candidates(0, comp, rep)
        self.assertEqual(int(vc.member_pixel_count.sum().item()), int(vc.blob_pixel_total.sum().item()))


class ValidChartMaskTest(unittest.TestCase):
    def test_threshold_derived_from_member_count_not_pixel_count(self):
        comp = torch.zeros((4, 4), dtype=torch.int64)
        rep = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 3, 4, 5],
            [2, 3, 4, 5],
        ], dtype=torch.int64)
        vc = build_view_chart_candidates(0, comp, rep)
        # Both blobs are the same component -> merged into one blob anyway
        # (rows 0-1 rep=1 touches row 2-3 which has distinct reps 2,3,4,5 but
        # same component id 0, so they connect). Sanity: at least 5 distinct
        # representatives total, mask should be all-True at min_members=5.
        self.assertTrue(bool(valid_chart_mask(vc, 5).all()))
        self.assertFalse(bool(valid_chart_mask(vc, 6).any()))


if __name__ == "__main__":
    unittest.main()
