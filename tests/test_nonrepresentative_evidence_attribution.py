from __future__ import annotations

import unittest

import torch

from osn_gs.surface.torch_nonrepresentative_evidence_attribution import (
    ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION,
    SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS,
    SUPPORTS_ONE_REPRESENTATIVE_COMPONENT,
    COMPONENT_RELATION_CATEGORIES,
    classify_pre_post_median,
    component_relation_category,
    finalize_component_co_support,
    view_contributor_component_pairs,
)


class ClassifyPrePostMedianTest(unittest.TestCase):
    def test_pre_and_post_events_split_correctly(self):
        # (H=1, W=3, K=2): pixel 0 has surfel 5 pre-median, pixel 1 has
        # surfel 5 post-median, pixel 2 is unused (-1 slots).
        contrib_ids = torch.tensor([[[5, -1], [5, -1], [-1, -1]]], dtype=torch.int64)
        contrib_post_median = torch.tensor([[[0, 0], [1, 0], [0, 0]]], dtype=torch.int64)
        ever_pre, ever_post = classify_pre_post_median(contrib_ids, contrib_post_median, count=10)
        self.assertTrue(bool(ever_pre[5].item()))
        self.assertTrue(bool(ever_post[5].item()))
        for other in (0, 1, 2, 3, 4, 6, 7, 8, 9):
            self.assertFalse(bool(ever_pre[other].item()))
            self.assertFalse(bool(ever_post[other].item()))

    def test_no_valid_slots_yields_all_false(self):
        contrib_ids = torch.full((1, 4, 3), -1, dtype=torch.int64)
        contrib_post_median = torch.zeros((1, 4, 3), dtype=torch.int64)
        ever_pre, ever_post = classify_pre_post_median(contrib_ids, contrib_post_median, count=5)
        self.assertFalse(bool(ever_pre.any()))
        self.assertFalse(bool(ever_post.any()))

    def test_mutually_exclusive_within_a_single_event(self):
        # A single event is either pre-or-at or post, never counted as both
        # from ONE slot -- only aggregation across multiple events can mark
        # a primitive true in both.
        contrib_ids = torch.tensor([[[7]]], dtype=torch.int64)
        contrib_post_median = torch.tensor([[[1]]], dtype=torch.int64)
        ever_pre, ever_post = classify_pre_post_median(contrib_ids, contrib_post_median, count=8)
        self.assertFalse(bool(ever_pre[7].item()))
        self.assertTrue(bool(ever_post[7].item()))


class ContributorComponentPairsTest(unittest.TestCase):
    def test_single_view_single_component_pair(self):
        # 2x1 image, K=1: pixel (0,0) has contributor 3, representative 0;
        # pixel (0,1) has contributor 3 again, representative 0 (same
        # component) -- must dedupe to one pair within this view.
        contrib_ids = torch.tensor([[[3], [3]]], dtype=torch.int64)
        representative_id = torch.tensor([[0, 0]], dtype=torch.int64)
        subset_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int64)  # node 0's component = 100
        pairs = view_contributor_component_pairs(contrib_ids, representative_id, subset_ids)
        self.assertEqual(int(pairs.shape[0]), 1)
        self.assertEqual(pairs[0].tolist(), [3, 100])

    def test_no_representative_at_pixel_yields_no_pair(self):
        contrib_ids = torch.tensor([[[3]]], dtype=torch.int64)
        representative_id = torch.tensor([[-1]], dtype=torch.int64)
        subset_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int64)
        pairs = view_contributor_component_pairs(contrib_ids, representative_id, subset_ids)
        self.assertEqual(int(pairs.shape[0]), 0)

    def test_two_distinct_components_in_one_view(self):
        # contributor 5 co-occurs with representative 0 (component 100) at
        # pixel A and representative 1 (component 200) at pixel B.
        contrib_ids = torch.tensor([[[5], [5]]], dtype=torch.int64)
        representative_id = torch.tensor([[0, 1]], dtype=torch.int64)
        subset_ids = torch.tensor([100, 200], dtype=torch.int64)
        pairs = view_contributor_component_pairs(contrib_ids, representative_id, subset_ids)
        pair_set = {tuple(row) for row in pairs.tolist()}
        self.assertEqual(pair_set, {(5, 100), (5, 200)})


class FinalizeComponentCoSupportTest(unittest.TestCase):
    def test_one_contributor_co_supports_exactly_one_component(self):
        pairs_view1 = torch.tensor([[3, 100]], dtype=torch.int64)
        pairs_view2 = torch.tensor([[3, 100]], dtype=torch.int64)  # same relation, different view
        result = finalize_component_co_support([pairs_view1, pairs_view2], count=10, subset_count=1000)
        self.assertEqual(int(result["distinct_component_count"][3].item()), 1)
        self.assertEqual(int(result["unique_pairs"].shape[0]), 1)

    def test_one_contributor_co_supports_multiple_components(self):
        pairs_view1 = torch.tensor([[3, 100]], dtype=torch.int64)
        pairs_view2 = torch.tensor([[3, 200]], dtype=torch.int64)
        pairs_view3 = torch.tensor([[3, 300]], dtype=torch.int64)
        result = finalize_component_co_support([pairs_view1, pairs_view2, pairs_view3], count=10, subset_count=1000)
        self.assertEqual(int(result["distinct_component_count"][3].item()), 3)

    def test_unassociated_contributor_has_zero_distinct_components(self):
        pairs_view1 = torch.tensor([[3, 100]], dtype=torch.int64)
        result = finalize_component_co_support([pairs_view1], count=10, subset_count=1000)
        self.assertEqual(int(result["distinct_component_count"][7].item()), 0)

    def test_empty_batches_yield_zero_everywhere(self):
        result = finalize_component_co_support([], count=5, subset_count=100)
        self.assertTrue(bool((result["distinct_component_count"] == 0).all()))
        self.assertEqual(int(result["unique_pairs"].shape[0]), 0)


class ComponentRelationCategoryTest(unittest.TestCase):
    def test_categories_assigned_correctly(self):
        distinct = torch.tensor([0, 1, 1, 2, 5], dtype=torch.int64)
        mask = torch.ones((5,), dtype=torch.bool)
        category = component_relation_category(distinct, mask)
        labels = [COMPONENT_RELATION_CATEGORIES[i] for i in category.tolist()]
        self.assertEqual(labels, [
            ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION,
            SUPPORTS_ONE_REPRESENTATIVE_COMPONENT,
            SUPPORTS_ONE_REPRESENTATIVE_COMPONENT,
            SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS,
            SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS,
        ])


if __name__ == "__main__":
    unittest.main()
