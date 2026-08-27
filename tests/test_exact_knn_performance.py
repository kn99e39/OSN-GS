import torch

from osn_gs.surface.torch_coverage_first_subset_partition import _knn
from osn_gs.surface.torch_exact_knn_performance import scipy_ckdtree_exact_knn


def test_scipy_exact_knn_matches_reference_on_distinct_random_points():
    torch.manual_seed(91)
    points = torch.rand(257, 3)
    expected_index, expected_distance = _knn(points, 8, 64, None)
    actual_index, actual_distance = scipy_ckdtree_exact_knn(points, 8, workers=1)
    assert torch.equal(actual_index, expected_index)
    assert torch.equal(actual_distance, expected_distance)


def test_scipy_exact_knn_excludes_row_not_zero_distance():
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    index, distance = scipy_ckdtree_exact_knn(points, 2, workers=1)
    rows = torch.arange(len(points))[:, None]
    assert not bool((index == rows).any())
    assert int(index[0, 0]) == 1
    assert float(distance[0, 0]) == 0.0
