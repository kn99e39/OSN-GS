import torch

from osn_gs.surface.torch_exact_knn_performance import scipy_ckdtree_exact_knn
from osn_gs.surface.torch_knn_reference_attribution import (
    CLASS_DUPLICATE_OR_SELF,
    CLASS_EXACT_DISTANCE_TIE,
    CLASS_FLOAT32_REFERENCE_TIE,
    CLASS_MATERIAL_DISAGREEMENT,
    CLASS_NEAR_TIE_ERROR_BOUND,
    CLASS_ORDER_ONLY,
    adversarial_knn_fixtures,
    boundary_margins,
    classify_neighbor_mismatches,
    observe_reference_knn,
)


def _classify(points, reference, candidate, reference_raw, candidate_raw):
    return classify_neighbor_mismatches(
        points, torch.tensor(reference), torch.tensor(reference_raw),
        torch.tensor(candidate), torch.tensor(candidate_raw),
    )


def test_reference_repeatability_order_membership_and_self_exclusion():
    points, k = adversarial_knn_fixtures()["S8_clustered_local_geometry"]
    first = observe_reference_knn(points, k, 37)
    second = observe_reference_knn(points, k, 37)
    assert torch.equal(first.neighbor_index, second.neighbor_index)
    assert torch.equal(first.neighbor_distance, second.neighbor_distance)
    assert torch.equal(first.boundary_index, second.boundary_index)
    assert torch.equal(first.boundary_raw_distance, second.boundary_raw_distance)
    rows = torch.arange(len(points))[:, None]
    assert not bool((first.neighbor_index == rows).any())
    assert bool((first.boundary_raw_distance[:, 1:] >= first.boundary_raw_distance[:, :-1]).all())


def test_default_cdist_path_matches_explicit_mm_when_shape_requires_mm():
    points, k = adversarial_knn_fixtures()["S1_well_separated_random"]
    default = observe_reference_knn(points, k, 64)
    explicit = observe_reference_knn(
        points, k, 64, compute_mode="use_mm_for_euclid_dist_if_necessary"
    )
    forced_mm = observe_reference_knn(
        points, k, 64, compute_mode="use_mm_for_euclid_dist"
    )
    assert torch.equal(default.boundary_index, explicit.boundary_index)
    assert torch.equal(default.boundary_raw_distance, explicit.boundary_raw_distance)
    assert torch.equal(default.boundary_index, forced_mm.boundary_index)
    assert torch.equal(default.boundary_raw_distance, forced_mm.boundary_raw_distance)


def test_exact_duplicate_is_valid_neighbor_and_not_self():
    points, k = adversarial_knn_fixtures()["S2_exact_duplicate_coordinates"]
    index, distance = scipy_ckdtree_exact_knn(points, k, workers=1)
    rows = torch.arange(len(points))[:, None]
    assert not bool((index == rows).any())
    assert int(index[0, 0]) == 1
    assert float(distance[0, 0]) == 0.0


def test_order_only_mismatch_classification():
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    result = _classify(
        points, [[1, 2], [0, 2], [1, 0]], [[2, 1], [0, 2], [1, 0]],
        [[1.0, 2.0, 3.0], [1.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
        [[2.0, 1.0], [1.0, 1.0], [1.0, 2.0]],
    )
    assert int(result["primary_class"][0]) == CLASS_ORDER_ONLY
    assert bool(result["order_only"][0])


def test_duplicate_effect_has_priority_over_order_only():
    points = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    result = _classify(
        points, [[1, 2], [0, 2], [0, 1]], [[2, 1], [0, 2], [0, 1]],
        [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    assert int(result["primary_class"][0]) == CLASS_DUPLICATE_OR_SELF


def test_exact_distance_tie_classification_at_k_boundary():
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [3.0, 0, 0]])
    result = _classify(
        points, [[1], [0], [0], [1]], [[2], [0], [0], [1]],
        [[1.0, 1.0], [1.0, 2.0], [1.0, 2.0], [2.0, 3.0]],
        [[1.0], [1.0], [1.0], [2.0]],
    )
    assert int(result["primary_class"][0]) == CLASS_EXACT_DISTANCE_TIE
    assert bool(result["boundary_only"][0])


def test_float32_reference_tie_with_mathematically_different_distances():
    one = torch.tensor(1.0)
    farther = torch.nextafter(one, torch.tensor(float("inf")))
    points = torch.tensor([[0.0, 0, 0], [float(farther), 0, 0], [1.0, 0, 0]])
    tied = float(torch.tensor(1.0))
    result = _classify(
        points, [[1], [2], [1]], [[2], [2], [1]],
        [[tied, 2.0], [float(farther - 1), 1.0], [float(farther - 1), 1.0]],
        [[tied], [0.0], [float(farther - 1)]],
    )
    assert int(result["primary_class"][0]) == CLASS_FLOAT32_REFERENCE_TIE


def test_derived_error_bound_separates_near_tie_from_material_disagreement():
    base = torch.tensor(1_000_000.0)
    step = torch.nextafter(base, torch.tensor(float("inf"))) - base
    points = torch.tensor([
        [float(base), 0, 0], [float(base + step), 0, 0],
        [float(base + 2 * step), 0, 0], [float(base + 100 * step), 0, 0],
    ])
    near = _classify(
        points, [[2], [0], [1], [2]], [[1], [0], [1], [2]],
        [[float(2 * step), 1.0], [float(step), 1.0], [float(step), 1.0], [float(98 * step), 100.0]],
        [[float(step)], [float(step)], [float(step)], [float(98 * step)]],
    )
    assert int(near["primary_class"][0]) == CLASS_NEAR_TIE_ERROR_BOUND

    material_points = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [1.0, 0, 0]])
    material = _classify(
        material_points, [[1], [2], [0]], [[2], [2], [0]],
        [[10.0, 11.0], [9.0, 10.0], [1.0, 9.0]],
        [[1.0], [9.0], [1.0]],
    )
    assert int(material["primary_class"][0]) == CLASS_MATERIAL_DISAGREEMENT


def test_k_boundary_margin_uses_k_and_kplus1_without_selection_epsilon():
    raw = torch.tensor([[1.0, 2.0, 2.5], [3.0, 3.0, 4.0]])
    absolute, relative = boundary_margins(raw, 2)
    torch.testing.assert_close(absolute, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(relative, torch.tensor([0.25, 1.0 / 3.0]))
