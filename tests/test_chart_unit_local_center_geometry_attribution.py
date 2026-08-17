from __future__ import annotations

import torch

from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
    LOCALLY_SINGLE_CURVED_SHEET,
    LOCALLY_THICK_UNIMODAL_SHEET,
    SPARSE_SATELLITE_OR_OUTLIER,
    TRUE_PERSISTENT_MULTI_LAYER,
    TRUE_PERSISTENT_TWO_LAYER,
    attribute_local_center_geometry,
)


def _grid(n: int, extent: float = 3.0) -> torch.Tensor:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def test_flat_single_sheet_is_locally_single_curved_sheet():
    uu, vv = _grid(8)
    points = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    result = attribute_local_center_geometry(points, list(range(points.shape[0])))
    assert result.primary_class == LOCALLY_SINGLE_CURVED_SHEET
    assert result.class_node_fractions[TRUE_PERSISTENT_TWO_LAYER] == 0.0
    assert result.class_node_fractions[TRUE_PERSISTENT_MULTI_LAYER] == 0.0


def test_globally_curved_bowl_is_not_misclassified_as_multilayer():
    # Worklog 91's confound: a bowl-shaped single sheet spans a wide global
    # depth range relative to ONE global plane, but every local neighborhood
    # is still a single thin locally-planar patch (barring a few
    # high-curvature neighborhoods, which correctly fall to
    # SPARSE_SATELLITE_OR_OUTLIER via their own tiny mode populations rather
    # than being promoted to a false persistent layer).
    uu, vv = _grid(10, extent=4.0)
    zz = 0.15 * (uu**2 + vv**2)
    points = torch.stack([uu, vv, zz], dim=1)
    result = attribute_local_center_geometry(points, list(range(points.shape[0])))
    assert result.class_node_fractions[TRUE_PERSISTENT_TWO_LAYER] < 0.1
    assert result.class_node_fractions[TRUE_PERSISTENT_MULTI_LAYER] < 0.1
    assert result.primary_class in (
        LOCALLY_SINGLE_CURVED_SHEET, LOCALLY_THICK_UNIMODAL_SHEET, SPARSE_SATELLITE_OR_OUTLIER,
    )


def test_thick_unimodal_sheet_is_not_called_true_layer():
    # One mode, but with a wide, continuous (not gapped) spread along Z --
    # thick single sheet, not two competing layers. Checked across several
    # seeds: a bare gap-ratio test is noisy at small local neighborhood
    # size (k~8), so this guards the silhouette-style side-spread check
    # that rejects spurious order-statistic gaps in smooth unimodal noise.
    uu, vv = _grid(8)
    for seed in range(5):
        torch.manual_seed(seed)
        zz = torch.rand_like(uu) * 4.0
        points = torch.stack([uu, vv, zz], dim=1)
        result = attribute_local_center_geometry(points, list(range(points.shape[0])))
        assert result.class_node_fractions[TRUE_PERSISTENT_TWO_LAYER] < 0.1
        assert result.class_node_fractions[TRUE_PERSISTENT_MULTI_LAYER] < 0.1


def test_two_persistent_interleaved_sheets_are_true_persistent_two_layer():
    # Two flat sheets sharing the same X-Y grid (so every local
    # neighborhood is genuinely mixed, not same-sheet-only under kNN),
    # offset by a Z gap that dominates each sheet's own near-zero internal
    # spread -- the split recurs across spatially neighboring
    # neighborhoods, satisfying persistence.
    uu, vv = _grid(10, extent=3.0)
    sheet_a = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    sheet_b = torch.stack([uu, vv, torch.full_like(uu, 0.05)], dim=1)
    points = torch.cat([sheet_a, sheet_b], dim=0)
    result = attribute_local_center_geometry(points, list(range(points.shape[0])))
    assert result.primary_class == TRUE_PERSISTENT_TWO_LAYER
    assert result.class_node_fractions[TRUE_PERSISTENT_TWO_LAYER] > 0.5
    assert result.persistent_layer_count == 2


def test_isolated_single_neighborhood_split_is_not_persistent():
    # One flat sheet with a single stray point offset far in Z from just
    # ONE location -- multi-modal at that one neighborhood only (if at
    # all -- a lone singleton outlier surrounded by a dominant flat sheet
    # is exactly the SPARSE_SATELLITE_OR_OUTLIER case this class exists
    # for), with no neighboring neighborhood also showing a split. Must
    # never be promoted to a true persistent layer either way.
    uu, vv = _grid(8)
    points = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    stray = torch.tensor([[0.0, 0.0, 5.0]])
    points = torch.cat([points, stray], dim=0)
    result = attribute_local_center_geometry(points, list(range(points.shape[0])))
    assert result.class_node_fractions[TRUE_PERSISTENT_TWO_LAYER] == 0.0
    assert result.class_node_fractions[TRUE_PERSISTENT_MULTI_LAYER] == 0.0
    assert result.class_by_member[-1] != TRUE_PERSISTENT_TWO_LAYER
    assert result.class_by_member[-1] != TRUE_PERSISTENT_MULTI_LAYER


def test_tiny_member_count_is_sparse_satellite_by_construction():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = attribute_local_center_geometry(points, [0, 1, 2])
    assert result.primary_class == SPARSE_SATELLITE_OR_OUTLIER
    assert all(cls == SPARSE_SATELLITE_OR_OUTLIER for cls in result.class_by_member)


def test_never_reads_covariance_signature():
    import ast
    import inspect

    from osn_gs.surface import torch_chart_unit_local_center_geometry_attribution as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("covariance" in name.lower() for name in imported_names)
    # No function signature in this module accepts a covariance argument.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            arg_names = [arg.arg for arg in node.args.args]
            assert not any("covariance" in name.lower() for name in arg_names)
