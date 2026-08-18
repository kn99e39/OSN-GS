from __future__ import annotations

import torch

from osn_gs.surface.torch_patch_identifiability import assess_patch_identifiability


def test_regularized_fit_not_categorically_rejected_below_36_samples():
    # A small (5-sample) chart against a 6x6=36-control-variable grid is
    # underdetermined, but the regularized solver never requires full
    # column rank -- as long as the 5 samples are themselves independent
    # (rank == sample_count), this must be reported identifiable, not
    # rejected outright by a sample-count cutoff.
    torch.manual_seed(0)
    uv = torch.rand(5, 2)
    report = assess_patch_identifiability(uv, 2, 2, 6, 6)
    assert report.sample_count == 5
    assert report.control_variable_count == 36
    assert report.identifiable is True
    assert report.effective_rank == report.achievable_rank == 5


def test_bspline_basis_rank_computation_matches_shape():
    coords = torch.linspace(0.0, 1.0, 10)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    report = assess_patch_identifiability(uv, 2, 2, 6, 6)
    assert report.identifiable is True
    assert report.effective_rank == 36
    assert report.condition_number is not None
    assert len(report.singular_values) == 36


def test_rank_deficiency_for_geometrically_collapsed_uv():
    # All samples collapse to the same single u value -- the design matrix
    # cannot resolve any u-direction variation regardless of sample count.
    torch.manual_seed(1)
    uv = torch.stack([torch.zeros(20), torch.rand(20)], dim=1)
    report = assess_patch_identifiability(uv, 2, 2, 4, 4)
    assert report.identifiable is False
    assert "degenerate_u_extent" in (report.invalid_reason or "")
    assert report.u_constrained is False


def test_valid_3x3_quadratic_identification():
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.5, 0.0]])
    report = assess_patch_identifiability(uv, 2, 2, 3, 3)
    assert report.identifiable is True
    assert report.effective_rank == report.achievable_rank


def test_valid_2x2_linear_identification():
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    report = assess_patch_identifiability(uv, 1, 1, 2, 2)
    assert report.identifiable is True
    assert report.control_variable_count == 4
    assert report.effective_rank == 4


def test_no_samples_is_not_identifiable():
    uv = torch.zeros((0, 2))
    report = assess_patch_identifiability(uv, 2, 2, 3, 3)
    assert report.identifiable is False
    assert report.invalid_reason == "no_samples"


def test_never_uses_fit_or_held_out_error():
    import ast
    import inspect

    from osn_gs.surface import torch_patch_identifiability as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(value="")
    code_only = ast.unparse(tree)
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)
    assert "held_out" not in code_only
    assert "extrapolat" not in code_only.lower()
    assert "unsafe" not in code_only.lower()
    assert "render" not in code_only.lower()


def test_uses_real_basis_table_machinery_not_reimplementation():
    import ast
    import inspect

    from osn_gs.surface import torch_patch_identifiability as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "TorchNURBSSurface" in imported_names
