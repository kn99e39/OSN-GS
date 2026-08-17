from __future__ import annotations

import torch

from osn_gs.surface.torch_chart_unit_surface_topology_temporal_lineage import (
    compute_center_geometry_layering,
    compute_covariance_only_ambiguity,
)


def test_single_sheet_centers_report_one_layer():
    # A flat 4x4 grid in the XY plane: centers all share one Z, so PCA-normal
    # depth clustering must find exactly one layer.
    xs, ys = torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij")
    points = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(16)], dim=1)
    layering = compute_center_geometry_layering(points, list(range(16)))
    assert layering.layer_count == 1
    assert layering.multilayer is False
    assert layering.depth_separation is None


def test_two_well_separated_sheets_report_two_layers():
    # Two flat sheets, each spanning a wide X-Y extent, offset by a large gap
    # along Z relative to their own in-sheet spacing -- true positional
    # multilayer. Using a wide (10x10) in-plane extent keeps the sheets'
    # own spread from competing with the inter-sheet Z gap for the PCA
    # least-variance axis.
    xs, ys = torch.meshgrid(torch.linspace(0, 10, 5), torch.linspace(0, 10, 5), indexing="ij")
    sheet_a = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(25)], dim=1)
    sheet_b = sheet_a.clone()
    sheet_b[:, 2] = 5.0
    points = torch.cat([sheet_a, sheet_b], dim=0)
    layering = compute_center_geometry_layering(points, list(range(50)))
    assert layering.layer_count == 2
    assert layering.multilayer is True
    assert layering.depth_separation is not None
    assert layering.depth_separation > 4.0


def test_tiny_member_count_is_single_layer_by_construction():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    layering = compute_center_geometry_layering(points, [0, 1])
    assert layering.member_count == 2
    assert layering.layer_count == 1
    assert layering.multilayer is False


def test_covariance_only_ambiguity_requires_single_sheet_centers():
    xs, ys = torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij")
    points = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(16)], dim=1)
    layering = compute_center_geometry_layering(points, list(range(16)))
    conflict_mask = torch.zeros(16, dtype=torch.bool)
    conflict_mask[:4] = True
    result = compute_covariance_only_ambiguity(layering, conflict_mask)
    assert result.covariance_only_ambiguous is True
    assert abs(result.covariance_only_ambiguous_node_fraction - 0.25) < 1e-6


def test_covariance_only_ambiguity_is_false_when_centers_are_true_multilayer():
    xs, ys = torch.meshgrid(torch.linspace(0, 10, 5), torch.linspace(0, 10, 5), indexing="ij")
    sheet_a = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(25)], dim=1)
    sheet_b = sheet_a.clone()
    sheet_b[:, 2] = 5.0
    points = torch.cat([sheet_a, sheet_b], dim=0)
    layering = compute_center_geometry_layering(points, list(range(50)))
    conflict_mask = torch.ones(50, dtype=torch.bool)
    result = compute_covariance_only_ambiguity(layering, conflict_mask)
    # True positional multilayer is present, so the covariance conflict here
    # is not classified as a pure representation artifact.
    assert result.covariance_only_ambiguous is False
    assert result.covariance_only_ambiguous_node_fraction == 0.0
