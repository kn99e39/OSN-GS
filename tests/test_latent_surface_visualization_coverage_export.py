from __future__ import annotations

import sys
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from osn_gs.surface.torch_latent_surface_coverage_audit import LatentSupportUnit  # noqa: E402
from osn_gs.surface.torch_latent_surface_visualization_coverage import (  # noqa: E402
    materialize_unit_with_subdivision,
)


def _mixed_unit() -> LatentSupportUnit:
    # A chain of 10 collinear points -- large enough to attempt a direct
    # fit; if it fails, subdivision must still account for every node.
    coords = torch.linspace(0.0, 1.0, 10)
    positions = torch.stack([coords, torch.zeros(10), torch.zeros(10)], dim=1)
    edges = tuple((i, i + 1) for i in range(9))
    return LatentSupportUnit(
        unit_id=5, node_indices=tuple(range(200, 210)),
        raw_positions=positions, latent_positions=positions,
        projection_displacement=torch.zeros_like(positions), normals=torch.zeros_like(positions), edges=edges,
    )


def test_certificate_accounting_identity_holds():
    unit = _mixed_unit()
    materialized, unrepresented = materialize_unit_with_subdivision(unit)
    represented_nodes = {node for fragment in materialized for node in fragment.node_indices}
    unrepresented_nodes = {node for fragment in unrepresented for node in fragment.node_indices}
    assert represented_nodes | unrepresented_nodes == set(unit.node_indices)
    assert not (represented_nodes & unrepresented_nodes)


def test_single_node_and_pair_units_report_unrepresented_not_crash():
    single = LatentSupportUnit(
        unit_id=6, node_indices=(500,),
        raw_positions=torch.tensor([[1.0, 2.0, 3.0]]), latent_positions=torch.tensor([[1.0, 2.0, 3.0]]),
        projection_displacement=torch.zeros((1, 3)), normals=torch.zeros((1, 3)), edges=(),
    )
    materialized, unrepresented = materialize_unit_with_subdivision(single)
    assert materialized == []
    assert len(unrepresented) == 1
    assert unrepresented[0].node_indices == (500,)
    assert "insufficient_points_for_any_surface" in unrepresented[0].reason
