"""Focused tests for the W174 reference-surface diagnostic utilities.

Synthetic fixtures validate the diagnostics only; they are not an architecture
experiment and no production behaviour is exercised or changed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from devtools.demo import worklog_174_surface_complex as sc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/174_reference_surface_complex_attribution"


def corners_for(cell, field):
    """Sample a scalar field at the eight corners in the frozen W154 order."""
    values = np.zeros(8, dtype=np.float64)
    for index, (dx, dy, dz) in enumerate(sc.CORNER_OFFSETS):
        values[index] = field(cell[0] + dx, cell[1] + dy, cell[2] + dz)
    return values


def build(cells, field, h=1.0):
    cells = np.asarray(cells, dtype=np.int64)
    corner_values = np.stack([corners_for(cell, field) for cell in cells])
    return sc.build_surface_complex(corner_values, cells, h)


PLANE = lambda x, y, z: float(z) - 0.5


def test_single_cell_plane_produces_contained_triangles():
    complex_data = build([[0, 0, 0]], PLANE)
    assert len(complex_data["triangles"]) > 0
    assert complex_data["unit_cell_containment_violations"] == 0
    assert complex_data["per_cell_patch_count"][0] == 1
    # Every vertex must sit inside the owning cell's world extent.
    assert np.all(complex_data["vertices"] >= 0.5 - 1e-9)
    assert np.all(complex_data["vertices"] <= 1.5 + 1e-9)


def test_adjacent_cells_weld_into_one_component():
    """The welding rule must join cells that cut the same lattice edge."""
    complex_data = build([[0, 0, 0], [1, 0, 0]], PLANE)
    topology = sc.complex_topology(complex_data)
    assert topology["triangle_connected_components"] == 1, "adjacent cells failed to weld"
    # Welding must not duplicate the shared crossing vertices.
    assert topology["welded_vertex_count"] < 4 * len(complex_data["triangles"])
    assert topology["non_manifold_edge_count"] == 0


def test_separated_cells_stay_distinct_components():
    """Cells that share no lattice edge must not be merged by proximity."""
    complex_data = build([[0, 0, 0], [5, 0, 0]], PLANE)
    topology = sc.complex_topology(complex_data)
    assert topology["triangle_connected_components"] == 2


def test_triangle_ownership_is_construction_native():
    cells = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    complex_data = build(cells, PLANE)
    owner = complex_data["triangle_owner_row"]
    assert len(owner) == len(complex_data["triangles"])
    assert set(owner.tolist()) <= {0, 1, 2}
    # Each owning row must account for exactly its own per-cell triangle count.
    for row in range(len(cells)):
        assert int((owner == row).sum()) == int(complex_data["per_cell_triangle_count"][row])


def test_component_labels_are_deterministic():
    complex_data = build([[0, 0, 0], [1, 0, 0], [6, 0, 0]], PLANE)
    first = sc.triangle_components(complex_data["triangles"])
    second = sc.triangle_components(complex_data["triangles"])
    assert np.array_equal(first, second)


def test_boundary_edges_counted_on_open_sheet():
    """A finite plane patch is an open sheet: it must report boundary edges."""
    complex_data = build([[0, 0, 0], [1, 0, 0]], PLANE)
    topology = sc.complex_topology(complex_data)
    assert topology["boundary_edge_count"] > 0
    assert topology["boundary_edge_components"] >= 1
    assert topology["edge_degree_histogram"].get(1, 0) == topology["boundary_edge_count"]


def test_surface_shortest_path_finds_and_rejects():
    connected = build([[0, 0, 0], [1, 0, 0], [2, 0, 0]], PLANE)
    owner = connected["triangle_owner_row"]
    path = sc.surface_shortest_path(
        connected["triangles"], np.flatnonzero(owner == 0), np.flatnonzero(owner == 2)
    )
    assert path is not None and path["edge_steps"] >= 1

    split = build([[0, 0, 0], [7, 0, 0]], PLANE)
    owner = split["triangle_owner_row"]
    missing = sc.surface_shortest_path(
        split["triangles"], np.flatnonzero(owner == 0), np.flatnonzero(owner == 1)
    )
    assert missing is None, "disconnected pieces must not report a path"


def test_shortest_path_is_minimal_on_a_strip():
    """BFS must return a minimal step count, not merely some walk."""
    cells = [[i, 0, 0] for i in range(6)]
    complex_data = build(cells, PLANE)
    owner = complex_data["triangle_owner_row"]
    path = sc.surface_shortest_path(
        complex_data["triangles"], np.flatnonzero(owner == 0), np.flatnonzero(owner == 5)
    )
    assert path is not None
    direct = sc.surface_shortest_path(
        complex_data["triangles"], np.flatnonzero(owner == 0), np.flatnonzero(owner == 1)
    )
    assert direct is not None
    assert path["edge_steps"] > direct["edge_steps"], "distance must grow along the strip"


def test_multi_patch_cell_is_detected():
    """A cell whose scalars produce two separated patches must report 2."""
    # Two parallel crossings inside one cell: value depends on z with a fold.
    values = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    single = sc.build_surface_complex(values[None, :], np.array([[0, 0, 0]]), 1.0)
    assert single["per_cell_patch_count"][0] == 1

    # Opposite corners negative produces two disjoint corner patches.
    diagonal = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    split = sc.build_surface_complex(diagonal[None, :], np.array([[0, 0, 0]]), 1.0)
    assert split["per_cell_patch_count"][0] >= 2


@pytest.mark.skipif(not (OUT / "worklog_174_report.json").exists(), reason="W174 outputs not generated")
class TestFrozenW174Outputs:
    @pytest.fixture(scope="class")
    def report(self):
        return json.loads((OUT / "worklog_174_report.json").read_text(encoding="utf-8"))

    def test_baseline_and_contract_preserved(self, report):
        assert report["baseline"]["selected_support"] == 15189
        assert report["baseline"]["w173_loops"] == 134
        assert report["baseline"]["w173_chart_holes"] == 134
        assert report["frozen_inputs_unchanged_after_run"] is True
        for flag in ("region_changed", "chart_changed", "support_changed",
                     "nurbs_fitted", "production_changed"):
            assert report[flag] is False, flag
        assert report["reference_geometry_contract"]["no_distance_or_threshold_matching"] is True

    def test_ownership_is_exact(self, report):
        surface = report["surface_complex"]
        assert surface["cells_producing_no_triangle"] == 0
        assert surface["unit_cell_containment_violations"] == 0

    def test_witness_reproduces_w173_collision(self, report):
        witness = report["witness"]
        assert witness["rows"] == [4043, 4051]
        assert witness["same_chart_bin"] is True
        # W173 measured 32.88685 h of chart-normal height separation.
        assert witness["chart_normal_height_difference_in_h"] == pytest.approx(32.88685, abs=1e-3)
        assert witness["same_triangle_connected_component"] is True
        assert witness["surface_path"]["edge_steps"] > 0
        assert report["witness_classification"] == "CHART_COLLAPSE_SAME_SURFACE"

    def test_cell_surface_relation_direction(self, report):
        relation = report["cell_surface_relation"]
        # Native connectivity over-states surface adjacency; never the reverse.
        assert relation["surface_adjacent_but_not_native_adjacent"] == 0
        assert relation["native_adjacent_but_not_surface_adjacent"] > 0
        assert (relation["of_those_still_same_triangle_component"]
                + relation["of_those_in_different_triangle_components"]
                == relation["native_adjacent_but_not_surface_adjacent"])

    def test_no_one_to_one_loop_correspondence_claimed(self, report):
        assert report["loop_correspondence"]["one_to_one_correspondence_established"] is False

    def test_attribution_is_not_overclaimed(self, report):
        attribution = report["architecture_attribution"]
        assert attribution["verdict"] == "MIXED_ATTRIBUTION"
        assert attribution["stopped_before_region_split_or_chart_redesign"] is True
        assert len(attribution["not_established"]) >= 3

    def test_review_artifacts_complete(self, report):
        validation = report["visualization"]["validation"]
        assert validation["complete"] is True
        assert validation["png_count"] >= 8
        assert not list((OUT / "review_views").rglob("*.ppm"))
        for readme in (OUT / "review_views").rglob("README.md"):
            text = readme.read_text(encoding="utf-8")
            assert text.strip(), readme
            if readme.parent.name != "review_views":
                assert "분석 및 평가" in text, readme

    def test_component_inventory_is_complete(self, report):
        inventory = (OUT / "loop_inventory.md").read_text(encoding="utf-8")
        rows = [
            line for line in inventory.splitlines()
            if line.startswith("| ") and re.fullmatch(r"C\d{3}", line.split("|")[2].strip())
        ]
        assert len(rows) == report["surface_complex"]["triangle_connected_components"]
        # The full tail must be present, including every single-triangle fragment.
        assert rows[-1].split("|")[3].strip() == "1"
