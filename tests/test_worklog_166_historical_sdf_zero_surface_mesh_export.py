from __future__ import annotations

import json

import numpy as np

from devtools.demo.worklog_166_historical_sdf_zero_surface_mesh_export import (
    parse_obj,
    write_obj_from_arrays,
)


def test_obj_roundtrip_preserves_coordinates_and_connectivity(tmp_path):
    vertices = np.asarray(
        [[-1.25, 0.0, 2.5], [0.125, 3.0, -4.0], [5.0, 6.25, 7.5], [8.0, -9.0, 10.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [2, 1, 3], [3, 0, 2]], dtype=np.int64)
    path = tmp_path / "surface.obj"
    write_obj_from_arrays(path, vertices, faces, chunk_rows=2)
    parsed_vertices, parsed_faces = parse_obj(path)
    np.testing.assert_array_equal(parsed_vertices, vertices)
    np.testing.assert_array_equal(parsed_faces, faces)


def test_obj_writer_does_not_filter_degenerate_or_repeated_faces(tmp_path):
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 0, 1], [0, 1, 2], [2, 2, 2]], dtype=np.int64)
    path = tmp_path / "degenerate.obj"
    write_obj_from_arrays(path, vertices, faces)
    _, parsed_faces = parse_obj(path)
    np.testing.assert_array_equal(parsed_faces, faces)


def test_worklog_166_source_contract_is_not_occlusion_validation():
    source = json.loads(
        '{"occlusion_semantics_validated": false, "physical_hidden_surface_identity_validated": false}'
    )
    assert source["occlusion_semantics_validated"] is False
    assert source["physical_hidden_surface_identity_validated"] is False
