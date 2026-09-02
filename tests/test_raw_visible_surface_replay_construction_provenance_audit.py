from types import SimpleNamespace

import numpy as np

from devtools.demo import raw_visible_surface_replay_construction_provenance_audit as wl153


def test_wl152_baseline_and_historical_core_are_identifiable():
    audit = wl153.build_audit()

    assert audit["baseline_reconciliation"]["checks"]
    assert all(audit["baseline_reconciliation"]["checks"].values())
    historical = audit["historical_raw_surface_contract"]
    assert historical["status"] == "IDENTIFIABLE"
    assert historical["identifiable"] is True
    assert historical["source_manifest"]["core_exact"] is True
    assert audit["available_artifact_contract"]["faces"] == 0
    assert audit["architecture_verdict"]["canonical_production_modified"] is False


def test_frozen_input_contract_keeps_camera_and_historical_parameters():
    manifest = wl153._input_manifest()

    assert manifest["checkpoint_iteration"] == 30000
    assert manifest["camera_contract"] == {
        "images": "images_8",
        "sparse_dir": "sparse/0",
        "resolution": -1,
        "llffhold": 8,
        "expected_train_cameras": 161,
    }
    assert manifest["files"]["checkpoint"]["status"] == "AVAILABLE_AND_HASHED"
    assert manifest["files"]["camera_image_dimensions"]["count"] == 185
    assert manifest["historical_h"] == wl153.EXPECTED["h"]
    assert manifest["historical_mu"] == wl153.EXPECTED["mu"]
    assert manifest["historical_tsdf"]["extraction_block"] == 64
    assert manifest["historical_tsdf"]["batch_blocks"] == 6


def test_native_topology_accounting_uses_faces_without_repair_or_selection():
    # Closed tetrahedron, a deliberately non-manifold open component, one
    # isolated vertex, and one degenerate-index face.  No repair/merge/filter
    # is permitted in the WL153 accounting function.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [3.0, 1.0, 1.0],
            [9.0, 9.0, 9.0],  # isolated
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],  # closed tetrahedron
            [4, 5, 6],
            [5, 4, 7],
            [4, 5, 8],  # edge (4,5) has three incident faces
            [10, 10, 11],  # degenerate index/zero area face
        ],
        dtype=np.int64,
    )
    result = wl153._topology_accounting(SimpleNamespace(vertices=vertices, faces=faces))

    assert result["vertex_count"] == 12
    assert result["face_count"] == 8
    assert result["connected_component_count"] == 4
    # The degenerate face contributes its native (10,10) self-edge to the
    # incidence accounting; this is intentionally not repaired away.
    assert result["boundary_edge_count"] == 7
    assert result["boundary_component_count"] == 2
    assert result["boundary_loop_count_when_well_defined"] is None
    assert result["non_manifold_edge_count"] == 1
    assert result["isolated_vertex_count"] == 1
    assert result["degenerate_index_face_count"] == 1
    assert result["zero_area_face_count"] == 1
    assert result["operation_contract"].startswith("faces adjacency only")


def test_provenance_review_does_not_invent_membership_or_event_mapping():
    provenance, event_probe, secondary = wl153._provenance_and_review()

    assert provenance["forbidden_posthoc_mapping"] is False
    assert provenance["physical_sheet_membership"]["status"] == "NOT_PROVIDED"
    assert provenance["physical_sheet_membership"]["new_membership"] is False
    assert provenance["observation_provenance"]["event_set_per_surface_element"] == "EVENT_LEVEL_NOT_AVAILABLE"
    assert event_probe["event_id"] == 1527
    assert event_probe["blacklist"] is False
    assert event_probe["surface_trace"] == "EVENT_LEVEL_NOT_AVAILABLE"
    assert all(case["classification"] == "NOT_REVIEWABLE" for case in secondary["review"]["cases"])
    assert secondary["boundary"]["promotion_justified"] is False


def test_numbered_output_contract_and_display_preview_are_noncanonical():
    assert wl153.OUTPUT_ROOT.name.startswith("153_")
    assert wl153.TEMP_ROOT.name.startswith("153_")
    assert "typed ``ExtractedSurface``" in wl153.__doc__
    assert "connectivity" in wl153.__doc__
