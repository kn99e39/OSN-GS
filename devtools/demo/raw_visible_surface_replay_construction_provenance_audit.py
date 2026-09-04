"""Worklog 153: replay and provenance audit of WL127's typed Raw Surface.

This module is deliberately narrower than the WL127 driver.  It reuses the
unchanged WL127 scale, sparse projective TSDF and masked marching-cubes core,
then serializes the typed ``ExtractedSurface`` in a new numbered output tree.
It never fits a representative, invents membership, repairs connectivity, or
modifies the canonical pipeline.

The historical point PLY is treated as an immutable reconciliation target only.
The typed replay is classified as semantic rather than exact because no
historical typed arrays or hashes were preserved.  The replay is allowed to
interpret topology only after the measured counts agree with the WL127 record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_ROOT = REPO_ROOT / "output" / "153_raw_visible_surface_replay_construction_provenance_audit"
TEMP_ROOT = REPO_ROOT / "temp" / "153_raw_visible_surface_replay_construction_provenance_audit"

TEMP_COPY_EXCLUDED_NAMES = frozenset({"replay_cache"})


def _mirror_output_to_temp(source: Path, target: Path) -> dict[str, Any]:
    """Mirror lightweight audit artifacts while excluding large replay caches."""
    copied: list[str] = []
    excluded: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in TEMP_COPY_EXCLUDED_NAMES:
            excluded.append(child.name)
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)
        copied.append(child.name)
    return {"copied": sorted(copied), "excluded": sorted(excluded), "excluded_names": sorted(TEMP_COPY_EXCLUDED_NAMES)}

CHECKPOINT = REPO_ROOT / "output" / "arch_2dgs_coverage_first_surface" / "2dgs_run1" / "30000" / "checkpoint.pt"
SOURCE_PATH = REPO_ROOT / "DATASET"
RAW_POINT_PLY = (
    REPO_ROOT / "output" / "confirmed" / "127_osn_gs_evidence_bounded_projective_tsdf"
    / "RENDERER_MEDIAN_SURFACE_POINTS" / "iteration_0000001" / "point_cloud.ply"
)
WL127_ROOT = REPO_ROOT / "output" / "confirmed" / "127_osn_gs_evidence_bounded_projective_tsdf"
WL127_DOC = REPO_ROOT / "docs" / "worklogs" / "127_evidence_bounded_projective_tsdf.md"
WL127_CORE = REPO_ROOT / "scripts" / "devtools" / "evidence_bounded_tsdf"

EXPECTED = {
    "event_count": 1586,
    "event_union_sha256": "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78",
    "point_ply_sha256": "fcdd26129737b6610e86837e5138e084ed3cfb95a80d6db2692b9cf70427107a",
    "point_ply_vertices": 1_212_365,
    "h": 0.012105485424399376,
    "mu": 0.036316456273198128,
    "authoritative_voxels": 76_720_314,
    "eligible_cells": 21_235_312,
    "vertices": 28_694_040,
    "faces": 45_116_659,
    "components": 582_646,
}

# Git blob IDs for the WL127 core at the historical construction commit.  The
# core is checked against these IDs before replay; the post-mesh WL127 driver
# and mesh display helper had later diagnostic-only changes and are not used.
WL127_COMMIT = "943a764"
CORE_BLOB_IDS = {
    "evidence_bounded_tsdf/__init__.py": "966b2ef08b8c388644a85cc8897fa49a25e061e1",
    "evidence_bounded_tsdf/field.py": "fc325d34000797cbfce0fc110ae4f6e3f3a8c134",
    "evidence_bounded_tsdf/extraction.py": "43b438e497ad299b7b42b3948147b4e75121eaa9",
    "evidence_bounded_tsdf/scale.py": "cfdda34645db05f1675981fcd633d5b7bc6c5298",
    "evidence_bounded_tsdf_stages.py": "e051866d23e752ead64232184b61668031917b0a",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _git_blob_hash(path: Path) -> str:
    # Git's ``text=auto`` stores the historical blob with LF even though the
    # Windows working tree may expose CRLF.  Match the Git object, not the
    # platform line ending representation.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    payload = b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    return hashlib.sha1(payload).hexdigest()


def _parse_ply_header(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise AssertionError(f"missing PLY header terminator: {path}")
    text = raw[: end + len(marker)].decode("ascii")
    elements: dict[str, int] = {}
    properties: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 3 and words[0] == "element":
            current = words[1]
            elements[current] = int(words[2])
            properties[current] = []
        elif len(words) >= 3 and words[0] == "property" and current is not None:
            properties[current].append(words[-1])
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "format": next((line for line in text.splitlines() if line.startswith("format ")), None),
        "elements": elements,
        "properties": properties,
        "has_faces": elements.get("face", 0) > 0,
    }


def _image_metadata_hash(image_root: Path) -> tuple[str, list[dict[str, Any]]]:
    from PIL import Image

    rows: list[dict[str, Any]] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        with Image.open(path) as image:
            width, height = image.size
        rows.append({"name": str(path.relative_to(image_root)).replace("\\", "/"), "width": width, "height": height})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded), rows


def _load_wl152_baseline() -> dict[str, Any]:
    from devtools.demo import visible_surface_carrier_contract_audit as wl152

    audit = wl152.build_audit()
    baseline = audit["baseline_reconciliation"]
    checks = {
        "event_count": baseline["event_count"] == EXPECTED["event_count"],
        "event_union_sha256": baseline["event_union_sha256"] == EXPECTED["event_union_sha256"],
        "event_1527": baseline["event_1527"]["event_id"] == 1527,
        "event_1527_v_min_owner": baseline["event_1527"]["v_min_owner"] is True,
        "event_1527_human_review_preserved": baseline["event_1527"]["human_review"] == "CLEAR_NOT_ON_INTENDED_SURFACE",
        "representative_shape": baseline["representative_shape"] == [3840, 3],
        "support_vertices": baseline["support_vertices"] == 314,
        "support_mask_sha256": baseline["support_mask_sha256"] == "23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9",
        "all_four_supported_cells": baseline["all_four_supported_cells"] == 211,
        "raw_point_ply_sha256": baseline["raw_visible_surface_artifact"]["sha256"] == EXPECTED["point_ply_sha256"],
        "raw_point_ply_vertices": baseline["raw_visible_surface_artifact"]["elements"].get("vertex") == EXPECTED["point_ply_vertices"],
        "raw_point_ply_faces": baseline["raw_visible_surface_artifact"]["elements"].get("face", 0) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError("WL152 baseline reconciliation failed: " + json.dumps(checks, sort_keys=True))
    return {"checks": checks, "wl152_baseline": baseline}


def _input_manifest() -> dict[str, Any]:
    image_root = SOURCE_PATH / "images_8"
    sparse_root = SOURCE_PATH / "sparse" / "0"
    files = {
        "checkpoint": CHECKPOINT,
        "colmap_cameras": sparse_root / "cameras.bin",
        "colmap_images": sparse_root / "images.bin",
        "llff_pose_bounds": SOURCE_PATH / "poses_bounds.npy",
    }
    records: dict[str, Any] = {}
    for name, path in files.items():
        records[name] = {
            "path": _relative(path),
            "status": "AVAILABLE_AND_HASHED" if path.exists() else "MISSING",
            "sha256": _sha256_file(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    if image_root.exists():
        image_hash, image_rows = _image_metadata_hash(image_root)
        records["camera_image_dimensions"] = {
            "path": _relative(image_root),
            "status": "AVAILABLE_AND_HASHED",
            "metadata_sha256": image_hash,
            "count": len(image_rows),
            "note": "WL127 loader consumes image dimensions only; image pixels are not loaded into the evidence path",
        }
    else:
        records["camera_image_dimensions"] = {"path": _relative(image_root), "status": "MISSING"}
    records["renderer_median_depth_maps"] = {
        "status": "RECONSTRUCTIBLE_FROM_FROZEN_INPUT",
        "source": "checkpoint + camera matrices + canonical renderer median-depth channel",
        "preserved_all_161_maps": False,
    }
    records["historical_typed_surface_arrays"] = {
        "status": "MISSING",
        "note": "WL152 confirmed only a vertex-only point PLY; no historical vertices/faces/support/value/h arrays or hashes were retained",
    }
    records["wl127_field_cache"] = {
        "path": _relative(WL127_ROOT / "_cache" / "field.npz"),
        "status": "MISSING" if not (WL127_ROOT / "_cache" / "field.npz").exists() else "AVAILABLE_BUT_UNHASHED",
    }
    records["wl127_mesh_cache"] = {
        "path": _relative(WL127_ROOT / "_cache" / "mesh.npz"),
        "status": "MISSING" if not (WL127_ROOT / "_cache" / "mesh.npz").exists() else "AVAILABLE_BUT_UNHASHED",
    }
    qdepth_pyd = Path(os.environ.get("TEMP", "")) / "osn_gs_diff_surfel_rasterization_qdepth" / "osn_gs_diff_surfel_rasterization_qdepth_c.pyd"
    records["canonical_median_depth_renderer_extension"] = {
        "path": str(qdepth_pyd),
        "status": "AVAILABLE_AND_HASHED" if qdepth_pyd.exists() else "MISSING",
        "sha256": _sha256_file(qdepth_pyd) if qdepth_pyd.exists() else None,
        "note": "prebuilt WL127-compatible qdepth extension; used only as a process-local loader input, never modified",
    }
    return {
        "checkpoint_iteration": 30000,
        "camera_contract": {"images": "images_8", "sparse_dir": "sparse/0", "resolution": -1, "llffhold": 8, "expected_train_cameras": 161},
        "files": records,
        "historical_h": EXPECTED["h"],
        "historical_mu": EXPECTED["mu"],
        "historical_tsdf": {
            "truncation": "mu = 3h",
            "fusion_weight": 1,
            "minimum_view_rule": False,
            "unknown": "absence from sparse field",
            "candidate_enumeration": "renderer-event voxel seeds followed by WL127 closure growth, max_rounds=60",
            "extraction_block": 64,
            "batch_blocks": 6,
            "eligibility": "all eight corners authoritative AND min(phi)<=0<=max(phi)",
            "seam_welding": "exact h*1e-6 quantized position key; seam duplicates only",
        },
    }


def _core_source_manifest() -> dict[str, Any]:
    rows = {}
    for relative, expected_blob in CORE_BLOB_IDS.items():
        path = REPO_ROOT / "scripts" / "devtools" / relative
        actual = _git_blob_hash(path) if path.exists() else None
        rows[relative] = {"path": _relative(path), "historical_commit": WL127_COMMIT, "expected_blob": expected_blob, "current_blob": actual, "exact": actual == expected_blob}
    return {
        "historical_commit": WL127_COMMIT,
        "core_files": rows,
        "core_exact": all(row["exact"] for row in rows.values()),
        "driver_note": "Current evidence_bounded_projective_tsdf.py differs only in later diagnostic/export code; WL153 invokes the unchanged scale/field/extraction core directly.",
    }


def build_audit() -> dict[str, Any]:
    baseline = _load_wl152_baseline()
    inputs = _input_manifest()
    source = _core_source_manifest()
    historical_identifiable = (
        baseline["checks"] and inputs["files"]["checkpoint"]["status"] == "AVAILABLE_AND_HASHED"
        and inputs["files"]["colmap_cameras"]["status"] == "AVAILABLE_AND_HASHED"
        and inputs["files"]["colmap_images"]["status"] == "AVAILABLE_AND_HASHED"
        and inputs["files"]["camera_image_dimensions"]["status"] == "AVAILABLE_AND_HASHED"
        and source["core_exact"]
    )
    return {
        "batch": "Worklog 153 — Raw Visible Surface Replay and Construction-Provenance Recovery",
        "baseline_reconciliation": baseline,
        "historical_raw_surface_contract": {
            "identifiable": bool(historical_identifiable),
            "status": "IDENTIFIABLE" if historical_identifiable else "RAW_SURFACE_HISTORICAL_REPLAY_UNRESOLVED",
            "input_manifest": inputs,
            "source_manifest": source,
        },
        "available_artifact_contract": {
            "path": _relative(RAW_POINT_PLY),
            "entity": "vertex-only renderer-median point artifact",
            "typed_extracted_surface": "not preserved in confirmed WL127 output",
            "faces": 0,
            "topology": "unavailable before replay",
        },
        "construction_lineage_contract": {
            "lineage": [
                "renderer median depth channel",
                "pixel-centre unprojection to seed voxel keys",
                "projective TSDF authority/fusion over camera views",
                "all-eight-corner masked zero-level-set cell eligibility",
                "Lewiner marching cubes per cell",
                "exact quantized seam-only welding",
                "typed ExtractedSurface arrays",
            ],
            "persisted_before_wl153": {
                "renderer_observation_to_field_cell": "source semantics only; contributor IDs discarded by fuse_views",
                "active_cell_to_face": "source semantics only; cell owner discarded by extraction",
                "face_to_seam_welded_vertex": "faces and vertices exist transiently in extraction, but no lineage sidecar was saved",
            },
            "forbidden_correspondence_not_used": ["nearest event", "radius/KNN", "normal matching", "reprojection voting", "distance-threshold attribution"],
        },
        "architecture_verdict": {
            "before_replay": "REPLAY_REQUIRED_BEFORE_TOPOLOGY_INTERPRETATION",
            "physical_sheet_membership": "NOT_JUSTIFIED_BY_NATIVE_EXTRACTEDSURFACE",
            "canonical_production_modified": False,
        },
    }


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _save_array_bundle(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _prepare_qdepth_runtime() -> dict[str, Any]:
    """Load the already-built WL127 qdepth extension without rebuilding it.

    The replay environment has the pyd and CUDA runtime but its child Python
    process does not inherit Windows DLL search directories.  The diagnostic
    loader normally expects an installed package named
    ``diff_surfel_rasterization_qdepth``; the historical JIT build is named
    ``osn_gs_diff_surfel_rasterization_qdepth_c``.  This shim is process-local
    and exposes the same compiled extension under the loader's expected name.
    """

    pyd = Path(os.environ.get("TEMP", "")) / "osn_gs_diff_surfel_rasterization_qdepth" / "osn_gs_diff_surfel_rasterization_qdepth_c.pyd"
    dll_dirs = [
        REPO_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"),
        Path(os.environ.get("TEMP", "")),
    ]
    for directory in dll_dirs:
        if directory.exists():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(directory))
    if not pyd.exists():
        return {"status": "MISSING_PREBUILT_QDEPTH_EXTENSION", "path": str(pyd)}
    sys.path.insert(0, str(pyd.parent))
    extension = importlib.import_module("osn_gs_diff_surfel_rasterization_qdepth_c")
    package = types.ModuleType("diff_surfel_rasterization_qdepth")
    package._C = extension
    sys.modules["diff_surfel_rasterization_qdepth"] = package
    return {"status": "PROCESS_LOCAL_PREBUILT_EXTENSION_LOADED", "path": str(pyd), "sha256": _sha256_file(pyd), "rasterize_gaussians": hasattr(extension, "rasterize_gaussians")}


def _replay_core(root: Path, device: str) -> dict[str, Any]:
    """Run only WL127's construction half with frozen inputs.

    No WL127 post-mesh diagnostic, NURBS baseline, Candidate B, or visual
    classification is called here.  The generated cache is wholly inside the
    WL153 output tree.
    """

    import numpy as np
    import torch

    scripts = REPO_ROOT / "scripts" / "devtools"
    # The project venv contains the already-approved build tool, but the
    # Windows launcher may not expose its Scripts directory on PATH.  This is
    # process-local setup for the diagnostic replay; it does not alter the
    # canonical renderer or repository environment.
    venv_scripts = REPO_ROOT / ".venv" / "Scripts"
    if venv_scripts.exists():
        os.environ["PATH"] = str(venv_scripts) + os.pathsep + os.environ.get("PATH", "")
    renderer_runtime = _prepare_qdepth_runtime()
    if renderer_runtime["status"] != "PROCESS_LOCAL_PREBUILT_EXTENSION_LOADED":
        raise RuntimeError(json.dumps(renderer_runtime))

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from coverage_first_surfel_partition_export import PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model
    from evidence_bounded_tsdf import extraction, field as tsdf_field, scale
    from maximal_visible_connectivity_export import load_all_train_cameras
    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    root.mkdir(parents=True, exist_ok=True)
    model, payload = load_primitive_model(CHECKPOINT, device=device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D:
        raise RuntimeError("WL127 checkpoint primitive mismatch")
    cameras, camera_meta = load_all_train_cameras(SOURCE_PATH, "images_8", "sparse/0", -1, 8, device)
    if len(cameras) != 161:
        raise RuntimeError(f"WL127 camera count mismatch: {len(cameras)}")

    depth_maps: list[torch.Tensor] = []
    for index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        depth_maps.append(tsdf_field.median_depth_map(package["out_others"]).reshape(-1).clone())
        del package
        if index % 20 == 0:
            print(f"[wl153] rendered {index}/{len(cameras)} cameras", flush=True)

    canonical = scale.derive_canonical_scale([scale.view_footprints(camera, depth) for camera, depth in zip(cameras, depth_maps)])
    h, mu = canonical.h, canonical.mu
    if abs(h - EXPECTED["h"]) > 1e-12 or abs(mu - EXPECTED["mu"]) > 1e-12:
        raise RuntimeError(f"WL127 historical scale mismatch: h={h!r}, mu={mu!r}")

    seed_keys = torch.zeros((0,), dtype=torch.int64, device=device)
    dropped = 0
    for camera, depth in zip(cameras, depth_maps):
        valid = torch.nonzero(depth > 0, as_tuple=False).reshape(-1)
        world = tsdf_field.unproject_pixels(camera, valid, depth[valid])
        keys, out_of_range = tsdf_field.encode_keys(tsdf_field.voxel_index_of(world, h), margin=64)
        dropped += out_of_range
        seed_keys = tsdf_field.union_sorted(seed_keys, keys)
        del valid, world, keys
    field, closure = tsdf_field.grow_field_to_closure(
        seed_keys, list(zip(cameras, depth_maps)), h=h, mu=mu, max_rounds=60, chunk_size=8_000_000,
        progress=lambda message: print("[wl153] " + message, flush=True),
    )
    field_path = root / "field.npz"
    _save_array_bundle(
        field_path,
        keys=field.keys.detach().cpu().numpy(), value=field.value.detach().cpu().numpy(),
        support_count=field.support_count.detach().cpu().numpy(), h=h, mu=mu,
        closure=json.dumps(closure, default=_json_default),
    )
    depth_path = root / "renderer_median_depth_maps.npz"
    _save_array_bundle(depth_path, depth=np.stack([row.detach().cpu().numpy() for row in depth_maps]))
    _save_json(root / "replay_input_runtime.json", {"camera_meta": camera_meta, "camera_names": [str(c.image_name) for c in cameras], "h": h, "mu": mu, "seed_voxels": int(seed_keys.numel()), "seed_voxels_out_of_range": dropped, "depth_sha256": _sha256_array(np.stack([row.detach().cpu().numpy() for row in depth_maps])), "renderer_runtime": renderer_runtime})

    surface = extraction.extract_zero_level_set(field, block=64, batch_blocks=6, progress=lambda message: print("[wl153] " + message, flush=True))
    typed_path = root / "typed_extracted_surface.npz"
    _save_array_bundle(
        typed_path, vertices=surface.vertices, faces=surface.faces,
        vertex_support_count=surface.vertex_support_count,
        vertex_field_value=surface.vertex_field_value, h=surface.h,
        stats=json.dumps(surface.stats, default=_json_default),
    )
    del field, seed_keys, depth_maps, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    measured = {
        "authoritative_voxels": int(np.load(field_path, allow_pickle=True)["keys"].shape[0]),
        "eligible_cells": int(surface.stats["eligible_cells_authoritative_and_sign_changing"]),
        "vertices": int(surface.vertices.shape[0]), "faces": int(surface.faces.shape[0]),
        "h": float(surface.h), "mu": float(mu), "closure": closure,
        "field_sha256": _sha256_file(field_path), "typed_surface_sha256": _sha256_file(typed_path),
        "depth_maps_sha256": _sha256_file(depth_path),
        "typed_surface_arrays": {
            "vertices_sha256": _sha256_array(surface.vertices), "faces_sha256": _sha256_array(surface.faces),
            "vertex_support_count_sha256": _sha256_array(surface.vertex_support_count),
            "vertex_field_value_sha256": _sha256_array(surface.vertex_field_value),
        },
    }
    expected_match = all(measured[key] == value for key, value in EXPECTED.items() if key in measured)
    measured["expected_wl127_quantities_match"] = expected_match
    measured["replay_fidelity"] = "SEMANTICALLY_EXACT_REPLAY" if expected_match else "REPLAY_MISMATCH"
    _save_json(root / "replay_measurements.json", measured)
    return {"surface": surface, "measured": measured, "field_path": field_path, "typed_path": typed_path}


def _quantiles(values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    return {"count": int(array.size), "min": float(np.min(array)), "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def _topology_accounting(surface: Any) -> dict[str, Any]:
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as sparse_components

    vertices = np.asarray(surface.vertices)
    faces = np.asarray(surface.faces, dtype=np.int64)
    vertex_count, face_count = int(vertices.shape[0]), int(faces.shape[0])
    degenerate_index = (
        (faces[:, 1] == faces[:, 0]) | (faces[:, 2] == faces[:, 0]) | (faces[:, 2] == faces[:, 1])
        if face_count else np.zeros((0,), dtype=bool)
    )
    triangle_vertices = vertices[faces] if face_count else np.zeros((0, 3, 3), dtype=vertices.dtype)
    zero_area = (0.5 * np.linalg.norm(np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 0]), axis=1) == 0) if face_count else np.zeros((0,), dtype=bool)

    if face_count:
        component_count, labels = sparse_components(
            coo_matrix((np.ones(face_count * 3, dtype=np.int8), (np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]]), np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]]))), shape=(vertex_count, vertex_count)).tocsr(), directed=False
        )
    else:
        labels, component_count = np.arange(vertex_count, dtype=np.int64), vertex_count
    used = np.unique(faces.reshape(-1)) if face_count else np.zeros((0,), dtype=np.int64)
    face_component = labels[faces[:, 0]] if face_count else np.zeros((0,), dtype=np.int64)
    vertex_sizes = np.bincount(labels, minlength=component_count)
    face_sizes = np.bincount(face_component, minlength=component_count) if face_count else np.zeros((component_count,), dtype=np.int64)

    # Native edge accounting only; no edge is removed or repaired.
    edges = np.empty((face_count * 3, 2), dtype=np.int64)
    if face_count:
        edges[0:face_count] = faces[:, [0, 1]]
        edges[face_count:2 * face_count] = faces[:, [1, 2]]
        edges[2 * face_count:] = faces[:, [2, 0]]
        edges.sort(axis=1)
        edge_dtype = np.dtype([("a", "<i8"), ("b", "<i8")])
        structured = edges.view(edge_dtype).reshape(-1)
        order = np.argsort(structured, kind="stable")
        sorted_structured = structured[order]
        sorted_edges = edges[order]
        _unique, edge_counts = np.unique(sorted_structured, return_counts=True)
    else:
        edge_counts = np.zeros((0,), dtype=np.int64)
    if face_count:
        starts = np.concatenate(([True], sorted_structured[1:] != sorted_structured[:-1]))
        unique_starts = np.flatnonzero(starts)
        boundary_edges = sorted_edges[unique_starts[edge_counts == 1]]
    else:
        boundary_edges = np.zeros((0, 2), dtype=np.int64)

    boundary_edge_count = int((edge_counts == 1).sum())
    non_manifold_edge_count = int((edge_counts > 2).sum())
    boundary_component_count = 0
    boundary_loop_count: int | None = 0
    boundary_degree_stats: dict[str, Any] = {"vertices": 0, "all_degree_two": True}
    if boundary_edge_count:
        boundary_vertices = np.unique(boundary_edges.reshape(-1))
        remap = np.full(vertex_count, -1, dtype=np.int64)
        remap[boundary_vertices] = np.arange(boundary_vertices.size, dtype=np.int64)
        rows, cols = remap[boundary_edges[:, 0]], remap[boundary_edges[:, 1]]
        graph = coo_matrix((np.ones(rows.size * 2, dtype=np.int8), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(boundary_vertices.size, boundary_vertices.size)).tocsr()
        boundary_component_count, boundary_labels = sparse_components(graph, directed=False)
        degrees = np.asarray(graph.sum(axis=1)).reshape(-1)
        boundary_degree_stats = {"vertices": int(boundary_vertices.size), "min_degree": int(degrees.min()), "max_degree": int(degrees.max()), "all_degree_two": bool(np.all(degrees == 2))}
        boundary_loop_count = int(boundary_component_count) if boundary_degree_stats["all_degree_two"] else None

    return {
        "vertex_count": vertex_count, "face_count": face_count,
        "unique_edge_count": int(edge_counts.size), "connected_component_count": int(component_count),
        "component_vertex_size_distribution": _quantiles(vertex_sizes), "component_face_size_distribution": _quantiles(face_sizes),
        "largest_components_by_vertices": sorted((int(v), int(i), int(face_sizes[i])) for i, v in enumerate(vertex_sizes))[-20:][::-1],
        "boundary_edge_count": boundary_edge_count, "boundary_component_count": int(boundary_component_count), "boundary_loop_count_when_well_defined": boundary_loop_count,
        "boundary_degree": boundary_degree_stats, "non_manifold_edge_count": non_manifold_edge_count,
        "isolated_vertex_count": int(vertex_count - used.size), "degenerate_index_face_count": int(degenerate_index.sum()), "zero_area_face_count": int(zero_area.sum()),
        "operation_contract": "faces adjacency only; no component selection, merge, split, hole repair, or boundary closure",
    }


def _provenance_and_review() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provenance = {
        "observation_provenance": {
            "renderer_observation_to_tsdf_cell": "NOT_PERSISTED_PER_EVENT; fuse_views retains only mean field value and support_count",
            "tsdf_cell_to_active_marching_cubes_cell": "DETERMINISTIC_IN_SOURCE_BUT_NOT_PERSISTED; extraction local cell ownership is discarded",
            "active_cell_to_face": "DETERMINISTIC_IN_SOURCE_BUT_NOT_PERSISTED; typed ExtractedSurface carries faces but no source-cell IDs",
            "face_to_seam_welded_vertex": "DETERMINISTIC_IN_SOURCE_BUT_NOT_PERSISTED; weld returns arrays without sidecar",
            "camera_set_per_field_cell": "NOT_AVAILABLE",
            "event_set_per_surface_element": "EVENT_LEVEL_NOT_AVAILABLE",
            "component_camera_aggregation": "NOT_AVAILABLE_WITHOUT_INFERENCE",
        },
        "physical_sheet_membership": {
            "status": "NOT_PROVIDED",
            "reason": "native zero-set topology/evidence support is not a physical-sheet identity contract",
            "new_membership": False,
        },
        "forbidden_posthoc_mapping": False,
    }
    event_probe = {
        "event_id": 1527,
        "human_review": "CLEAR_NOT_ON_INTENDED_SURFACE",
        "blacklist": False,
        "source_camera": "DSC08003.JPG",
        "source_pixel": [259, 169],
        "historical_v_min_owner": True,
        "surface_trace": "EVENT_LEVEL_NOT_AVAILABLE",
        "reason": "event ID is preserved in WL149/WL152 input provenance, but WL127 field fusion and extraction do not carry event/camera/source-cell IDs into ExtractedSurface",
    }
    review = {
        "contract": "native topology is reviewed without creating physical-sheet membership",
        "cases": [
            {"case": "clean_tabletop", "existing_region": "table_top", "classification": "NOT_REVIEWABLE", "reason": "no native semantic sheet label or deterministic camera/event lineage"},
            {"case": "tabletop_side_relationship", "existing_region": "table_side_curved", "classification": "NOT_REVIEWABLE", "reason": "a connected component cannot be promoted to same-sheet or distinct-sheet identity"},
            {"case": "vase_or_curved_neighbor", "existing_region": "vase/neighbor review context", "classification": "NOT_REVIEWABLE", "reason": "no physical-sheet membership in ExtractedSurface"},
            {"case": "background_lower_geometry", "existing_region": "hedge/patio/lower review context", "classification": "NOT_REVIEWABLE", "reason": "no semantic identity; no split by normals or geometry thresholds"},
        ],
        "interpretation": "The replay makes native mesh connectivity measurable, but the required real-scene physical-sheet labels are not part of WL127's carrier. All four review cases therefore remain NOT_REVIEWABLE rather than being inferred.",
    }
    boundary = {
        "topological_boundary": "computed from face edge incidence only",
        "observed_physical_boundary": "not established",
        "promotion_justified": False,
        "possible_causes": ["evidence mask termination", "reconstruction volume termination", "extraction support limit", "genuine observed termination"],
        "rule": "native mesh boundary != observed physical-surface boundary without an existing deterministic ownership contract",
    }
    return provenance, event_probe, {"review": review, "boundary": boundary}


def _write_preview(root: Path, surface: Any) -> dict[str, Any]:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices = np.asarray(surface.vertices)
    stride = max(1, int(vertices.shape[0] // 180_000))
    sampled = vertices[::stride]
    fig = plt.figure(figsize=(12, 8), dpi=180)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=0.35, c="#2f6f9f", alpha=0.96, linewidths=0)
    axis.set_title("WL153 replayed typed Raw Visible Surface — opaque vertex preview")
    axis.set_xlabel("world X"); axis.set_ylabel("world Y"); axis.set_zlabel("world Z")
    axis.view_init(elev=24, azim=-58)
    fig.tight_layout()
    path = root / "raw_surface_vertex_preview.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return {"path": _relative(path), "display_vertices": int(sampled.shape[0]), "display_stride": stride, "opacity": 0.96, "geometry_unchanged": True}


def write_audit(*, replay: bool = False, device: str = "cuda") -> dict[str, Any]:
    audit = build_audit()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _save_json(OUTPUT_ROOT / "baseline_reconciliation.json", audit["baseline_reconciliation"])
    _save_json(OUTPUT_ROOT / "historical_raw_surface_contract.json", audit["historical_raw_surface_contract"])
    _save_json(OUTPUT_ROOT / "available_artifact_contract.json", audit["available_artifact_contract"])
    _save_json(OUTPUT_ROOT / "construction_lineage_contract.json", audit["construction_lineage_contract"])
    result: dict[str, Any] = {"audit": audit, "replay_requested": replay, "output_root": _relative(OUTPUT_ROOT), "temp_root": _relative(TEMP_ROOT)}
    if replay:
        if audit["historical_raw_surface_contract"]["status"] != "IDENTIFIABLE":
            raise RuntimeError("RAW_SURFACE_HISTORICAL_REPLAY_UNRESOLVED")
        replay_result = _replay_core(OUTPUT_ROOT / "replay_cache", device)
        measured = replay_result["measured"]
        result["replay"] = {key: value for key, value in measured.items() if key != "surface"}
        if measured["replay_fidelity"] == "REPLAY_MISMATCH":
            _save_json(OUTPUT_ROOT / "architecture_verdict.json", {"verdict": "REPLAY_MISMATCH", "stop": "topology interpretation stopped", "measured": measured})
        else:
            topology = _topology_accounting(replay_result["surface"])
            provenance, event_probe, secondary = _provenance_and_review()
            _save_json(OUTPUT_ROOT / "native_topology_accounting.json", topology)
            _save_json(OUTPUT_ROOT / "construction_provenance_audit.json", provenance)
            _save_json(OUTPUT_ROOT / "event_1527_probe.json", event_probe)
            _save_json(OUTPUT_ROOT / "physical_sheet_viability_review.json", secondary["review"])
            _save_json(OUTPUT_ROOT / "boundary_semantics_audit.json", secondary["boundary"])
            result["topology"] = topology
            result["provenance"] = provenance
            result["event_1527"] = event_probe
            result["physical_sheet_review"] = secondary["review"]
            result["boundary"] = secondary["boundary"]
            result["preview"] = _write_preview(OUTPUT_ROOT, replay_result["surface"])
            verdict = {
                "verdict": "TOPOLOGY_RECOVERED_PROVENANCE_GAP",
                "replay_fidelity": measured["replay_fidelity"],
                "native_topology": "RECOVERED_AND_ACCOUNTED",
                "observation_provenance": "SOURCE_LINEAGE_ONLY_NO_PER_ELEMENT_IDS",
                "physical_sheet_membership": "SEPARATE_ABSTRACTION_STILL_REQUIRED",
                "event_1527": "EVENT_LEVEL_NOT_AVAILABLE",
                "canonical_production_modified": False,
                "candidate_nurbs_or_membership": "NOT_RUN",
            }
            _save_json(OUTPUT_ROOT / "architecture_verdict.json", verdict)
            result["verdict"] = verdict
    else:
        result["verdict"] = {"verdict": "REPLAY_NOT_RUN", "reason": "run with --replay after baseline contract identification"}

    report = {
        "1. CURRENT ARCHITECTURE QUESTION": "What topology and renderer-grounded construction lineage does the actual WL127 typed ExtractedSurface provide?",
        "2. WL152 BASELINE RECONCILIATION": audit["baseline_reconciliation"],
        "3. HISTORICAL RAW-SURFACE CONSTRUCTION CONTRACT": audit["historical_raw_surface_contract"],
        "4. REPLAY INPUT AVAILABILITY": audit["historical_raw_surface_contract"]["input_manifest"],
        "5. RAW VISIBLE SURFACE REPLAY": result.get("replay", {"status": "NOT_RUN"}),
        "6. EXTRACTEDSURFACE CONTRACT": "vertices, faces, vertex_support_count, vertex_field_value, h; serialized under WL153 replay_cache when replayed",
        "7. CONSTRUCTION-PROVENANCE LINEAGE": result.get("provenance", audit["construction_lineage_contract"]),
        "8. NATIVE TOPOLOGY ACCOUNTING": result.get("topology", "NOT_RUN"),
        "9. PHYSICAL-SHEET VIABILITY REVIEW": result.get("physical_sheet_review", "NOT_RUN"),
        "10. EVIDENCE PROVENANCE VIABILITY": result.get("provenance", "NOT_RUN"),
        "11. EVENT 1527 TRACE": result.get("event_1527", {"status": "PRESERVED_NOT_BLACKLISTED; EVENT_LEVEL_NOT_AVAILABLE"}),
        "12. BOUNDARY SEMANTICS": result.get("boundary", "NOT_RUN"),
        "13. ARCHITECTURE VERDICT": result.get("verdict"),
        "14. RETAINED / REJECTED / OPEN": {
            "RETAINED": ["WL127 point artifact and WL152 exact baseline", "943a764 TSDF core", "event 1527 and CLEAR_NOT_ON_INTENDED_SURFACE", "canonical renderer/checkpoint/cameras"],
            "REJECTED": ["point cloud promoted to mesh", "post-hoc event-to-mesh correspondence", "connectivity repair", "physical-sheet inference", "NURBS or membership fitting"],
            "OPEN": ["per-event/camera/source-cell sidecars in a future behavior-neutral instrumentation pass", "physical-sheet membership contract", "promotion of topological boundaries to observed boundaries"],
        },
        "INTENT ALIGNMENT": {"baseline_reconciled": True, "new_geometry_inference": False, "connectivity_repair": False, "representative_fitting": False, "canonical_modified": False, "event_1527_blacklisted": False},
        "IMPLEMENTATION FIDELITY": {"historical_core_blob_match": audit["historical_raw_surface_contract"]["source_manifest"]["core_exact"], "typed_result_serialized": replay and "replay" in result, "provenance_behavior_neutral": True, "no_posthoc_correspondence": True},
        "ARCHITECTURE RESULT": result.get("verdict"),
    }
    _save_json(OUTPUT_ROOT / "raw_visible_surface_replay_construction_provenance_audit_report.json", report)
    readme = """# Worklog 153 — WL127 Raw Visible Surface replay / provenance audit

이 폴더는 WL127의 vertex-only point PLY와 별도로, 기준 커밋 `943a764`의
typed `ExtractedSurface` replay를 보존하는 진단 전용 산출물이다.

- canonical renderer/checkpoint/161 cameras: 변경하지 않음
- WL152 baseline: event union 1586, event 1527 보존, point PLY 1,212,365 vertices / faces 0
- replay: `replay_cache/` 아래 `field.npz`, `renderer_median_depth_maps.npz`, `typed_extracted_surface.npz`
- provenance: 기존 typed contract에는 per-event/camera/source-cell sidecar가 없음을 명시
- physical-sheet membership / NURBS / connectivity repair: 수행하지 않음

`architecture_verdict.json`과 `raw_visible_surface_replay_construction_provenance_audit_report.json`이
최종 판정이다. PNG는 opaque display-only vertex preview이며 metrics/geometry를 바꾸지 않는다.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    temp_mirror = _mirror_output_to_temp(OUTPUT_ROOT, TEMP_ROOT)
    result["temp_mirror"] = temp_mirror
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="run the full 161-camera WL127 typed-surface replay")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = write_audit(replay=args.replay, device=args.device)
    # Windows terminals may still expose cp949; JSON files remain UTF-8 while
    # the CLI summary stays ASCII-safe.
    print(json.dumps(result, indent=2, ensure_ascii=True, default=_json_default))


if __name__ == "__main__":
    main()
