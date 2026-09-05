from __future__ import annotations

"""Worklog 166: export the historical W153 TSDF zero-level surface unchanged.

This module is deliberately an export/validation tool, not a new surface
constructor.  The default input is W153's replayed typed ``ExtractedSurface``
NPZ, whose arrays were produced by the historical ``943a764``
renderer-median -> projective-TSDF -> all-eight-corner Lewiner extraction
contract.  The raw NPZ is copied byte-for-byte; the OBJ is a text
serialization of the same vertices and faces.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib import format as npy_format


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "output/confirmed/153_raw_visible_surface_replay_construction_provenance_audit"
    / "replay_cache/typed_extracted_surface.npz"
)
DEFAULT_SOURCE_ROOT = DEFAULT_SOURCE.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "output/166_historical_sdf_zero_surface_mesh_export"
DEFAULT_ACCOUNTING = DEFAULT_SOURCE_ROOT / "native_topology_accounting.json"

RAW_NAME = "historical_sdf_zero_surface_raw.npz"
OBJ_NAME = "historical_sdf_zero_surface.obj"
REPORT_NAME = "mesh_export_report.json"
README_NAME = "README.md"


def _sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray, *, chunk_rows: int = 1_000_000) -> str:
    """Hash C-order array bytes without making a second whole-array copy."""

    array = np.asarray(array)
    digest = hashlib.sha256()
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_rows):
        digest.update(np.ascontiguousarray(flat[start : start + chunk_rows]).tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def array_descriptors(source: Path) -> dict[str, dict[str, Any]]:
    """Read NPZ NPY headers and return metadata without materializing values."""

    descriptors: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(source, "r") as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            name = Path(member.filename).stem
            with archive.open(member, "r") as handle:
                version = npy_format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
                elif version == (3, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_3_0(handle)
                else:
                    raise ValueError(f"unsupported NPY header version {version}")
            if fortran_order:
                raise ValueError(f"Fortran-order source array is not supported: {name}")
            descriptors[name] = {
                "shape": [int(v) for v in shape],
                "dtype": str(dtype),
                "nbytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize),
            }
    return descriptors


def _write_vertex_lines(handle: Any, vertices: np.ndarray, chunk_rows: int) -> None:
    for start in range(0, vertices.shape[0], chunk_rows):
        chunk = np.asarray(vertices[start : start + chunk_rows], dtype=np.float64)
        np.savetxt(handle, chunk, fmt="v %.17g %.17g %.17g")


def _write_face_lines(handle: Any, faces: np.ndarray, chunk_rows: int) -> None:
    for start in range(0, faces.shape[0], chunk_rows):
        chunk = np.asarray(faces[start : start + chunk_rows], dtype=np.int64) + 1
        np.savetxt(handle, chunk, fmt="f %d %d %d")


def write_obj_from_arrays(
    path: Path, vertices: np.ndarray, faces: np.ndarray, *, chunk_rows: int = 100_000
) -> None:
    """Write a triangular OBJ without changing row order or connectivity."""

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
        raise ValueError("face index is outside the source vertex range")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# Worklog 166 historical SDF/TSDF zero-level surface; unchanged row order\n")
        handle.write("o historical_sdf_zero_surface\n")
        _write_vertex_lines(handle, vertices, chunk_rows)
        _write_face_lines(handle, faces, chunk_rows)


def export_obj_from_npz(source: Path, destination: Path, *, chunk_rows: int = 100_000) -> None:
    """Materialize only the source arrays needed for the OBJ serialization."""

    with np.load(source, allow_pickle=False) as bundle:
        vertices = bundle["vertices"]
        faces = bundle["faces"]
        write_obj_from_arrays(destination, vertices, faces, chunk_rows=chunk_rows)


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Small exact parser used by focused tests and small external fixtures."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0] in {"#", "o", "g", "s"}:
                continue
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError(f"unsupported vertex line: {line!r}")
                vertices.append([float(value) for value in fields[1:]])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"only triangular faces are supported: {line!r}")
                indices = [int(value.split("/", 1)[0]) for value in fields[1:]]
                if any(value <= 0 for value in indices):
                    raise ValueError("OBJ parser expects positive 1-based face indices")
                faces.append([value - 1 for value in indices])
            else:
                raise ValueError(f"unsupported OBJ record {fields[0]!r}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _stream_roundtrip_compare(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    obj_path: Path,
) -> dict[str, Any]:
    """Compare every OBJ record to source arrays without retaining parsed OBJ."""

    vertex_index = 0
    face_index = 0
    max_abs_coordinate_error = 0.0
    vertex_mismatch_count = 0
    face_mismatch_count = 0
    obj_min = np.full((3,), np.inf, dtype=np.float64)
    obj_max = np.full((3,), -np.inf, dtype=np.float64)
    with obj_path.open("rb") as handle:
        for raw in handle:
            fields = raw.split()
            if not fields or fields[0] in {b"#", b"o", b"g", b"s"}:
                continue
            if fields[0] == b"v":
                if vertex_index >= source_vertices.shape[0] or len(fields) != 4:
                    vertex_mismatch_count += 1
                    continue
                parsed = np.asarray([float(v) for v in fields[1:]], dtype=np.float64)
                error = float(np.max(np.abs(parsed - source_vertices[vertex_index])))
                max_abs_coordinate_error = max(max_abs_coordinate_error, error)
                obj_min = np.minimum(obj_min, parsed)
                obj_max = np.maximum(obj_max, parsed)
                if not np.array_equal(parsed, source_vertices[vertex_index]):
                    vertex_mismatch_count += 1
                vertex_index += 1
            elif fields[0] == b"f":
                if face_index >= source_faces.shape[0] or len(fields) != 4:
                    face_mismatch_count += 1
                    continue
                parsed = np.asarray(
                    [int(v.split(b"/", 1)[0]) - 1 for v in fields[1:]], dtype=np.int64
                )
                if not np.array_equal(parsed, source_faces[face_index]):
                    face_mismatch_count += 1
                face_index += 1
            else:
                raise ValueError(f"unsupported OBJ record {fields[0]!r}")
    return {
        "source_vertex_count": int(source_vertices.shape[0]),
        "obj_vertex_count": int(vertex_index),
        "source_face_count": int(source_faces.shape[0]),
        "obj_face_count": int(face_index),
        "vertex_count_match": vertex_index == source_vertices.shape[0],
        "face_count_match": face_index == source_faces.shape[0],
        "vertex_connectivity_or_coordinate_mismatch_count": int(vertex_mismatch_count),
        "face_connectivity_mismatch_count": int(face_mismatch_count),
        "max_abs_coordinate_error": max_abs_coordinate_error,
        "connectivity_exact": face_mismatch_count == 0 and face_index == source_faces.shape[0],
        "coordinates_finite": bool(np.isfinite(source_vertices).all()),
        "obj_bounds": {"min": obj_min.tolist(), "max": obj_max.tolist()},
        "bounds_match_source": bool(
            vertex_index == source_vertices.shape[0]
            and np.array_equal(obj_min, source_vertices.min(axis=0))
            and np.array_equal(obj_max, source_vertices.max(axis=0))
        ),
    }


def roundtrip_compare(source: Path, obj_path: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as bundle:
        vertices = bundle["vertices"]
        faces = bundle["faces"]
        bounds = {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()}
        result = _stream_roundtrip_compare(vertices, faces, obj_path)
        result["source_bounds"] = bounds
        result["roundtrip_pass"] = bool(
            result["vertex_count_match"]
            and result["face_count_match"]
            and result["vertex_connectivity_or_coordinate_mismatch_count"] == 0
            and result["face_connectivity_mismatch_count"] == 0
            and result["coordinates_finite"]
            and result["bounds_match_source"]
        )
        return result


def _source_hashes(source: Path) -> dict[str, str]:
    with np.load(source, allow_pickle=False) as bundle:
        return {
            name: _sha256_array(bundle[name])
            for name in ("vertices", "faces", "vertex_support_count", "vertex_field_value")
        }


def _load_component_accounting(path: Path, source: Path) -> dict[str, Any]:
    accounting = json.loads(path.read_text(encoding="utf-8"))
    descriptors = array_descriptors(source)
    count_match = {
        "vertex_count": accounting.get("vertex_count") == descriptors["vertices"]["shape"][0],
        "face_count": accounting.get("face_count") == descriptors["faces"]["shape"][0],
    }
    if not all(count_match.values()):
        raise ValueError(f"W153 component accounting count mismatch: {count_match}")
    return {
        "source_report": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "verified_against_source_array_shapes": count_match,
        "operation_contract": accounting.get("operation_contract"),
        "native_faces_adjacency_accounting": accounting,
    }


def _read_source_scalar_metadata(source: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as bundle:
        stats = json.loads(str(bundle["stats"].item()))
        return {"h": float(bundle["h"].item()), "stats": stats}


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    counts = report["source_arrays"]
    source_contract = report["historical_source_contract"]
    path.write_text(
        "\n".join(
            [
                "# Worklog 166 — Historical SDF/TSDF Zero-Level Surface Export",
                "",
                "이 directory는 W153 replay의 historical projective TSDF zero-level surface를 외부 3D viewer에서 직접 확인하기 위한 export이다.",
                "",
                "## Source and preservation",
                "",
                f"- Source: `{report['source_npz']}`; historical core commit: `{source_contract['historical_commit']}`.",
                "- Source contract: renderer-median depth seed → projective TSDF (`mu=3h`, unit fusion, sparse UNKNOWN) → all-eight-corner authoritative sign-changing cells → Lewiner Marching Cubes → quantized seam-only weld.",
                "- Vertices and faces are copied in source row order. No smoothing, repair, remesh, decimation, welding beyond the historical source weld, filtering, recentering, normalization, axis rotation, component selection, or topology change was performed here.",
                f"- Source arrays: `{counts['vertices']['shape']}` vertices and `{counts['faces']['shape']}` triangular faces; raw NPZ is a byte-for-byte copy of the typed source NPZ.",
                "",
                "## Files",
                "",
                f"- `{OBJ_NAME}` — 1-based triangular OBJ serialization for direct inspection.",
                f"- `{RAW_NAME}` — raw typed arrays, including `vertices`, `faces`, `vertex_support_count`, `vertex_field_value`, `h`, and `stats`.",
                f"- `{REPORT_NAME}` — source provenance, bounds, component accounting, and OBJ round-trip result.",
                "",
                "## Interpretation limits",
                "",
                "This is a historical observed-visible-surface reconstruction artifact. It is not a repaired/watertight object mesh, does not establish physical hidden-surface identity, and does not validate Observed/Occluded semantics or occlusion truth. The W153 component accounting is faces-adjacency accounting only; its fragmented/open/non-manifold geometry is retained rather than corrected.",
                "",
                f"FBX was not emitted: `{report['fbx']['status']}`. No new heavy exporter dependency was installed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_export(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"historical typed surface source does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)

    raw_path = output / RAW_NAME
    obj_path = output / OBJ_NAME
    report_path = output / REPORT_NAME
    readme_path = output / README_NAME

    source_descriptors = array_descriptors(source)
    raw_source_sha256 = _sha256_file(source)
    shutil.copyfile(source, raw_path)
    raw_copy_exact = _sha256_file(raw_path) == raw_source_sha256

    export_obj_from_npz(source, obj_path)
    roundtrip = roundtrip_compare(source, obj_path)
    source_hashes = _source_hashes(source)
    source_scalar_metadata = _read_source_scalar_metadata(source)
    accounting = _load_component_accounting(DEFAULT_ACCOUNTING, source)
    vertices_shape = source_descriptors["vertices"]["shape"]
    faces_shape = source_descriptors["faces"]["shape"]
    with np.load(source, allow_pickle=False) as bundle:
        bounds = {
            "min": bundle["vertices"].min(axis=0).tolist(),
            "max": bundle["vertices"].max(axis=0).tolist(),
        }

    report: dict[str, Any] = {
        "status": "COMPLETE_HISTORICAL_ZERO_SURFACE_EXPORT" if raw_copy_exact and roundtrip["roundtrip_pass"] else "EXPORT_VALIDATION_FAILED",
        "worklog": 166,
        "intent_alignment": "diagnostic/export-only; exact historical SDF/TSDF zero-level surface for direct external inspection",
        "source_npz": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_npz_sha256": raw_source_sha256,
        "raw_export": {
            "path": RAW_NAME,
            "byte_for_byte_source_copy": raw_copy_exact,
            "sha256": _sha256_file(raw_path),
            "arrays_sha256": source_hashes,
        },
        "obj_export": {
            "path": OBJ_NAME,
            "sha256": _sha256_file(obj_path),
            "bytes": int(obj_path.stat().st_size),
            "format": "triangular OBJ; vertices serialized with %.17g; faces are source indices + 1",
        },
        "source_arrays": source_descriptors,
        "exported_geometry": {
            "vertex_count": int(vertices_shape[0]),
            "face_count": int(faces_shape[0]),
            "bounds_world": bounds,
            "coordinates_frame": "historical world-space coordinates; no recenter/normalize/axis transform",
            "geometry_modification": "NONE",
        },
        "historical_source_contract": {
            "historical_commit": "943a764",
            "source_artifact": "W153 replay_cache/typed_extracted_surface.npz",
            "pipeline": [
                "renderer median depth seed",
                "projective TSDF sparse field",
                "all-eight-corner authoritative and sign-changing cell eligibility",
                "Lewiner Marching Cubes",
                "h*1e-6 quantized seam-only weld",
            ],
            "h": source_scalar_metadata["h"],
            "mu": 3.0 * source_scalar_metadata["h"],
            "extraction_stats": source_scalar_metadata["stats"],
        },
        "component_accounting": accounting,
        "obj_roundtrip": roundtrip,
        "fbx": {
            "status": "FBX_EXPORT_UNAVAILABLE",
            "reason": "No existing repository exporter was available for this artifact; no heavy dependency or external converter was installed.",
        },
        "semantic_boundary": {
            "occlusion_semantics_validated": False,
            "physical_hidden_surface_identity_validated": False,
            "watertightness_or_repair_performed": False,
        },
        "retained_risks": [
            "W153 is a semantically exact replay with source/input-identifiable provenance, not a byte hash of the unavailable original historical typed array.",
            "The retained mesh has the historical fragmented/open/non-manifold/degenerate topology recorded by W153; these are not export errors and were not repaired.",
            "OBJ is a large text serialization and may be slow/heavy for viewers; NPZ is the authoritative raw-array preservation artifact.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value) + "\n", encoding="utf-8")
    _write_readme(readme_path, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_export(args.source, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output), "obj_roundtrip": report["obj_roundtrip"]}, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE_HISTORICAL_ZERO_SURFACE_EXPORT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
