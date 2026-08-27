"""Full-scene WL119-4 reference-exact KNN semantics attribution."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from coverage_first_surfel_partition_export import (  # noqa: E402
    PRIMITIVE_SURFEL_2D, checkpoint_primitive, load_primitive_model,
)
from osn_gs.surface.torch_coverage_first_subset_partition import (  # noqa: E402
    CoverageFirstPartitionConfig, _auto_chunk_size, _connected_component_roots, _knn,
)
from osn_gs.surface.torch_exact_knn_performance import (  # noqa: E402
    candidate_graph_from_neighbors, scipy_ckdtree_exact_knn,
)
from osn_gs.surface.torch_knn_reference_attribution import (  # noqa: E402
    CLASS_NAMES, adversarial_knn_fixtures, boundary_margins,
    classify_neighbor_mismatches, distance_arithmetic_variants,
    observe_reference_knn,
)
from osn_gs.surface.torch_surfel_surface_orientation import (  # noqa: E402
    derive_surface_orientation_from_surfel,
)


def _progress(message: str) -> None:
    print(f"[wl119-4-knn] {message}", flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _identity(expected: Any, actual: Any) -> dict[str, Any]:
    shape = tuple(expected.shape) == tuple(actual.shape)
    if not shape:
        return {"shape_exact": False, "value_exact": False, "mismatch_count": None}
    mismatch = expected != actual
    return {
        "shape_exact": True, "value_exact": bool(not mismatch.any()),
        "mismatch_count": int(mismatch.sum()),
    }


def _graph_identity(expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "local_spacing": _identity(expected.local_spacing, actual.local_spacing),
        "candidate_edges": _identity(expected.candidate_edges, actual.candidate_edges),
        "spatial_edge_mask": _identity(expected.spatial_edge_mask, actual.spatial_edge_mask),
        "normal_compatible_mask": _identity(expected.normal_compatible_mask, actual.normal_compatible_mask),
        "normal_alignment": _identity(expected.normal_alignment, actual.normal_alignment),
        "accepted_edges": _identity(expected.accepted_edges, actual.accepted_edges),
    }


def _knn_explicit_mode(
    positions: Any, k: int, chunk_size: int, compute_mode: str
) -> tuple[Any, Any, float]:
    indices, distances = [], []
    started = time.perf_counter()
    count = int(positions.shape[0])
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        matrix = torch.cdist(
            positions[start:end], positions, compute_mode=compute_mode
        )
        rows = torch.arange(end - start, device=positions.device)
        matrix[rows, torch.arange(start, end, device=positions.device)] = float("inf")
        _, index = torch.topk(matrix, k, dim=1, largest=False)
        exact = (positions[start:end, None, :] - positions[index]).norm(dim=-1)
        indices.append(index)
        distances.append(exact)
        del matrix
        if (start // chunk_size) % 200 == 0:
            _progress(f"explicit mode rows {end}/{count} mode={compute_mode}")
    _sync(positions.device)
    return torch.cat(indices), torch.cat(distances), time.perf_counter() - started


def _distribution(values: Any) -> dict[str, Any]:
    array = values.detach().to(torch.float64).cpu().numpy()
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size), "min": float(array.min()),
        "q01": float(np.percentile(array, 1)), "q05": float(np.percentile(array, 5)),
        "q25": float(np.percentile(array, 25)), "median": float(np.median(array)),
        "q75": float(np.percentile(array, 75)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "max": float(array.max()),
        "zero_count": int((array == 0).sum()),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception as error:
        nvidia_smi = f"unavailable: {error}"
    return {
        "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "nvidia_smi": nvidia_smi,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _production_chunk_mode_audit(positions: Any, k: int, chunk_size: int) -> dict[str, Any]:
    """Compare all raw distances for fixed production chunk zero."""

    end = min(chunk_size, int(positions.shape[0]))
    query = positions[:end]
    _progress(f"production chunk mode audit rows=0:{end}")
    default = torch.cdist(query, positions)
    explicit_mm = torch.cdist(query, positions, compute_mode="use_mm_for_euclid_dist")
    raw_default_mm = _identity(default, explicit_mm)
    rows = torch.arange(end, device=positions.device)
    default[rows, rows] = float("inf")
    explicit_mm[rows, rows] = float("inf")
    d_raw, d_index = torch.topk(default, k + 1, dim=1, largest=False, sorted=True)
    m_raw, m_index = torch.topk(explicit_mm, k + 1, dim=1, largest=False, sorted=True)
    del default, explicit_mm
    direct = torch.cdist(
        query, positions, compute_mode="donot_use_mm_for_euclid_dist"
    )
    direct[rows, rows] = float("inf")
    n_raw, n_index = torch.topk(direct, k + 1, dim=1, largest=False, sorted=True)
    del direct
    return {
        "chunk_start": 0, "chunk_end": end,
        "default_vs_explicit_mm_raw_matrix": raw_default_mm,
        "default_vs_explicit_mm_topk_index": _identity(d_index, m_index),
        "default_vs_explicit_mm_topk_raw": _identity(d_raw, m_raw),
        "default_vs_direct_topk_index": _identity(d_index, n_index),
        "default_vs_direct_topk_raw": _identity(d_raw, n_raw),
        "default_vs_direct_row_mismatch_count": int((d_index != n_index).any(dim=1).sum()),
        "default_vs_direct_max_raw_delta_at_default_ids": float(
            (d_raw - n_raw).abs().max()
        ),
    }


def _synthetic_audit(device: torch.device) -> dict[str, Any]:
    results = {}
    for name, (points, k) in adversarial_knn_fixtures(device).items():
        chunk = int(points.shape[0])
        default = observe_reference_knn(points, k, chunk)
        conditional = observe_reference_knn(
            points, k, chunk, compute_mode="use_mm_for_euclid_dist_if_necessary"
        )
        mm = observe_reference_knn(
            points, k, chunk, compute_mode="use_mm_for_euclid_dist"
        )
        direct = observe_reference_knn(
            points, k, chunk, compute_mode="donot_use_mm_for_euclid_dist"
        )
        ckd_index, _ = scipy_ckdtree_exact_knn(points, k, workers=1)
        results[name] = {
            "point_count": int(points.shape[0]), "k": k,
            "default_vs_conditional_index": _identity(default.neighbor_index, conditional.neighbor_index),
            "default_vs_conditional_raw": _identity(default.boundary_raw_distance, conditional.boundary_raw_distance),
            "default_vs_forced_mm_index": _identity(default.neighbor_index, mm.neighbor_index),
            "default_vs_forced_mm_raw": _identity(default.boundary_raw_distance, mm.boundary_raw_distance),
            "default_vs_direct_index": _identity(default.neighbor_index, direct.neighbor_index),
            "default_vs_direct_raw": _identity(default.boundary_raw_distance, direct.boundary_raw_distance),
            "default_vs_ckdtree_ordered": _identity(default.neighbor_index, ckd_index),
            "default_vs_ckdtree_set_row_mismatch_count": int((
                default.neighbor_index.sort(dim=1).values
                != ckd_index.sort(dim=1).values
            ).any(dim=1).sum()),
        }
    return results


def _sample_arithmetic(
    positions: Any, attribution: dict[str, Any], production_raw_ref: Any,
    production_raw_candidate: Any,
) -> list[dict[str, Any]]:
    primary = attribution["primary_class"]
    rows_to_sample: list[int] = []
    for class_id in sorted(CLASS_NAMES):
        # Order-only rows have no entering/leaving pair. Their synthetic
        # pair slots are placeholders, not valid arithmetic samples.
        if CLASS_NAMES[class_id].startswith("A_"):
            continue
        rows = torch.nonzero(primary == class_id, as_tuple=False).reshape(-1)
        rows = rows[attribution["membership_mismatch"][rows]]
        rows_to_sample.extend(int(value) for value in rows[:32].cpu())
    material_id = next(key for key, value in CLASS_NAMES.items() if value.startswith("F_"))
    material_rows = torch.nonzero(primary == material_id, as_tuple=False).reshape(-1)
    if material_rows.numel():
        gaps = attribution["mathematical_squared_gap"][material_rows]
        order = torch.argsort(gaps, descending=True)
        rows_to_sample.extend(int(value) for value in material_rows[order[:32]].cpu())
    rows_to_sample = sorted(set(rows_to_sample))
    output = []
    for row in rows_to_sample:
        ref_index = int(attribution["reference_only_index"][row])
        candidate_index = int(attribution["candidate_only_index"][row])
        output.append({
            "row": row, "class": CLASS_NAMES[int(primary[row])],
            "reference_only_index": ref_index, "candidate_only_index": candidate_index,
            "production_reference_raw": float(attribution["reference_only_raw"][row]),
            "production_candidate_raw": float(attribution["candidate_only_raw"][row]),
            "reference_pair_variants": distance_arithmetic_variants(positions, row, ref_index),
            "candidate_pair_variants": distance_arithmetic_variants(positions, row, candidate_index),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    if checkpoint_primitive(payload) != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("WL119-4 requires the frozen 2DGS surfel checkpoint")
    visible = torch.nonzero(~model.is_uncertain.reshape(-1).to(torch.bool), as_tuple=False).reshape(-1)
    full = derive_surface_orientation_from_surfel(model)
    orientation = replace(
        full,
        gaussian_ids=full.gaussian_ids[visible], positions=full.positions[visible],
        tangent_axis_u=full.tangent_axis_u[visible], tangent_axis_v=full.tangent_axis_v[visible],
        surface_normal=full.surface_normal[visible], tangent_scale_u=full.tangent_scale_u[visible],
        tangent_scale_v=full.tangent_scale_v[visible],
    )
    positions = orientation.positions
    device = positions.device
    count = int(positions.shape[0])
    config = CoverageFirstPartitionConfig()
    k = min(int(config.neighbor_count), count - 1)
    chunk_size = int(config.knn_chunk_size) or _auto_chunk_size(count, device)
    environment = _environment(device)

    _progress("authoritative default topk(K) reference run 1")
    started = time.perf_counter()
    reference_index, reference_distance = _knn(positions, k, chunk_size, _progress)
    _sync(device)
    reference_search_seconds = time.perf_counter() - started
    graph_started = time.perf_counter()
    reference_graph = candidate_graph_from_neighbors(
        orientation, config, reference_index, reference_distance, progress=_progress
    )
    reference_roots = _connected_component_roots(count, reference_graph.accepted_edges, config)
    _sync(device)
    reference_graph_seconds = time.perf_counter() - graph_started

    _progress("historical cKDTree K and diagnostic K+1")
    ckd_index, ckd_distance = scipy_ckdtree_exact_knn(positions, k, workers=-1, progress=_progress)
    ckd_boundary_index, _ = scipy_ckdtree_exact_knn(positions, k + 1, workers=-1, progress=None)
    extra_mask = ~(
        ckd_boundary_index[:, :, None] == ckd_index[:, None, :]
    ).any(dim=2)
    extra_rank = extra_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(count, device=device)
    extra_index = ckd_boundary_index[rows, extra_rank]
    probe_index = torch.cat((reference_index, ckd_index, extra_index[:, None]), dim=1)

    _progress("default raw topk(K+1) and probe observation")
    observation = observe_reference_knn(
        positions, k, chunk_size, probe_index=probe_index, progress=_progress
    )
    raw_for_reference = observation.probe_raw_distance[:, :k]
    raw_for_candidate = observation.probe_raw_distance[:, k : 2 * k]
    topk_size_sensitivity = _identity(reference_index, observation.neighbor_index)

    _progress("same-environment default topk(K) repeat run 2")
    repeat_started = time.perf_counter()
    repeat_index, repeat_distance = _knn(positions, k, chunk_size, _progress)
    _sync(device)
    repeat_seconds = time.perf_counter() - repeat_started
    repeat_graph = candidate_graph_from_neighbors(
        orientation, config, repeat_index, repeat_distance
    )
    repeat_roots = _connected_component_roots(count, repeat_graph.accepted_edges, config)
    repeatability = {
        "neighbor_index": _identity(reference_index, repeat_index),
        "neighbor_distance": _identity(reference_distance, repeat_distance),
        "graph": _graph_identity(reference_graph, repeat_graph),
        "partition_roots": _identity(reference_roots, repeat_roots),
        "repeat_search_seconds": repeat_seconds,
    }
    del repeat_index, repeat_distance, repeat_graph, repeat_roots

    _progress("full-scene explicit use_mm_for_euclid_dist")
    mm_index, mm_distance, mm_seconds = _knn_explicit_mode(
        positions, k, chunk_size, "use_mm_for_euclid_dist"
    )
    explicit_mm_full = {
        "neighbor_index": _identity(reference_index, mm_index),
        "neighbor_distance": _identity(reference_distance, mm_distance),
        "seconds": mm_seconds,
    }
    del mm_index, mm_distance

    attribution = classify_neighbor_mismatches(
        positions, reference_index, raw_for_reference,
        ckd_index, raw_for_candidate,
    )
    mismatch = attribution["mismatch"]
    absolute_margin, relative_margin = boundary_margins(
        observation.boundary_raw_distance, k
    )
    mismatch_rows = torch.nonzero(mismatch, as_tuple=False).reshape(-1)
    class_counts = {
        CLASS_NAMES[class_id]: int((attribution["primary_class"] == class_id).sum())
        for class_id in sorted(CLASS_NAMES)
    }
    margin_report = {
        "reporting_epsilon_only": 1e-30,
        "matched_absolute": _distribution(absolute_margin[~mismatch]),
        "mismatched_absolute": _distribution(absolute_margin[mismatch]),
        "matched_relative": _distribution(relative_margin[~mismatch]),
        "mismatched_relative": _distribution(relative_margin[mismatch]),
        "membership_mismatched_absolute": _distribution(
            absolute_margin[attribution["membership_mismatch"]]
        ),
        "membership_mismatched_relative": _distribution(
            relative_margin[attribution["membership_mismatch"]]
        ),
    }

    candidate_graph = candidate_graph_from_neighbors(
        orientation, config, ckd_index, ckd_distance, progress=_progress
    )
    candidate_roots = _connected_component_roots(count, candidate_graph.accepted_edges, config)
    candidate_equivalence = {
        "neighbor_index": _identity(reference_index, ckd_index),
        "neighbor_row_mismatch_count": int(mismatch.sum()),
        "graph": _graph_identity(reference_graph, candidate_graph),
        "partition_roots": _identity(reference_roots, candidate_roots),
    }

    arithmetic_samples = _sample_arithmetic(
        positions, attribution, raw_for_reference, raw_for_candidate
    )
    production_chunk_modes = _production_chunk_mode_audit(
        positions, k, chunk_size
    )
    synthetic = _synthetic_audit(device)

    artifact = {
        "schema": "wl119-4-reference-mismatch-attribution-v2",
        "row_id": mismatch_rows.cpu(),
        "primary_class": attribution["primary_class"][mismatch].cpu(),
        "membership_mismatch": attribution["membership_mismatch"][mismatch].cpu(),
        "pair_fields_valid": attribution["membership_mismatch"][mismatch].cpu(),
        "boundary_only": attribution["boundary_only"][mismatch].cpu(),
        "duplicate_effect": attribution["duplicate_effect"][mismatch].cpu(),
        "reference_only_index": attribution["reference_only_index"][mismatch].cpu(),
        "candidate_only_index": attribution["candidate_only_index"][mismatch].cpu(),
        "reference_only_raw": attribution["reference_only_raw"][mismatch].cpu(),
        "candidate_only_raw": attribution["candidate_only_raw"][mismatch].cpu(),
        "reference_squared_float64": attribution["reference_squared_float64"][mismatch].cpu(),
        "candidate_squared_float64": attribution["candidate_squared_float64"][mismatch].cpu(),
        "mathematical_squared_gap": attribution["mathematical_squared_gap"][mismatch].cpu(),
        "derived_squared_error_bound": attribution["derived_squared_error_bound"][mismatch].cpu(),
        "ulp_separation": attribution["ulp_separation"][mismatch].cpu(),
        "reference_k_margin_abs": absolute_margin[mismatch].cpu(),
        "reference_k_margin_rel": relative_margin[mismatch].cpu(),
    }
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, arguments.artifact)

    all_repeat_exact = (
        repeatability["neighbor_index"]["value_exact"]
        and repeatability["neighbor_distance"]["value_exact"]
        and all(value["value_exact"] for value in repeatability["graph"].values())
        and repeatability["partition_roots"]["value_exact"]
    )
    mismatch_explained = sum(class_counts.values()) == int(mismatch.sum()) and class_counts["H_UNATTRIBUTED"] == 0
    report = {
        "batch": "Worklog 119-4 Reference-Exact KNN Semantics Attribution",
        "checkpoint": str(arguments.checkpoint), "iteration": payload.get("iteration"),
        "reference_contract": {
            "source": "osn_gs/surface/torch_coverage_first_subset_partition.py::_knn",
            "shape": list(positions.shape), "dtype": str(positions.dtype), "device": str(device),
            "k": k, "chunk_size": chunk_size,
            "cdist_call": "torch.cdist(query_chunk, all_positions), default compute_mode",
            "self_exclusion": "row index set to +inf before topk",
            "topk": {"largest": False, "sorted": True, "authoritative_k": k},
            "returned_distance": "direct gathered coordinate difference .norm, not raw cdist",
        },
        "environment": environment,
        "runtime_seconds": {
            "authoritative_reference_search": reference_search_seconds,
            "reference_graph_and_roots": reference_graph_seconds,
            "raw_kplus1_observation": observation.elapsed_seconds,
            "repeat_reference_search": repeat_seconds,
            "explicit_mm_search": mm_seconds,
        },
        "repeatability": repeatability,
        "reference_repeatability_exact": all_repeat_exact,
        "topk_k_vs_kplus1_first_k": topk_size_sensitivity,
        "explicit_mm_full_scene": explicit_mm_full,
        "production_chunk_compute_mode_audit": production_chunk_modes,
        "historical_ckdtree_equivalence": candidate_equivalence,
        "mismatch_attribution": {
            "ordered_element_mismatch_count": int((reference_index != ckd_index).sum()),
            "row_mismatch_count": int(mismatch.sum()),
            "order_only_count": int(attribution["order_only"].sum()),
            "membership_mismatch_count": int(attribution["membership_mismatch"].sum()),
            "k_boundary_only_count": int(attribution["boundary_only"].sum()),
            "class_counts": class_counts,
            "classification_complete": mismatch_explained,
            "ulp_separation_membership_mismatches": _distribution(
                attribution["ulp_separation"][attribution["membership_mismatch"]]
            ),
            "mathematical_squared_gap_membership_mismatches": _distribution(
                attribution["mathematical_squared_gap"][attribution["membership_mismatch"]]
            ),
            "derived_error_bound_membership_mismatches": _distribution(
                attribution["derived_squared_error_bound"][attribution["membership_mismatch"]]
            ),
            "artifact": str(arguments.artifact),
        },
        "k_boundary_margin": margin_report,
        "distance_arithmetic_samples": arithmetic_samples,
        "synthetic_adversarial_contracts": synthetic,
        "attribution_gate_inputs": {
            "reference_deterministic": all_repeat_exact,
            "mismatch_classification_complete": mismatch_explained,
            "default_full_scene_equals_explicit_mm": explicit_mm_full["neighbor_index"]["value_exact"],
            "clean_exact_reproduction_path_demonstrated": False,
        },
        "attribution_gate_verdict": {
            "passed": False,
            "outcome": "C_REFERENCE_SEMANTICS_TOO_IMPLEMENTATION_COUPLED_FOR_A_CLEAN_EXACT_REPLACEMENT",
            "accelerated_backend_implemented": False,
            "reasons": [
                "Production default is exact with explicit MM only at the frozen call shape/chunking.",
                "Changing topk output size from K to K+1 changes first-K output.",
                "Small fixtures show shape-dependent default direct/MM behavior.",
                "Candidate pruning changes GEMM shape/arithmetic and lacks an exact ordered-output proof.",
            ],
        },
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(json.dumps({
        "out": str(arguments.out), "artifact": str(arguments.artifact),
        "repeatability_exact": all_repeat_exact,
        "class_counts": class_counts,
        "boundary_only": int(attribution["boundary_only"].sum()),
        "explicit_mm_exact": explicit_mm_full["neighbor_index"]["value_exact"],
        "partition_exact_ckdtree": candidate_equivalence["partition_roots"]["value_exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
