"""Full-scene adoption gate for WL119 exact-KNN Performance Track candidates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

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
    CoverageFirstPartitionConfig, _connected_component_roots, build_candidate_graph,
)
from osn_gs.surface.torch_exact_knn_performance import (  # noqa: E402
    candidate_graph_from_neighbors, scipy_ckdtree_exact_knn,
)
from osn_gs.surface.torch_surfel_surface_orientation import (  # noqa: E402
    derive_surface_orientation_from_surfel,
)


def _progress(message: str) -> None:
    print(f"[wl119-exact-knn] {message}", flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_identity(expected: Any, actual: Any) -> dict[str, Any]:
    shape_equal = tuple(expected.shape) == tuple(actual.shape)
    if not shape_equal:
        return {"shape_exact": False, "value_exact": False, "mismatch_count": None}
    unequal = expected != actual
    return {
        "shape_exact": True, "value_exact": bool(not unequal.any()),
        "mismatch_count": int(unequal.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=-1)
    arguments = parser.parse_args()

    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError("Full-scene KNN gate requires the frozen 2DGS surfel checkpoint")
    uncertain = model.is_uncertain.reshape(-1).to(torch.bool)
    visible = torch.nonzero(~uncertain, as_tuple=False).reshape(-1)
    full = derive_surface_orientation_from_surfel(model)
    orientation = replace(
        full,
        gaussian_ids=full.gaussian_ids[visible], positions=full.positions[visible],
        tangent_axis_u=full.tangent_axis_u[visible], tangent_axis_v=full.tangent_axis_v[visible],
        surface_normal=full.surface_normal[visible],
        tangent_scale_u=full.tangent_scale_u[visible], tangent_scale_v=full.tangent_scale_v[visible],
    )
    positions = orientation.positions
    device = positions.device
    count = int(positions.shape[0])
    config = CoverageFirstPartitionConfig()
    k = min(int(config.neighbor_count), count - 1)

    _progress(f"reference torch.cdist full scene count={count} k={k}")
    _sync(device)
    reference_started = time.perf_counter()
    reference = build_candidate_graph(
        orientation, config, retain_neighbor_index=True, progress=_progress
    )
    _sync(device)
    reference_seconds = time.perf_counter() - reference_started

    _progress("candidate scipy cKDTree full scene")
    _sync(device)
    candidate_started = time.perf_counter()
    candidate_index, candidate_distance = scipy_ckdtree_exact_knn(
        positions, k, workers=arguments.workers, progress=_progress
    )
    candidate = candidate_graph_from_neighbors(
        orientation, config, candidate_index, candidate_distance, progress=_progress
    )
    _sync(device)
    candidate_seconds = time.perf_counter() - candidate_started

    _progress("connected-component partition identity")
    reference_roots = _connected_component_roots(count, reference.accepted_edges, config)
    candidate_roots = _connected_component_roots(count, candidate.accepted_edges, config)

    neighbor_identity = _tensor_identity(reference.neighbor_index, candidate.neighbor_index)
    neighbor_row_mismatch = int(
        (reference.neighbor_index != candidate.neighbor_index).any(dim=1).sum()
    )
    distance_difference = (reference.local_spacing - candidate.local_spacing).abs()
    evidence = {
        "neighbor_index": {**neighbor_identity, "row_mismatch_count": neighbor_row_mismatch},
        "local_spacing": {
            **_tensor_identity(reference.local_spacing, candidate.local_spacing),
            "max_abs_delta": float(distance_difference.max()),
        },
        "candidate_edges": _tensor_identity(reference.candidate_edges, candidate.candidate_edges),
        "spatial_edge_mask": _tensor_identity(reference.spatial_edge_mask, candidate.spatial_edge_mask),
        "normal_compatible_mask": _tensor_identity(reference.normal_compatible_mask, candidate.normal_compatible_mask),
        "accepted_edges": _tensor_identity(reference.accepted_edges, candidate.accepted_edges),
        "partition_roots": _tensor_identity(reference_roots, candidate_roots),
    }
    adopted = all(
        evidence[key]["value_exact"]
        for key in (
            "neighbor_index", "candidate_edges", "spatial_edge_mask",
            "normal_compatible_mask", "accepted_edges", "partition_roots",
        )
    )
    report = {
        "track": "WL119 Performance Track / exact KNN adoption gate",
        "checkpoint": str(arguments.checkpoint), "iteration": payload.get("iteration"),
        "count": count, "k": k, "device": str(device),
        "reference_backend": "immutable-torch-cdist",
        "candidate_backend": "scipy-cKDTree-exact-eps0-float64",
        "workers": arguments.workers,
        "runtime_dependent_splitting": False, "silent_backend_fallback": False,
        "runtime_seconds": {
            "reference": reference_seconds, "candidate": candidate_seconds,
        },
        "candidate_speedup": reference_seconds / candidate_seconds,
        "equivalence": evidence,
        "adoption_gate_passed": adopted,
        "adoption_decision": "adopt" if adopted else "reject-contract-mismatch",
        "tie_semantics_policy": (
            "Any reference/candidate neighbor mismatch is reported as a contract issue; "
            "the candidate is not modified or silently accepted."
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _progress(json.dumps({
        "out": str(arguments.out), "runtime": report["runtime_seconds"],
        "speedup": report["candidate_speedup"], "adopted": adopted,
        "neighbor_row_mismatch": neighbor_row_mismatch,
        "partition_exact": evidence["partition_roots"]["value_exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
