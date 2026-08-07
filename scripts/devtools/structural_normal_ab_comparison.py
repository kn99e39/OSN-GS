"""Worklog 75: covariance_normal vs structural_normal A/B (bounded decision run).

Worklog 74 ended by showing the orientation stage is an independent
cycle-destruction bottleneck: of 34 distance-stage cycle edges only 16 survived
to mutuality, 13 died on tangent incompatibility, and that tangent is
`cross(normal, outward)` over the covariance eigenframe normal. This script is
the single bounded comparison that closes the normal-source question -- it is
not another diagnostic round.

Mode A (`covariance_normal`): the current covariance eigenframe
`normal_candidate` path, unchanged.

Mode B (`structural_normal`): `torch_structural_normal.compute_structural_normals`
-- local PCA over region-owned observed point POSITIONS only (no scale,
rotation, covariance, SH, opacity, renderer, or optimizer state), then the
existing missing-sector/outward-direction logic re-derives the boundary tangent
from that normal.

FROZEN across A and B (mode B never re-extracts anything):
  * candidate ids and xyz -- mode B rebuilds the orientation of mode A's
    already-admitted candidate set via `rebuild_candidate_orientation`.
  * region ownership -- computed once per scene, before either mode.
  * boundary reasons, full-evidence sampling scale.
  * connectivity/distance thresholds, ambiguity and mutuality logic, topology
    acceptance -- both modes go through the SAME unmodified
    `torch_region_owned_dense_boundary_support._connect` and the SAME unmodified
    worklog 73/74 diagnostics.

Nothing here is wired into the optimizer, renderer, trainer, or checkpoint
path; `boundary_support_spacing` is not activated or redesigned; no hull, PCA
rectangle, forced closure, gap bridging, cross-region merge, or geometric
fallback is introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import osn_gs.core.torch_pipeline  # noqa: F401 -- resolve osn_gs's own circular-import order first
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_dense_boundary_connectivity_diagnostics import diagnose_dense_boundary_connectivity
from osn_gs.surface.torch_dense_boundary_scale_diagnostics import diagnose_scale_domain
from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation, extract_covariance_frame
from osn_gs.surface.torch_nurbs import pca_parameterize_points
from osn_gs.surface.torch_region_owned_dense_boundary_support import (
    _connect,
    extract_dense_boundary_support,
)
from osn_gs.surface.torch_single_chart_uv_validity import interior_within_boundary
from osn_gs.surface.torch_structural_normal import (
    compute_structural_normals,
    normal_angular_disagreement_degrees,
    rebuild_candidate_orientation,
)

MODE_COVARIANCE = "covariance_normal"
MODE_STRUCTURAL = "structural_normal"


def _stats(values: torch.Tensor) -> dict:
    if int(values.numel()) == 0:
        return {"median": None, "p90": None, "max": None, "mean": None}
    array = values.detach().cpu().numpy()
    return {
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def _closed_components(result) -> list:
    return [c for c in result.components if c.closed]


def _containment_for_loop(component, candidates_by_id, evidence_positions) -> dict | None:
    """interior_outside_boundary for a recovered loop, in a single shared
    PCA-UV frame over (loop vertices + region evidence) -- the same convention
    worklog 69/70/71 used. Only ever called when a valid closed loop exists."""

    loop_positions = torch.tensor(
        [candidates_by_id[sid].position for sid in component.stable_ids],
        dtype=evidence_positions.dtype, device=evidence_positions.device,
    )
    combined = torch.cat((loop_positions, evidence_positions), dim=0)
    uv = pca_parameterize_points(combined)
    loop_uv = uv[: int(loop_positions.shape[0])]
    evidence_uv = uv[int(loop_positions.shape[0]) :]
    loop_id_set = set(component.stable_ids)
    interior_mask = torch.tensor(
        [sid not in loop_id_set for sid in candidates_by_id["__evidence_ids__"]], dtype=torch.bool,
    ) if "__evidence_ids__" in candidates_by_id else None
    interior_uv = evidence_uv if interior_mask is None else evidence_uv[interior_mask]
    return interior_within_boundary(interior_uv, loop_uv)


def run_mode(
    mode: str, points: torch.Tensor, normals: torch.Tensor, stable_ids: list,
    representative_scale: float | None, frozen_candidates=None,
) -> dict:
    """One mode on one region. ``frozen_candidates`` is None for mode A (which
    defines the candidate set) and mode A's candidates for mode B (which only
    rebuilds their orientation frame)."""

    # Orientation stage: mode A extracts the candidate set (which is what
    # defines the frozen set), mode B only rebuilds that set's frame. Both are
    # then connected by the SAME `_connect` call, timed identically -- mode A
    # is re-connected explicitly rather than reusing the result already
    # computed inside extraction, so the two connectivity timings are
    # apples-to-apples rather than one of them reading ~0.
    torch.cuda.synchronize() if points.is_cuda else None
    normal_start = time.perf_counter()
    if frozen_candidates is None:
        support = extract_dense_boundary_support(points, normals, stable_ids, representative_scale=representative_scale)
        candidates = support.candidates
        rebuild_diagnostics = None
    else:
        candidates, rebuild_diagnostics = rebuild_candidate_orientation(points, normals, stable_ids, frozen_candidates)
    torch.cuda.synchronize() if points.is_cuda else None
    normal_seconds = time.perf_counter() - normal_start

    connect_start = time.perf_counter()
    connected = _connect(candidates, representative_scale)
    torch.cuda.synchronize() if points.is_cuda else None
    connect_seconds = time.perf_counter() - connect_start

    connectivity = diagnose_dense_boundary_connectivity(candidates)
    scale_domain = diagnose_scale_domain(candidates, representative_scale)
    stages = connectivity.get("stages", {})
    outcomes = connectivity.get("terminal_outcomes", {})

    def stage(name, key):
        return stages.get(name, {}).get(key)

    closed = _closed_components(connected)
    return {
        "mode": mode,
        "candidate_count": len(candidates),
        "normal_rejection_count": outcomes.get("normal_incompatible", 0),
        "tangent_rejection_count": outcomes.get("tangent_incompatible", 0),
        "pairwise_rejections": dict(connected.rejection_counts),
        "distance_valid_edge_survival": {
            "distance": stage("distance_local_scale", "surviving_directional_proposals"),
            "reason": stage("reason", "surviving_directional_proposals"),
            "normal": stage("normal", "surviving_directional_proposals"),
            "tangent": stage("tangent", "surviving_directional_proposals"),
            "ambiguity": stage("ambiguity", "surviving_directional_proposals"),
            "mutuality": stage("mutuality", "surviving_directional_proposals"),
        },
        "distance_stage_cycle_survival": {
            "distance": stage("distance_local_scale", "closed_cycles"),
            "normal": stage("normal", "closed_cycles"),
            "tangent": stage("tangent", "closed_cycles"),
            "ambiguity": stage("ambiguity", "closed_cycles"),
            "mutuality": stage("mutuality", "closed_cycles"),
        },
        "final_closed_loop_count": len(closed),
        "final_component_status_counts": _status_counts(connected.components),
        "directional_coverage": connectivity.get("directional_coverage", {}),
        "terminal_outcomes": outcomes,
        "full_evidence_scale": connected.full_evidence_scale,
        "candidate_angular_largest_gap_degrees": scale_domain.get("candidate_angular_largest_gap_degrees"),
        "rebuild_diagnostics": rebuild_diagnostics,
        "normal_generation_seconds": normal_seconds,
        "boundary_connectivity_seconds": connect_seconds,
        "_closed_components": closed,
        "_candidates": candidates,
    }


def _status_counts(components) -> dict:
    counts: dict[str, int] = {}
    for component in components:
        counts[component.status] = counts.get(component.status, 0) + 1
    return counts


def compare_region(
    region_label: str, points: torch.Tensor, covariance: torch.Tensor, stable_ids: list,
    representative_scale: float | None,
) -> dict:
    covariance_normals = extract_covariance_frame(covariance).normal_candidate
    structural_start = time.perf_counter()
    structural_normals = compute_structural_normals(points)
    structural_only_seconds = time.perf_counter() - structural_start

    mode_a = run_mode(MODE_COVARIANCE, points, covariance_normals, stable_ids, representative_scale)
    mode_b = run_mode(
        MODE_STRUCTURAL, points, structural_normals, stable_ids, representative_scale,
        frozen_candidates=mode_a["_candidates"],
    )

    disagreement = normal_angular_disagreement_degrees(covariance_normals, structural_normals)
    candidate_indices = [
        i for i, sid in enumerate(stable_ids)
        if sid in {c.stable_id for c in mode_a["_candidates"]}
    ]
    candidate_disagreement = disagreement[torch.tensor(candidate_indices, dtype=torch.long)] if candidate_indices else disagreement[:0]

    containment = {}
    evidence_ids_holder = {"__evidence_ids__": stable_ids}
    for mode_result in (mode_a, mode_b):
        by_id = {c.stable_id: c for c in mode_result["_candidates"]}
        by_id.update(evidence_ids_holder)
        loops = mode_result["_closed_components"]
        containment[mode_result["mode"]] = (
            [_containment_for_loop(loop, by_id, points) for loop in loops] if loops else None
        )

    element_bytes = points.element_size()
    return {
        "region": region_label,
        "point_count": int(points.shape[0]),
        "representative_scale": representative_scale,
        MODE_COVARIANCE: {k: v for k, v in mode_a.items() if not k.startswith("_")},
        MODE_STRUCTURAL: {k: v for k, v in mode_b.items() if not k.startswith("_")},
        "normal_angular_disagreement_degrees_all_points": _stats(disagreement),
        "normal_angular_disagreement_degrees_at_candidates": _stats(candidate_disagreement),
        "containment_interior_outside_boundary": containment,
        "structural_normal_only_seconds": structural_only_seconds,
        "structural_normal_extra_bytes": int(points.shape[0]) * 3 * element_bytes,
        "structural_normal_persistent_state": "none (recomputed per call, not stored on model/checkpoint/optimizer)",
    }


# --------------------------------------------------------------------------
# Scene 1/2: existing synthetic fixtures (no new dataset built).
# --------------------------------------------------------------------------


def run_synthetic_scene(scene_name: str) -> dict:
    from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene

    scene = make_gaussian_reliability_scene(scene_name, seed=0)
    points = scene.positions.to(torch.float32)
    covariance = scene.covariances.to(torch.float32)
    stable_ids = list(range(int(points.shape[0])))

    # Region ownership is frozen BEFORE either mode and never derived from any
    # normal: for a labelled fixture it is the fixture's own surface labels, and
    # for an unlabelled single-patch fixture the whole scene is one region.
    labels = scene.group_labels or ("region",) * int(points.shape[0])
    regions: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        regions.setdefault(label, []).append(index)

    region_reports = []
    for label, indices in sorted(regions.items()):
        if len(indices) < 4:
            continue
        selector = torch.tensor(indices, dtype=torch.long)
        region_reports.append(compare_region(
            f"{scene_name}:{label}", points[selector], covariance[selector],
            [stable_ids[i] for i in indices], representative_scale=None,
        ))
    return {"scene": scene_name, "description": scene.description, "regions": region_reports}


# --------------------------------------------------------------------------
# Scene 3: the real checkpoint that currently exhibits boundary-closure failure.
# --------------------------------------------------------------------------


def _load_checkpoint_model(checkpoint_dir: Path, device: str):
    from osn_gs.gaussian.torch_model import TorchGaussianModel

    payload = torch.load(checkpoint_dir / "checkpoint.pt", map_location=device, weights_only=False)
    raw = payload["model_raw"]
    rest_dim = int(raw["features_rest"].shape[-2])
    degree = 0
    while (degree + 1) ** 2 - 1 < rest_dim:
        degree += 1
    model = TorchGaussianModel(sh_degree=degree, device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"), surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    return model


def _render_fingerprint(model) -> dict:
    """Render one fixed view and fingerprint it, to verify both modes leave the
    render path bit-identical (the structural normal must never reach it)."""

    from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
    from osn_gs.render.torch_fallback import TorchCamera

    device = model.device
    world_view = torch.eye(4, device=device)
    world_view[2, 3] = 4.0
    camera = TorchCamera(
        image_height=64, image_width=64, world_view_transform=world_view,
        full_proj_transform=world_view.clone(), camera_center=torch.zeros(3, device=device),
    )
    renderer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=True, allow_fallback=True))
    with torch.no_grad():
        output = renderer.render(camera, model)
    image = output["render"] if isinstance(output, dict) and "render" in output else next(iter(output.values()))
    image = image.detach().to(torch.float64)
    return {
        "backend": renderer.backend_source,
        "shape": list(image.shape),
        "sum": float(image.sum()),
        "abs_sum": float(image.abs().sum()),
        "max": float(image.max()),
        "_tensor": image,
    }


def run_real_checkpoint(checkpoint_dir: Path, cap: int, device: str, max_regions: int) -> dict:
    model = _load_checkpoint_model(checkpoint_dir, device)
    render_before = _render_fingerprint(model)

    pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=cap), device=device)
    points = model.get_xyz.detach()
    stable_ids = list(range(int(points.shape[0])))
    with torch.no_grad():
        covariance = covariance_from_scale_rotation(model.get_scaling.detach(), model.get_rotation.detach())
        bundle = pipeline._construct_canonical_with_full_evidence(
            points, covariance, torch.sigmoid(model.get_opacity.detach()).reshape(-1), stable_ids,
        )
    regions = bundle.construction.surface_regions
    rep_stable_ids = bundle.representative_stable_ids

    # Region ownership: computed ONCE here, identical for both modes.
    cluster_ids = torch.tensor(regions.node_region_id, dtype=torch.long, device=points.device)
    propagated, _diag = pipeline._propagate_with_evidence_gating(points, covariance, bundle, cluster_ids)
    owned: dict[int, list[int]] = {}
    for full_index, region_id in enumerate(propagated.detach().cpu().tolist()):
        if region_id >= 0:
            owned.setdefault(region_id, []).append(full_index)

    mean_spacing = bundle.evidence.mean_spacing
    region_reports = []
    for region in regions.regions:
        indices = owned.get(region.region_id, [])
        if len(indices) < 4:
            continue
        member_local = [rep_stable_ids.index(sid) for sid in region.member_ids if sid in rep_stable_ids]
        representative_scale = (
            float(mean_spacing[torch.tensor(member_local, dtype=torch.long, device=mean_spacing.device)].median())
            if member_local else None
        )
        selector = torch.tensor(indices, dtype=torch.long, device=points.device)
        print(f"  region {region.region_id}: evidence={len(indices)} ...", flush=True)
        region_reports.append(compare_region(
            f"real:region{region.region_id}", points[selector], covariance[selector],
            [stable_ids[i] for i in indices], representative_scale,
        ))
        if len(region_reports) >= max_regions:
            break

    render_after = _render_fingerprint(model)
    identical = bool(torch.equal(render_before["_tensor"], render_after["_tensor"]))
    return {
        "scene": f"real:{checkpoint_dir.name}",
        "description": "Real OSN-GS checkpoint exhibiting the dense boundary-closure failure (worklog 72-74).",
        "regions": region_reports,
        "render_identity": {
            "bitwise_identical": identical,
            "backend": render_before["backend"],
            "before": {k: v for k, v in render_before.items() if k != "_tensor"},
            "after": {k: v for k, v in render_after.items() if k != "_tensor"},
        },
    }


def _aggregate(scene_reports: list) -> dict:
    totals = {
        mode: {
            "candidate_count": 0, "normal_rejection_count": 0, "tangent_rejection_count": 0,
            "distance_edges": 0, "tangent_edges": 0, "mutuality_edges": 0,
            "distance_cycles": 0, "tangent_cycles": 0, "final_closed_loops": 0,
            "normal_seconds": 0.0, "connect_seconds": 0.0,
        }
        for mode in (MODE_COVARIANCE, MODE_STRUCTURAL)
    }
    disagreements = []
    region_count = 0
    for scene in scene_reports:
        for region in scene["regions"]:
            region_count += 1
            median = region["normal_angular_disagreement_degrees_at_candidates"]["median"]
            if median is not None:
                disagreements.append(median)
            for mode in (MODE_COVARIANCE, MODE_STRUCTURAL):
                block = region[mode]
                bucket = totals[mode]
                bucket["candidate_count"] += block["candidate_count"]
                bucket["normal_rejection_count"] += block["normal_rejection_count"]
                bucket["tangent_rejection_count"] += block["tangent_rejection_count"]
                bucket["distance_edges"] += block["distance_valid_edge_survival"]["distance"] or 0
                bucket["tangent_edges"] += block["distance_valid_edge_survival"]["tangent"] or 0
                bucket["mutuality_edges"] += block["distance_valid_edge_survival"]["mutuality"] or 0
                bucket["distance_cycles"] += block["distance_stage_cycle_survival"]["distance"] or 0
                bucket["tangent_cycles"] += block["distance_stage_cycle_survival"]["tangent"] or 0
                bucket["final_closed_loops"] += block["final_closed_loop_count"]
                bucket["normal_seconds"] += block["normal_generation_seconds"]
                bucket["connect_seconds"] += block["boundary_connectivity_seconds"]
    return {
        "region_count": region_count,
        "per_mode_totals": totals,
        "median_candidate_normal_disagreement_degrees": (
            float(np.median(disagreements)) if disagreements else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/extent_ab/val64/baseline_compatible/2900"))
    parser.add_argument("--max_real_regions", type=int, default=4)
    parser.add_argument("--synthetic", nargs="+", default=["box_face", "cylinder"])
    parser.add_argument("--skip_real", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/extent_ab/val75/structural_normal_ab.json"))
    args = parser.parse_args()

    scenes = []
    for name in args.synthetic:
        print(f"analyzing synthetic scene {name} ...", flush=True)
        scenes.append(run_synthetic_scene(name))
    if not args.skip_real and (args.checkpoint / "checkpoint.pt").exists():
        print(f"analyzing real checkpoint {args.checkpoint} ...", flush=True)
        scenes.append(run_real_checkpoint(args.checkpoint, args.cap, "cuda", args.max_real_regions))

    report = {"scenes": scenes, "aggregate": _aggregate(scenes)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
