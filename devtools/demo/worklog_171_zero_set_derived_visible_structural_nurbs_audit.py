from __future__ import annotations

"""Worklog 171: zero-set-derived visible structural NURBS audit.

This is an isolated diagnostic.  Geometry comes only from frozen TSDF
zero-set samples.  Existing Gaussian region IDs organize real support but
never provide fit points.  A case materializes only when the complete selected
support is one native-cell component with one valid boundary chart and a
full-rank frozen WL139/W154 8x4 degree-2 fit.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_155_intrinsic_normal_gaussian_region_viability_audit as w155  # noqa: E402
from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167  # noqa: E402
from devtools.demo import worklog_168_raw_zero_set_first_hit_positive_occlusion_evidence_audit as w168  # noqa: E402
from devtools.demo import worklog_170_construction_native_conservative_blocker_certificate_audit as w170  # noqa: E402
from osn_gs.surface.torch_gaussian_region_owned_tsdf import (  # noqa: E402
    ABSTAIN_REPRESENTATIVE,
    MATERIALIZED_REPRESENTATIVE,
    ObservedSupportBoundary,
    RegionOwnedTSDFSupport,
    TSDFVisibleSurfaceSamples,
    build_native_tsdf_support_components,
    derive_native_support_boundary,
    extract_tsdf_zero_surface_samples,
    fit_boundary_first_region_representative,
)


DEFAULT_W145 = REPO_ROOT / "output/confirmed/145_genuine_physical_sheet_oracle_clean_support_representative_audit"
DEFAULT_W154 = REPO_ROOT / "output/confirmed/154_gaussian_region_owned_tsdf_boundary_first_nurbs"
DEFAULT_DATASET = REPO_ROOT / "DATASET"
DEFAULT_OUT = REPO_ROOT / "output/171_zero_set_derived_visible_structural_nurbs_audit"

REAL_CASE_CONTRACT = {
    "real_tabletop_coherent": {
        "review_key": "tabletop",
        "frozen_case": "tabletop_broad_planar_clean",
        "organization": "prior_dominant_region",
    },
    "real_curved_vase_coherent": {
        "review_key": "vase_neighbor",
        "frozen_case": "table_rim_curved_interior_candidate",
        "organization": "prior_dominant_region",
    },
    "real_mixed_contact_stress": {
        "review_key": "table_side",
        "frozen_case": "tabletop_near_vase_boundary_candidate",
        "organization": "all_accepted_regions",
    },
}

SYNTHETIC_CASES = (
    "synthetic_planar_single_sheet",
    "synthetic_curved_single_sheet",
    "synthetic_layered_non_single_chart",
)
CASE_ORDER = SYNTHETIC_CASES + tuple(REAL_CASE_CONTRACT)
REVIEW_FAMILIES = (
    "support_common_world",
    "boundary_and_domain",
    "nurbs_fit_common_world",
    "residual_review",
    "image_reference_overlay",
)

FIT_CONTRACT = {
    "family": "WL139_boundary_chart_seeded_visible_surface_lsq",
    "resolution_u": 8,
    "resolution_v": 4,
    "degree_u": 2,
    "degree_v": 2,
    "smoothness_lambda": 1.0e-4,
    "tikhonov_lambda": 1.0e-4,
    "correction_rounds": 2,
    "projection_iterations": 2,
    "tuned_for_w171": False,
}

SUPPORT_RGB = "#17becf"
CONTEXT_RGB = "#9aa0a6"
BOUNDARY_RGB = "#ff7f0e"
DOMAIN_RGB = "#6baed6"
SURFACE_RGB = "#2563eb"
CONTROL_RGB = "#d946ef"


@dataclass
class CaseRuntime:
    name: str
    kind: str
    samples: TSDFVisibleSurfaceSamples
    selection: dict[str, Any]
    context_xyz: np.ndarray
    region_ids: np.ndarray | None = None
    components: tuple[Any, ...] = tuple()
    component_ids: Any | None = None
    boundary: ObservedSupportBoundary | None = None
    representative: Any | None = None
    result: dict[str, Any] | None = None
    fitted_xyz: np.ndarray | None = None
    residual: np.ndarray | None = None
    dense_surface_xyz: np.ndarray | None = None


def _jsonable(value: Any) -> Any:
    import torch

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _distribution(values: np.ndarray, h: float | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        world = {key: None for key in ("min", "median", "mean", "p95", "max")}
        world["count"] = 0
    else:
        world = {
            "count": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(array.max()),
        }
    if h is None:
        return world
    normalized = dict(world)
    for key in ("min", "median", "mean", "p95", "max"):
        normalized[key] = None if world[key] is None else float(world[key] / h)
    return {"world_units": world, "normalized_by_h": normalized}


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _stable_plot_subset(points: np.ndarray, limit: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    if count <= limit:
        indices = np.arange(count, dtype=np.int64)
    else:
        step = int(math.ceil(count / limit))
        indices = np.arange(0, count, step, dtype=np.int64)
    return points[indices], indices


def _all_accepted_support(samples: TSDFVisibleSurfaceSamples, region_ids: np.ndarray | None = None) -> RegionOwnedTSDFSupport:
    import torch

    count = int(samples.source_cell_keys.numel())
    if region_ids is None:
        region = torch.zeros((count,), dtype=torch.int64)
    else:
        region = torch.as_tensor(np.asarray(region_ids), dtype=torch.int64)
    accepted = torch.ones((count,), dtype=torch.bool)
    # Status strings are not consumed by native topology.  Empty tuples avoid
    # allocating millions of duplicate Python strings for the real stress case.
    return RegionOwnedTSDFSupport(
        nearest_region_id=region,
        nearest_membership_status=tuple(),
        owned_region_id=region,
        membership_status=tuple(),
        accepted_mask=accepted,
        accounting={"accepted": count, "selection_stage": "frozen_case_organization"},
    )


def _build_curved_graph_field() -> tuple[Any, dict[str, Any]]:
    """Finite parabolic graph in the same sparse TSDF carrier as W167."""

    import torch
    from evidence_bounded_tsdf.field import SparseProjectiveTSDF

    h = w167.SYNTHETIC_H
    mu = w167.SYNTHETIC_MU
    # Even cardinality makes voxel centers exactly origin-symmetric; this keeps
    # the finite-domain centroid from tilting the canonical chart independently
    # of the graph curvature being audited.
    xy = np.arange(-22, 22, dtype=np.int64)
    z = np.arange(-20, 21, dtype=np.int64)
    gx, gy, gz = np.meshgrid(xy, xy, z, indexing="ij")
    indices = np.column_stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)))
    points = (indices.astype(np.float64) + 0.5) * h
    # A fixed analytic graph, not a tuned fit target.  Its maximum neighboring
    # height change is below one cell, retaining a connected native-cell sheet.
    surface_z = 0.10 * points[:, 0] ** 2 + 0.06 * points[:, 1] ** 2 - 0.25
    signed = points[:, 2] - surface_z
    phi = np.clip(signed / mu, -1.0, 1.0).astype(np.float32)
    order = np.argsort(w167._encode_keys(indices), kind="stable")
    keys = w167._encode_keys(indices[order])
    field = SparseProjectiveTSDF(
        keys=torch.as_tensor(keys, dtype=torch.int64),
        value=torch.as_tensor(phi[order], dtype=torch.float32),
        support_count=torch.ones((len(keys),), dtype=torch.int32),
        h=h,
        mu=mu,
    )
    return field, {
        "construction": "fixed finite parabolic graph in historical SparseProjectiveTSDF carrier",
        "equation": "z = 0.10*x^2 + 0.06*y^2 - 0.25",
        "h": h,
        "mu": mu,
        "authoritative_voxels": int(len(keys)),
        "unknown_outside_finite_xy_support": True,
    }


def _synthetic_runtime(name: str) -> CaseRuntime:
    if name == "synthetic_planar_single_sheet":
        field, construction = w167._synthetic_field(w167._plane_surface("w171_planar"))
    elif name == "synthetic_curved_single_sheet":
        field, construction = _build_curved_graph_field()
    elif name == "synthetic_layered_non_single_chart":
        extracted, construction = w168._layered_fixture()
        fixture = {"analytic": {"surfaces": w168._layered_surfaces()}, "surface": extracted}
        field = w170._reconstruct_field("layered_two_sheet", fixture)
    else:
        raise ValueError(name)
    samples = extract_tsdf_zero_surface_samples(field, device="cpu")
    return CaseRuntime(
        name=name,
        kind="synthetic",
        samples=samples,
        selection={
            "rule": "complete all-eight-corner zero-straddling support from fixed synthetic field",
            "construction": construction,
            "region_ids_are_geometry": False,
            "gaussian_centers_used_as_geometry": False,
        },
        context_xyz=np.empty((0, 3), dtype=np.float32),
    )


def prior_real_case_specs(w154_root: Path, w145_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve W171 real cases exclusively from pre-W171 frozen artifacts."""

    report_path = w154_root / "candidate_f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report["qualitative_review"]["cases"]
    result: dict[str, dict[str, Any]] = {}
    for case_name, contract in REAL_CASE_CONTRACT.items():
        prior = cases[contract["review_key"]]
        if prior["frozen_case"] != contract["frozen_case"]:
            raise ValueError(f"prior frozen case mismatch for {case_name}")
        clouds = sorted(prior["clouds"], key=lambda row: Path(row["path"]).parent.name)
        if not clouds:
            raise ValueError(f"no prior review cloud for {case_name}")
        # Lexicographically first camera is a fixed mechanical rule, applied
        # before any W171 topology, fit, metric, or visualization exists.
        cloud = clouds[0]
        cloud_path = Path(cloud["path"])
        if not cloud_path.is_absolute():
            cloud_path = w145_root / contract["frozen_case"] / "per_view_renderer_median_events" / cloud_path
        with np.load(cloud_path, allow_pickle=False) as payload:
            points = np.asarray(payload["event_points_xyz"], dtype=np.float64)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError(f"invalid prior event envelope for {case_name}")
        region_id = None
        if contract["organization"] == "prior_dominant_region":
            candidates = cloud["candidate_region_ids_by_nearest_gaussian"]
            if not candidates:
                raise ValueError(f"missing prior region accounting for {case_name}")
            region_id = int(candidates[0][0])
        result[case_name] = {
            "frozen_case": contract["frozen_case"],
            "review_key": contract["review_key"],
            "camera": cloud_path.parent.name,
            "event_cloud": str(cloud_path),
            "event_count": int(len(points)),
            "aabb_min": points.min(axis=0),
            "aabb_max": points.max(axis=0),
            "organization": contract["organization"],
            "prior_region_id": region_id,
            "selection_frozen_before_w171_fit": True,
            "selection_uses_w171_visual_success": False,
            "selection_uses_fit_residual": False,
        }
    return result


def _load_real_runtimes(w154_root: Path, w145_root: Path) -> dict[str, CaseRuntime]:
    specs = prior_real_case_specs(w154_root, w145_root)
    sample_path = w154_root / "candidate_f_tsdf_surface_samples.npz"
    support_path = w154_root / "candidate_f_region_owned_support.npz"

    with np.load(sample_path, allow_pickle=False) as payload:
        world_all = np.asarray(payload["world_xyz"])
    with np.load(support_path, allow_pickle=False) as payload:
        owned_all = np.asarray(payload["owned_region_id"])
        accepted_all = np.asarray(payload["accepted_mask"], dtype=bool)

    selected: dict[str, np.ndarray] = {}
    context: dict[str, np.ndarray] = {}
    selected_regions: dict[str, np.ndarray] = {}
    selection_accounting: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        in_box = np.all(world_all >= spec["aabb_min"], axis=1) & np.all(world_all <= spec["aabb_max"], axis=1)
        envelope_mask = accepted_all & in_box
        values, counts = np.unique(owned_all[envelope_mask], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        if spec["organization"] == "prior_dominant_region":
            mask = envelope_mask & (owned_all == spec["prior_region_id"])
        else:
            mask = envelope_mask
        indices = np.flatnonzero(mask)
        selected[name] = indices
        selected_regions[name] = owned_all[indices].astype(np.int64, copy=True)
        context_indices = np.flatnonzero(envelope_mask & ~mask)
        context_points, _ = _stable_plot_subset(world_all[context_indices], limit=10_000)
        context[name] = context_points.astype(np.float32, copy=True)
        selection_accounting[name] = {
            **spec,
            "envelope_accepted_count": int(envelope_mask.sum()),
            "selected_support_count": int(len(indices)),
            "excluded_existing_region_context_count": int(len(context_indices)),
            "envelope_region_counts": {str(int(values[i])): int(counts[i]) for i in order},
            "aabb_is_exact_event_point_extent": True,
            "aabb_padding_or_tuned_margin": None,
            "support_deletion_after_selection": False,
        }
    del world_all, owned_all, accepted_all

    arrays: dict[str, dict[str, np.ndarray]] = {name: {} for name in specs}
    for key in ("source_cell_keys", "cell_indices", "world_xyz", "normals"):
        with np.load(sample_path, allow_pickle=False) as payload:
            full = np.asarray(payload[key])
            for name, indices in selected.items():
                arrays[name][key] = full[indices].copy()
        del full

    report = json.loads((w154_root / "candidate_f_report.json").read_text(encoding="utf-8"))
    h = float(report["observation_branch"].get("h", report["inputs"].get("h", w167.HISTORICAL_H)))
    import torch

    result: dict[str, CaseRuntime] = {}
    for name in specs:
        data = arrays[name]
        count = len(data["source_cell_keys"])
        samples = TSDFVisibleSurfaceSamples(
            source_cell_keys=torch.from_numpy(data["source_cell_keys"]),
            cell_indices=torch.from_numpy(data["cell_indices"]),
            world_xyz=torch.from_numpy(data["world_xyz"]),
            normals=torch.from_numpy(data["normals"]),
            corner_values=torch.empty((0, 8), dtype=torch.float32),
            corner_support_count=torch.empty((0, 8), dtype=torch.int32),
            h=h,
            stats={
                "surface_sample_count": count,
                "source": str(sample_path),
                "region_ownership_source": str(support_path),
                "zero_set_support_authoritative": True,
                "gaussian_centers_used_as_geometry": False,
            },
        )
        selection_accounting[name]["selected_support_hash_before_analysis"] = _sha256_arrays(
            data["source_cell_keys"], data["world_xyz"]
        )
        selection_accounting[name]["selected_region_count"] = int(len(np.unique(selected_regions[name])))
        result[name] = CaseRuntime(
            name=name,
            kind="real",
            samples=samples,
            selection=selection_accounting[name],
            context_xyz=context[name],
            region_ids=selected_regions[name],
        )
    return result


def _design_rank(samples: TSDFVisibleSurfaceSamples, component: Any, boundary: ObservedSupportBoundary) -> tuple[int | None, int]:
    if not boundary.eligible:
        return None, FIT_CONTRACT["resolution_u"] * FIT_CONTRACT["resolution_v"]
    import torch
    from osn_gs.surface.torch_nurbs import TorchNURBSSurface

    points = samples.world_xyz[component.sample_indices.to(torch.int64)]
    projected = torch.stack(
        ((points - boundary.chart_origin) @ boundary.tangent_u, (points - boundary.chart_origin) @ boundary.tangent_v), dim=1
    )
    boundary_projected = torch.stack(
        (
            (boundary.boundary_world - boundary.chart_origin) @ boundary.tangent_u,
            (boundary.boundary_world - boundary.chart_origin) @ boundary.tangent_v,
        ),
        dim=1,
    )
    mins = boundary_projected.amin(dim=0)
    spans = boundary_projected.amax(dim=0) - mins
    if bool(torch.any(spans <= 1.0e-12)):
        return 0, FIT_CONTRACT["resolution_u"] * FIT_CONTRACT["resolution_v"]
    uv = (projected - mins) / spans
    probe = TorchNURBSSurface(
        control_grid=torch.zeros((FIT_CONTRACT["resolution_u"], FIT_CONTRACT["resolution_v"], 3), dtype=points.dtype),
        weights=torch.ones((FIT_CONTRACT["resolution_u"], FIT_CONTRACT["resolution_v"]), dtype=points.dtype),
        degree_u=FIT_CONTRACT["degree_u"],
        degree_v=FIT_CONTRACT["degree_v"],
    )
    basis_u, basis_v = probe._basis_values(uv)
    design = torch.einsum("qi,qj->qij", basis_u, basis_v).reshape(points.shape[0], -1)
    return int(torch.linalg.matrix_rank(design).item()), int(design.shape[1])


def _component_accounting(components: tuple[Any, ...]) -> dict[str, Any]:
    sizes = np.asarray([int(component.sample_indices.numel()) for component in components], dtype=np.int64)
    regions = Counter(int(component.region_id) for component in components)
    return {
        "count": int(len(components)),
        "by_existing_region": {str(key): int(value) for key, value in sorted(regions.items())},
        "size": _distribution(sizes),
        "all_components_retained": True,
        "largest_component_selection": False,
        "fragment_filtering": False,
    }


def analyze_case(runtime: CaseRuntime) -> dict[str, Any]:
    import torch
    from scipy.spatial import cKDTree

    samples = runtime.samples
    count = int(samples.source_cell_keys.numel())
    region_ids = runtime.region_ids
    if region_ids is None:
        prior = runtime.selection.get("prior_region_id")
        if prior is not None:
            region_ids = np.full((count,), int(prior), dtype=np.int64)
    support = _all_accepted_support(samples, region_ids)
    components, component_ids = build_native_tsdf_support_components(samples, support)
    runtime.components = components
    runtime.component_ids = component_ids
    component_summary = _component_accounting(components)

    boundary = None
    representative = None
    rank = None
    required_rank = FIT_CONTRACT["resolution_u"] * FIT_CONTRACT["resolution_v"]
    if len(components) != 1:
        status = ABSTAIN_REPRESENTATIVE
        reason = "multiple_native_tsdf_components" if len(components) > 1 else "no_native_tsdf_component"
    else:
        boundary = derive_native_support_boundary(samples, components[0])
        runtime.boundary = boundary
        rank, required_rank = _design_rank(samples, components[0], boundary)
        representative = fit_boundary_first_region_representative(samples, components[0], boundary)
        runtime.representative = representative
        status = representative.status
        reason = representative.reason

    boundary_summary = {
        "derived": boundary is not None,
        "loop_count": 0 if boundary is None else int(len(boundary.loops)),
        "closed": False if boundary is None else bool(boundary.closed),
        "ordered_boundary_valid": False if boundary is None else bool(boundary.eligible),
        "reason": "not_derived_for_multi_component_case" if boundary is None else boundary.reason,
        "vertex_count": 0 if boundary is None else int(boundary.boundary_world.shape[0]),
    }

    fit_residual = _distribution(np.empty((0,)), samples.h)
    support_to_surface = _distribution(np.empty((0,)), samples.h)
    if representative is not None and representative.status == MATERIALIZED_REPRESENTATIVE:
        fitted = representative.surface.evaluate(representative.uv).detach().cpu().numpy()
        points = samples.world_xyz.detach().cpu().numpy()
        residual = np.linalg.norm(fitted - points, axis=1)
        grid_u = torch.linspace(0.0, 1.0, 64)
        grid_v = torch.linspace(0.0, 1.0, 32)
        uu, vv = torch.meshgrid(grid_u, grid_v, indexing="ij")
        dense_uv = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=1)
        dense = representative.surface.evaluate(dense_uv).detach().cpu().numpy()
        nearest, _ = cKDTree(dense).query(points, k=1, workers=1)
        runtime.fitted_xyz = fitted
        runtime.residual = residual
        runtime.dense_surface_xyz = dense
        fit_residual = _distribution(residual, samples.h)
        support_to_surface = _distribution(nearest, samples.h)

    keys_after = samples.source_cell_keys.detach().cpu().numpy()
    xyz_after = samples.world_xyz.detach().cpu().numpy()
    support_hash_after = _sha256_arrays(keys_after, xyz_after)
    support_hash_before = runtime.selection.get("selected_support_hash_before_analysis", support_hash_after)
    result = {
        "case": runtime.name,
        "kind": runtime.kind,
        "support": {
            "count": count,
            "h": float(samples.h),
            "source_cell_key_min": int(keys_after.min()) if count else None,
            "source_cell_key_max": int(keys_after.max()) if count else None,
            "hash_before_analysis": support_hash_before,
            "hash_after_analysis": support_hash_after,
            "preserved_exactly": support_hash_before == support_hash_after,
            "zero_set_xyz_is_fit_geometry": True,
            "gaussian_centers_used_as_geometry": False,
        },
        "selection": runtime.selection,
        "components": component_summary,
        "boundary": boundary_summary,
        "solvability": {
            "design_matrix_rank": rank,
            "required_rank": required_rank,
            "full_rank": rank == required_rank if rank is not None else False,
        },
        "fit_residual": fit_residual,
        "support_to_surface_dense_reference": support_to_surface,
        "result": status,
        "reason": reason,
        "fit_contract": FIT_CONTRACT,
        "no_rescue": {
            "component_deletion": False,
            "top_k": False,
            "trusted_subset": False,
            "parameter_sweep": False,
            "manual_cleanup": False,
            "smoothing": False,
            "hole_fill": False,
        },
    }
    runtime.result = result
    return result


def _prepare_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
    })
    return plt


def _equal_3d_axes(ax: Any, points: np.ndarray) -> None:
    if not len(points):
        return
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.55, 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _label_common_world(ax: Any) -> None:
    ax.set_xlabel("world X")
    ax.set_ylabel("world Y")
    ax.set_zlabel("world Z")


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
    fig.clf()


def _plot_support_common_world(runtime: CaseRuntime, path: Path) -> None:
    plt = _prepare_matplotlib()
    points = runtime.samples.world_xyz.detach().cpu().numpy()
    shown, _ = _stable_plot_subset(points)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    if len(runtime.context_xyz):
        ax.scatter(*runtime.context_xyz.T, s=1.0, c=CONTEXT_RGB, alpha=0.18, label="existing-region context")
    ax.scatter(*shown.T, s=1.2, c=SUPPORT_RGB, alpha=0.72, label="selected zero-set support")
    ax.set_title(f"{runtime.name} — support_common_world\nshown {len(shown):,} / complete {len(points):,}")
    _label_common_world(ax)
    _equal_3d_axes(ax, np.concatenate((shown, runtime.context_xyz), axis=0) if len(runtime.context_xyz) else shown)
    ax.legend(loc="upper right", fontsize=7)
    _save_figure(fig, path)
    plt.close(fig)


def _plot_boundary_and_domain(runtime: CaseRuntime, path: Path) -> None:
    plt = _prepare_matplotlib()
    points = runtime.samples.world_xyz.detach().cpu().numpy()
    shown, shown_indices = _stable_plot_subset(points)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    boundary = runtime.boundary
    if boundary is not None:
        import torch

        shown_tensor = torch.from_numpy(shown)
        projected = torch.stack(
            (
                (shown_tensor - boundary.chart_origin) @ boundary.tangent_u,
                (shown_tensor - boundary.chart_origin) @ boundary.tangent_v,
            ),
            dim=1,
        ).numpy() / runtime.samples.h
        ax.scatter(projected[:, 0], projected[:, 1], s=2.0, c=DOMAIN_RGB, alpha=0.55, label="native support chart")
        for loop_index, loop in enumerate(boundary.loops):
            loop_projected = np.column_stack((
                ((loop - boundary.chart_origin) @ boundary.tangent_u).numpy(),
                ((loop - boundary.chart_origin) @ boundary.tangent_v).numpy(),
            )) / runtime.samples.h
            closed = np.vstack((loop_projected, loop_projected[:1]))
            ax.plot(closed[:, 0], closed[:, 1], color=BOUNDARY_RGB, linewidth=1.6, label="ordered boundary" if loop_index == 0 else None)
        ax.set_xlabel("boundary chart u / h")
        ax.set_ylabel("boundary chart v / h")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=7)
        subtitle = f"loops={len(boundary.loops)}, closed={boundary.closed}, valid={boundary.eligible}"
    else:
        component_ids = runtime.component_ids.detach().cpu().numpy()[shown_indices]
        ax.scatter(shown[:, 0], shown[:, 1], s=2.0, c=component_ids, cmap="tab20", alpha=0.55)
        ax.set_xlabel("world X")
        ax.set_ylabel("world Y")
        ax.set_aspect("equal", adjustable="box")
        subtitle = f"no single chart: native components={len(runtime.components)}"
    ax.set_title(f"{runtime.name} — boundary_and_domain\n{subtitle}")
    ax.grid(alpha=0.2)
    _save_figure(fig, path)
    plt.close(fig)


def _plot_nurbs_fit_common_world(runtime: CaseRuntime, path: Path) -> None:
    plt = _prepare_matplotlib()
    points = runtime.samples.world_xyz.detach().cpu().numpy()
    shown, _ = _stable_plot_subset(points)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*shown.T, s=1.0, c=SUPPORT_RGB, alpha=0.38, label="zero-set support")
    all_points = shown
    if runtime.representative is not None and runtime.representative.status == MATERIALIZED_REPRESENTATIVE:
        dense = runtime.dense_surface_xyz
        control = runtime.representative.surface.control_grid.detach().cpu().numpy()
        ax.scatter(*dense.T, s=3.0, c=SURFACE_RGB, alpha=0.72, label="bounded NURBS")
        for row in control:
            ax.plot(row[:, 0], row[:, 1], row[:, 2], color=CONTROL_RGB, linewidth=0.8, alpha=0.8)
        for col in np.swapaxes(control, 0, 1):
            ax.plot(col[:, 0], col[:, 1], col[:, 2], color=CONTROL_RGB, linewidth=0.8, alpha=0.8)
        all_points = np.concatenate((shown, dense, control.reshape(-1, 3)), axis=0)
        subtitle = "MATERIALIZED — fixed WL139/W154 fit"
    else:
        subtitle = f"ABSTAIN — {runtime.result['reason']}"
        ax.text2D(0.03, 0.03, "No NURBS geometry was created.", transform=ax.transAxes, color="#b91c1c")
    ax.set_title(f"{runtime.name} — nurbs_fit_common_world\n{subtitle}")
    _label_common_world(ax)
    _equal_3d_axes(ax, all_points)
    ax.legend(loc="upper right", fontsize=7)
    _save_figure(fig, path)
    plt.close(fig)


def _plot_residual_review(runtime: CaseRuntime, path: Path) -> None:
    plt = _prepare_matplotlib()
    points = runtime.samples.world_xyz.detach().cpu().numpy()
    shown, shown_indices = _stable_plot_subset(points)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    if runtime.residual is not None:
        values = runtime.residual[shown_indices] / runtime.samples.h
        scatter = ax.scatter(*shown.T, s=2.0, c=values, cmap="viridis", alpha=0.8)
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
        colorbar.set_label("fit residual / h")
        subtitle = f"median={np.median(runtime.residual) / runtime.samples.h:.4g} h, p95={np.percentile(runtime.residual, 95) / runtime.samples.h:.4g} h"
    else:
        ax.scatter(*shown.T, s=1.2, c=CONTEXT_RGB, alpha=0.55)
        subtitle = f"ABSTAIN — residual undefined ({runtime.result['reason']})"
        ax.text2D(0.03, 0.03, "Residual is intentionally absent.", transform=ax.transAxes, color="#b91c1c")
    ax.set_title(f"{runtime.name} — residual_review\n{subtitle}")
    _label_common_world(ax)
    _equal_3d_axes(ax, shown)
    _save_figure(fig, path)
    plt.close(fig)


def _synthetic_image_overlay(runtime: CaseRuntime, path: Path) -> None:
    plt = _prepare_matplotlib()
    points = runtime.samples.world_xyz.detach().cpu().numpy()
    shown, _ = _stable_plot_subset(points)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_facecolor("#f3f4f6")
    ax.scatter(shown[:, 0], shown[:, 2], s=2.0, c=SUPPORT_RGB, alpha=0.55, label="zero-set support")
    if runtime.dense_surface_xyz is not None:
        dense = runtime.dense_surface_xyz
        ax.scatter(dense[:, 0], dense[:, 2], s=5.0, c=SURFACE_RGB, alpha=0.7, label="bounded NURBS")
    ax.set_xlabel("synthetic camera horizontal / world units")
    ax.set_ylabel("synthetic camera depth / world units")
    ax.set_title(f"{runtime.name} — synthetic reference projection\nnot a real RGB observation")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.2)
    _save_figure(fig, path)
    plt.close(fig)


def _real_camera_overlays(runtimes: dict[str, CaseRuntime], dataset: Path, out_dir: Path) -> dict[str, Any]:
    from PIL import Image

    plt = _prepare_matplotlib()
    cameras, metadata = w155._build_named_cameras(dataset, "images_8", "sparse/0", -1, 8, "cpu")
    colors = {
        "real_tabletop_coherent": "#00bcd4",
        "real_curved_vase_coherent": "#f59e0b",
        "real_mixed_contact_stress": "#ef4444",
    }
    accounting: dict[str, Any] = {}
    for camera_name in w155.REVIEW_CAMERAS:
        image_path = dataset / "images_8" / camera_name
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.imshow(rgb)
        camera_rows: dict[str, Any] = {}
        for case_name in REAL_CASE_CONTRACT:
            runtime = runtimes[case_name]
            points = runtime.samples.world_xyz.detach().cpu().numpy()
            shown, _ = _stable_plot_subset(points, limit=12_000)
            projection = w155._project_world_points(shown, cameras[camera_name])
            valid = projection["valid"]
            ax.scatter(projection["x"][valid], projection["y"][valid], s=2.0, c=colors[case_name], alpha=0.35, label=case_name)
            fit_visible = 0
            if runtime.dense_surface_xyz is not None:
                fit_projection = w155._project_world_points(runtime.dense_surface_xyz, cameras[camera_name])
                fit_valid = fit_projection["valid"]
                fit_visible = int(fit_valid.sum())
                ax.scatter(
                    fit_projection["x"][fit_valid], fit_projection["y"][fit_valid], s=8.0,
                    facecolors="none", edgecolors="#ffffff", linewidths=0.45,
                )
            camera_rows[case_name] = {"support_shown": int(len(shown)), "support_projected": int(valid.sum()), "fit_projected": fit_visible}
        ax.set_xlim(0, rgb.shape[1] - 1)
        ax.set_ylim(rgb.shape[0] - 1, 0)
        ax.set_axis_off()
        ax.set_title(f"{camera_name} — zero-set case support overlay (RGB is reference only)")
        ax.legend(loc="lower left", fontsize=6, framealpha=0.75)
        output_path = out_dir / f"{Path(camera_name).stem}.png"
        _save_figure(fig, output_path)
        plt.close(fig)
        accounting[camera_name] = {"image": str(image_path), "camera": metadata[camera_name], "cases": camera_rows}
    return accounting


def _readme_texts() -> dict[str, str]:
    shared = (
        "모든 point geometry는 frozen historical TSDF의 raw zero-set sample에서 온다. Existing Gaussian region은 real case의 "
        "local support organization에만 쓰였고 Gaussian center는 fit point가 아니다. Plot의 stable stride는 review 표시 수만 줄이며 "
        "topology, boundary, rank, fit, residual 계산에는 complete support를 사용한다.\n"
    )
    return {
        "support_common_world": (
            "# support_common_world\n\n각 PNG는 case별 selected raw zero-set support를 동일한 world XYZ 좌표계에서 보여 준다. Cyan point는 "
            "NURBS 입력인 selected support이고 gray point는 coherent case envelope 안에 있었지만 prior existing-region organization으로 "
            "선택되지 않은 context다. Gray context는 geometry 수정이나 fit 입력이 아니다. " + shared
        ),
        "boundary_and_domain": (
            "# boundary_and_domain\n\nSingle native-cell component이면 W154의 canonical local chart에 투영한 support와 ordered boundary loop를 표시한다. "
            "Light-blue point는 chart occupancy support, orange line은 ordered boundary다. Component가 여러 개이면 world X-Y projection과 "
            "component label만 표시하며 single domain을 만들지 않고 ABSTAIN한다. Color는 component 구분용이며 품질 점수가 아니다. " + shared
        ),
        "nurbs_fit_common_world": (
            "# nurbs_fit_common_world\n\nCyan은 complete selected zero-set support, blue는 materialized bounded NURBS sample, magenta는 고정 "
            "8x4 control net이다. ABSTAIN case는 NURBS geometry를 만들지 않았으므로 support만 표시한다. 동일 world frame의 3D view가 "
            "구조 판단의 주 evidence이며 RGB overlay보다 우선한다. " + shared
        ),
        "residual_review": (
            "# residual_review\n\nMaterialized case의 각 zero-set support point를 fitted UV에서의 Euclidean fit residual / h로 색칠한다. "
            "Viridis colorbar의 값이 클수록 fitted surface와 support 사이 거리가 크다. ABSTAIN case에는 residual 자체가 정의되지 않아 "
            "neutral gray support와 abstain reason만 표시한다. Dense nearest-surface 통계는 report JSON에 별도로 기록한다. " + shared
        ),
        "image_reference_overlay": (
            "# image_reference_overlay\n\nReal PNG는 fixed W145-W155 camera RGB 위에 세 real case의 zero-set support를 직접 투영한 secondary review다. "
            "Cyan=tabletop, orange=curved/vase, red=mixed/contact이며 white outline은 materialized NURBS sample이다. RGB는 geometry source가 "
            "아니며 visibility/occlusion 판정을 새로 만들지 않는다. Synthetic PNG는 real RGB가 없는 control의 fixed X-Z reference projection이다. "
            "Camera별 파일은 이 directory에 `<camera_name>.png`로 직접 저장하며 하위 camera directory는 없다. " + shared
        ),
    }


def _write_output_readmes(out: Path, runtimes: dict[str, CaseRuntime]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "# Worklog 171 diagnostic artifacts\n\n"
        "이 output은 frozen raw zero-set observed support가 bounded visible structural NURBS representative를 직접 materialize할 수 있는지 "
        "감사한다. Production behavior, W161 blocker pause, occluded continuation은 변경하지 않는다. `review_views`의 다섯 family와 "
        "`case_artifacts`의 case별 README를 함께 확인한다. 모든 review image는 PNG primary artifact다.\n",
        encoding="utf-8",
    )
    review_root = out / "review_views"
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "README.md").write_text(
        "# Review views\n\n이 directory는 W171의 support, boundary/domain, fitted NURBS, residual, image-reference overlay를 분리해 "
        "보여 준다. 각 하위 directory의 README가 palette, state semantics, shared condition과 limitation을 개별적으로 설명한다.\n",
        encoding="utf-8",
    )
    for family, text in _readme_texts().items():
        directory = review_root / family
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(text, encoding="utf-8")

    case_root = out / "case_artifacts"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "README.md").write_text(
        "# Case artifacts\n\n각 하위 case는 result JSON, fit이 materialize된 경우 control grid/weights/UV, 그리고 해당 case의 "
        "selection 및 visualization 의미를 설명하는 README를 가진다. Full W154 replay cache는 복사하지 않는다.\n",
        encoding="utf-8",
    )
    for name, runtime in runtimes.items():
        directory = case_root / name
        directory.mkdir(parents=True, exist_ok=True)
        status = runtime.result["result"]
        reason = runtime.result["reason"]
        (directory / "README.md").write_text(
            f"# {name}\n\n이 case의 결과는 `{status}`이며 reason은 `{reason}`이다. `support_common_world`는 raw zero-set "
            "support와 기존-region context, `boundary_and_domain`은 native component에서 유도 가능한 ordered domain, "
            "`nurbs_fit_common_world`는 같은 world frame의 bounded NURBS/control net, `residual_review`는 fit residual / h, "
            "`image_reference_overlay`는 fixed camera 또는 synthetic reference projection을 뜻한다. Geometry 계산은 complete support를 "
            "사용하며 display stride는 visualization에만 적용된다. Gaussian center, component deletion, top-k, trusted subset, smoothing, "
            "hole filling은 사용하지 않았다.\n",
            encoding="utf-8",
        )


def generate_review_exports(out: Path, runtimes: dict[str, CaseRuntime], dataset: Path) -> dict[str, Any]:
    review_root = out / "review_views"
    for name in CASE_ORDER:
        runtime = runtimes[name]
        _plot_support_common_world(runtime, review_root / "support_common_world" / f"{name}.png")
        _plot_boundary_and_domain(runtime, review_root / "boundary_and_domain" / f"{name}.png")
        _plot_nurbs_fit_common_world(runtime, review_root / "nurbs_fit_common_world" / f"{name}.png")
        _plot_residual_review(runtime, review_root / "residual_review" / f"{name}.png")
    overlay_dir = review_root / "image_reference_overlay"
    for name in SYNTHETIC_CASES:
        _synthetic_image_overlay(runtimes[name], overlay_dir / f"{name}.png")
    camera_accounting = _real_camera_overlays(runtimes, dataset, overlay_dir)
    return camera_accounting


def _write_case_artifacts(out: Path, runtimes: dict[str, CaseRuntime]) -> None:
    for name, runtime in runtimes.items():
        directory = out / "case_artifacts" / name
        _write_json(directory / "result.json", runtime.result)
        if runtime.representative is not None and runtime.representative.status == MATERIALIZED_REPRESENTATIVE:
            np.savez_compressed(
                directory / "bounded_nurbs.npz",
                control_grid=runtime.representative.surface.control_grid.detach().cpu().numpy(),
                weights=runtime.representative.surface.weights.detach().cpu().numpy(),
                uv=runtime.representative.uv.detach().cpu().numpy(),
                source_cell_keys=runtime.samples.source_cell_keys.detach().cpu().numpy(),
            )


def validate_visual_artifacts(out: Path) -> dict[str, Any]:
    missing: list[str] = []
    review_root = out / "review_views"
    for family in REVIEW_FAMILIES:
        if not (review_root / family / "README.md").is_file():
            missing.append(str(review_root / family / "README.md"))
    for family in REVIEW_FAMILIES[:4]:
        for case in CASE_ORDER:
            if not (review_root / family / f"{case}.png").is_file():
                missing.append(str(review_root / family / f"{case}.png"))
    for case in SYNTHETIC_CASES:
        if not (review_root / "image_reference_overlay" / f"{case}.png").is_file():
            missing.append(str(review_root / "image_reference_overlay" / f"{case}.png"))
    for camera in w155.REVIEW_CAMERAS:
        if not (review_root / "image_reference_overlay" / f"{Path(camera).stem}.png").is_file():
            missing.append(str(review_root / "image_reference_overlay" / f"{Path(camera).stem}.png"))
    ppm = [str(path) for path in out.rglob("*.ppm")]
    return {
        "complete": not missing and not ppm,
        "missing": missing,
        "ppm_files": ppm,
        "png_count": len(list(out.rglob("*.png"))),
        "readme_count": len(list(out.rglob("README.md"))),
        "camera_directory_nesting": False,
    }


def _architecture_result(case_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required_materialized = (
        "synthetic_planar_single_sheet",
        "synthetic_curved_single_sheet",
        "real_tabletop_coherent",
        "real_curved_vase_coherent",
    )
    required_abstain = (
        "synthetic_layered_non_single_chart",
        "real_mixed_contact_stress",
    )
    materialized_checks = {name: case_results[name]["result"] == MATERIALIZED_REPRESENTATIVE for name in required_materialized}
    abstain_checks = {name: case_results[name]["result"] == ABSTAIN_REPRESENTATIVE for name in required_abstain}
    yes = all(materialized_checks.values()) and all(abstain_checks.values())
    if yes:
        verdict = "YES_COHERENT_ZERO_SET_SUPPORT_YIELDS_BOUNDED_VISIBLE_NURBS_WITH_MIXED_ABSTENTION"
        interpretation = (
            "The tested coherent synthetic and real zero-set support cases materialize under the frozen boundary-first fit, "
            "while mixed/non-single-chart controls abstain without support deletion."
        )
    else:
        verdict = "NO_CURRENT_ZERO_SET_SUPPORT_NOT_RELIABLY_STRUCTURALLY_FIT_READY"
        interpretation = (
            "At least one required coherent case does not satisfy complete-support component, boundary, or solvability contracts. "
            "No fragment selection or fit rescue was applied."
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "required_materialized": materialized_checks,
        "required_abstain": abstain_checks,
        "automatic_production_promotion": False,
        "occluded_continuation": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    runtimes: dict[str, CaseRuntime] = {name: _synthetic_runtime(name) for name in SYNTHETIC_CASES}
    if args.synthetic_only:
        selected_names = SYNTHETIC_CASES
    else:
        runtimes.update(_load_real_runtimes(args.wl154, args.wl145))
        selected_names = CASE_ORDER

    case_results: dict[str, dict[str, Any]] = {}
    for name in selected_names:
        print(f"[W171] analyze {name}: {int(runtimes[name].samples.source_cell_keys.numel()):,} support", flush=True)
        case_results[name] = analyze_case(runtimes[name])
        print(f"[W171] {name}: {case_results[name]['result']} ({case_results[name]['reason']})", flush=True)

    if args.synthetic_only:
        return {
            "status": "SYNTHETIC_ONLY",
            "cases": case_results,
            "runtimes": runtimes,
        }

    _write_output_readmes(args.out, runtimes)
    _write_case_artifacts(args.out, runtimes)
    camera_accounting = generate_review_exports(args.out, runtimes, args.dataset)
    visual_validation = validate_visual_artifacts(args.out)
    if not visual_validation["complete"]:
        raise RuntimeError(f"incomplete visual artifact contract: {visual_validation}")

    architecture = _architecture_result(case_results)
    report = {
        "status": "COMPLETE_ZERO_SET_DERIVED_VISIBLE_STRUCTURAL_NURBS_AUDIT",
        "batch": "Worklog 171 — Zero-Set-Derived Visible Structural NURBS Audit with Review Visualizations",
        "intent_alignment": {
            "diagnostic_only": True,
            "zero_set_support_authoritative_geometry": True,
            "gaussian_region_mapping_organizational_only": True,
            "gaussian_centers_used_as_geometry": False,
            "occluded_continuation": False,
            "production_behavior_changed": False,
            "w161_pause_preserved": True,
        },
        "frozen_inputs": {
            "w145": str(args.wl145.resolve()),
            "w154": str(args.wl154.resolve()),
            "real_zero_set_samples": str((args.wl154 / "candidate_f_tsdf_surface_samples.npz").resolve()),
            "real_region_ownership": str((args.wl154 / "candidate_f_region_owned_support.npz").resolve()),
            "synthetic_construction": "W167/W168 historical SparseProjectiveTSDF carrier and all-eight-corner zero-set sample extraction",
            "w167_geometry_preserved": True,
            "historical_tsdf_and_extraction_preserved": True,
        },
        "case_selection": {
            "rule": "W154 prior frozen W145 case; lexicographically first prior camera; exact event-point AABB; prior dominant existing region for coherent cases; all accepted regions for stress case",
            "resolved_before_w171_fit_or_visualization": True,
            "visual_success_selection": False,
            "fit_residual_selection": False,
            "manual_cleanup": False,
        },
        "fit_contract": FIT_CONTRACT,
        "cases": case_results,
        "camera_overlay_accounting": camera_accounting,
        "visualization": {
            "families": list(REVIEW_FAMILIES),
            "validation": visual_validation,
            "common_world_primary": True,
            "rgb_overlay_secondary": True,
            "output_format": "PNG",
            "display_downsampling_affects_computation": False,
            "readme_at_each_visualization_directory": True,
        },
        "architecture_result": architecture,
        "retained": [
            "W167 raw zero-set geometry semantics",
            "historical TSDF construction/extraction",
            "all selected native support components",
            "W154 boundary-first WL139 fit family",
            "W161 pause and BEHIND_ZEROSET representation fact",
        ],
        "rejected": [
            "Gaussian centers as fitted geometry",
            "component filtering or largest-component rescue",
            "top-k/trusted subset/manual cleanup",
            "smoothing/hole fill/occluded continuation",
            "automatic production promotion",
        ],
        "open": (
            [] if architecture["verdict"].startswith("YES_") else
            ["The failed coherent cases require a future architecture decision; W171 does not repair or tune them."]
        ),
        "outputs": {
            "report": str((args.out / "worklog_171_report.json").resolve()),
            "review_views": str((args.out / "review_views").resolve()),
            "case_artifacts": str((args.out / "case_artifacts").resolve()),
            "w153_replay_cache_copied": False,
        },
        "runtime_seconds": {"total": time.time() - started},
    }
    _write_json(args.out / "worklog_171_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl145", type=Path, default=DEFAULT_W145)
    parser.add_argument("--wl154", type=Path, default=DEFAULT_W154)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    if report["status"] == "SYNTHETIC_ONLY":
        summary = {name: row["result"] for name, row in report["cases"].items()}
    else:
        summary = {
            "status": report["status"],
            "verdict": report["architecture_result"]["verdict"],
            "cases": {name: row["result"] for name, row in report["cases"].items()},
            "png_count": report["visualization"]["validation"]["png_count"],
        }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
