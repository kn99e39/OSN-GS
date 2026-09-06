from __future__ import annotations

"""Worklog 168: strict raw-zero-set first-hit positive-occlusion audit.

This diagnostic replays W167's frozen synthetic construction and first-hit
primitive, adds one deterministic two-sheet fixture, and measures the complete
query-depth disagreement intervals induced by ``z_s`` versus analytic ``z*``.
It changes no mesh, component population, Candidate-B state, or production
visibility behavior.
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167  # noqa: E402


DEFAULT_W167_REPORT = REPO_ROOT / "output/167_raw_zero_set_ray_blocker_audit/worklog_167_report.json"
DEFAULT_W167_REAL_REPORT = REPO_ROOT / "output/167_raw_zero_set_ray_blocker_audit/real_scene/real_scene_replay_report.json"
DEFAULT_OUT = REPO_ROOT / "output/168_raw_zero_set_first_hit_positive_per_view_occlusion_evidence_audit"

FIXTURE_NAMES = ("fronto_parallel_plane", "oblique_plane", "curved_sphere", "layered_two_sheet")
SINGLE_SURFACE_NAMES = FIXTURE_NAMES[:3]
LAYER_FRONT_Z = -0.55
LAYER_REAR_Z = 0.65
LAYER_REAR_QUERY_OFFSET = 0.50

VERDICT_SUPPORTED = "RAW_ZEROSET_FIRST_HIT_CANONICAL_POSITIVE_PER_VIEW_EVIDENCE_SUPPORTED"
VERDICT_REJECTED = "RAW_ZEROSET_FIRST_HIT_NOT_PROMOTED_STRICT_PREMATURE_BLOCKER_COUNTEREXAMPLE"


def _write_json(path: Path, value: Any) -> None:
    w167._write_json(path, value)


def zero_set_blocked(first_hit_depth: np.ndarray | float, query_depth: np.ndarray | float, valid_hit: np.ndarray | bool = True) -> np.ndarray:
    """Return the proposed strict positive evidence predicate, without epsilon."""

    z_s = np.asarray(first_hit_depth, dtype=np.float64)
    z_q = np.asarray(query_depth, dtype=np.float64)
    valid = np.asarray(valid_hit, dtype=bool)
    z_s, z_q, valid = np.broadcast_arrays(z_s, z_q, valid)
    return valid & np.isfinite(z_s) & np.isfinite(z_q) & (z_s > 0.0) & (z_s < z_q)


def analytic_blocked(first_blocker_depth: np.ndarray | float, query_depth: np.ndarray | float, valid_hit: np.ndarray | bool = True) -> np.ndarray:
    """Return strict analytic ground truth for the first physical blocker."""

    z_star = np.asarray(first_blocker_depth, dtype=np.float64)
    z_q = np.asarray(query_depth, dtype=np.float64)
    valid = np.asarray(valid_hit, dtype=bool)
    z_star, z_q, valid = np.broadcast_arrays(z_star, z_q, valid)
    return valid & np.isfinite(z_star) & np.isfinite(z_q) & (z_star > 0.0) & (z_star < z_q)


def _layered_surfaces() -> tuple[w167.AnalyticSurface, w167.AnalyticSurface]:
    tangent_u = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    tangent_v = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    normal = np.cross(tangent_u, tangent_v)
    front = w167.AnalyticSurface(
        "layered_front_sheet",
        "plane_rectangle",
        np.asarray((0.0, 0.0, LAYER_FRONT_Z), dtype=np.float64),
        tangent_u,
        tangent_v,
        normal,
        w167.SYNTHETIC_PLANE_HALF,
        w167.SYNTHETIC_PLANE_HALF,
    )
    rear = w167.AnalyticSurface(
        "layered_rear_sheet",
        "plane_rectangle",
        np.asarray((0.0, 0.0, LAYER_REAR_Z), dtype=np.float64),
        tangent_u,
        tangent_v,
        normal,
        w167.SYNTHETIC_PLANE_HALF,
        w167.SYNTHETIC_PLANE_HALF,
    )
    return front, rear


def _layered_fixture() -> tuple[Any, dict[str, Any]]:
    """Construct two zero sheets with W167's historical-style TSDF carrier."""

    import torch
    from evidence_bounded_tsdf.field import SparseProjectiveTSDF

    front, rear = _layered_surfaces()
    values = np.arange(-40, 41, dtype=np.int64)
    gx, gy, gz = np.meshgrid(values, values, values, indexing="ij")
    indices = np.column_stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)))
    points = (indices.astype(np.float64) + 0.5) * w167.SYNTHETIC_H
    support = front.support_mask(points) & rear.support_mask(points)
    indices = indices[support]
    points = points[support]
    # Exact signed distance to the boundary of the slab between the two sheets.
    # Sign orientation is immaterial to zero extraction; both sheets remain.
    signed = np.minimum(points[:, 2] - LAYER_FRONT_Z, LAYER_REAR_Z - points[:, 2])
    phi = np.clip(signed / w167.SYNTHETIC_MU, -1.0, 1.0).astype(np.float32)
    order = np.argsort(w167._encode_keys(indices), kind="stable")
    keys = w167._encode_keys(indices[order])
    field = SparseProjectiveTSDF(
        keys=torch.as_tensor(keys, dtype=torch.int64),
        value=torch.as_tensor(phi[order], dtype=torch.float32),
        support_count=torch.ones((len(keys),), dtype=torch.int32),
        h=w167.SYNTHETIC_H,
        mu=w167.SYNTHETIC_MU,
    )
    extracted = w167._historical_extract(field)
    return extracted, {
        "h": w167.SYNTHETIC_H,
        "mu": w167.SYNTHETIC_MU,
        "grid_index_min": int(values.min()),
        "grid_index_max": int(values.max()),
        "authoritative_voxels": int(len(keys)),
        "unknown_outside_plane_support": True,
        "front_sheet_world_z": LAYER_FRONT_Z,
        "rear_sheet_world_z": LAYER_REAR_Z,
        "construction": "SparseProjectiveTSDF with clipped analytic signed distance to a two-sheet slab; W167 frozen all-eight-corner extraction",
        "extraction_stats": extracted.stats,
    }


def _analytic_fixture(name: str, rays: w167.RayBundle) -> dict[str, Any]:
    if name == "fronto_parallel_plane":
        surfaces = (w167._plane_surface(name),)
    elif name == "oblique_plane":
        surfaces = (w167._plane_surface(name, True),)
    elif name == "curved_sphere":
        surfaces = (w167._sphere_surface(),)
    elif name == "layered_two_sheet":
        surfaces = _layered_surfaces()
    else:
        raise ValueError(name)

    depths = []
    hits = []
    points = []
    for surface in surfaces:
        depth, hit, point = surface.intersect(rays)
        depths.append(depth)
        hits.append(hit)
        points.append(point)
    stacked = np.stack(depths, axis=0)
    valid = np.stack(hits, axis=0)
    safe = np.where(valid, stacked, np.inf)
    first_index = np.argmin(safe, axis=0)
    first_depth = np.min(safe, axis=0)
    first_hit = np.isfinite(first_depth)
    first_depth = np.where(first_hit, first_depth, np.nan)
    return {
        "surfaces": surfaces,
        "surface_depths": stacked,
        "surface_hits": valid,
        "surface_points": np.stack(points, axis=0),
        "first_surface_index": np.where(first_hit, first_index, -1),
        "first_depth": first_depth,
        "first_hit": first_hit,
    }


def _discrete_support_boundary(analytic_hit: np.ndarray, image: int = w167.SYNTHETIC_IMAGE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split fixed raster hits using exact 8-neighbor support adjacency."""

    hit = np.asarray(analytic_hit, dtype=bool).reshape(image, image)
    padded = np.pad(hit, 1, mode="constant", constant_values=False)
    all_neighbors_hit = np.ones_like(hit)
    any_neighbor_hit = np.zeros_like(hit)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = padded[1 + dy:1 + dy + image, 1 + dx:1 + dx + image]
            all_neighbors_hit &= shifted
            any_neighbor_hit |= shifted
    interior = hit & all_neighbors_hit
    boundary = hit & ~interior
    exterior_adjacent = ~hit & any_neighbor_hit
    return interior.reshape(-1), boundary.reshape(-1), exterior_adjacent.reshape(-1)


def _interval_masks(
    analytic_depth: np.ndarray,
    analytic_hit: np.ndarray,
    zero_depth: np.ndarray,
    zero_status: np.ndarray,
) -> dict[str, np.ndarray]:
    z_star = np.asarray(analytic_depth, dtype=np.float64).reshape(-1)
    z_s = np.asarray(zero_depth, dtype=np.float64).reshape(-1)
    analytic = np.asarray(analytic_hit, dtype=bool).reshape(-1) & np.isfinite(z_star) & (z_star > 0.0)
    zero = (np.asarray(zero_status).reshape(-1) == w167.STATUS_HIT) & np.isfinite(z_s) & (z_s > 0.0)
    both = analytic & zero
    return {
        "analytic_hit": analytic,
        "zero_hit": zero,
        "both_hit": both,
        "false_zero_hit": ~analytic & zero,
        "missed_zero_hit": analytic & ~zero,
        "premature": both & (z_s < z_star),
        "delayed": both & (z_s > z_star),
        "exact": both & (z_s == z_star),
        "ambiguous": np.asarray(zero_status).reshape(-1) == w167.STATUS_AMBIGUOUS,
    }


def disagreement_interval_accounting(
    analytic_depth: np.ndarray,
    analytic_hit: np.ndarray,
    zero_depth: np.ndarray,
    zero_status: np.ndarray,
    h: float,
    interior: np.ndarray | None = None,
    boundary: np.ndarray | None = None,
    exterior_adjacent: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure the complete strict-query disagreement implied by z_s vs z*."""

    z_star = np.asarray(analytic_depth, dtype=np.float64).reshape(-1)
    z_s = np.asarray(zero_depth, dtype=np.float64).reshape(-1)
    masks = _interval_masks(z_star, analytic_hit, z_s, zero_status)
    n = len(z_star)
    interior = masks["analytic_hit"] if interior is None else np.asarray(interior, dtype=bool).reshape(-1)
    boundary = np.zeros((n,), dtype=bool) if boundary is None else np.asarray(boundary, dtype=bool).reshape(-1)
    exterior_adjacent = np.zeros((n,), dtype=bool) if exterior_adjacent is None else np.asarray(exterior_adjacent, dtype=bool).reshape(-1)
    if any(len(value) != n for value in (interior, boundary, exterior_adjacent)):
        raise ValueError("boundary masks must align with depth arrays")

    premature_length = z_star - z_s
    delayed_length = z_s - z_star

    def domain_report(domain: np.ndarray) -> dict[str, Any]:
        premature = masks["premature"] & domain
        delayed = masks["delayed"] & domain
        return {
            "analytic_hit_rays": int(np.count_nonzero(masks["analytic_hit"] & domain)),
            "both_hit_rays": int(np.count_nonzero(masks["both_hit"] & domain)),
            "premature_ray_count": int(np.count_nonzero(premature)),
            "premature_interval_length": w167._distribution(premature_length[premature]),
            "premature_interval_length_over_h": w167._distribution(premature_length[premature] / h),
            "delayed_ray_count": int(np.count_nonzero(delayed)),
            "delayed_interval_length": w167._distribution(delayed_length[delayed]),
            "delayed_interval_length_over_h": w167._distribution(delayed_length[delayed] / h),
            "missed_zero_hit_rays": int(np.count_nonzero(masks["missed_zero_hit"] & domain)),
        }

    false_hits = masks["false_zero_hit"]
    strict_counterexample_count = int(np.count_nonzero(masks["premature"]) + np.count_nonzero(false_hits))
    return {
        "total_rays": n,
        "analytic_hit_rays": int(np.count_nonzero(masks["analytic_hit"])),
        "analytic_no_hit_rays": int(np.count_nonzero(~masks["analytic_hit"])),
        "zero_set_hit_rays": int(np.count_nonzero(masks["zero_hit"])),
        "zero_set_no_hit_or_ambiguous_rays": int(np.count_nonzero(~masks["zero_hit"])),
        "both_hit_rays": int(np.count_nonzero(masks["both_hit"])),
        "false_zero_set_hit_rays": int(np.count_nonzero(false_hits)),
        "false_zero_set_hit_exterior_adjacent_to_support": int(np.count_nonzero(false_hits & exterior_adjacent)),
        "false_zero_set_hit_exterior_nonadjacent": int(np.count_nonzero(false_hits & ~exterior_adjacent)),
        "missed_zero_set_hit_rays": int(np.count_nonzero(masks["missed_zero_hit"])),
        "ambiguous_zero_set_rays": int(np.count_nonzero(masks["ambiguous"])),
        "exact_first_depth_rays": int(np.count_nonzero(masks["exact"])),
        "premature_blocker_ray_count": int(np.count_nonzero(masks["premature"])),
        "premature_blocker_interval_length": w167._distribution(premature_length[masks["premature"]]),
        "premature_blocker_interval_length_over_h": w167._distribution(premature_length[masks["premature"]] / h),
        "delayed_blocker_ray_count": int(np.count_nonzero(masks["delayed"])),
        "delayed_blocker_interval_length": w167._distribution(delayed_length[masks["delayed"]]),
        "delayed_blocker_interval_length_over_h": w167._distribution(delayed_length[masks["delayed"]] / h),
        "strict_positive_evidence_counterexample_count": strict_counterexample_count,
        "strict_positive_evidence_sound_on_fixture": strict_counterexample_count == 0,
        "complete_query_depth_consequence": {
            "premature": "for every z_q in (z_s, z*], ZEROSET_BLOCKED=true and GT_BLOCKED=false",
            "delayed": "for every z_q in (z*, z_s], GT_BLOCKED=true and ZEROSET_BLOCKED=false",
            "false_zero_set_hit": "for every finite z_q>z_s, ZEROSET_BLOCKED=true while GT_BLOCKED=false; disagreement is unbounded above",
            "missed_zero_set_hit": "for every finite z_q>z*, GT_BLOCKED=true while ZEROSET_BLOCKED has no positive evidence; disagreement is unbounded above",
            "exact": "when z_s==z*, strict predicates agree for every z_q and exact surface coincidence is not blocked",
        },
        "interior": domain_report(interior),
        "silhouette_or_support_boundary": domain_report(boundary),
        "boundary_definition": "exact fixed-raster 8-neighbor analytic-hit adjacency; no depth epsilon or semantic threshold",
    }


def _build_fixture(name: str) -> dict[str, Any]:
    rays = w167._synthetic_rays()
    if name == "layered_two_sheet":
        extracted, construction = _layered_fixture()
    else:
        _surface, extracted, frozen_meta = w167._synthetic_fixture(name)
        construction = frozen_meta
    vertices = np.asarray(extracted.vertices, dtype=np.float64)
    faces = np.asarray(extracted.faces, dtype=np.int64)
    _vertex_components, face_components, component_sizes = w167._component_labels(faces, len(vertices))
    zero_hit = w167.intersect_first_hit_bruteforce(
        rays,
        vertices,
        faces,
        face_components,
        triangle_chunk=20_000,
        ray_chunk=16,
    )
    analytic = _analytic_fixture(name, rays)
    interior, boundary, exterior_adjacent = _discrete_support_boundary(analytic["first_hit"])
    intervals = disagreement_interval_accounting(
        analytic["first_depth"],
        analytic["first_hit"],
        zero_hit.depth,
        zero_hit.status,
        w167.SYNTHETIC_H,
        interior,
        boundary,
        exterior_adjacent,
    )
    return {
        "name": name,
        "rays": rays,
        "extracted": extracted,
        "vertices": vertices,
        "faces": faces,
        "component_sizes": component_sizes,
        "zero_hit": zero_hit,
        "analytic": analytic,
        "interior": interior,
        "boundary": boundary,
        "exterior_adjacent": exterior_adjacent,
        "intervals": intervals,
        "construction": construction,
    }


def _multi_surface_result(fixture: dict[str, Any]) -> dict[str, Any]:
    analytic = fixture["analytic"]
    zero_hit = fixture["zero_hit"]
    front_depth = analytic["surface_depths"][0]
    rear_depth = analytic["surface_depths"][1]
    both_surfaces = analytic["surface_hits"][0] & analytic["surface_hits"][1]
    valid_zero = zero_hit.status == w167.STATUS_HIT
    auditable = both_surfaces & valid_zero
    first_world_z = zero_hit.world_xyz[:, 2]
    zero_first_nearer_front = np.abs(first_world_z - LAYER_FRONT_Z) < np.abs(first_world_z - LAYER_REAR_Z)
    rear_surface_query = rear_depth
    deeper_query = rear_depth + LAYER_REAR_QUERY_OFFSET
    return {
        "front_sheet_world_z": LAYER_FRONT_Z,
        "rear_sheet_world_z": LAYER_REAR_Z,
        "analytic_both_surface_hit_rays": int(np.count_nonzero(both_surfaces)),
        "auditable_zero_set_first_hit_rays": int(np.count_nonzero(auditable)),
        "zero_set_first_hit_nearer_front_sheet": int(np.count_nonzero(auditable & zero_first_nearer_front)),
        "zero_set_first_hit_nearer_rear_sheet": int(np.count_nonzero(auditable & ~zero_first_nearer_front)),
        "rear_surface_associated_query": {
            "query_depth_definition": "exact analytic rear-sheet depth",
            "gt_blocked_by_front_surface_count": int(np.count_nonzero(analytic_blocked(front_depth, rear_surface_query, both_surfaces))),
            "candidate_blocked_by_first_zero_set_count": int(np.count_nonzero(zero_set_blocked(zero_hit.depth, rear_surface_query, valid_zero) & both_surfaces)),
            "exact_rear_coincidence_is_not_used_as_the_blocker": True,
        },
        "deeper_rear_associated_query": {
            "query_depth_definition": f"analytic rear-sheet depth + {LAYER_REAR_QUERY_OFFSET}",
            "gt_blocked_by_front_surface_count": int(np.count_nonzero(analytic_blocked(front_depth, deeper_query, both_surfaces))),
            "candidate_blocked_by_first_zero_set_count": int(np.count_nonzero(zero_set_blocked(zero_hit.depth, deeper_query, valid_zero) & both_surfaces)),
        },
        "surface_selection_heuristic": False,
        "target_identity_used": False,
        "interpretation": "the first geometric sheet blocks a query associated with the rear sheet; component and target identity are irrelevant to the predicate",
    }


def _save_signed_error_map(directory: Path, fixture: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = fixture["name"]
    analytic = fixture["analytic"]
    zero_hit = fixture["zero_hit"]
    masks = _interval_masks(analytic["first_depth"], analytic["first_hit"], zero_hit.depth, zero_hit.status)
    value = np.full((w167.SYNTHETIC_IMAGE * w167.SYNTHETIC_IMAGE,), np.nan, dtype=np.float64)
    value[masks["both_hit"]] = (zero_hit.depth[masks["both_hit"]] - analytic["first_depth"][masks["both_hit"]]) / w167.SYNTHETIC_H
    image = value.reshape(w167.SYNTHETIC_IMAGE, w167.SYNTHETIC_IMAGE)
    finite = np.abs(value[np.isfinite(value)])
    limit = max(float(finite.max()) if finite.size else 0.0, np.finfo(np.float64).eps)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=130)
    display = ax.imshow(image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    miss = masks["missed_zero_hit"].reshape(w167.SYNTHETIC_IMAGE, w167.SYNTHETIC_IMAGE)
    false = masks["false_zero_hit"].reshape(w167.SYNTHETIC_IMAGE, w167.SYNTHETIC_IMAGE)
    if np.any(miss):
        y, x = np.nonzero(miss)
        ax.scatter(x, y, s=8, c="#f4c542", marker="s", label="analytic hit / zero-set miss")
    if np.any(false):
        y, x = np.nonzero(false)
        ax.scatter(x, y, s=8, c="#c53cff", marker="x", label="false zero-set hit")
    ax.set_title(f"{name}: signed first-hit error (z_s - z*) / h")
    ax.set_xlabel("pixel column")
    ax.set_ylabel("pixel row")
    fig.colorbar(display, ax=ax, label="signed depth error / h; red=delayed, blue=premature")
    if np.any(miss) or np.any(false):
        ax.legend(loc="best")
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", format="png")
    plt.close(fig)


def _save_interval_distribution(directory: Path, fixture: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = fixture["name"]
    analytic = fixture["analytic"]
    zero_hit = fixture["zero_hit"]
    masks = _interval_masks(analytic["first_depth"], analytic["first_hit"], zero_hit.depth, zero_hit.status)
    premature = (analytic["first_depth"] - zero_hit.depth)[masks["premature"]] / w167.SYNTHETIC_H
    delayed = (zero_hit.depth - analytic["first_depth"])[masks["delayed"]] / w167.SYNTHETIC_H
    positive = np.concatenate((premature[premature > 0.0], delayed[delayed > 0.0]))
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=130)
    if positive.size:
        lower = float(positive.min())
        upper = float(positive.max())
        if lower == upper:
            bins = np.asarray([lower * 0.5, upper * 1.5])
        else:
            bins = np.geomspace(lower, upper, 36)
        if premature.size:
            ax.hist(premature, bins=bins, alpha=0.68, color="#2b6cb0", label="premature: z* - z_s")
        if delayed.size:
            ax.hist(delayed, bins=bins, alpha=0.68, color="#d64f3c", label="delayed: z_s - z*")
        ax.set_xscale("log")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No non-zero disagreement interval", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{name}: complete query-depth disagreement intervals")
    ax.set_xlabel("interval length / h (log scale)")
    ax.set_ylabel("ray count")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", format="png")
    plt.close(fig)


def _write_fixture_readmes(root: Path, name: str) -> None:
    common = (
        f"공통 조건: fixed 64×64 synthetic camera, h={w167.SYNTHETIC_H}, mu={w167.SYNTHETIC_MU}, "
        "historical-style SparseProjectiveTSDF, frozen all-eight-corner zero-set extraction, W167 two-sided exact first-hit primitive. "
        "zero-set, component, iso-value, h/mu를 수정하거나 filtering하지 않았다."
    )
    limitation = (
        "Review limitation: analytic geometry는 synthetic fixture에서만 ground truth다. PNG는 정량 NPZ/JSON의 review projection이며 "
        "real-scene physical hidden-surface truth나 global Observed/Occluded state를 의미하지 않는다."
    )
    (root / "README.md").write_text(
        f"# W168 `{name}`\n\n이 directory는 `{name}`의 analytic first blocker `z*`와 raw zero-set first hit `z_s`를 strict ordering으로 비교한다. "
        "`fixture_audit.npz`가 ray별 authoritative numeric artifact이고, 하위 PNG는 동일 값을 표시한다.\n\n"
        f"{common}\n\nPalette/legend는 각 visualization README에 별도로 기록한다. {limitation}\n",
        encoding="utf-8",
    )
    signed = root / "signed_first_hit_error"
    signed.mkdir(parents=True, exist_ok=True)
    (signed / "README.md").write_text(
        f"# Signed First-Hit Error — `{name}`\n\n각 pixel에서 `(z_s-z*)/h`를 표시한다. blue는 premature first hit(`z_s<z*`), red는 delayed first hit(`z_s>z*`)이며, yellow square는 analytic hit를 zero-set이 놓친 ray, magenta x는 analytic surface가 없는데 zero-set hit가 있는 ray다. gray/blank는 양쪽 first hit를 함께 비교할 수 없는 위치다.\n\n{common}\n\n{limitation}\n",
        encoding="utf-8",
    )
    dist = root / "disagreement_interval_distribution"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "README.md").write_text(
        f"# Disagreement Interval Distribution — `{name}`\n\ncomplete query-depth consequence의 interval length를 `h`로 정규화해 log x-axis histogram으로 표시한다. blue는 false-positive territory `(z_s,z*]`의 길이 `z*-z_s`, red는 false-negative territory `(z*,z_s]`의 길이 `z_s-z*`다. bar가 없으면 non-zero finite interval이 없다는 뜻이다.\n\n{common}\n\n이 histogram은 semantic threshold를 만들거나 작은 interval을 제거하지 않는다. {limitation}\n",
        encoding="utf-8",
    )


def _export_fixture(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    name = fixture["name"]
    fixture_root = root / name
    fixture_root.mkdir(parents=True, exist_ok=True)
    _write_fixture_readmes(fixture_root, name)
    analytic = fixture["analytic"]
    zero_hit = fixture["zero_hit"]
    masks = _interval_masks(analytic["first_depth"], analytic["first_hit"], zero_hit.depth, zero_hit.status)
    np.savez_compressed(
        fixture_root / "fixture_audit.npz",
        analytic_first_depth=analytic["first_depth"],
        analytic_hit=analytic["first_hit"],
        analytic_first_surface_index=analytic["first_surface_index"],
        analytic_surface_depths=analytic["surface_depths"],
        analytic_surface_hits=analytic["surface_hits"],
        zero_set_first_hit_depth=zero_hit.depth,
        zero_set_status=zero_hit.status.astype(str),
        zero_set_world_xyz=zero_hit.world_xyz,
        triangle_id=zero_hit.triangle_id,
        component_id=zero_hit.component_id,
        interior=fixture["interior"],
        silhouette_or_support_boundary=fixture["boundary"],
        false_zero_hit=masks["false_zero_hit"],
        missed_zero_hit=masks["missed_zero_hit"],
        premature=masks["premature"],
        delayed=masks["delayed"],
        exact=masks["exact"],
    )
    _save_signed_error_map(fixture_root / "signed_first_hit_error", fixture)
    _save_interval_distribution(fixture_root / "disagreement_interval_distribution", fixture)
    fixture_report = {
        "fixture": name,
        "construction": fixture["construction"],
        "zero_set_geometry": {
            "vertex_count": int(len(fixture["vertices"])),
            "face_count": int(len(fixture["faces"])),
            "component_count": int(len(fixture["component_sizes"])),
            "component_filtering": False,
        },
        "interval_accounting": fixture["intervals"],
        "raw_artifact": str((fixture_root / "fixture_audit.npz").resolve()),
    }
    _write_json(fixture_root / "fixture_report.json", fixture_report)
    return fixture_report


def _w167_reproduction(fixture_reports: dict[str, dict[str, Any]], frozen_report: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "analytic_hit_rays": "analytic_hit_rays",
        "zero_set_hit_rays": "zero_set_hit_rays",
        "missed_blocker_count": "missed_zero_set_hit_rays",
        "false_zero_set_blocker_count_among_analytic_no_hit": "false_zero_set_hit_rays",
        "premature_blocker_count_raw": "premature_blocker_ray_count",
        "delayed_blocker_count_raw": "delayed_blocker_ray_count",
    }
    per_fixture: dict[str, Any] = {}
    frozen = frozen_report["synthetic_analytic_blocker_result"]
    for name in SINGLE_SURFACE_NAMES:
        old = frozen[name]["metrics"]
        new = fixture_reports[name]["interval_accounting"]
        comparisons = {old_key: {"w167": int(old[old_key]), "w168": int(new[new_key]), "exact": int(old[old_key]) == int(new[new_key])} for old_key, new_key in keys.items()}
        per_fixture[name] = {"all_exact": all(value["exact"] for value in comparisons.values()), "counts": comparisons}
    return {"all_exact": all(value["all_exact"] for value in per_fixture.values()), "per_fixture": per_fixture}


def _real_scene_non_oracle(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    aggregate: Counter[int] = Counter()
    per_camera: dict[str, Any] = {}
    total_rays = 0
    total_hits = 0
    total_non_top20 = 0
    for camera_name, record in source["per_camera"].items():
        rays = int(record["query_ray_count"])
        hits = int(record["status_counts"][w167.STATUS_HIT])
        non_top20 = int(record["fragment_first_hit_count"])
        for row in record["component_provenance"]:
            aggregate[int(row["component_id"])] += int(row["first_hit_count"])
        per_camera[camera_name] = {
            "ray_count": rays,
            "first_hit_count": hits,
            "non_top20_first_hit_count_attribution_only": non_top20,
            "unique_first_hit_component_count": len(record["component_provenance"]),
            "all_components_active_as_blockers": True,
        }
        total_rays += rays
        total_hits += hits
        total_non_top20 += non_top20
    component_rows = [{"component_id": int(component), "first_hit_count": int(count)} for component, count in sorted(aggregate.items())]
    return {
        "status": "REUSED_W167_NON_ORACLE_VIABILITY_CHECK",
        "source": str(path.resolve()),
        "physical_hidden_surface_ground_truth": "NONE",
        "total_sampled_rays": total_rays,
        "total_first_hits": total_hits,
        "unique_first_hit_component_count": len(component_rows),
        "non_top20_first_hit_count_attribution_only": total_non_top20,
        "component_provenance_all_hits": component_rows,
        "per_camera": per_camera,
        "component_filtering": False,
        "top20_used_as_semantics": False,
        "single_depth_sheet_required": False,
        "all_non_top20_components_remain_active_blockers": True,
        "interpretation": "viability/non-oracle only; it cannot rescue or refute strict synthetic soundness",
    }


def _prepare_output(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for directory in (out / "synthetic", out / "real_scene"):
        if directory.exists():
            shutil.rmtree(directory)
    for name in ("worklog_168_report.json", "README.md"):
        path = out / name
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _prepare_output(args.out)
    frozen_w167 = json.loads(args.w167_report.read_text(encoding="utf-8"))
    synthetic_root = args.out / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    synthetic_root.joinpath("README.md").write_text(
        "# W168 Synthetic Strict First-Hit Audit\n\n"
        "이 directory는 W167의 fronto-parallel plane, oblique plane, sphere를 같은 historical-style TSDF/extraction/ray contract로 재생하고 two-sheet layered fixture를 추가한다. 각 fixture는 complete query-depth disagreement interval과 ray별 NPZ를 보존한다.\n\n"
        "Palette: signed error map의 blue=premature, red=delayed, yellow=missed zero-set hit, magenta=false zero-set hit. Histogram blue=premature false-positive interval, red=delayed false-negative interval.\n\n"
        "공통 조건은 각 fixture 및 visualization README에 중복 기록했다. Synthetic analytic truth만 사용하며 real-scene physical truth, Candidate-B, global aggregation, Eligibility/continuation을 정의하지 않는다.\n",
        encoding="utf-8",
    )

    fixtures: dict[str, dict[str, Any]] = {}
    fixture_reports: dict[str, dict[str, Any]] = {}
    for name in FIXTURE_NAMES:
        print(f"[worklog 168] replaying {name}", flush=True)
        fixture = _build_fixture(name)
        fixtures[name] = fixture
        fixture_reports[name] = _export_fixture(synthetic_root, fixture)

    reproduction = _w167_reproduction(fixture_reports, frozen_w167)
    if not reproduction["all_exact"]:
        raise RuntimeError("W168 failed to reproduce frozen W167 single-surface counts")
    multi_surface = _multi_surface_result(fixtures["layered_two_sheet"])
    real_check = _real_scene_non_oracle(args.w167_real_report)
    total_counterexamples = sum(int(value["interval_accounting"]["strict_positive_evidence_counterexample_count"]) for value in fixture_reports.values())
    supported = total_counterexamples == 0
    verdict = VERDICT_SUPPORTED if supported else VERDICT_REJECTED
    per_fixture_intervals = {name: value["interval_accounting"] for name, value in fixture_reports.items()}
    interior_boundary = {
        name: {
            "interior": value["interval_accounting"]["interior"],
            "silhouette_or_support_boundary": value["interval_accounting"]["silhouette_or_support_boundary"],
        }
        for name, value in fixture_reports.items()
    }
    report = {
        "status": "COMPLETE_WL168_RAW_ZERO_SET_FIRST_HIT_POSITIVE_PER_VIEW_OCCLUSION_EVIDENCE_AUDIT",
        "worklog": 168,
        "1_intent_alignment": {
            "status": "PASS",
            "question": "Can strict z_s < z_q from the unfiltered historical raw zero-set be canonical positive per-view DirectObservationUnavailable evidence?",
            "geometry_improved": False,
            "trusted_subset_selected": False,
            "fragmentation_deferred": True,
            "production_behavior_modified": False,
        },
        "2_implementation_fidelity": {
            "historical_zero_set_modified": False,
            "single_surface_replay": "W167 helpers and frozen historical-style SparseProjectiveTSDF/all-eight-corner extraction",
            "w167_count_reproduction": reproduction,
            "ray_first_hit_contract": "W167 two-sided exact ray-triangle first hit; strict positive depth and deterministic depth/triangle-ID ordering",
            "candidate_predicate": "valid z_s and 0 < z_s < z_q",
            "ground_truth_predicate": "valid z* and 0 < z* < z_q",
            "semantic_epsilon": None,
            "h_or_mu_tuned": False,
            "iso_value_changed": False,
            "component_filtering": False,
            "top20_semantics": False,
            "renderer_median_used": False,
        },
        "3_synthetic_first_blocker_result": {
            "fixtures": fixture_reports,
            "all_components_active": True,
            "w167_single_surface_counts_exactly_reproduced": reproduction["all_exact"],
        },
        "4_arbitrary_query_disagreement_intervals": {
            "method": "complete interval consequence; no sampled query-density approximation",
            "strict_boundary": "z_s < z_q; z_q == z_s is not OCCLUDED",
            "per_fixture": per_fixture_intervals,
            "total_strict_positive_counterexample_rays": total_counterexamples,
            "promotion_soundness_requires": "zero false-zero-set-hit rays and zero premature intervals across every unfiltered fixture ray",
        },
        "5_multi_surface_result": multi_surface,
        "6_interior_vs_boundary_failures": {
            "boundary_definition": "fixed synthetic raster exact 8-neighbor analytic-hit adjacency; attribution only, never a trusted subset",
            "per_fixture": interior_boundary,
            "boundary_failures_excluded_from_verdict": False,
        },
        "7_real_scene_non_oracle_check": real_check,
        "8_architecture_result": {
            "verdict": verdict,
            "promoted": supported,
            "strict_counterexample_count": total_counterexamples,
            "reason": (
                "No false-positive query-depth territory was found in the complete unfiltered synthetic audit."
                if supported
                else "At least one valid raw zero-set first hit is strictly premature (or false), creating a non-empty query-depth interval where ZEROSET_BLOCKED is true and analytic GT_BLOCKED is false; no semantic epsilon is allowed."
            ),
            "qualitative_real_scene_result_can_override_synthetic_counterexample": False,
        },
        "9_promoted_retained_rejected_open": {
            "promoted": ["strict per-view raw-zero-set positive blocker relation"] if supported else [],
            "retained": [
                "historical raw zero-set mesh/extraction and all components",
                "W167 ray-triangle first-hit contract and synthetic/real evidence",
                "Candidate-B as historical baseline only",
                "W161 spatial-domain pause",
                "primitive-observation versus point-query semantic separation",
                "strict exact-boundary handling",
            ],
            "rejected": [] if supported else ["z_s < z_q as canonical positive per-view DirectObservationUnavailable evidence under the required strict no-epsilon contract"],
            "open": [
                "real-scene physical hidden-surface ground truth remains unavailable",
                "W161 global spatial domain",
                "global OBSERVED/OCCLUDED aggregation and final UNRESOLVED semantics",
                "fragmentation/component organization, Eligibility, and continuation remain deferred",
            ],
        },
        "outputs": {
            "root": str(args.out.resolve()),
            "fixture_png_count": len(list(synthetic_root.rglob("*.png"))),
            "ppm_count": len(list(args.out.rglob("*.ppm"))),
            "readme_count": len(list(args.out.rglob("README.md"))) + 1,
            "w153_replay_cache_copied_to_temp": False,
        },
        "runtime_seconds": time.time() - started,
    }
    real_root = args.out / "real_scene"
    real_root.mkdir(parents=True, exist_ok=True)
    real_root.joinpath("README.md").write_text(
        "# W168 Real-Scene Non-Oracle Check\n\n"
        "이 directory는 새 ray replay나 physical hidden-surface label을 만들지 않고 W167 frozen real-scene report의 first-hit/component provenance를 읽기 전용으로 요약한다. 모든 component와 non-top-20 hit는 positive blocker 후보에서 그대로 active이며 component ID는 attribution일 뿐 filtering semantics가 아니다.\n\n"
        "Palette/legend: 새 PNG를 생성하지 않는다. W167의 existing `component_provenance`, `first_hit_surface`, `blocker_relation` PNG와 각 README가 qualitative reference다.\n\n"
        "공통 조건: W167의 3개 fixed camera/ROI/stride/ray primitive를 그대로 보존한다. 실제 scene ground truth가 없으므로 이 결과는 viability check이며 synthetic strict counterexample를 뒤집을 수 없다.\n",
        encoding="utf-8",
    )
    _write_json(real_root / "real_scene_non_oracle_check.json", real_check)
    report["outputs"]["readme_count"] = len(list(args.out.rglob("README.md"))) + 1
    _write_json(args.out / "worklog_168_report.json", report)
    args.out.joinpath("README.md").write_text(
        "# Worklog 168 — Raw Zero-Set First-Hit Positive Per-View Occlusion Evidence Audit\n\n"
        "이 output은 unfiltered historical raw zero-set first hit `z_s`와 analytic first blocker `z*`의 complete query-depth consequence를 strict no-epsilon contract로 감사한다. W167의 세 single-surface fixture를 exact count로 재현하고 layered two-sheet fixture를 추가했다.\n\n"
        f"Architecture verdict: `{verdict}`. Candidate-B, renderer median, W161 domain, global aggregation, Eligibility, continuation, NURBS 및 production behavior는 변경하지 않았다.\n\n"
        "`synthetic/<fixture>/fixture_audit.npz`가 authoritative ray-level artifact다. PNG는 review용이며 모든 visualization directory에 input/state semantics, palette, 공통 조건, limitation을 설명하는 UTF-8 README가 있다. PPM은 생성하지 않았고 W153 replay cache를 temp에 복사하지 않았다.\n",
        encoding="utf-8",
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w167-report", type=Path, default=DEFAULT_W167_REPORT)
    parser.add_argument("--w167-real-report", type=Path, default=DEFAULT_W167_REAL_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "verdict": report["8_architecture_result"]["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
