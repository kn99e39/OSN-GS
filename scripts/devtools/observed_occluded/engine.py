from __future__ import annotations

"""Worklog 120 -- SHARED evaluation engine.

This module runs ONE query bank against ONE camera set and collects, per view,
the four candidates' independent verdicts. It is deliberately dumb: it renders,
gathers, hands each candidate exactly the primitive that candidate's own module
asked for, and records what comes back. It contains no visibility boundary, no
blocker rule, no surface-hit rule, no T semantics and no median semantics; every
decision line lives in `candidate_a..d`.

The same engine runs the synthetic S1-S7 contracts and the real-scene bank, so
the two can never diverge in aggregation, relevance, or metric semantics.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from . import candidate_a_surface_hit as candidate_a
from . import candidate_b_median_depth as candidate_b
from . import candidate_c_geometric_visibility as candidate_c
from . import candidate_d_renderer_reachability as candidate_d
from .shared import (
    STATE_NON_RELEVANT,
    aggregate_global,
    assign_query_depth_slots,
    canonical_geometric_support_rho_max,
    project_queries,
    reconstruct_direct_surfel_intersection_world_point,
)

CANDIDATE_NAMES = ("A", "B", "C", "D")


@dataclass
class EvaluationResult:
    per_view_states: dict[str, np.ndarray]        # candidate -> (N, V) int8
    global_states: dict[str, np.ndarray]          # candidate -> (N,) int8
    relevance_code: np.ndarray                    # (N, V) int8
    query_depth: np.ndarray                       # (N, V) float32
    provenance: dict[str, np.ndarray] = field(default_factory=dict)
    view_names: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_geometric_support(model: Any) -> candidate_c.GeometricSceneSupport:
    """Candidate C's blocker set, built from the trained model's own tensors and
    the canonical alpha cutoff. Placed in the engine only because it is a
    once-per-run precomputation; every semantic line is in candidate_c."""

    with torch.no_grad():
        rotation = model.get_rotation_matrix
        scaling = model.get_scaling
        return candidate_c.GeometricSceneSupport(
            centers=model.get_xyz.detach(),
            normals=rotation[:, :, 2].detach(),
            tangent_u=rotation[:, :, 0].detach(),
            tangent_v=rotation[:, :, 1].detach(),
            scale_u=scaling[:, 0].detach(),
            scale_v=scaling[:, 1].detach(),
            rho_max=canonical_geometric_support_rho_max(model.get_opacity.detach()),
            opacity=model.get_opacity.detach().reshape(-1),
        )


def evaluate(
    model: Any,
    cameras: list[Any],
    positions: torch.Tensor,
    *,
    support: candidate_c.GeometricSceneSupport | None = None,
    chunk_bytes: int = 384 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
    enable_candidates: tuple[str, ...] = CANDIDATE_NAMES,
) -> EvaluationResult:
    from osn_gs.render.torch_surfel_query_depth_diagnostics import MAX_QUERY_SLOTS, render_with_query_depth_probe

    device = positions.device
    count = int(positions.shape[0])
    views = len(cameras)
    if support is None and "C" in enable_candidates:
        support = build_geometric_support(model)

    with torch.no_grad():
        positions_full = model.get_xyz.detach()
        rotation_full = model.get_rotation_matrix.detach()
        tangent_u_full = rotation_full[:, :, 0].contiguous()
        tangent_v_full = rotation_full[:, :, 1].contiguous()
        scaling_full = model.get_scaling.detach()
        scale_u_full = scaling_full[:, 0].contiguous()
        scale_v_full = scaling_full[:, 1].contiguous()

    per_view_states = {name: np.full((count, views), STATE_NON_RELEVANT, dtype=np.int8) for name in enable_candidates}
    relevance_code = np.zeros((count, views), dtype=np.int8)
    query_depth = np.zeros((count, views), dtype=np.float32)
    provenance: dict[str, np.ndarray] = {
        "A_hit_distance": np.full((count, views), np.nan, dtype=np.float32),
        "A_event_depth": np.full((count, views), np.nan, dtype=np.float32),
        "A_event_branch": np.full((count, views), -1, dtype=np.int8),  # 0 = rho3d, 1 = rho2d
        "A_event_surfel": np.full((count, views), -1, dtype=np.int64),
        "B_median_depth": np.full((count, views), np.nan, dtype=np.float32),
        "C_blocker_count": np.zeros((count, views), dtype=np.int32),
        "C_nearest_blocker_t": np.full((count, views), np.nan, dtype=np.float32),
        "C_max_blocker_opacity": np.full((count, views), np.nan, dtype=np.float32),
        "D_transmittance": np.full((count, views), np.nan, dtype=np.float32),
        "D_reached": np.full((count, views), -1, dtype=np.int8),
        "D_prefix_count": np.full((count, views), -1, dtype=np.int32),
    }
    view_names: list[str] = []
    total_render_passes = 0
    max_slot_rank = 0
    # Frozen-state fingerprint, a free by-product of the sweep: the union over
    # all views of surfels that were ever a median surface representative.
    # Worklog 119 measured exactly 785,937 for this checkpoint + camera set, so
    # reproducing that number is direct evidence that the model, the camera set
    # and the renderer are all untouched by this batch.
    ever_representative = torch.zeros((int(positions_full.shape[0]),), dtype=torch.bool, device=device)

    for view_index, camera in enumerate(cameras):
        view_names.append(str(getattr(camera, "image_name", f"view_{view_index}")))
        height, width = int(camera.image_height), int(camera.image_width)
        geometry = project_queries(camera, positions)
        relevance_code[:, view_index] = geometry.relevance_code.detach().cpu().numpy()
        query_depth[:, view_index] = geometry.depth.detach().cpu().numpy()

        pixel_index_np = geometry.pixel_index.detach().cpu().numpy()
        ranks = assign_query_depth_slots(pixel_index_np, MAX_QUERY_SLOTS)
        max_slot_rank = max(max_slot_rank, int(ranks.max()) if ranks.size else -1)
        passes = 1 if ranks.max(initial=-1) < 0 else int(ranks.max()) // MAX_QUERY_SLOTS + 1

        terminated = torch.full((count,), -1, dtype=torch.int32, device=device)
        transmittance = torch.full((count,), float("nan"), dtype=torch.float32, device=device)
        reached = torch.full((count,), -1, dtype=torch.int32, device=device)
        prefix_count = torch.full((count,), -1, dtype=torch.int32, device=device)
        canonical: dict[str, Any] | None = None

        need_probe = "D" in enable_candidates
        for pass_index in range(passes if need_probe else 1):
            query_map = None
            selected = np.zeros(0, dtype=np.int64)
            if need_probe:
                lower, upper = pass_index * MAX_QUERY_SLOTS, (pass_index + 1) * MAX_QUERY_SLOTS
                selected = np.nonzero((ranks >= lower) & (ranks < upper))[0]
                query_map = torch.zeros((height * width * MAX_QUERY_SLOTS,), dtype=torch.float32, device=device)
                if selected.size:
                    slots = torch.as_tensor(
                        pixel_index_np[selected] * MAX_QUERY_SLOTS + (ranks[selected] - lower),
                        dtype=torch.int64, device=device,
                    )
                    query_map[slots] = geometry.depth[torch.as_tensor(selected, dtype=torch.int64, device=device)]
                query_map = query_map.reshape(height, width, MAX_QUERY_SLOTS)

            package = render_with_query_depth_probe(camera, model, query_depths=query_map)
            total_render_passes += 1
            if canonical is None:
                canonical = package
            if need_probe and selected.size:
                rows = torch.as_tensor(selected, dtype=torch.int64, device=device)
                flat_slots = torch.as_tensor(
                    pixel_index_np[selected] * MAX_QUERY_SLOTS + (ranks[selected] - pass_index * MAX_QUERY_SLOTS),
                    dtype=torch.int64, device=device,
                )
                terminated[rows] = package["query_terminated"].reshape(-1)[flat_slots]
                transmittance[rows] = package["query_T"].reshape(-1)[flat_slots]
                reached[rows] = package["query_reached"].reshape(-1)[flat_slots]
                prefix_count[rows] = package["query_prefix_count"].reshape(-1)[flat_slots]
            if package is not canonical:
                del package

        assert canonical is not None
        representative = canonical["representative_id"].reshape(-1).to(torch.int64)
        event_valid = representative >= 0
        ever_representative[torch.unique(representative[event_valid])] = True

        if "A" in enable_candidates:
            event_world = reconstruct_direct_surfel_intersection_world_point(
                representative, canonical["median_s_u"], canonical["median_s_v"],
                positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
            )
            homogeneous = torch.cat([event_world, torch.ones((event_world.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
            event_depth = (homogeneous @ camera.world_view_transform)[:, 2].contiguous()
            result_a = candidate_a.classify_view(geometry, event_world, event_depth, event_valid, positions)
            per_view_states["A"][:, view_index] = result_a["states"].detach().cpu().numpy()
            provenance["A_hit_distance"][:, view_index] = result_a["hit_distance"].detach().cpu().numpy()
            provenance["A_event_depth"][:, view_index] = result_a["event_depth"].detach().cpu().numpy()
            index = geometry.pixel_index.clamp(min=0)
            rho3d = canonical["median_rho3d"].reshape(-1)[index]
            rho2d = canonical["median_rho2d"].reshape(-1)[index]
            branch = torch.where(rho3d <= rho2d, torch.zeros_like(rho3d), torch.ones_like(rho3d)).to(torch.int8)
            branch = torch.where(event_valid[index] & geometry.relevant, branch, torch.full_like(branch, -1))
            provenance["A_event_branch"][:, view_index] = branch.detach().cpu().numpy()
            surfel = torch.where(event_valid[index] & geometry.relevant, representative[index], torch.full_like(representative[index], -1))
            provenance["A_event_surfel"][:, view_index] = surfel.detach().cpu().numpy()
            del event_world, homogeneous, event_depth

        if "B" in enable_candidates:
            median_flat = candidate_b.median_depth_map(canonical["out_others"]).reshape(-1)
            result_b = candidate_b.classify_view(geometry, median_flat)
            per_view_states["B"][:, view_index] = result_b["states"].detach().cpu().numpy()
            provenance["B_median_depth"][:, view_index] = result_b["median_depth"].detach().cpu().numpy()

        if "C" in enable_candidates:
            assert support is not None
            result_c = candidate_c.classify_view(
                geometry, positions, camera.camera_center, camera.world_view_transform, support,
                chunk_bytes=chunk_bytes,
            )
            per_view_states["C"][:, view_index] = result_c["states"].detach().cpu().numpy()
            provenance["C_blocker_count"][:, view_index] = result_c["blocker_count"].detach().cpu().numpy()
            provenance["C_nearest_blocker_t"][:, view_index] = result_c["nearest_blocker_t"].detach().cpu().numpy()
            provenance["C_max_blocker_opacity"][:, view_index] = result_c["max_blocker_opacity"].detach().cpu().numpy()

        if "D" in enable_candidates:
            result_d = candidate_d.classify_view(geometry, terminated, transmittance, reached, prefix_count)
            per_view_states["D"][:, view_index] = result_d["states"].detach().cpu().numpy()
            provenance["D_transmittance"][:, view_index] = transmittance.detach().cpu().numpy()
            provenance["D_reached"][:, view_index] = reached.detach().cpu().numpy().astype(np.int8)
            provenance["D_prefix_count"][:, view_index] = prefix_count.detach().cpu().numpy()

        del canonical
        if progress is not None and (view_index % 10 == 0 or view_index == views - 1):
            progress(f"evaluated view {view_index + 1}/{views} ({view_names[-1]})")

    global_states = {name: aggregate_global(per_view_states[name]) for name in enable_candidates}
    return EvaluationResult(
        per_view_states=per_view_states,
        global_states=global_states,
        relevance_code=relevance_code,
        query_depth=query_depth,
        provenance=provenance,
        view_names=view_names,
        diagnostics={
            "render_passes": total_render_passes,
            "max_queries_per_pixel_rank": max_slot_rank + 1,
            "query_depth_slots_per_pass": MAX_QUERY_SLOTS,
            "median_surface_representatives_union": int(ever_representative.sum().item()),
            "total_model_surfels": int(positions_full.shape[0]),
        },
    )
