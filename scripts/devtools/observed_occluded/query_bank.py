from __future__ import annotations

"""Worklog 120 -- SHARED real-scene query bank (directive 9B).

One deterministic bank, built once, used verbatim by all four candidates. No
RNG anywhere: every selection is a fixed stride over a fixed ordering (view
index, then raster order, then bank order).

The bank's R1 anchors are placed at renderer median surface events. That is a
bank definition the directive itself mandates ("Use renderer-observed surface
points with provenance", 9B/R1), not a candidate decision -- but it is also a
KNOWN BIAS toward candidates A and B, whose primitive is that same event, and
the worklog's Shared-Code Semantic Audit says so explicitly. The bank
deliberately also carries query classes that no candidate's primitive
generated: the ray ladder (R3/R4), region gap midpoints (R5) and out-of-frustum
controls (R6).

Offsets are never tuned. The ray ladder steps are integer/half multiples of the
ANCHOR SURFEL'S OWN canonical geometric support radius
(`sqrt(rho_max) * max(scale_u, scale_v)`, from `shared.canonical_geometric_support_rho_max`),
i.e. a quantity the trained model and the canonical alpha cutoff already define
per surfel. The multiplier set is fixed a priori and was never adjusted.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .shared import (
    KIND_R1_ANCHOR_RHO2D,
    KIND_R1_ANCHOR_RHO3D,
    KIND_R3_BEHIND,
    KIND_R4_FRONT,
    KIND_R5_REGION_GAP,
    KIND_R6_OUT_OF_FRUSTUM,
    QueryBank,
    canonical_geometric_support_rho_max,
    reconstruct_direct_surfel_intersection_world_point,
)

# Fixed a priori. Every one of these is a selection stride or a count, never a
# threshold that any decision depends on.
ANCHOR_VIEW_STRIDE = 16          # 161 train views -> views 0, 16, ..., 160 (11 views)
ANCHORS_PER_VIEW_PER_BRANCH = 120
LADDER_ANCHOR_STRIDE = 12        # every 12th R1 anchor also seeds a ray ladder
LADDER_STEPS = (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)
REGION_GAP_PER_REGION = 60
OUT_OF_FRUSTUM_EXTENT_MULTIPLES = (4.0, 8.0)

# Region probe anchors -- identical fractions to worklog 108-119's own
# `ANCHOR_FRACTIONS`, reused verbatim so region accounting stays comparable
# across the whole lineage. WORKING INTERPRETATION ONLY, never ground truth.
ANCHOR_FRACTIONS: dict[str, list[tuple[float, float]]] = {
    "table_top": [(0.5, 0.46)],
    "table_side_curved": [(0.26, 0.50), (0.74, 0.50)],
    "table_legs": [(0.45, 0.60), (0.55, 0.60)],
    "patio": [(0.15, 0.85), (0.85, 0.9)],
    "hedge": [(0.1, 0.1), (0.9, 0.15), (0.5, 0.05)],
}
REGION_LABELS = list(ANCHOR_FRACTIONS.keys())


@dataclass
class BankBuildReport:
    anchor_views: list[int]
    anchor_view_names: list[str]
    per_view_valid_pixels: list[int]
    per_view_rho3d_pixels: list[int]
    per_view_rho2d_pixels: list[int]
    notes: dict[str, Any]


def region_of_surfel(model: Any, preview_camera: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """Nearest-anchor region label per FULL-model surfel, reusing worklog
    108-119's own anchor mechanism unchanged (a working interpretation for
    reporting, never a decision input)."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    device = model.device
    package = render_with_query_depth_probe(preview_camera, model, query_depths=None)
    representative = package["representative_id"].to(torch.int64)
    height, width = representative.shape
    positions = model.get_xyz.detach()

    anchor_ids: dict[str, list[int]] = {}
    for label, fractions in ANCHOR_FRACTIONS.items():
        ids: list[int] = []
        for fx, fy in fractions:
            px = min(width - 1, int(fx * width))
            py = min(height - 1, int(fy * height))
            surfel = int(representative[py, px].item())
            if surfel >= 0:
                ids.append(surfel)
        anchor_ids[label] = ids

    count = int(positions.shape[0])
    region = torch.full((count,), -1, dtype=torch.int64, device=device)
    for index, label in enumerate(REGION_LABELS):
        ids = anchor_ids.get(label, [])
        if not ids:
            continue
        own = torch.cdist(positions, positions[torch.tensor(ids, device=device)]).min(dim=1).values
        others = sum((anchor_ids.get(other, []) for other in REGION_LABELS if other != label), [])
        if others:
            rival = torch.cdist(positions, positions[torch.tensor(others, device=device)]).min(dim=1).values
            mask = own < rival
        else:
            mask = torch.ones((count,), dtype=torch.bool, device=device)
        region[mask] = index
    return region, {"preview_camera": str(getattr(preview_camera, "image_name", "?")), "anchor_surfel_ids": anchor_ids}


def build_bank(
    model: Any,
    cameras: list[Any],
    region_index: torch.Tensor,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[QueryBank, BankBuildReport]:
    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    device = model.device
    with torch.no_grad():
        positions_full = model.get_xyz.detach()
        rotation_full = model.get_rotation_matrix.detach()
        tangent_u_full = rotation_full[:, :, 0].contiguous()
        tangent_v_full = rotation_full[:, :, 1].contiguous()
        scaling_full = model.get_scaling.detach()
        scale_u_full = scaling_full[:, 0].contiguous()
        scale_v_full = scaling_full[:, 1].contiguous()
        rho_max_full = canonical_geometric_support_rho_max(model.get_opacity.detach())
        support_radius_full = torch.sqrt(torch.clamp(rho_max_full, min=0.0)) * torch.maximum(scale_u_full, scale_v_full)

    anchor_views = list(range(0, len(cameras), ANCHOR_VIEW_STRIDE))
    positions: list[torch.Tensor] = []
    kinds: list[str] = []
    source_view: list[int] = []
    source_surfel: list[int] = []
    per_view_valid: list[int] = []
    per_view_rho3d: list[int] = []
    per_view_rho2d: list[int] = []

    for view_index in anchor_views:
        camera = cameras[view_index]
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        representative = package["representative_id"].reshape(-1).to(torch.int64)
        valid = representative >= 0
        rho3d = package["median_rho3d"].reshape(-1)
        rho2d = package["median_rho2d"].reshape(-1)
        world = reconstruct_direct_surfel_intersection_world_point(
            representative, package["median_s_u"], package["median_s_v"],
            positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
        )
        finite = valid & torch.isfinite(world).all(dim=1)
        rho3d_branch = torch.nonzero(finite & (rho3d <= rho2d), as_tuple=False).reshape(-1)
        rho2d_branch = torch.nonzero(finite & (rho3d > rho2d), as_tuple=False).reshape(-1)
        per_view_valid.append(int(finite.sum()))
        per_view_rho3d.append(int(rho3d_branch.numel()))
        per_view_rho2d.append(int(rho2d_branch.numel()))

        for pixels, kind in ((rho3d_branch, KIND_R1_ANCHOR_RHO3D), (rho2d_branch, KIND_R1_ANCHOR_RHO2D)):
            total = int(pixels.numel())
            if total == 0:
                continue
            take = min(ANCHORS_PER_VIEW_PER_BRANCH, total)
            # Deterministic uniform stride over raster order; no RNG.
            picks = pixels[torch.linspace(0, total - 1, steps=take, device=device).round().to(torch.int64)]
            picks = torch.unique(picks, sorted=True)
            positions.append(world[picks].clone())
            kinds.extend([kind] * int(picks.numel()))
            source_view.extend([view_index] * int(picks.numel()))
            source_surfel.extend(representative[picks].tolist())
        if progress is not None:
            progress(f"bank: anchors from view {view_index} ({getattr(camera, 'image_name', '?')})")
        del package, world

    anchor_positions = torch.cat(positions, dim=0) if positions else torch.zeros((0, 3), device=device)
    anchor_count = int(anchor_positions.shape[0])
    anchor_view_np = np.asarray(source_view, dtype=np.int64)
    anchor_surfel_np = np.asarray(source_surfel, dtype=np.int64)
    anchor_kind = list(kinds)

    # ------------------------------------------------------------ ray ladder
    ladder_rows = np.arange(0, anchor_count, LADDER_ANCHOR_STRIDE, dtype=np.int64)
    ladder_positions: list[torch.Tensor] = []
    ladder_kind: list[str] = []
    ladder_view: list[int] = []
    ladder_surfel: list[int] = []
    ladder_step: list[float] = []
    ladder_radius: list[float] = []
    for row in ladder_rows:
        surfel = int(anchor_surfel_np[row])
        view_index = int(anchor_view_np[row])
        radius = float(support_radius_full[surfel].item())
        origin = cameras[view_index].camera_center.reshape(3)
        anchor = anchor_positions[row]
        direction = anchor - origin
        norm = float(direction.norm().item())
        if norm <= 0 or radius <= 0:
            continue
        unit = direction / norm
        for step in LADDER_STEPS:
            ladder_positions.append(anchor + unit * (step * radius))
            ladder_kind.append(KIND_R3_BEHIND if step > 0 else KIND_R4_FRONT)
            ladder_view.append(view_index)
            ladder_surfel.append(surfel)
            ladder_step.append(step)
            ladder_radius.append(radius)

    # --------------------------------------------------------- region gaps
    gap_positions: list[torch.Tensor] = []
    gap_kind: list[str] = []
    gap_view: list[int] = []
    gap_surfel: list[int] = []
    anchor_region = region_index[torch.as_tensor(anchor_surfel_np, dtype=torch.int64, device=device)] if anchor_count else torch.zeros((0,), dtype=torch.int64, device=device)
    gap_pair_distance: list[float] = []
    for label_index, _label in enumerate(REGION_LABELS):
        rows = torch.nonzero(anchor_region == label_index, as_tuple=False).reshape(-1)
        if int(rows.numel()) < 2:
            continue
        subset = anchor_positions[rows]
        take = min(REGION_GAP_PER_REGION, int(rows.numel()))
        seeds = torch.linspace(0, int(rows.numel()) - 1, steps=take, device=device).round().to(torch.int64)
        seeds = torch.unique(seeds, sorted=True)
        distances = torch.cdist(subset[seeds], subset)
        distances[torch.arange(seeds.numel(), device=device), seeds] = float("inf")
        nearest = distances.argmin(dim=1)
        gap_positions.append((subset[seeds] + subset[nearest]) * 0.5)
        gap_pair_distance.extend(distances[torch.arange(seeds.numel(), device=device), nearest].tolist())
        gap_kind.extend([KIND_R5_REGION_GAP] * int(seeds.numel()))
        gap_view.extend(anchor_view_np[rows[seeds].cpu().numpy()].tolist())
        gap_surfel.extend(anchor_surfel_np[rows[seeds].cpu().numpy()].tolist())

    # --------------------------------------------------- out-of-frustum controls
    lower = positions_full.min(dim=0).values
    upper = positions_full.max(dim=0).values
    centre = (lower + upper) * 0.5
    extent = (upper - lower).max()
    control_positions: list[torch.Tensor] = []
    for multiple in OUT_OF_FRUSTUM_EXTENT_MULTIPLES:
        for axis in range(3):
            for sign in (-1.0, 1.0):
                offset = torch.zeros(3, device=device)
                offset[axis] = sign * multiple * float(extent.item())
                control_positions.append(centre + offset)

    all_positions = [anchor_positions]
    all_kinds = list(anchor_kind)
    all_view = list(anchor_view_np.tolist())
    all_surfel = list(anchor_surfel_np.tolist())
    all_step = [0.0] * anchor_count
    all_radius = [float(support_radius_full[int(s)].item()) for s in anchor_surfel_np] if anchor_count else []

    if ladder_positions:
        all_positions.append(torch.stack(ladder_positions))
        all_kinds.extend(ladder_kind)
        all_view.extend(ladder_view)
        all_surfel.extend(ladder_surfel)
        all_step.extend(ladder_step)
        all_radius.extend(ladder_radius)
    if gap_positions:
        stacked = torch.cat(gap_positions, dim=0)
        all_positions.append(stacked)
        all_kinds.extend(gap_kind)
        all_view.extend(gap_view)
        all_surfel.extend(gap_surfel)
        all_step.extend([float("nan")] * int(stacked.shape[0]))
        all_radius.extend([float("nan")] * int(stacked.shape[0]))
    if control_positions:
        stacked = torch.stack(control_positions)
        all_positions.append(stacked)
        all_kinds.extend([KIND_R6_OUT_OF_FRUSTUM] * int(stacked.shape[0]))
        all_view.extend([-1] * int(stacked.shape[0]))
        all_surfel.extend([-1] * int(stacked.shape[0]))
        all_step.extend([float("nan")] * int(stacked.shape[0]))
        all_radius.extend([float("nan")] * int(stacked.shape[0]))

    positions_tensor = torch.cat(all_positions, dim=0).to(torch.float32).contiguous()
    surfel_np = np.asarray(all_surfel, dtype=np.int64)
    region_np = np.full(surfel_np.shape[0], -1, dtype=np.int64)
    known = surfel_np >= 0
    if known.any():
        region_np[known] = region_index[torch.as_tensor(surfel_np[known], dtype=torch.int64, device=device)].cpu().numpy()

    bank = QueryBank(
        positions=positions_tensor,
        kind=all_kinds,
        source_view=np.asarray(all_view, dtype=np.int64),
        source_surfel=surfel_np,
        region=region_np,
        ladder_step=np.asarray(all_step, dtype=np.float32),
        support_radius=np.asarray(all_radius, dtype=np.float32),
    )
    report = BankBuildReport(
        anchor_views=anchor_views,
        anchor_view_names=[str(getattr(cameras[i], "image_name", i)) for i in anchor_views],
        per_view_valid_pixels=per_view_valid,
        per_view_rho3d_pixels=per_view_rho3d,
        per_view_rho2d_pixels=per_view_rho2d,
        notes={
            "anchor_view_stride": ANCHOR_VIEW_STRIDE,
            "anchors_per_view_per_branch": ANCHORS_PER_VIEW_PER_BRANCH,
            "ladder_anchor_stride": LADDER_ANCHOR_STRIDE,
            "ladder_steps_in_units_of_anchor_surfel_canonical_support_radius": list(LADDER_STEPS),
            "region_gap_per_region": REGION_GAP_PER_REGION,
            "out_of_frustum_extent_multiples": list(OUT_OF_FRUSTUM_EXTENT_MULTIPLES),
            "scene_extent": float(extent.item()),
            "region_gap_pair_distance_samples": len(gap_pair_distance),
            "rng": "none -- every selection is a fixed stride over a fixed ordering",
        },
    )
    return bank, report
