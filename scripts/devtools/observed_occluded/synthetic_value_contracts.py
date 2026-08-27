from __future__ import annotations

"""Worklog 121 -- SUPPLEMENTAL synthetic value contracts S-D1 / S-C1 / S-B1
(directive section 13).

Worklog 120's S1-S7 are preserved exactly and re-run unchanged. These three add
FOCUSED DIAGNOSTICS of the quantities under the decisions -- no new candidate
semantics, no new state, no tolerance. Every fixture's geometry is fixed here
before any result direction is read.

  S-D1  a controlled traversal in which the canonical CENTRE-depth tile order
        and the per-pixel accepted-event depth order genuinely disagree, so the
        D probe's resolution point can be compared against strict physical-depth
        ordering.
  S-C1  several overlapping surfel footprints forming ONE surface, with the
        renderer's own median event on that surface as the query, so the
        camera-nearest and query-nearest blockers are separable by construction.
  S-B1  the renderer's exact median event reconstructed into 3D and projected
        back, measuring the float32 `query_depth - median_depth` discrepancy
        without adding any tolerance anywhere.
"""

import math
from typing import Any

import torch

from .synthetic_contracts import (
    IMAGE, PLANE_DEPTH, front_camera, make_plane_stack, pixel_center_world_offset,
    surface_event_world_point,
)

# Fixed a priori.
S_D1_TILT_DEGREES = 80.0          # near-edge-on plane: tiny lateral offsets move its intersection depth a lot
S_D1_TILTED_SCALE = 3.0           # large enough that the offset pixel is still inside the rho3d footprint
S_D1_FLAT_CENTRE_Z = 0.05         # centre BEHIND the tilted surfel's centre, yet in FRONT of its pixel intersection
S_D1_QUERY_DEPTH = 4.10           # between the two accepted events' per-pixel depths
S_C1_LAYERS = 8
S_C1_LAYER_SPACING = 0.01
S_C1_LAYER_OPACITY = 0.15         # keeps T above 0.5 for several layers, so the median lands mid-stack


def _tilt_quaternion_about_y(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def build_s_d1(device: str = "cuda") -> dict[str, Any]:
    """S-D1: accepted-event depth inversion inside the canonical traversal.

    Surfel 0 is a near-edge-on plane whose CENTRE sits at camera depth 4.00 but
    whose ray/plane intersection at the probed pixel lands BEHIND depth 4.05.
    Surfel 1 is an ordinary front-facing plane whose centre AND intersection are
    at 4.05. The canonical tile list is sorted by centre depth, so surfel 0 is
    composited first even though its per-pixel event is physically deeper --
    exactly the traversal-order/physical-depth mismatch the audit must quantify.
    """

    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
    from osn_gs.render.torch_surfel_query_depth_diagnostics import MAX_QUERY_SLOTS, render_with_query_depth_probe

    model = TorchGaussianSurfelModel(sh_degree=0, device=device)
    tilted = _tilt_quaternion_about_y(-S_D1_TILT_DEGREES)
    model.initialize(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, S_D1_FLAT_CENTRE_Z]], dtype=torch.float32),
        colors=torch.tensor([[0.6, 0.5, 0.4], [0.4, 0.5, 0.6]], dtype=torch.float32),
        opacities=torch.tensor([[0.6], [0.6]], dtype=torch.float32),
        scales=torch.tensor([[S_D1_TILTED_SCALE, S_D1_TILTED_SCALE], [1.0, 1.0]], dtype=torch.float32),
        rotations=torch.tensor([list(tilted), [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    model.active_sh_degree = 0
    camera = front_camera(device)
    centre = IMAGE // 2

    query = torch.zeros((IMAGE, IMAGE, MAX_QUERY_SLOTS), dtype=torch.float32, device=device)
    query[centre, centre, 0] = S_D1_QUERY_DEPTH
    package = render_with_query_depth_probe(camera, model, query_depths=query)

    flat = centre * IMAGE + centre
    return {
        "name": "S_D1_accepted_event_depth_inversion",
        "description": (
            "Two accepted contributors whose canonical centre-depth tile order is the reverse of their "
            "per-pixel intersection-depth order; one probe at depth 4.10 between them."
        ),
        "fixture": {
            "tilt_degrees": S_D1_TILT_DEGREES, "tilted_scale": S_D1_TILTED_SCALE,
            "flat_centre_z": S_D1_FLAT_CENTRE_Z, "query_depth": S_D1_QUERY_DEPTH,
        },
        "pixel_inversion_count": int(package["pixel_inversion_count"].reshape(-1)[flat].item()),
        "pixel_max_backward_jump": float(package["pixel_max_backward_jump"].reshape(-1)[flat].item()),
        "probe": {
            "query_T_pre": float(package["query_T"][centre, centre, 0].item()),
            "terminated": int(package["query_terminated"][centre, centre, 0].item()),
            "reached": int(package["query_reached"][centre, centre, 0].item()),
            "accepted_prefix_count": int(package["query_prefix_count"][centre, centre, 0].item()),
            "resolution_event_depth": float(package["query_resolution_depth"][centre, centre, 0].item()),
            "late_front_count": int(package["query_late_front_count"][centre, centre, 0].item()),
        },
        "median_depth": float(package["out_others"][5].reshape(-1)[flat].item()),
        "accepted_contributor_count_at_pixel": int(package["contrib_count"].reshape(-1)[flat].item()),
    }


def build_s_c1(device: str = "cuda") -> dict[str, Any]:
    """S-C1: one surface made of several overlapping footprints, probed at the
    renderer's own median event on it. Camera-nearest and query-nearest blockers
    are then separated by construction, with a known world thickness."""

    from .candidate_c_geometric_visibility import GeometricSceneSupport
    from .shared import canonical_geometric_support_rho_max, project_queries
    from .value_diagnostics import candidate_c_blocker_values

    layers = [S_C1_LAYER_SPACING * i for i in range(S_C1_LAYERS)]
    model = make_plane_stack(layers, opacity=S_C1_LAYER_OPACITY, device=device)
    camera = front_camera(device)
    centre = IMAGE // 2
    event, provenance = surface_event_world_point(model, camera, centre, centre)

    with torch.no_grad():
        rotation = model.get_rotation_matrix
        scaling = model.get_scaling
        support = GeometricSceneSupport(
            centers=model.get_xyz.detach(), normals=rotation[:, :, 2].detach(),
            tangent_u=rotation[:, :, 0].detach(), tangent_v=rotation[:, :, 1].detach(),
            scale_u=scaling[:, 0].detach(), scale_v=scaling[:, 1].detach(),
            rho_max=canonical_geometric_support_rho_max(model.get_opacity.detach()),
            opacity=model.get_opacity.detach().reshape(-1),
        )
    positions = event.reshape(1, 3)
    geometry = project_queries(camera, positions)
    blocker = candidate_c_blocker_values(
        geometry, positions, camera.camera_center, camera.world_view_transform, support, None, None,
    )
    return {
        "name": "S_C1_same_surface_overlapping_footprints",
        "description": (
            f"{S_C1_LAYERS} overlapping opacity-{S_C1_LAYER_OPACITY} footprints spaced {S_C1_LAYER_SPACING} apart "
            "forming ONE surface; the query is the renderer's own median event on that surface."
        ),
        "fixture": {"layers": S_C1_LAYERS, "spacing": S_C1_LAYER_SPACING, "opacity": S_C1_LAYER_OPACITY},
        "median_event": provenance,
        "blocker_count": int(blocker["blocker_count"][0].item()),
        "camera_nearest_blocker_t": float(blocker["camera_nearest_blocker_t"][0].item()),
        "query_nearest_blocker_t": float(blocker["query_nearest_blocker_t"][0].item()),
        "camera_nearest_blocker_world_gap": float(blocker["camera_nearest_blocker_world_gap"][0].item()),
        "query_nearest_blocker_world_gap": float(blocker["query_nearest_blocker_world_gap"][0].item()),
        "blocker_region_thickness": float(blocker["blocker_region_thickness"][0].item()),
        "camera_nearest_blocker_surfel": int(blocker["camera_nearest_blocker_surfel"][0].item()),
        "query_nearest_blocker_surfel": int(blocker["query_nearest_blocker_surfel"][0].item()),
    }


def build_s_b1(device: str = "cuda") -> dict[str, Any]:
    """S-B1: the exact median event reconstructed into 3D and projected back.
    Measures the float32 `query_depth - median_depth` discrepancy. NO tolerance
    is introduced anywhere -- the number is reported, not repaired."""

    from .shared import project_queries

    records = []
    for label, offset_pixels in (("on_pixel_centre", 0.0), ("half_pixel_offset", 0.5)):
        offset = pixel_center_world_offset(IMAGE // 2 + offset_pixels, PLANE_DEPTH)
        model = make_plane_stack([0.0], x_offset=offset, device=device)
        camera = front_camera(device)
        centre = IMAGE // 2
        event, provenance = surface_event_world_point(model, camera, centre, centre)
        geometry = project_queries(camera, event.reshape(1, 3))
        query_depth = float(geometry.depth[0].item())
        median = provenance["median_depth"]
        records.append({
            "case": label,
            "branch": provenance["branch"],
            "query_depth": query_depth,
            "median_depth": median,
            "absolute_delta": query_depth - median,
            "relative_delta": (query_depth - median) / median if median else float("nan"),
            "reprojected_pixel_row": int(geometry.pixel_row[0].item()),
            "reprojected_pixel_col": int(geometry.pixel_col[0].item()),
        })
    return {
        "name": "S_B1_median_event_roundtrip_float32_discrepancy",
        "description": (
            "The renderer-defined median-surface event under the canonical pre-update T > 0.5 rule, "
            "reconstructed into 3D (worklog 119 G2) and projected back into the same camera."
        ),
        "cases": records,
        "note": "Reported, never repaired: no tolerance is added to candidate B anywhere in this batch.",
    }


def run_value_contracts(device: str = "cuda") -> dict[str, Any]:
    return {
        "S_D1": build_s_d1(device=device),
        "S_C1": build_s_c1(device=device),
        "S_B1": build_s_b1(device=device),
    }
