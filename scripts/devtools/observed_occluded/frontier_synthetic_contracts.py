from __future__ import annotations

"""Worklog 122 -- SYNTHETIC KNOWN-GEOMETRY CONTRACTS S1-S5 for the candidate B
median frontier (directive section 9).

Real-scene renderer self-consistency alone does not prove physical validity, so
these fixtures supply semantic ground truth on controlled OPAQUE surfaces.
Geometry and expectations are fixed here before any result direction is read.
Candidate B's decision function is called unmodified; no tolerance is added.

  S1  single exposed surface: camera-side OBSERVED, frontier OBSERVED,
      behind-surface OCCLUDED.
  S2  fully hidden rear surface, hidden from EVERY relevant camera -> global
      OCCLUDED.
  S3  rear surface hidden in view A but directly exposed in view B -> global
      OBSERVED (frozen aggregation, no view-count threshold).
  S4  one opaque physical surface represented by MANY overlapping soft splats:
      does the frontier stay on the intended surface while post-median
      contribution remains representation redundancy?
  S5  two genuinely distinct depth layers: where do first contributor, median
      event, post-median contribution and termination fall, and does B place the
      frontier on a semantically acceptable visible layer? Median is NOT
      required to equal the physical first hit.

A genuinely translucent fixture is included ONLY as an explicitly labelled
OUT-OF-SCOPE / AMBIGUOUS semantic control. The architecture is not redefined
around transparency in this batch.
"""

from typing import Any

import torch

from . import candidate_b_median_depth as candidate_b
from .frontier_validation import POST_MEDIAN_CATEGORIES
from .shared import STATE_NAMES, STATE_OBSERVED, STATE_OCCLUDED, aggregate_global, project_queries
from .synthetic_contracts import (
    IMAGE, OPAQUE, PLANE_DEPTH, back_camera, front_camera, make_plane_stack,
    surface_event_world_point,
)

# Fixed a priori.
OPAQUE_STACK = [0.0, 0.02, 0.04, 0.06]      # canonically opaque in the renderer's own terms
S4_SPLAT_COUNT = 12                          # one physical surface, many soft overlapping splats
S4_SPLAT_SPACING = 0.004                     # total thickness 0.044, far below the layer gap in S5
S4_SPLAT_OPACITY = 0.25
S5_LAYER_GAP = 1.5                           # two genuinely distinct surfaces
S5_LAYER_OPACITY = 0.9


def _classify(model: Any, cameras: list[Any], positions: torch.Tensor) -> dict[str, Any]:
    """Candidate B only, called through its own unmodified decision function."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    import numpy as np

    count = int(positions.shape[0])
    per_view = np.full((count, len(cameras)), -1, dtype=np.int8)
    margins: list[list[float]] = []
    medians: list[list[float]] = []
    for view_index, camera in enumerate(cameras):
        package = render_with_query_depth_probe(camera, model, query_depths=None)
        geometry = project_queries(camera, positions)
        result = candidate_b.classify_view(geometry, candidate_b.median_depth_map(package["out_others"]).reshape(-1))
        per_view[:, view_index] = result["states"].detach().cpu().numpy()
        medians.append([float(v) for v in result["median_depth"].detach().cpu().tolist()])
        margins.append([float(a - b) for a, b in zip(geometry.depth.tolist(), result["median_depth"].tolist())])
        del package
    global_states = aggregate_global(per_view)
    return {
        "per_view": [[STATE_NAMES[int(v)] for v in row] for row in per_view],
        "global": [STATE_NAMES[int(v)] for v in global_states],
        "median_depth_per_view": medians,
        "signed_margin_per_view": margins,
    }


def build_s1(device: str = "cuda") -> dict[str, Any]:
    model = make_plane_stack(OPAQUE_STACK, device=device)
    camera = front_camera(device)
    centre = IMAGE // 2
    event, provenance = surface_event_world_point(model, camera, centre, centre)
    origin = camera.camera_center.reshape(3)
    direction = event.reshape(3) - origin
    positions = torch.stack([
        origin + direction * 0.25,   # camera-side free space
        origin + direction * 0.75,   # camera-side free space, closer to the surface
        event.reshape(3),            # the frontier event itself
        origin + direction * 1.25,   # behind the surface
        origin + direction * 2.00,   # far behind the surface
    ])
    labels = ["free_space_25pct", "free_space_75pct", "frontier_event", "behind_125pct", "behind_200pct"]
    expected = ["OBSERVED", "OBSERVED", "OBSERVED", "OCCLUDED", "OCCLUDED"]
    outcome = _classify(model, [camera], positions)
    return {
        "name": "S1_single_exposed_surface",
        "description": "camera -> free space -> canonically opaque surface.",
        "median_event": provenance,
        "queries": [
            {"label": label, "expected_global": want, "actual_global": got,
             "matches": got == want, "signed_margin": outcome["signed_margin_per_view"][0][index],
             "median_depth": outcome["median_depth_per_view"][0][index]}
            for index, (label, want, got) in enumerate(zip(labels, expected, outcome["global"]))
        ],
        "pass": all(got == want for want, got in zip(expected, outcome["global"])),
    }


def build_s2(device: str = "cuda") -> dict[str, Any]:
    """Rear surface hidden behind an opaque foreground from EVERY relevant view."""

    layers = OPAQUE_STACK + [1.5, 1.52, 1.54, 1.56]
    model = make_plane_stack(layers, device=device)
    cameras = [front_camera(device, name="front_a")]
    # A second front camera, laterally offset, still sees the foreground first.
    rotation = torch.eye(3, dtype=torch.float32)
    from .synthetic_contracts import make_camera

    cameras.append(make_camera(rotation, torch.tensor([0.35, 0.0, PLANE_DEPTH]), "front_b", device))
    rear = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 1.56], [0.0, 0.0, 2.0]], dtype=torch.float32, device=device)
    outcome = _classify(model, cameras, rear)
    labels = ["rear_surface_front_face", "rear_surface_back_face", "behind_rear_surface"]
    expected = ["OCCLUDED"] * 3
    return {
        "name": "S2_fully_hidden_rear_surface",
        "description": "opaque foreground stack in front of a rear surface, from every relevant camera.",
        "camera_count": len(cameras),
        "queries": [
            {"label": label, "expected_global": want, "actual_global": got, "matches": got == want,
             "per_view": outcome["per_view"][index],
             "signed_margin_per_view": [outcome["signed_margin_per_view"][v][index] for v in range(len(cameras))]}
            for index, (label, want, got) in enumerate(zip(labels, expected, outcome["global"]))
        ],
        "pass": all(got == want for want, got in zip(expected, outcome["global"])),
    }


def build_s3(device: str = "cuda") -> dict[str, Any]:
    """Rear surface hidden from the front camera, directly exposed to a camera
    on the far side. Frozen aggregation must return OBSERVED."""

    model = make_plane_stack(OPAQUE_STACK + [1.5], device=device)
    front = front_camera(device, name="front_blocked")
    back = back_camera(device, distance=4.0, name="back_exposed")
    centre = IMAGE // 2
    event, provenance = surface_event_world_point(model, back, centre, centre)
    outcome = _classify(model, [front, back], event.reshape(1, 3))
    return {
        "name": "S3_cross_view_disocclusion",
        "description": "rear surface occluded from the front camera, its own median event from the back camera.",
        "median_event_from_back_camera": provenance,
        "per_view": outcome["per_view"][0],
        "expected_global": "OBSERVED",
        "actual_global": outcome["global"][0],
        "signed_margin_per_view": [outcome["signed_margin_per_view"][v][0] for v in range(2)],
        "pass": outcome["global"][0] == "OBSERVED",
    }


def build_s4(device: str = "cuda") -> dict[str, Any]:
    """ONE opaque physical surface represented by many overlapping soft splats.
    Measures whether the frontier stays on that surface and how much
    contribution sits behind it as representation redundancy."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    layers = [S4_SPLAT_SPACING * i for i in range(S4_SPLAT_COUNT)]
    model = make_plane_stack(layers, opacity=S4_SPLAT_OPACITY, device=device)
    camera = front_camera(device)
    centre = IMAGE // 2
    flat = centre * IMAGE + centre
    # Every splat belongs to the SAME physical surface, so give them one component.
    component = torch.zeros((len(model),), dtype=torch.int32, device=device)
    package = render_with_query_depth_probe(camera, model, query_depths=None, primitive_component=component)
    counts = package["post_median_counts"].reshape(-1, len(POST_MEDIAN_CATEGORIES))[flat].tolist()
    weights = package["post_median_weights"].reshape(-1, len(POST_MEDIAN_CATEGORIES))[flat].tolist()
    total = float(package["total_accepted_weight"].reshape(-1)[flat].item())
    median_depth = float(package["out_others"][5].reshape(-1)[flat].item())
    surface_span = [PLANE_DEPTH, PLANE_DEPTH + S4_SPLAT_SPACING * (S4_SPLAT_COUNT - 1)]
    return {
        "name": "S4_overlapping_splats_one_opaque_surface",
        "description": f"{S4_SPLAT_COUNT} overlapping opacity-{S4_SPLAT_OPACITY} splats spanning {surface_span[1] - surface_span[0]:.3f} depth, one physical surface.",
        "physical_surface_depth_span": surface_span,
        "median_depth": median_depth,
        "frontier_inside_physical_surface_span": bool(surface_span[0] <= median_depth <= surface_span[1]),
        "accepted_contributors": int(package["contrib_count"].reshape(-1)[flat].item()),
        "post_median_counts": {name: int(v) for name, v in zip(
            POST_MEDIAN_CATEGORIES, counts)},
        "post_median_mass": weights[0],
        "total_accepted_mass": total,
        "post_median_fraction_of_contribution": weights[0] / max(total, 1e-20),
        "post_median_same_component_share": weights[1] / max(weights[0], 1e-20),
        "post_median_cross_component_share": weights[2] / max(weights[0], 1e-20),
        "interpretation": (
            "All post-median contribution here is same-component by construction -- it is redundant "
            "representation of ONE surface, not independent visible-surface evidence behind the frontier."
        ),
    }


def build_s5(device: str = "cuda") -> dict[str, Any]:
    """Two genuinely distinct opaque layers, separated far beyond any single
    splat's extent. Reports where first contributor / median / post-median /
    termination fall. Median is NOT required to be the physical first hit."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import MAX_QUERY_SLOTS, render_with_query_depth_probe

    front_layer = [0.0, 0.02]
    rear_layer = [S5_LAYER_GAP, S5_LAYER_GAP + 0.02]
    model = make_plane_stack(front_layer + rear_layer, opacity=S5_LAYER_OPACITY, device=device)
    camera = front_camera(device)
    centre = IMAGE // 2
    flat = centre * IMAGE + centre
    component = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    query = torch.zeros((IMAGE, IMAGE, MAX_QUERY_SLOTS), dtype=torch.float32, device=device)
    query[centre, centre, 0] = PLANE_DEPTH + S5_LAYER_GAP + 0.5  # behind the rear layer
    package = render_with_query_depth_probe(camera, model, query_depths=query, primitive_component=component)
    counts = package["post_median_counts"].reshape(-1, len(POST_MEDIAN_CATEGORIES))[flat].tolist()
    weights = package["post_median_weights"].reshape(-1, len(POST_MEDIAN_CATEGORIES))[flat].tolist()
    total = float(package["total_accepted_weight"].reshape(-1)[flat].item())
    median_depth = float(package["out_others"][5].reshape(-1)[flat].item())
    probes = torch.tensor([
        [0.0, 0.0, -0.5],                     # in front of the near layer
        [0.0, 0.0, S5_LAYER_GAP * 0.5],       # free space BETWEEN the two layers
        [0.0, 0.0, S5_LAYER_GAP + 0.5],       # behind the rear layer
    ], dtype=torch.float32, device=device)
    outcome = _classify(model, [camera], probes)
    return {
        "name": "S5_two_distinct_depth_layers",
        "description": f"two opaque layers separated by {S5_LAYER_GAP} world units.",
        "near_layer_camera_depth": PLANE_DEPTH,
        "rear_layer_camera_depth": PLANE_DEPTH + S5_LAYER_GAP,
        "first_contributor_depth": PLANE_DEPTH,
        "median_depth": median_depth,
        "median_on_near_layer": bool(abs(median_depth - PLANE_DEPTH) < S5_LAYER_GAP * 0.5),
        "termination_probe": {
            "query_depth": float(query[centre, centre, 0].item()),
            "terminated": int(package["query_terminated"][centre, centre, 0].item()),
            "T_pre": float(package["query_T"][centre, centre, 0].item()),
            "termination_alpha": float(package["query_termination_alpha"][centre, centre, 0].item()),
        },
        "post_median_counts": {name: int(v) for name, v in zip(
            POST_MEDIAN_CATEGORIES, counts)},
        "post_median_fraction_of_contribution": weights[0] / max(total, 1e-20),
        "post_median_cross_component_share": weights[2] / max(weights[0], 1e-20),
        "probes": [
            {"label": label, "global": outcome["global"][index],
             "signed_margin": outcome["signed_margin_per_view"][0][index]}
            for index, label in enumerate(["in_front_of_near_layer", "free_space_between_layers", "behind_rear_layer"])
        ],
        "interpretation_note": (
            "B places the frontier on the NEAR visible layer. The free space BETWEEN the layers is classified "
            "OCCLUDED, which is correct for a surface-observation frontier (that space is behind the visible "
            "surface from this camera) and is NOT a claim about physical emptiness."
        ),
    }


def build_translucent_control(device: str = "cuda") -> dict[str, Any]:
    """EXPLICIT OUT-OF-SCOPE / AMBIGUOUS semantic control. A genuinely
    translucent front sheet in front of an opaque surface. Reported only; the
    architecture is NOT redefined around transparency in this batch."""

    model = make_plane_stack([0.0, 1.0, 1.02, 1.04], opacity=0.12, device=device)
    with torch.no_grad():
        opacity = model.get_opacity.detach().clone()
        opacity[1:] = OPAQUE
        model._opacity.data.copy_(torch.log(opacity / (1 - opacity)))
    camera = front_camera(device)
    probes = torch.tensor([
        [0.0, 0.0, 0.5],   # between the translucent sheet and the opaque surface
        [0.0, 0.0, 1.5],   # behind the opaque surface
    ], dtype=torch.float32, device=device)
    outcome = _classify(model, [camera], probes)
    return {
        "name": "OUT_OF_SCOPE_translucent_front_sheet",
        "status": "OUT-OF-SCOPE / AMBIGUOUS SEMANTIC CONTROL",
        "description": "opacity-0.12 sheet in front of an opaque surface; the paper's scene model does not claim transparency support.",
        "probes": [
            {"label": label, "global": outcome["global"][index],
             "signed_margin": outcome["signed_margin_per_view"][0][index],
             "median_depth": outcome["median_depth_per_view"][0][index]}
            for index, label in enumerate(["between_sheet_and_surface", "behind_opaque_surface"])
        ],
        "note": "Reported only. No architecture decision is derived from this fixture.",
    }


def run_frontier_contracts(device: str = "cuda") -> dict[str, Any]:
    return {
        "S1": build_s1(device=device),
        "S2": build_s2(device=device),
        "S3": build_s3(device=device),
        "S4": build_s4(device=device),
        "S5": build_s5(device=device),
        "OUT_OF_SCOPE_translucent": build_translucent_control(device=device),
    }
