from __future__ import annotations

"""Worklog 120 -- SHARED synthetic contract fixtures S1-S7 (directive 9A).

Minimal controlled scenes whose visibility/occlusion semantics are analytically
known. Every fixture is built ANALYTICALLY and its expectations written down
BEFORE any candidate is run; nothing here was adjusted after seeing which
candidate performed better (directive section 9A, S6).

Two separate expectation columns are recorded for every contract, and they are
never conflated:

  `expected_global`  -- the DIRECTIVE's paper-level expectation (S1-S5 only).
                        A candidate missing it is a semantic finding about the
                        hypothesis, not necessarily an implementation defect.
  `predicted_<X>`    -- what candidate X's OWN written contract predicts, worked
                        out by hand from that module's docstring. A candidate
                        missing THIS is an implementation-fidelity defect and is
                        asserted in `tests/test_observed_occluded_volumetric_audit.py`.

Synthetic PASS establishes semantic correctness only. It does NOT prove
architecture viability (directive section 9A).
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .shared import STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED

IMAGE = 64
FOV = 0.7
# Canonical camera-space depth of the primary surface plane in every fixture.
PLANE_DEPTH = 4.0
# Opacity that reaches the kernel's own alpha ceiling at the ray centre, so a
# stacked blocker is opaque in the canonical renderer's own terms.
OPAQUE = 0.99


def _projection_matrix(device: str) -> torch.Tensor:
    from osn_gs.data.colmap_scene import projection_matrix

    return projection_matrix(0.01, 100.0, FOV, FOV, device=device)


def make_camera(rotation: torch.Tensor, translation: torch.Tensor, name: str, device: str) -> Any:
    """Graphdeco/OSN-GS transposed-matrix camera, identical convention to
    `osn_gs.data.colmap_scene.camera_matrices`."""

    from osn_gs.render.torch_fallback import TorchCamera

    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    world_view = world_view.transpose(0, 1).contiguous().to(device)
    projection = _projection_matrix(device).transpose(0, 1).contiguous()
    full_proj = (world_view.unsqueeze(0) @ projection.unsqueeze(0)).squeeze(0)
    center = (-rotation.T @ translation).to(device)
    return TorchCamera(
        image_height=IMAGE, image_width=IMAGE, world_view_transform=world_view,
        full_proj_transform=full_proj, camera_center=center, FoVx=FOV, FoVy=FOV, image_name=name,
    )


def front_camera(device: str, name: str = "front") -> Any:
    """At world (0, 0, -PLANE_DEPTH), looking along +z. A world point at z = Z
    has camera-space depth Z + PLANE_DEPTH."""

    return make_camera(torch.eye(3, dtype=torch.float32), torch.tensor([0.0, 0.0, PLANE_DEPTH]), name, device)


def back_camera(device: str, distance: float = PLANE_DEPTH, name: str = "back") -> Any:
    """At world (0, 0, +distance), looking along -z (180 degrees about y)."""

    rotation = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32)
    return make_camera(rotation, torch.tensor([0.0, 0.0, distance]), name, device)


def make_plane_stack(
    z_values: list[float], *, scale: float = 1.0, opacity: float = OPAQUE,
    x_offset: float = 0.0, device: str = "cuda", rotation_wxyz: tuple[float, float, float, float] | None = None,
) -> Any:
    """A model of axis-aligned surfels, one per z, all facing the front camera."""

    from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel

    count = len(z_values)
    model = TorchGaussianSurfelModel(sh_degree=0, device=device)
    quaternion = rotation_wxyz if rotation_wxyz is not None else (1.0, 0.0, 0.0, 0.0)
    model.initialize(
        positions=torch.tensor([[x_offset, 0.0, z] for z in z_values], dtype=torch.float32),
        colors=torch.tensor([[0.6, 0.5, 0.4]] * count, dtype=torch.float32),
        opacities=torch.tensor([[opacity]] * count, dtype=torch.float32),
        scales=torch.tensor([[scale, scale]] * count, dtype=torch.float32),
        rotations=torch.tensor([list(quaternion)] * count, dtype=torch.float32),
    )
    model.active_sh_degree = 0
    return model


def pixel_center_world_offset(pixel_coordinate: float, depth: float) -> float:
    """World-space x of the ray through continuous pixel coordinate
    `pixel_coordinate` at camera-space depth `depth`, for the front camera.
    Inverts the rasterizer's own `ndc2Pix`."""

    ndc = (2.0 * pixel_coordinate + 1.0) / IMAGE - 1.0
    return ndc * math.tan(FOV * 0.5) * depth


def surface_event_world_point(model: Any, camera: Any, row: int, col: int) -> tuple[torch.Tensor, dict[str, float]]:
    """The renderer's own median surface event at one pixel, as a world point
    (worklog 119 G2), plus its rho3d/rho2d provenance."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import render_with_query_depth_probe

    from .shared import reconstruct_direct_surfel_intersection_world_point

    package = render_with_query_depth_probe(camera, model, query_depths=None)
    representative = package["representative_id"].reshape(-1).to(torch.int64)
    with torch.no_grad():
        rotation = model.get_rotation_matrix
        scaling = model.get_scaling
        world = reconstruct_direct_surfel_intersection_world_point(
            representative, package["median_s_u"], package["median_s_v"],
            model.get_xyz.detach(), rotation[:, :, 0].detach(), rotation[:, :, 1].detach(),
            scaling[:, 0].detach(), scaling[:, 1].detach(),
        )
    flat = row * IMAGE + col
    provenance = {
        "representative_id": int(representative[flat].item()),
        "median_rho3d": float(package["median_rho3d"].reshape(-1)[flat].item()),
        "median_rho2d": float(package["median_rho2d"].reshape(-1)[flat].item()),
        "median_depth": float(package["out_others"][5].reshape(-1)[flat].item()),
        "median_s_u": float(package["median_s_u"].reshape(-1)[flat].item()),
        "median_s_v": float(package["median_s_v"].reshape(-1)[flat].item()),
    }
    provenance["branch"] = "rho3d" if provenance["median_rho3d"] <= provenance["median_rho2d"] else "rho2d"
    return world[flat].clone(), provenance


@dataclass
class SyntheticContract:
    name: str
    description: str
    model: Any
    cameras: list[Any]
    positions: torch.Tensor
    query_labels: list[str]
    expected_global: list[int | None]
    predicted: dict[str, list[int]]
    provenance: dict[str, Any] = field(default_factory=dict)


def build_contracts(device: str = "cuda") -> list[SyntheticContract]:
    """All seven contracts. Fixture geometry and every expectation below are
    fixed a priori; see this module's docstring."""

    contracts: list[SyntheticContract] = []
    centre = IMAGE // 2

    # ---------------------------------------------------------------- S1
    model = make_plane_stack([0.0], device=device)
    camera = front_camera(device)
    event, provenance = surface_event_world_point(model, camera, centre, centre)
    contracts.append(SyntheticContract(
        name="S1_directly_exposed_surface",
        description="Camera -> Surface x. x IS the renderer's own surface event.",
        model=model, cameras=[camera], positions=event.reshape(1, 3),
        query_labels=["surface_event"],
        expected_global=[STATE_OBSERVED],
        predicted={"A": [STATE_OBSERVED], "B": [STATE_OBSERVED], "C": [STATE_OBSERVED], "D": [STATE_OBSERVED]},
        provenance={"event": provenance},
    ))

    # ---------------------------------------------------------------- S2
    # Camera -> x -> Surface: x is exposed empty space in front of the surface.
    # A is EXPECTED to fail this: a surface-only mechanism has no way to call
    # free space observed. That failure is the measurement (directive 10E).
    model = make_plane_stack([0.0], device=device)
    camera = front_camera(device)
    event, provenance = surface_event_world_point(model, camera, centre, centre)
    halfway = (camera.camera_center.reshape(3) + event.reshape(3)) * 0.5
    contracts.append(SyntheticContract(
        name="S2_exposed_free_space_before_surface",
        description="Camera -> x -> Surface. x is exposed empty space, halfway along the ray.",
        model=model, cameras=[camera], positions=halfway.reshape(1, 3),
        query_labels=["free_space_midpoint"],
        expected_global=[STATE_OBSERVED],
        predicted={"A": [STATE_UNRESOLVED], "B": [STATE_OBSERVED], "C": [STATE_OBSERVED], "D": [STATE_OBSERVED]},
        provenance={"event": provenance},
    ))

    # ---------------------------------------------------------------- S3a
    # Canonically opaque blocker: four stacked opacity-0.99 surfels, enough for
    # the kernel's own `T * (1 - alpha) < 0.0001` termination to fire.
    model = make_plane_stack([0.0, 0.02, 0.04, 0.06], device=device)
    camera = front_camera(device)
    behind = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device=device)
    contracts.append(SyntheticContract(
        name="S3a_behind_canonically_opaque_blocker",
        description="Camera -> Foreground(4 stacked opaque surfels, canonical traversal terminates) -> x.",
        model=model, cameras=[camera], positions=behind,
        query_labels=["behind_opaque_stack"],
        expected_global=[STATE_OCCLUDED],
        predicted={"A": [STATE_OCCLUDED], "B": [STATE_OCCLUDED], "C": [STATE_OCCLUDED], "D": [STATE_OCCLUDED]},
    ))

    # ---------------------------------------------------------------- S3b
    # A priori companion (NOT a post-hoc variant): the same geometry with a
    # SINGLE opacity-0.99 surfel, which is a blocker geometrically but leaves
    # T = 0.01 -- canonical traversal never terminates. Declared before running.
    model = make_plane_stack([0.0], device=device)
    camera = front_camera(device)
    contracts.append(SyntheticContract(
        name="S3b_behind_single_primitive_blocker",
        description="Camera -> Foreground(1 surfel, T stays 0.01, canonical traversal never terminates) -> x.",
        model=model, cameras=[camera], positions=behind.clone(),
        query_labels=["behind_single_surfel"],
        expected_global=[STATE_OCCLUDED],
        predicted={"A": [STATE_OCCLUDED], "B": [STATE_OCCLUDED], "C": [STATE_OCCLUDED], "D": [STATE_OBSERVED]},
    ))

    # ---------------------------------------------------------------- S4
    # Cross-view disocclusion. Target surface at z = 0; blocker slab at z = +2
    # sits between the BACK camera and the target, and behind the target from
    # the FRONT camera. x is the front camera's own surface event on the target.
    model = make_plane_stack([0.0, 2.0, 2.02, 2.04, 2.06], device=device)
    front = front_camera(device, name="front_sees_x")
    back = back_camera(device, distance=6.0, name="back_blocked")
    event, provenance = surface_event_world_point(model, front, centre, centre)
    contracts.append(SyntheticContract(
        name="S4_cross_view_disocclusion",
        description="View A: Foreground -> x (occluded). View B: x directly exposed. GLOBAL must be OBSERVED.",
        model=model, cameras=[front, back], positions=event.reshape(1, 3),
        query_labels=["disoccluded_surface_event"],
        expected_global=[STATE_OBSERVED],
        predicted={"A": [STATE_OBSERVED], "B": [STATE_OBSERVED], "C": [STATE_OBSERVED], "D": [STATE_OBSERVED]},
        provenance={"event": provenance},
    ))

    # ---------------------------------------------------------------- S5
    model = make_plane_stack([0.0], device=device)
    camera = front_camera(device)
    outside = torch.tensor([[100.0, 100.0, 0.0], [0.0, 0.0, -100.0]], dtype=torch.float32, device=device)
    contracts.append(SyntheticContract(
        name="S5_outside_camera_support",
        description="Queries with no relevant view at all (far outside the frustum; behind the camera).",
        model=model, cameras=[camera], positions=outside,
        query_labels=["far_off_axis", "behind_camera"],
        expected_global=[STATE_UNRESOLVED, STATE_UNRESOLVED],
        predicted={name: [STATE_UNRESOLVED, STATE_UNRESOLVED] for name in ("A", "B", "C", "D")},
    ))

    # ---------------------------------------------------------------- S6
    # Layered soft compositing: first contributor, median crossing and canonical
    # termination land at three DISTINCT depths. With alpha = 0.3 per layer,
    # T = 0.7^n: the median (T > 0.5) is still set at layer 2 (T = 0.7) but not
    # layer 3 (T = 0.49), and `T * (1 - alpha) < 0.0001` first fires at layer 26.
    layers = [0.05 * i for i in range(30)]
    model = make_plane_stack(layers, opacity=0.3, device=device)
    camera = front_camera(device)
    probes = torch.tensor(
        [[0.0, 0.0, 0.02], [0.0, 0.0, 0.60], [0.0, 0.0, 2.00]], dtype=torch.float32, device=device
    )
    contracts.append(SyntheticContract(
        name="S6_layered_soft_compositing",
        description=(
            "30 alpha-0.3 layers: first contributor at depth 4.00, median crossing at 4.05, canonical "
            "termination at 5.25. Probe depths 4.02 / 4.60 / 6.00 straddle all three."
        ),
        model=model, cameras=[camera], positions=probes,
        query_labels=["between_first_and_median", "between_median_and_termination", "past_termination"],
        expected_global=[None, None, None],
        predicted={
            "A": [STATE_UNRESOLVED, STATE_OCCLUDED, STATE_OCCLUDED],
            "B": [STATE_OBSERVED, STATE_OCCLUDED, STATE_OCCLUDED],
            "C": [STATE_OCCLUDED, STATE_OCCLUDED, STATE_OCCLUDED],
            "D": [STATE_OBSERVED, STATE_OBSERVED, STATE_OCCLUDED],
        },
    ))

    # ---------------------------------------------------------------- S7
    # Low-pass vs true-footprint provenance, kept in two SEPARATE scenes so the
    # two event kinds can never contaminate each other. Both surfels sit at the
    # same world position, projecting to continuous pixel coordinate centre+0.5,
    # so the nearest pixel is exactly half a pixel away: rho2d = 2 * 0.5^2 = 0.5,
    # while rho3d = (half-pixel world offset / scale)^2 -- tiny for the large
    # surfel, enormous for the sub-pixel one.
    offset = pixel_center_world_offset(centre + 0.5, PLANE_DEPTH)
    for label, scale in (("rho3d_true_footprint", 1.0), ("rho2d_low_pass", 0.0005)):
        model = make_plane_stack([0.0], scale=scale, x_offset=offset, device=device)
        camera = front_camera(device)
        event, provenance = surface_event_world_point(model, camera, centre, centre)
        behind_event = camera.camera_center.reshape(3) + (event.reshape(3) - camera.camera_center.reshape(3)) * 1.2
        contracts.append(SyntheticContract(
            name=f"S7_{label}",
            description=f"Isolated {label} median event (scale={scale}); the event itself and a probe 20% further along its ray.",
            model=model, cameras=[camera],
            positions=torch.stack([event.reshape(3), behind_event]),
            query_labels=["event_itself", "behind_event"],
            expected_global=[STATE_OBSERVED, None],
            predicted={
                "A": [STATE_OBSERVED, STATE_OCCLUDED],
                "B": [STATE_OBSERVED, STATE_OCCLUDED],
                # C's `behind_event` prediction is branch-dependent, and this is
                # the ONE place where the a-priori hand-derivation was corrected
                # -- with the original recorded, not overwritten. First pass
                # predicted OCCLUDED for both branches; that is wrong for the
                # rho2d scene, because a sub-pixel surfel accepted ONLY through
                # the screen-space low-pass has its true geometric support
                # (rho3d <= rho_max) nowhere near the ray, so a purely geometric
                # line-of-sight test correctly finds nothing there. See the
                # worklog's Synthetic Contracts section.
                "C": [STATE_OBSERVED, STATE_OCCLUDED if scale > 0.01 else STATE_OBSERVED],
                "D": [STATE_OBSERVED, STATE_OBSERVED],
            },
            provenance={
                "event": provenance,
                "scale": scale,
                "prediction_note": (
                    None if scale > 0.01 else
                    "C/behind_event: a-priori hand-derivation said OCCLUDED; corrected to OBSERVED before the "
                    "real-scene run. The correction is to the PREDICTION, not to the fixture or the implementation "
                    "-- a rho2d-only median event has no geometric support on the ray at all."
                ),
            },
        ))

    return contracts


def run_contracts(device: str = "cuda", progress=None) -> dict[str, Any]:
    """Run every contract through the SAME shared engine the real scene uses."""

    from .engine import evaluate
    from .shared import STATE_NAMES

    results: dict[str, Any] = {}
    for contract in build_contracts(device=device):
        outcome = evaluate(contract.model, contract.cameras, contract.positions, progress=None)
        entry: dict[str, Any] = {
            "description": contract.description,
            "camera_count": len(contract.cameras),
            "queries": [],
            "provenance": contract.provenance,
        }
        for query_index, label in enumerate(contract.query_labels):
            record: dict[str, Any] = {
                "query": label,
                "world_position": [float(v) for v in contract.positions[query_index].tolist()],
                "expected_global_directive": (
                    STATE_NAMES[contract.expected_global[query_index]]
                    if contract.expected_global[query_index] is not None else None
                ),
                "per_view_relevance": [int(v) for v in outcome.relevance_code[query_index].tolist()],
            }
            for name in ("A", "B", "C", "D"):
                actual = int(outcome.global_states[name][query_index])
                predicted = contract.predicted[name][query_index]
                record[name] = {
                    "global": STATE_NAMES[actual],
                    "per_view": [STATE_NAMES[int(v)] for v in outcome.per_view_states[name][query_index].tolist()],
                    "predicted_by_own_contract": STATE_NAMES[predicted],
                    "matches_own_contract": bool(actual == predicted),
                    "matches_directive_expectation": (
                        None if contract.expected_global[query_index] is None
                        else bool(actual == contract.expected_global[query_index])
                    ),
                }
            record["D_transmittance_at_query"] = [
                float(v) for v in outcome.provenance["D_transmittance"][query_index].tolist()
            ]
            record["B_median_depth"] = [float(v) for v in outcome.provenance["B_median_depth"][query_index].tolist()]
            record["C_blocker_count"] = [int(v) for v in outcome.provenance["C_blocker_count"][query_index].tolist()]
            record["A_hit_distance"] = [float(v) for v in outcome.provenance["A_hit_distance"][query_index].tolist()]
            record["query_depth_per_view"] = [float(v) for v in outcome.query_depth[query_index].tolist()]
            entry["queries"].append(record)
        results[contract.name] = entry
        if progress is not None:
            progress(f"synthetic contract {contract.name} done")
        del contract
    return results


def contract_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Per-contract, per-candidate pass/fail -- never collapsed into one scalar."""

    summary: dict[str, Any] = {}
    for contract_name, entry in results.items():
        per_candidate = {}
        for name in ("A", "B", "C", "D"):
            own = [bool(query[name]["matches_own_contract"]) for query in entry["queries"]]
            directive = [query[name]["matches_directive_expectation"] for query in entry["queries"]]
            directive = [value for value in directive if value is not None]
            per_candidate[name] = {
                "implementation_fidelity_pass": all(own),
                "directive_semantics_pass": (all(directive) if directive else None),
                "states": [query[name]["global"] for query in entry["queries"]],
            }
        summary[contract_name] = per_candidate
    return summary
