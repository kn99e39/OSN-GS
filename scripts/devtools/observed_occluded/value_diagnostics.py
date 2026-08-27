from __future__ import annotations

"""Worklog 121 -- SUPPLEMENTAL VALUE DIAGNOSTICS for the frozen worklog 120
candidates.

This module measures the ACTUAL surface / depth / blocker / transmittance
quantities underlying worklog 120's A/B/C/D decisions. It NEVER decides a state:
every per-view verdict here comes from calling worklog 120's own unmodified
`candidate_a..d.classify_view`, and `evaluate_with_values` asserts bit-identity
against the historical arrays before any value is interpreted.

Three provenance corrections found during independent source review are
implemented here, in the REPORTING layer only:

  1. Worklog 120's `C_nearest_blocker_t` is MAX(t) -- the blocker nearest the
     QUERY, not the camera. This module reports BOTH `camera_nearest_blocker_t`
     (MIN valid t) and `query_nearest_blocker_t` (MAX valid t), plus their world
     gaps, opacities and surfel ids, and never reuses the ambiguous old name.
  2. Candidate C's support is the `rho3d geometric footprint derived from the
     canonical alpha cutoff` -- NOT the renderer's complete contribution
     support, because canonical acceptance is `rho = min(rho3d, rho2d)` and a
     rho2d low-pass event can be accepted outside the rho3d footprint. Wording
     is fixed everywhere; the geometry is unchanged.
  3. Candidate D's probe measures `canonical traversal-order reachability`, and
     `query_T` is the `pre-update traversal transmittance at the recorded
     resolution event`. At a termination event the quantity the kernel compared
     against 1e-4 is `T_pre * (1 - alpha)`, which the worklog 121 probe fields
     now expose directly.

`GeometricSceneSupport`, `SEGMENT_EPSILON` and the rho_max formula are imported
from the frozen candidate C module rather than re-derived, so the value pass and
the decision pass cannot drift apart.
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
    STATE_OCCLUDED,
    aggregate_global,
    assign_query_depth_slots,
    project_queries,
    reconstruct_direct_surfel_intersection_world_point,
)

CANDIDATE_NAMES = ("A", "B", "C", "D")

# Candidate D resolution reasons (directive section 7). Derived purely from the
# probe's own `terminated` / `reached` flags -- no new condition is introduced.
REASON_UNRESOLVED = 0
REASON_REACHED_ACCEPTED_EVENT = 1
REASON_TERMINATED_BEFORE_QUERY = 2
REASON_CONTRIBUTOR_LIST_EXHAUSTED = 3
REASON_NAMES = {
    REASON_UNRESOLVED: "UNRESOLVED",
    REASON_REACHED_ACCEPTED_EVENT: "REACHED_ACCEPTED_EVENT",
    REASON_TERMINATED_BEFORE_QUERY: "TERMINATED_BEFORE_QUERY",
    REASON_CONTRIBUTOR_LIST_EXHAUSTED: "CONTRIBUTOR_LIST_EXHAUSTED",
}

# The canonical kernel's own termination constant, `if (test_T < 0.0001f)` in
# the vendored forward.cu. Used ONLY to verify the recorded contract, never to
# decide anything.
CANONICAL_TERMINATION_TEST_T = 1e-4


@dataclass
class ValueEvaluationResult:
    per_view_states: dict[str, np.ndarray]
    global_states: dict[str, np.ndarray]
    relevance_code: np.ndarray
    query_depth: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)
    view_names: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def d_resolution_reason(terminated: np.ndarray, reached: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Directive section 7's four reasons, read straight off the probe flags."""

    reason = np.full(terminated.shape, REASON_UNRESOLVED, dtype=np.int8)
    written = relevant & (terminated >= 0)
    reason[written & (terminated == 1)] = REASON_TERMINATED_BEFORE_QUERY
    reason[written & (terminated == 0) & (reached == 1)] = REASON_REACHED_ACCEPTED_EVENT
    reason[written & (terminated == 0) & (reached == 0)] = REASON_CONTRIBUTOR_LIST_EXHAUSTED
    return reason


def candidate_c_blocker_values(
    geometry,
    query_positions: torch.Tensor,
    camera_center: torch.Tensor,
    world_view_transform: torch.Tensor,
    support: candidate_c.GeometricSceneSupport,
    component_of_surfel: torch.Tensor | None,
    source_component: torch.Tensor | None,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> dict[str, torch.Tensor]:
    """Corrected blocker provenance for ONE view.

    Reuses candidate C's own `GeometricSceneSupport` and `SEGMENT_EPSILON`
    verbatim, so the blocked set computed here is by construction the same set
    candidate C decided on -- `evaluate_with_values` asserts that equality
    against the frozen decision function's own output.

    Returned per query (NaN / -1 where no blocker or non-relevant):
      camera_nearest_blocker_t / _world_gap / _opacity / _surfel / _component
      query_nearest_blocker_t  / _world_gap / _opacity / _surfel / _component
      blocker_count, max_blocker_opacity, blocker_region_thickness,
      same_component_blocker_count, ray_length

    World gaps are measured ALONG THE RAY toward the query: a blocker at
    parametric t sits `(1 - t) * ||x - camera||` world units in FRONT of the
    query. `blocker_region_thickness` is `(t_max - t_min) * ||x - camera||`.
    """

    device = query_positions.device
    count = int(query_positions.shape[0])

    def _empty(fill: float, dtype=torch.float32):
        return torch.full((count,), fill, dtype=dtype, device=device)

    out = {
        "camera_nearest_blocker_t": _empty(float("nan")),
        "query_nearest_blocker_t": _empty(float("nan")),
        "camera_nearest_blocker_opacity": _empty(float("nan")),
        "query_nearest_blocker_opacity": _empty(float("nan")),
        "camera_nearest_blocker_surfel": _empty(-1, torch.int64),
        "query_nearest_blocker_surfel": _empty(-1, torch.int64),
        "max_blocker_opacity": _empty(float("nan")),
        "blocker_count": _empty(0, torch.int32),
        "same_component_blocker_count": _empty(-1, torch.int32),
        "ray_length": _empty(float("nan")),
    }

    relevant_rows = torch.nonzero(geometry.relevant, as_tuple=False).reshape(-1)
    if relevant_rows.numel() == 0:
        return out

    origin = camera_center.reshape(3).to(torch.float32)
    out["ray_length"][relevant_rows] = (query_positions[relevant_rows] - origin.reshape(1, 3)).norm(dim=1)
    max_depth = float(geometry.depth[relevant_rows].max().item())

    center_view_z = (
        support.centers @ world_view_transform[:3, 2].reshape(3) + world_view_transform[3, 2]
    )
    candidate = support.nonempty & ((center_view_z - support.support_radius) < max_depth)
    candidate_rows = torch.nonzero(candidate, as_tuple=False).reshape(-1)
    out["blocker_count"][relevant_rows] = 0
    if candidate_rows.numel() == 0:
        return out

    normals = support.normals[candidate_rows]
    tangent_u = support.tangent_u[candidate_rows]
    tangent_v = support.tangent_v[candidate_rows]
    inv_su = support.inv_scale_u[candidate_rows]
    inv_sv = support.inv_scale_v[candidate_rows]
    rho_max = support.rho_max[candidate_rows]
    opacity = support.opacity[candidate_rows]
    numerator = support.center_dot_normal[candidate_rows] - (normals @ origin)
    base_u = (origin @ tangent_u.T) - support.center_dot_tu[candidate_rows]
    base_v = (origin @ tangent_v.T) - support.center_dot_tv[candidate_rows]
    candidate_component = (
        component_of_surfel[candidate_rows] if component_of_surfel is not None else None
    )

    primitives = int(candidate_rows.shape[0])
    chunk = max(1, min(int(relevant_rows.shape[0]), chunk_bytes // max(1, 4 * primitives)))
    for start in range(0, int(relevant_rows.shape[0]), chunk):
        rows = relevant_rows[start:start + chunk]
        direction = query_positions[rows] - origin.reshape(1, 3)
        denominator = direction @ normals.T
        t = numerator.unsqueeze(0) / denominator
        hit = (denominator != 0) & (t > candidate_c.SEGMENT_EPSILON) & (t < 1.0 - candidate_c.SEGMENT_EPSILON)
        del denominator
        local = ((base_u.unsqueeze(0) + t * (direction @ tangent_u.T)) * inv_su.unsqueeze(0)) ** 2
        local = local + ((base_v.unsqueeze(0) + t * (direction @ tangent_v.T)) * inv_sv.unsqueeze(0)) ** 2
        hit = hit & (local <= rho_max.unsqueeze(0))
        del local

        counts = hit.sum(dim=1)
        any_hit = counts > 0
        out["blocker_count"][rows] = counts.to(torch.int32)

        # CORRECTION (directive section 6): min(t) is the CAMERA-nearest
        # blocker; max(t) is the QUERY-nearest blocker. Worklog 120 reported
        # only max(t) under the ambiguous name `nearest_blocker_t`.
        t_camera, index_camera = torch.where(hit, t, torch.full_like(t, float("inf"))).min(dim=1)
        t_query, index_query = torch.where(hit, t, torch.full_like(t, float("-inf"))).max(dim=1)
        nan = torch.full_like(t_camera, float("nan"))
        out["camera_nearest_blocker_t"][rows] = torch.where(any_hit, t_camera, nan)
        out["query_nearest_blocker_t"][rows] = torch.where(any_hit, t_query, nan)
        out["camera_nearest_blocker_opacity"][rows] = torch.where(any_hit, opacity[index_camera], nan)
        out["query_nearest_blocker_opacity"][rows] = torch.where(any_hit, opacity[index_query], nan)
        out["max_blocker_opacity"][rows] = torch.where(
            any_hit, torch.where(hit, opacity.unsqueeze(0).expand_as(hit), torch.zeros_like(t)).max(dim=1).values, nan
        )
        minus_one = torch.full_like(index_camera, -1)
        out["camera_nearest_blocker_surfel"][rows] = torch.where(any_hit, candidate_rows[index_camera], minus_one)
        out["query_nearest_blocker_surfel"][rows] = torch.where(any_hit, candidate_rows[index_query], minus_one)

        if candidate_component is not None and source_component is not None:
            own = source_component[rows]
            same = hit & (candidate_component.unsqueeze(0) == own.unsqueeze(1)) & (own.unsqueeze(1) >= 0)
            out["same_component_blocker_count"][rows] = torch.where(
                own >= 0, same.sum(dim=1).to(torch.int32), torch.full_like(counts, -1, dtype=torch.int32)
            )
            del same
        del hit, t, direction

    ray = out["ray_length"]
    out["camera_nearest_blocker_world_gap"] = (1.0 - out["camera_nearest_blocker_t"]) * ray
    out["query_nearest_blocker_world_gap"] = (1.0 - out["query_nearest_blocker_t"]) * ray
    out["blocker_region_thickness"] = (out["query_nearest_blocker_t"] - out["camera_nearest_blocker_t"]) * ray
    return out


def evaluate_with_values(
    model: Any,
    cameras: list[Any],
    positions: torch.Tensor,
    *,
    support: candidate_c.GeometricSceneSupport,
    component_of_surfel: torch.Tensor | None = None,
    source_surfel: np.ndarray | None = None,
    chunk_bytes: int = 256 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
) -> ValueEvaluationResult:
    """One sweep producing worklog 120's four state arrays (from the UNCHANGED
    decision functions) plus the supplemental value table."""

    from osn_gs.render.torch_surfel_query_depth_diagnostics import MAX_QUERY_SLOTS, render_with_query_depth_probe

    device = positions.device
    count = int(positions.shape[0])
    views = len(cameras)

    with torch.no_grad():
        positions_full = model.get_xyz.detach()
        rotation_full = model.get_rotation_matrix.detach()
        tangent_u_full = rotation_full[:, :, 0].contiguous()
        tangent_v_full = rotation_full[:, :, 1].contiguous()
        scaling_full = model.get_scaling.detach()
        scale_u_full = scaling_full[:, 0].contiguous()
        scale_v_full = scaling_full[:, 1].contiguous()

    source_component = None
    if component_of_surfel is not None and source_surfel is not None:
        source_component = torch.full((count,), -1, dtype=torch.int64, device=device)
        known = torch.as_tensor(source_surfel >= 0, device=device)
        ids = torch.as_tensor(np.maximum(source_surfel, 0), dtype=torch.int64, device=device)
        source_component = torch.where(known, component_of_surfel[ids], source_component)

    per_view_states = {name: np.full((count, views), STATE_NON_RELEVANT, dtype=np.int8) for name in CANDIDATE_NAMES}
    relevance_code = np.zeros((count, views), dtype=np.int8)
    query_depth = np.zeros((count, views), dtype=np.float32)

    float_fields = (
        "A_event_depth", "A_hit_distance", "B_median_depth",
        "C_camera_nearest_blocker_t", "C_query_nearest_blocker_t",
        "C_camera_nearest_blocker_world_gap", "C_query_nearest_blocker_world_gap",
        "C_blocker_region_thickness", "C_camera_nearest_blocker_opacity",
        "C_query_nearest_blocker_opacity", "C_max_blocker_opacity", "C_ray_length",
        "D_traversal_T_pre", "D_resolution_event_depth", "D_termination_alpha", "D_termination_test_T",
        "pixel_max_backward_jump",
    )
    int_fields = (
        "A_event_surfel", "A_event_branch", "C_blocker_count", "C_same_component_blocker_count",
        "C_camera_nearest_blocker_surfel", "C_query_nearest_blocker_surfel",
        "C_camera_nearest_blocker_component", "C_query_nearest_blocker_component",
        "D_reached", "D_prefix_count", "D_resolution_reason", "D_late_front_count",
        "pixel_inversion_count",
    )
    values: dict[str, np.ndarray] = {name: np.full((count, views), np.nan, dtype=np.float32) for name in float_fields}
    values.update({name: np.full((count, views), -1, dtype=np.int64) for name in int_fields})

    view_names: list[str] = []
    c_decision_mismatches = 0
    termination_contract_violations = 0
    total_render_passes = 0

    for view_index, camera in enumerate(cameras):
        view_names.append(str(getattr(camera, "image_name", f"view_{view_index}")))
        height, width = int(camera.image_height), int(camera.image_width)
        geometry = project_queries(camera, positions)
        relevance_code[:, view_index] = geometry.relevance_code.detach().cpu().numpy()
        query_depth[:, view_index] = geometry.depth.detach().cpu().numpy()

        pixel_index_np = geometry.pixel_index.detach().cpu().numpy()
        ranks = assign_query_depth_slots(pixel_index_np, MAX_QUERY_SLOTS)
        passes = 1 if ranks.max(initial=-1) < 0 else int(ranks.max()) // MAX_QUERY_SLOTS + 1

        probe = {
            key: torch.full((count,), fill, dtype=dtype, device=device)
            for key, fill, dtype in (
                ("terminated", -1, torch.int32), ("reached", -1, torch.int32),
                ("prefix_count", -1, torch.int32), ("late_front", -1, torch.int32),
                ("T", float("nan"), torch.float32), ("resolution_depth", float("nan"), torch.float32),
                ("termination_alpha", float("nan"), torch.float32),
            )
        }
        canonical: dict[str, Any] | None = None
        for pass_index in range(passes):
            lower, upper = pass_index * MAX_QUERY_SLOTS, (pass_index + 1) * MAX_QUERY_SLOTS
            selected = np.nonzero((ranks >= lower) & (ranks < upper))[0]
            query_map = torch.zeros((height * width * MAX_QUERY_SLOTS,), dtype=torch.float32, device=device)
            if selected.size:
                slots = torch.as_tensor(
                    pixel_index_np[selected] * MAX_QUERY_SLOTS + (ranks[selected] - lower),
                    dtype=torch.int64, device=device,
                )
                query_map[slots] = geometry.depth[torch.as_tensor(selected, dtype=torch.int64, device=device)]
            package = render_with_query_depth_probe(camera, model, query_depths=query_map.reshape(height, width, MAX_QUERY_SLOTS))
            total_render_passes += 1
            if canonical is None:
                canonical = package
            if selected.size:
                rows = torch.as_tensor(selected, dtype=torch.int64, device=device)
                flat = torch.as_tensor(
                    pixel_index_np[selected] * MAX_QUERY_SLOTS + (ranks[selected] - lower),
                    dtype=torch.int64, device=device,
                )
                probe["terminated"][rows] = package["query_terminated"].reshape(-1)[flat]
                probe["reached"][rows] = package["query_reached"].reshape(-1)[flat]
                probe["prefix_count"][rows] = package["query_prefix_count"].reshape(-1)[flat]
                probe["late_front"][rows] = package["query_late_front_count"].reshape(-1)[flat]
                probe["T"][rows] = package["query_T"].reshape(-1)[flat]
                probe["resolution_depth"][rows] = package["query_resolution_depth"].reshape(-1)[flat]
                probe["termination_alpha"][rows] = package["query_termination_alpha"].reshape(-1)[flat]
            if package is not canonical:
                del package

        assert canonical is not None
        representative = canonical["representative_id"].reshape(-1).to(torch.int64)
        event_valid = representative >= 0
        index = geometry.pixel_index.clamp(min=0)

        # ---------------------------------------------------------- A (frozen)
        event_world = reconstruct_direct_surfel_intersection_world_point(
            representative, canonical["median_s_u"], canonical["median_s_v"],
            positions_full, tangent_u_full, tangent_v_full, scale_u_full, scale_v_full,
        )
        homogeneous = torch.cat([event_world, torch.ones((event_world.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
        event_depth = (homogeneous @ camera.world_view_transform)[:, 2].contiguous()
        result_a = candidate_a.classify_view(geometry, event_world, event_depth, event_valid, positions)
        per_view_states["A"][:, view_index] = result_a["states"].detach().cpu().numpy()
        values["A_hit_distance"][:, view_index] = result_a["hit_distance"].detach().cpu().numpy()
        values["A_event_depth"][:, view_index] = result_a["event_depth"].detach().cpu().numpy()
        rho3d = canonical["median_rho3d"].reshape(-1)[index]
        rho2d = canonical["median_rho2d"].reshape(-1)[index]
        branch = torch.where(rho3d <= rho2d, torch.zeros_like(rho3d), torch.ones_like(rho3d)).to(torch.int64)
        has_event = event_valid[index] & geometry.relevant
        values["A_event_branch"][:, view_index] = torch.where(has_event, branch, torch.full_like(branch, -1)).cpu().numpy()
        values["A_event_surfel"][:, view_index] = torch.where(
            has_event, representative[index], torch.full_like(representative[index], -1)
        ).cpu().numpy()

        # ---------------------------------------------------------- B (frozen)
        median_flat = candidate_b.median_depth_map(canonical["out_others"]).reshape(-1)
        result_b = candidate_b.classify_view(geometry, median_flat)
        per_view_states["B"][:, view_index] = result_b["states"].detach().cpu().numpy()
        values["B_median_depth"][:, view_index] = result_b["median_depth"].detach().cpu().numpy()

        # ---------------------------------------------------------- C (frozen decision + corrected values)
        result_c = candidate_c.classify_view(
            geometry, positions, camera.camera_center, camera.world_view_transform, support, chunk_bytes=chunk_bytes,
        )
        per_view_states["C"][:, view_index] = result_c["states"].detach().cpu().numpy()
        blocker = candidate_c_blocker_values(
            geometry, positions, camera.camera_center, camera.world_view_transform, support,
            component_of_surfel, source_component, chunk_bytes=chunk_bytes,
        )
        # Invariance guard: the value pass must reproduce the frozen decision's
        # own blocked set exactly, or the values do not describe that decision.
        decided_blocked = result_c["states"] == STATE_OCCLUDED
        value_blocked = geometry.relevant & (blocker["blocker_count"] > 0)
        c_decision_mismatches += int((decided_blocked != value_blocked).sum().item())
        for source_key, target_key in (
            ("camera_nearest_blocker_t", "C_camera_nearest_blocker_t"),
            ("query_nearest_blocker_t", "C_query_nearest_blocker_t"),
            ("camera_nearest_blocker_world_gap", "C_camera_nearest_blocker_world_gap"),
            ("query_nearest_blocker_world_gap", "C_query_nearest_blocker_world_gap"),
            ("blocker_region_thickness", "C_blocker_region_thickness"),
            ("camera_nearest_blocker_opacity", "C_camera_nearest_blocker_opacity"),
            ("query_nearest_blocker_opacity", "C_query_nearest_blocker_opacity"),
            ("max_blocker_opacity", "C_max_blocker_opacity"),
            ("ray_length", "C_ray_length"),
        ):
            values[target_key][:, view_index] = blocker[source_key].detach().cpu().numpy()
        values["C_blocker_count"][:, view_index] = blocker["blocker_count"].detach().cpu().numpy()
        values["C_same_component_blocker_count"][:, view_index] = blocker["same_component_blocker_count"].detach().cpu().numpy()
        values["C_camera_nearest_blocker_surfel"][:, view_index] = blocker["camera_nearest_blocker_surfel"].detach().cpu().numpy()
        values["C_query_nearest_blocker_surfel"][:, view_index] = blocker["query_nearest_blocker_surfel"].detach().cpu().numpy()
        if component_of_surfel is not None:
            for surfel_key, component_key in (
                ("camera_nearest_blocker_surfel", "C_camera_nearest_blocker_component"),
                ("query_nearest_blocker_surfel", "C_query_nearest_blocker_component"),
            ):
                ids = blocker[surfel_key]
                known = ids >= 0
                values[component_key][:, view_index] = torch.where(
                    known, component_of_surfel[ids.clamp(min=0)], torch.full_like(ids, -1)
                ).cpu().numpy()

        # ---------------------------------------------------------- D (frozen decision + corrected values)
        result_d = candidate_d.classify_view(
            geometry, probe["terminated"], probe["T"], probe["reached"], probe["prefix_count"]
        )
        per_view_states["D"][:, view_index] = result_d["states"].detach().cpu().numpy()
        values["D_traversal_T_pre"][:, view_index] = probe["T"].detach().cpu().numpy()
        values["D_reached"][:, view_index] = probe["reached"].detach().cpu().numpy()
        values["D_prefix_count"][:, view_index] = probe["prefix_count"].detach().cpu().numpy()
        values["D_late_front_count"][:, view_index] = probe["late_front"].detach().cpu().numpy()
        resolution_depth = probe["resolution_depth"].detach().cpu().numpy()
        resolution_depth[resolution_depth < 0] = np.nan
        values["D_resolution_event_depth"][:, view_index] = resolution_depth
        alpha = probe["termination_alpha"].detach().cpu().numpy()
        alpha[alpha < 0] = np.nan
        values["D_termination_alpha"][:, view_index] = alpha
        t_pre = probe["T"].detach().cpu().numpy()
        test_t = t_pre * (1.0 - alpha)
        values["D_termination_test_T"][:, view_index] = test_t
        finite = np.isfinite(test_t)
        termination_contract_violations += int((finite & ~(test_t < CANONICAL_TERMINATION_TEST_T)).sum())
        values["D_resolution_reason"][:, view_index] = d_resolution_reason(
            probe["terminated"].detach().cpu().numpy(),
            probe["reached"].detach().cpu().numpy(),
            geometry.relevant.detach().cpu().numpy(),
        )
        pixel_inversion = canonical["pixel_inversion_count"].reshape(-1)[index]
        pixel_jump = canonical["pixel_max_backward_jump"].reshape(-1)[index]
        relevant_mask = geometry.relevant
        values["pixel_inversion_count"][:, view_index] = torch.where(
            relevant_mask, pixel_inversion.to(torch.int64), torch.full_like(pixel_inversion, -1, dtype=torch.int64)
        ).cpu().numpy()
        values["pixel_max_backward_jump"][:, view_index] = torch.where(
            relevant_mask, pixel_jump, torch.full_like(pixel_jump, float("nan"))
        ).cpu().numpy()

        del canonical, event_world, homogeneous, event_depth
        if progress is not None and (view_index % 20 == 0 or view_index == views - 1):
            progress(f"value sweep {view_index + 1}/{views} ({view_names[-1]})")

    global_states = {name: aggregate_global(per_view_states[name]) for name in CANDIDATE_NAMES}
    return ValueEvaluationResult(
        per_view_states=per_view_states, global_states=global_states,
        relevance_code=relevance_code, query_depth=query_depth, values=values,
        view_names=view_names,
        diagnostics={
            "render_passes": total_render_passes,
            "candidate_C_value_vs_decision_mismatches": c_decision_mismatches,
            "D_termination_contract_violations": termination_contract_violations,
            "canonical_termination_test_T": CANONICAL_TERMINATION_TEST_T,
        },
    )


def assert_historical_state_replay(result: ValueEvaluationResult, reference: Any) -> dict[str, Any]:
    """Directive section 2's baseline replay gate. Every listed array must match
    the stored worklog 120 artifact EXACTLY; anything else stops the batch."""

    checks: dict[str, Any] = {}
    failures: list[str] = []

    def _check(name: str, produced: np.ndarray, stored: np.ndarray) -> None:
        same = bool(produced.shape == stored.shape and np.array_equal(produced, stored))
        checks[name] = {"identical": same, "shape": list(produced.shape)}
        if not same:
            differing = int((produced != stored).sum()) if produced.shape == stored.shape else -1
            checks[name]["differing_entries"] = differing
            failures.append(name)

    for name in CANDIDATE_NAMES:
        _check(f"states_{name}", result.per_view_states[name], reference[f"states_{name}"])
        _check(f"global_{name}", result.global_states[name], reference[f"global_{name}"])
    _check("relevance_code", result.relevance_code, reference["relevance_code"])
    checks["relevant_view_counts_identical"] = bool(
        np.array_equal((result.relevance_code == 0).sum(axis=1), (reference["relevance_code"] == 0).sum(axis=1))
    )
    if not checks["relevant_view_counts_identical"]:
        failures.append("relevant_view_counts")
    checks["query_depth_identical"] = bool(np.array_equal(result.query_depth, reference["query_depth"]))
    if not checks["query_depth_identical"]:
        failures.append("query_depth")
    checks["failures"] = failures
    checks["gate"] = "PASS" if not failures else "FAIL"
    return checks


def bank_replay_check(bank, reference: Any) -> dict[str, Any]:
    """The bank itself must be regenerated identically before its states can be
    compared (directive section 2: count, ordering, positions, provenance)."""

    checks = {
        "query_count": {"produced": len(bank), "stored": int(reference["positions"].shape[0])},
        "positions_identical": bool(np.array_equal(bank.positions.detach().cpu().numpy(), reference["positions"])),
        "kind_identical": bool(np.array_equal(np.asarray(bank.kind), reference["kind"])),
        "source_view_identical": bool(np.array_equal(bank.source_view, reference["source_view"])),
        "source_surfel_identical": bool(np.array_equal(bank.source_surfel, reference["source_surfel"])),
        "region_identical": bool(np.array_equal(bank.region, reference["region"])),
        "ladder_step_identical": bool(
            np.array_equal(bank.ladder_step, reference["ladder_step"], equal_nan=True)
        ),
        "support_radius_identical": bool(
            np.array_equal(bank.support_radius, reference["support_radius"], equal_nan=True)
        ),
    }
    checks["query_count_identical"] = checks["query_count"]["produced"] == checks["query_count"]["stored"]
    failures = [key for key, value in checks.items() if key.endswith("_identical") and value is False]
    checks["failures"] = failures
    checks["gate"] = "PASS" if not failures else "FAIL"
    return checks
