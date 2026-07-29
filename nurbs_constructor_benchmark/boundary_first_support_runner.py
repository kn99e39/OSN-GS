"""Export the isolated Boundary-first support experiment for visual review.

This runner is deliberately separate from the benchmark dispatcher.  It makes
the current experiment inspectable without changing the production-like
``--constructor boundary_first`` route.
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Any

from nurbs_constructor_benchmark.boundary_first import renderer_payload, write_point_cloud_ply
from nurbs_constructor_benchmark.boundary_first_support import construct_boundary_first_support
from nurbs_constructor_benchmark.ground_truth import gt_nurbs_payload
from nurbs_constructor_benchmark.scenes import ALL_SCENE_NAMES, SCENE_NAMES, make_scene
from osn_gs.surface.torch_boundary_review_geometry import (
    CROSSING_SCOPE_REPRESENTATIVE_BUNDLE,
    NOT_YET_CHECKED_CROSSING_CATEGORIES,
    REVIEW_SCHEMA_VERSION,
    combine_ordered_patch_boundary,
    control_polygon_entity,
    correspondence_chord_entity,
    detect_support_curve_crossings,
    evaluate_interior_iso_curve,
    evaluate_iso_edge,
)
from osn_gs.surface.torch_boundary_surface_quality import measure_boundary_first_surface_quality

# Sample count for exporting a per-patch boundary/support edge -- enough to
# show cubic curvature without inflating the export; combine_ordered_patch_boundary
# drops the duplicate shared endpoint per patch, so the final closed-loop
# sample count is ``patch_count * (_BOUNDARY_EDGE_SAMPLES - 1)``.
_BOUNDARY_EDGE_SAMPLES = 5
# Deterministic, CONFIGURABLE interior-u sampling policy for the observed-anchor
# fan's representative support-curve bundle (see build_parser's
# --interior-support-curve-fractions). Not a canonical constant -- a default.
_DEFAULT_INTERIOR_SUPPORT_CURVE_FRACTIONS = (0.25, 0.5, 0.75)
_NOT_MATERIALIZED_CROSSING = {
    "state": "not_checked",
    "reason": "surface_not_materialized",
    "scope": CROSSING_SCOPE_REPRESENTATIVE_BUNDLE,
    "pairs": [],
    "has_invalid_crossing": False,
    "not_checked_categories": list(NOT_YET_CHECKED_CROSSING_CATEGORIES),
}


def _stable_tolerance_scale(item: Any) -> float | None:
    """A geometric crossing-tolerance basis independent of curve sampling resolution.

    Prefers the observed component's own median nearest-neighbor point
    spacing (``source_boundary_fidelity.local_spacing_median``, already
    computed by the builder from the raw observed points) over anything
    derived from how finely the review curves themselves were sampled.
    Returns ``None`` when unavailable so the caller falls back to the
    geometry module's own (resolution-dependent) convenience default rather
    than silently inventing a scale.
    """
    fidelity = item.provenance.get("source_boundary_fidelity")
    if not fidelity:
        return None
    spacing = fidelity.get("local_spacing_median")
    return float(spacing) if spacing is not None and float(spacing) > 0.0 else None


def _patch_payload(surface: Any, patch_id: int) -> dict[str, Any]:
    knots_u, knots_v = surface.knot_vectors()
    return {
        "patch_id": int(patch_id),
        "control_grid_shape": list(surface.control_grid.shape),
        "control_grid": surface.control_grid.detach().cpu().tolist(),
        "weights": surface.weights.detach().cpu().tolist(),
        "degree_u": int(surface.degree_u),
        "degree_v": int(surface.degree_v),
        "knots_u": knots_u.detach().cpu().tolist(),
        "knots_v": knots_v.detach().cpu().tolist(),
        "uv_support": None,
        "chart_kind": "boundary_first_support_seam_patch",
    }


def _empty_review_layers(provenance: dict[str, Any], crossing: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "boundary_roles": provenance.get("boundary_roles"),
        "correspondence": (provenance.get("network") or {}).get("correspondence"),
        "observed_outer_boundary": provenance.get("observed_outer_boundary"),
        "observed_inner_boundary": provenance.get("observed_inner_boundary"),
        "observed_interior_anchor": provenance.get("observed_interior_anchor"),
        "support_control_polygons": [],
        "evaluated_support_curves": [],
        "outer_boundary_control_polygons": [],
        "inner_boundary_control_polygons": [],
        "reconstructed_outer_boundary": None,
        "reconstructed_inner_boundary": None,
        "support_correspondence_chords": [],
        "support_crossing": crossing,
        "pole_metadata": None,
    }


def _closed_wedge_review_layers(surfaces: tuple[Any, ...], layers: dict[str, Any], *, tolerance_scale: float | None) -> tuple[dict[str, Any], list[Any]]:
    """Closed inner/outer support-curve-network case: seams ARE the actual radial
    evaluated support curves (patch k's u=0 edge == patch k-1's u=1 edge exactly,
    a general clamped-NURBS boundary property, verified by the crossing check
    below rather than assumed)."""
    support_control_polygons = []
    evaluated_support_curves = []
    outer_control_polygons = []
    inner_control_polygons = []
    outer_edges = []
    inner_edges = []
    seam_entities = []
    for index, surface in enumerate(surfaces):
        radial_samples = int(surface.control_grid.shape[1])
        support_control_polygons.append(control_polygon_entity(surface, "u0", patch_id=index, role="support_curve", entity_id=f"seam{index}:support_control_polygon").payload())
        seam_curve = evaluate_iso_edge(surface, "u0", samples=radial_samples, patch_id=index, role="support_curve", entity_id=f"seam{index}:evaluated_support_curve")
        evaluated_support_curves.append(seam_curve.payload())
        seam_entities.append(seam_curve)
        outer_control_polygons.append(control_polygon_entity(surface, "v1", patch_id=index, role="outer_boundary", entity_id=f"patch{index}:outer_control_polygon").payload())
        inner_control_polygons.append(control_polygon_entity(surface, "v0", patch_id=index, role="inner_boundary", entity_id=f"patch{index}:inner_control_polygon").payload())
        outer_edges.append(evaluate_iso_edge(surface, "v1", samples=_BOUNDARY_EDGE_SAMPLES, patch_id=index, role="outer_boundary", entity_id=f"patch{index}:outer_evaluated_edge", closed=False))
        inner_edges.append(evaluate_iso_edge(surface, "v0", samples=_BOUNDARY_EDGE_SAMPLES, patch_id=index, role="inner_boundary", entity_id=f"patch{index}:inner_evaluated_edge", closed=False))
    reconstructed_outer = combine_ordered_patch_boundary(outer_edges, entity_id="reconstructed_outer_boundary", role="outer_boundary", closed=True)
    reconstructed_inner = combine_ordered_patch_boundary(inner_edges, entity_id="reconstructed_inner_boundary", role="inner_boundary", closed=True)
    # Representative check: one curve per seam position -- not the wedge
    # interiors between seams, not a continuous non-intersection guarantee.
    crossing = detect_support_curve_crossings([entity["points"] for entity in evaluated_support_curves], tolerance_scale=tolerance_scale)
    layers.update(
        {
            "support_control_polygons": support_control_polygons,
            "evaluated_support_curves": evaluated_support_curves,
            "outer_boundary_control_polygons": outer_control_polygons,
            "inner_boundary_control_polygons": inner_control_polygons,
            "reconstructed_outer_boundary": reconstructed_outer.payload(),
            "reconstructed_inner_boundary": reconstructed_inner.payload(),
            "support_correspondence_chords": [],
            "support_crossing": crossing,
            "pole_metadata": None,
        }
    )
    return layers, seam_entities


def _anchor_fan_review_layers(
    surfaces: tuple[Any, ...], layers: dict[str, Any], *,
    tolerance_scale: float | None,
    interior_support_curve_fractions: tuple[float, ...] = _DEFAULT_INTERIOR_SUPPORT_CURVE_FRACTIONS,
) -> tuple[dict[str, Any], list[Any]]:
    """Observed-anchor fan case.

    The pole<->corner probe is a diagnostic correspondence chord (only its two
    endpoints are real geometry landmarks; a straight 2-point line is not "the"
    support curve through a patch's interior) and is kept in
    ``support_correspondence_chords`` ONLY. The seam used for ``patch_boundaries``
    is instead a SEPARATE, properly-sampled ``evaluate_iso_edge`` entity on the
    same shared ``u0`` edge -- numerically the same straight line (degree_v=1),
    but a distinct semantic entity, per instruction: never reuse a chord as a
    seam even when they coincide.

    The actual interior support curve is a representative BUNDLE of interior
    iso-curves (default u in (0.25, 0.5, 0.75), configurable), pole to the
    curved outer edge, at each patch -- not just its midpoint -- so the
    crossing check covers more of each patch's interior than a single sample.
    """
    outer_control_polygons = []
    outer_edges = []
    evaluated_support_curves = []
    correspondence_chords = []
    seam_entities = []
    pole_point = None
    for index, surface in enumerate(surfaces):
        outer_control_polygons.append(control_polygon_entity(surface, "v1", patch_id=index, role="outer_boundary", entity_id=f"patch{index}:outer_control_polygon").payload())
        outer_edges.append(evaluate_iso_edge(surface, "v1", samples=_BOUNDARY_EDGE_SAMPLES, patch_id=index, role="outer_boundary", entity_id=f"patch{index}:outer_evaluated_edge", closed=False))
        for fraction in interior_support_curve_fractions:
            interior_curve = evaluate_interior_iso_curve(
                surface, fixed_direction="u", fixed_value=float(fraction), samples=_BOUNDARY_EDGE_SAMPLES,
                patch_id=index, role="support_curve", entity_id=f"patch{index}:evaluated_interior_support_curve:u={fraction}",
            )
            evaluated_support_curves.append(interior_curve.payload())
        # The actual seam entity: a proper evaluated edge, sampled independently
        # of the 2-point correspondence chord below (never the same object).
        seam_curve = evaluate_iso_edge(surface, "u0", samples=_BOUNDARY_EDGE_SAMPLES, patch_id=index, role="support_curve_seam", entity_id=f"seam{index}:evaluated_seam")
        seam_entities.append(seam_curve)
        corner = evaluate_iso_edge(surface, "u0", samples=2, patch_id=index, role="support_curve_seam", entity_id=f"seam{index}:corner_probe")
        chord = correspondence_chord_entity(corner.points, entity_id=f"seam{index}:support_correspondence_chord", role="support_curve_seam", patch_id=index)
        correspondence_chords.append(chord)
        if pole_point is None:
            pole_point = corner.points[0]
    reconstructed_outer = combine_ordered_patch_boundary(outer_edges, entity_id="reconstructed_outer_boundary", role="outer_boundary", closed=True)
    # Representative check ONLY: a finite bundle of interior iso-curves, not
    # every point of the patch interior, and not a continuous guarantee.
    crossing = detect_support_curve_crossings(
        [entity["points"] for entity in evaluated_support_curves],
        tolerance_scale=tolerance_scale,
        expected_shared_point=pole_point.detach().cpu().tolist(),
        expected_shared_kind="pole",
    )
    layers.update(
        {
            "support_control_polygons": [],
            "evaluated_support_curves": evaluated_support_curves,
            "outer_boundary_control_polygons": outer_control_polygons,
            "inner_boundary_control_polygons": [],
            "reconstructed_outer_boundary": reconstructed_outer.payload(),
            "reconstructed_inner_boundary": None,
            "support_correspondence_chords": [chord.payload() for chord in correspondence_chords],
            "support_crossing": crossing,
            "pole_metadata": {
                "has_central_pole": True,
                "singularity_kind": "shared_observed_anchor_pole",
                "pole_aware_regularity_contract": "torch_boundary_surface_quality.measure_boundary_first_surface_quality:pole_excluded_minimum_jacobian_norm",
                "parameter_direction": "v",
                "pole_point": pole_point.detach().cpu().tolist(),
                "interior_support_curve_fractions": list(interior_support_curve_fractions),
            },
        }
    )
    return layers, seam_entities


def _boundary_first_review_layers(
    item: Any, *, interior_support_curve_fractions: tuple[float, ...] = _DEFAULT_INTERIOR_SUPPORT_CURVE_FRACTIONS,
) -> tuple[dict[str, Any], list[Any]]:
    """Derive observed/reconstructed boundary review layers for one component.

    Observed evidence (``observed_outer_boundary``/``observed_inner_boundary``/
    ``observed_interior_anchor``) comes straight from the builder's provenance so
    it is available even for ``review_required``/``unsupported`` items -- a
    rejected anchor still keeps its evidence on record. The reconstructed
    layers are real ``surface.evaluate()`` samples along actual iso-parametric
    edges/curves, never raw control-grid rows relabeled as curve geometry;
    ``support_control_polygons``/``outer_boundary_control_polygons``/
    ``inner_boundary_control_polygons`` carry the raw control data separately
    for anyone who wants the fitting representation instead. The second
    return value is the ordered list of seam entities (both cases now use a
    properly evaluated iso-parametric edge, never a correspondence chord)
    used to populate ``patch_boundaries``.
    """
    provenance = item.provenance
    layers = _empty_review_layers(provenance, dict(_NOT_MATERIALIZED_CROSSING))
    if item.surface_result is None:
        return layers, []
    surfaces = tuple(item.surface_result.surfaces)
    tolerance_scale = _stable_tolerance_scale(item)
    if "network" in provenance:
        return _closed_wedge_review_layers(surfaces, layers, tolerance_scale=tolerance_scale)
    if provenance.get("materialization") == "shared_observed_anchor_cubic_fan":
        return _anchor_fan_review_layers(
            surfaces, layers, tolerance_scale=tolerance_scale,
            interior_support_curve_fractions=interior_support_curve_fractions,
        )
    return layers, []


def _seam_payloads(seam_entities: list[Any], patch_id_offset: int) -> list[dict[str, Any]]:
    """Shift one component's local seam indices into the export's global patch ids.

    Seam ``index`` connects patch ``(index-1) % count``'s ``u1`` edge with
    patch ``index``'s ``u0`` edge -- always the SAME control column by
    construction (both wedge and fan builders reuse the identical support/
    outer-boundary sample for both), so ``same_orientation`` is always
    ``True`` and never independently re-derived per seam.
    """
    count = len(seam_entities)
    payloads = []
    for index, entity in enumerate(seam_entities):
        local_a, local_b = (index - 1) % count, index
        patch_a, patch_b = patch_id_offset + local_a, patch_id_offset + local_b
        payloads.append(
            {
                "boundary_id": f"p{patch_b}:seam:{entity.entity_id}",
                "patch_a": patch_a,
                "patch_b": patch_b,
                "edge_a": "u1",
                "edge_b": entity.parameter_edge,
                "parameter_direction": entity.parameter_direction,
                "same_orientation": True,
                "patch_id": patch_b,
                "adjacent_patch_id": patch_a,
                "adjacent_boundary_id": f"p{patch_a}:seam:{entity.entity_id}",
                "source_kind": "boundary_first_support_seam",
                "representation_kind": entity.representation_kind,
                "closed": True,
                "shared_endpoint_policy": "clamped_control_column_identity",
                "parameter_samples": None if entity.parameter_samples is None else list(entity.parameter_samples),
                "world": torch.as_tensor(entity.points).detach().cpu().tolist(),
            }
        )
    return payloads


def _component_quality_state(item: Any, layers: dict[str, Any]) -> list[str | None]:
    """Canonical per-component quality verdict, kept SEPARATE from materialization.

    ``materialization_state`` (``materialized``/``not_materialized``, already
    on ``item``) answers "did a patch/control structure get built at all".
    This answers a different question -- "is the already-materialized result
    trustworthy" -- and must never collapse into the same value: a
    materialized surface with an invalid support crossing is
    ``materialized`` + ``ineligible``, not the same state as a component that
    was never materialized (``not_materialized`` + ``unsupported``).

    Returned as a mutable ``[state, reason]`` pair (not a tuple) because the
    caller downgrades ``review_required`` further once per-component RMS
    fidelity is known, later in the same scene evaluation.

    ``eligible`` is a defined vocabulary value that no path here produces
    yet: crossing detection is only a representative sampled diagnostic (see
    ``CROSSING_SCOPE_REPRESENTATIVE_BUNDLE``/``NOT_YET_CHECKED_CROSSING_CATEGORIES``),
    and bidirectional fidelity/false-hole hardening are not implemented, so
    calling anything "eligible" would overclaim. This is an isolated review
    diagnostic label only -- it is not, and must never be read as, production
    Gate approval.
    """
    if item.materialization_state != "materialized":
        return ["unsupported", item.reason]
    crossing = layers.get("support_crossing", {})
    if crossing.get("state") == "checked" and crossing.get("has_invalid_crossing"):
        return ["ineligible", "invalid_support_crossing"]
    if crossing.get("state") != "checked":
        return ["review_required", "support_crossing_not_checked"]
    return ["review_required", "diagnostic_gates_incomplete_no_eligible_path_yet"]


def _scene_quality_projection(component_quality: list[list[str | None]]) -> tuple[str, str | None]:
    states = [state for state, _ in component_quality]
    if "ineligible" in states:
        reason = next(reason for state, reason in component_quality if state == "ineligible")
        return "ineligible", reason
    if "review_required" in states:
        reason = next((reason for state, reason in component_quality if state == "review_required" and reason), None)
        return "review_required", reason
    if states and all(state == "unsupported" for state in states):
        return "unsupported", None
    return "review_required", "diagnostic_gates_incomplete_no_eligible_path_yet"


def _source_point_fidelity(surface_result: Any, points: Any, resolution: int = 12) -> dict[str, Any]:
    values = torch.as_tensor(points)
    samples = []
    for surface in surface_result.surfaces:
        lin = torch.linspace(0.0, 1.0, resolution, dtype=values.dtype, device=values.device)
        u, v = torch.meshgrid(lin, lin, indexing='ij')
        samples.append(surface.evaluate(torch.stack((u.reshape(-1), v.reshape(-1)), dim=1)).detach())
    distances = torch.cdist(values, torch.cat(samples, dim=0)).min(dim=1).values
    return {'source_point_rms': float(distances.square().mean().sqrt()), 'source_point_max': float(distances.max()), 'source_point_count': int(values.shape[0]), 'sample_resolution': int(resolution)}
def evaluate_scene(
    scene: Any, *, curve_count: int, samples_per_curve: int, boundary_resolution: int = 96,
    max_source_point_rms: float | None = None,
    interior_support_curve_fractions: tuple[float, ...] = _DEFAULT_INTERIOR_SUPPORT_CURVE_FRACTIONS,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    construction = construct_boundary_first_support(
        scene, curve_count=curve_count, samples_per_curve=samples_per_curve, boundary_resolution=boundary_resolution
    )
    visible = construction.visible_results
    review_results = [
        _boundary_first_review_layers(item, interior_support_curve_fractions=interior_support_curve_fractions)
        for item in visible
    ]
    review_layers = [layers for layers, _ in review_results]
    seam_entities_by_item = [seam_entities for _, seam_entities in review_results]
    # component_quality[i] = [state, reason]; mutated in place once per-component
    # RMS fidelity is known below (constructed items only).
    component_quality = [_component_quality_state(item, layers) for item, layers in zip(visible, review_layers)]
    constructed = [item for item in visible if item.state == "constructed"]
    scene_materialization_state = (
        "materialized" if all(item.materialization_state == "materialized" for item in visible) else "not_materialized"
    )
    record: dict[str, Any] = {
        "scene": scene.name,
        "raw_component_count": construction.raw_component_count,
        "recovery_edges": [edge.payload() for edge in construction.recovery_edges],
        "recovered_region_count": len(construction.recovered_regions),
        # Canonical: materialization ("did a patch get built") vs quality
        # ("is the result trustworthy") are two independent axes -- never
        # collapse "not built" and "built but quality-failed" into one state.
        "materialization_state": scene_materialization_state,
        "visible_results": [
            {
                "state": item.state,  # compatibility projection only, see below -- not canonical
                "materialization_state": item.materialization_state,
                "quality_state": quality_state,
                "quality_reason": quality_reason,
                "topology": item.topology, "reason": item.reason, "provenance": item.provenance, "review": layers,
            }
            for item, layers, (quality_state, quality_reason) in zip(visible, review_layers, component_quality)
        ],
    }
    if not constructed or len(constructed) != len(visible):
        # ``record["state"]`` (constructed/review_required/unsupported) is kept
        # ONLY as a compatibility projection for older readers of this report;
        # ``materialization_state``/``quality_state``/``quality_reason`` above
        # and below are the canonical fields.
        record["state"] = "review_required" if any(item.state == "review_required" for item in visible) else "unsupported"
        record["quality_state"], record["quality_reason"] = _scene_quality_projection(component_quality)
        return record, None
    patches = []
    patch_boundaries: list[dict[str, Any]] = []
    qualities = []
    for index, (region, item, seam_entities) in enumerate(zip(construction.recovered_regions, visible, seam_entities_by_item)):
        if item.state != 'constructed':
            continue
        quality = measure_boundary_first_surface_quality(item.surface_result).payload()
        quality.update(_source_point_fidelity(item.surface_result, scene.points[region.component.gaussian_indices]))
        qualities.append(quality)
        if (
            max_source_point_rms is not None
            and quality['source_point_rms'] > float(max_source_point_rms)
            and component_quality[index][0] != "ineligible"
        ):
            component_quality[index] = ["review_required", "source_point_rms_exceeds_threshold"]
        patch_id_offset = len(patches)
        patches.extend(_patch_payload(surface, patch_id_offset + local) for local, surface in enumerate(item.surface_result.surfaces))
        patch_boundaries.extend(_seam_payloads(seam_entities, patch_id_offset))
    # Reflect any RMS-driven downgrade back into the exported per-component records.
    for entry, (quality_state, quality_reason) in zip(record["visible_results"], component_quality):
        entry["quality_state"], entry["quality_reason"] = quality_state, quality_reason
    rms_values = [quality['source_point_rms'] for quality in qualities]
    has_invalid_crossing = any(bool(layers.get('support_crossing', {}).get('has_invalid_crossing')) for layers in review_layers)
    rms_pass = max_source_point_rms is None or max(rms_values) <= float(max_source_point_rms)
    gate_pass = rms_pass and not has_invalid_crossing
    scene_quality_state, scene_quality_reason = _scene_quality_projection(component_quality)
    record.update({
        'state': 'constructed' if gate_pass else 'review_required',  # compatibility projection only, not canonical
        'quality_state': scene_quality_state,
        'quality_reason': scene_quality_reason,
        'quality': qualities[0] if len(qualities) == 1 else qualities,
        'patch_count': len(patches),
        'constructed_component_count': len(constructed),
        'fidelity_gate': {
            'max_source_point_rms': max_source_point_rms, 'observed_max_source_point_rms': max(rms_values),
            'has_invalid_support_crossing': has_invalid_crossing, 'passed': gate_pass,
        },
    })
    return record, renderer_payload(scene.name, patches, patch_boundaries, boundary_first_review=review_layers)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export isolated Boundary-first support surfaces for review.")
    parser.add_argument("--scenes", nargs="+", choices=ALL_SCENE_NAMES, default=list(SCENE_NAMES), help="Review every requested scene (current or legacy). Unsupported topology is recorded, not hidden or treated as a runner failure.")
    parser.add_argument("--output", type=Path, default=Path("nurbs_constructor_benchmark/results_boundary_first_support"))
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curve-count", type=int, default=8)
    parser.add_argument("--samples-per-curve", type=int, default=8)
    parser.add_argument("--boundary-resolution", type=int, default=96, help="Isolated observed-support raster resolution.")
    parser.add_argument("--max-source-point-rms", type=float, default=None, help="Review gate: export remains available but results above this RMS are marked review_required.")
    parser.add_argument(
        "--interior-support-curve-fractions", nargs="+", type=float, default=list(_DEFAULT_INTERIOR_SUPPORT_CURVE_FRACTIONS),
        help="Configurable deterministic interior-u sampling policy for the observed-anchor fan's representative support-curve crossing bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    renderer_root = output / "NURBS_output"
    records = []
    for name in args.scenes:
        scene = make_scene(name, args.points, args.seed)
        record, payload = evaluate_scene(
            scene, curve_count=args.curve_count, samples_per_curve=args.samples_per_curve,
            boundary_resolution=args.boundary_resolution, max_source_point_rms=args.max_source_point_rms,
            interior_support_curve_fractions=tuple(args.interior_support_curve_fractions),
        )
        records.append(record)
        scene_dir = renderer_root / name
        gt_dir = renderer_root / f"{name}_gt"
        scene_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        write_point_cloud_ply(scene, scene_dir / "point_cloud.ply")
        write_point_cloud_ply(scene, gt_dir / "point_cloud.ply")
        (scene_dir / "boundary_first_support_status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if payload is not None:
            (scene_dir / "nurbs_surface.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (gt_dir / "nurbs_surface.json").write_text(json.dumps(gt_nurbs_payload(scene), indent=2), encoding="utf-8")
    report = {"type": "isolated_boundary_first_support_review", "run": vars(args) | {"output": str(output)}, "results": records}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"renderer output={renderer_root}")
    # This is an inspection runner, not an eligibility gate. Unsupported topologies remain visible in the report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
