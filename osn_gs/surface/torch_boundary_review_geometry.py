from __future__ import annotations

"""Boundary-first isolated review-export geometry semantics.

A control polygon/control lattice is fitting *data*, not the surface itself.
Reading a NURBS patch's raw ``control_grid`` rows and calling that "the
reconstructed curve" silently assumes the corresponding axis is degree-1 and
non-rational (the one case where control points happen to sit on the curve).
That assumption already breaks for any cubic (``degree=3``) iso-parametric
edge, where only the two Bezier-segment endpoints lie on the curve and the
two interior control points do not.

This module gives the isolated Boundary-first review/export path a single
place to keep those concepts distinct:

- ``observed_evidence_points`` -- raw observed loop/anchor samples, unprocessed.
- ``resampled_observed_evidence`` -- the same evidence resampled/reordered by
  correspondence (still evidence, not control or curve data).
- ``control_polygon`` -- raw control-net rows/columns of a materialized patch.
  Lossless with respect to the *fit*, not with respect to the *curve*.
- ``correspondence_chord`` -- a diagnostic straight-line pairing between two
  evidence points (e.g. an observed-anchor fan's pole-to-corner spoke). It may
  coincide with a real (degenerate, degree-1) iso-parametric edge, but it is
  exported under this label because its role is correspondence/seam
  bookkeeping, not "the" support curve through a patch's interior.
- ``evaluated_curve`` -- actual ``TorchNURBSSurface.evaluate()`` samples along
  a real parameter path (an iso-parametric edge, or an interior iso-curve).
  This is the only representation kind that reflects degree/weight/knot
  effects.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

REPRESENTATION_OBSERVED_EVIDENCE = "observed_evidence_points"
REPRESENTATION_RESAMPLED_EVIDENCE = "resampled_observed_evidence"
REPRESENTATION_CONTROL_POLYGON = "control_polygon"
REPRESENTATION_CORRESPONDENCE_CHORD = "correspondence_chord"
REPRESENTATION_EVALUATED_CURVE = "evaluated_curve"

REVIEW_SCHEMA_VERSION = "boundary_first_review/2"

CROSSING_VALID_SHARED_POLE = "valid_shared_pole"
CROSSING_VALID_SHARED_ENDPOINT = "valid_shared_boundary_endpoint"
CROSSING_TRANSVERSAL_INTERSECTION = "transversal_intersection"
CROSSING_OVERLAPPING_SUPPORT_PATH = "overlapping_support_path"
CROSSING_NEAR_TOUCHING = "near_touching_ambiguous"
CROSSING_NO_CROSSING = "no_crossing"
CROSSING_NOT_CHECKED = "not_checked"
CROSSING_INVALID_CLASSIFICATIONS = (CROSSING_TRANSVERSAL_INTERSECTION, CROSSING_OVERLAPPING_SUPPORT_PATH)

# Crossing detection only checks a finite, deterministic bundle of actual
# evaluated support curves -- never a continuous non-intersection proof over
# the whole support family, and never patch-interior self-overlap, seam
# tangent/Jacobian consistency, or foldover. Those stay explicitly not_checked
# until their own dedicated diagnostics exist.
CROSSING_SCOPE_REPRESENTATIVE_BUNDLE = "representative_support_curve_bundle_crossing"
NOT_YET_CHECKED_CROSSING_CATEGORIES = (
    "patch_overlap",
    "jacobian_foldover",
    "seam_inconsistency",
    "full_support_family_crossing",
)


def _tensor_points(points: Any) -> Any:
    torch = require_torch()
    return torch.as_tensor(points).detach()


@dataclass(frozen=True)
class ReviewGeometryEntity:
    """One provenance-bearing geometry entity in the isolated review export.

    ``patch_ids`` is a tuple because a reconstructed boundary/support curve
    combined across several patches (``combine_ordered_patch_boundary``)
    keeps every contributing patch id, in deterministic order.
    """

    entity_id: str
    representation_kind: str
    role: str
    points: Any
    source_component_id: int | None = None
    source_loop_id: int | str | None = None
    source_anchor_id: int | None = None
    patch_ids: tuple[int, ...] = ()
    coordinate_space: str = "world"
    ordered: bool = True
    closed: bool = False
    orientation: str | None = None
    source_point_indices: tuple[int, ...] | None = None
    correction_applied: bool = False
    correction_reason: str | None = None
    parameter_edge: str | None = None
    parameter_direction: str | None = None
    parameter_samples: tuple[float, ...] | None = None
    sampling_policy: str | None = None

    def payload(self) -> dict[str, Any]:
        points = _tensor_points(self.points).cpu().tolist()
        return {
            "entity_id": self.entity_id,
            "representation_kind": self.representation_kind,
            "role": self.role,
            "source_component_id": self.source_component_id,
            "source_loop_id": self.source_loop_id,
            "source_anchor_id": self.source_anchor_id,
            "patch_ids": list(self.patch_ids),
            "coordinate_space": self.coordinate_space,
            "ordered": bool(self.ordered),
            "closed": bool(self.closed),
            "orientation": self.orientation,
            "source_point_indices": None if self.source_point_indices is None else list(self.source_point_indices),
            "correction_applied": bool(self.correction_applied),
            "correction_reason": self.correction_reason,
            "parameter_edge": self.parameter_edge,
            "parameter_direction": self.parameter_direction,
            "parameter_samples": None if self.parameter_samples is None else list(self.parameter_samples),
            "sampling_policy": self.sampling_policy,
            "points": points,
        }


_EDGE_DIRECTION = {"u0": ("u", 0.0), "u1": ("u", 1.0), "v0": ("v", 0.0), "v1": ("v", 1.0)}


def _evaluate_fixed_parameter_curve(
    surface: Any, *, fixed_direction: str, fixed_value: float, samples: int
) -> tuple[Any, Any]:
    """Sample ``S`` along the varying parameter with the other one held fixed.

    Always goes through ``surface.evaluate()`` -- the actual NURBS evaluator --
    never through raw ``control_grid`` rows, so degree/weight/knot effects are
    never silently assumed away.
    """
    torch = require_torch()
    if fixed_direction not in ("u", "v"):
        raise ValueError(f"fixed_direction must be 'u' or 'v', got {fixed_direction!r}")
    if int(samples) < 2:
        raise ValueError("evaluating a parametric curve requires at least two samples.")
    dtype, device = surface.control_grid.dtype, surface.control_grid.device
    varying = torch.linspace(0.0, 1.0, int(samples), dtype=dtype, device=device)
    uv = torch.empty((int(samples), 2), dtype=dtype, device=device)
    if fixed_direction == "u":
        uv[:, 0] = float(fixed_value)
        uv[:, 1] = varying
    else:
        uv[:, 1] = float(fixed_value)
        uv[:, 0] = varying
    return surface.evaluate(uv).detach(), varying


def evaluate_iso_edge(
    surface: Any,
    edge: str,
    *,
    samples: int,
    patch_id: int,
    role: str,
    entity_id: str,
    closed: bool = False,
    orientation: str | None = None,
) -> ReviewGeometryEntity:
    """Sample a real boundary iso-parametric edge (``S(0,v)``/``S(1,v)``/``S(u,0)``/``S(u,1)``)
    through the patch's own NURBS evaluator -- never through raw control data."""
    if edge not in _EDGE_DIRECTION:
        raise ValueError(f"Unknown iso-parametric edge: {edge!r}")
    fixed_direction, fixed_value = _EDGE_DIRECTION[edge]
    points, varying = _evaluate_fixed_parameter_curve(
        surface, fixed_direction=fixed_direction, fixed_value=fixed_value, samples=samples
    )
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_EVALUATED_CURVE,
        role=role,
        points=points,
        patch_ids=(int(patch_id),),
        closed=closed,
        orientation=orientation,
        parameter_edge=edge,
        parameter_direction="v" if fixed_direction == "u" else "u",
        parameter_samples=tuple(float(x) for x in varying.detach().cpu().tolist()),
        sampling_policy=f"linspace_{int(samples)}",
    )


def evaluate_interior_iso_curve(
    surface: Any,
    *,
    fixed_direction: str,
    fixed_value: float,
    samples: int,
    patch_id: int,
    role: str,
    entity_id: str,
    closed: bool = False,
) -> ReviewGeometryEntity:
    """Sample a real interior iso-parametric curve (fixed parameter not at 0/1).

    Used for e.g. an observed-anchor fan patch's actual radial support curve
    (pole to the curved outer edge at one interior ``u``), which a corner-only
    spoke/chord never represents.
    """
    points, varying = _evaluate_fixed_parameter_curve(
        surface, fixed_direction=fixed_direction, fixed_value=fixed_value, samples=samples
    )
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_EVALUATED_CURVE,
        role=role,
        points=points,
        patch_ids=(int(patch_id),),
        closed=closed,
        parameter_edge=None,
        parameter_direction="v" if fixed_direction == "u" else "u",
        parameter_samples=tuple(float(x) for x in varying.detach().cpu().tolist()),
        sampling_policy=f"linspace_{int(samples)}_fixed_{fixed_direction}={fixed_value}",
    )


def control_polygon_entity(
    surface: Any,
    edge: str,
    *,
    patch_id: int,
    role: str,
    entity_id: str,
) -> ReviewGeometryEntity:
    """Raw control-net row/column for one patch edge -- fitting data, not curve data."""
    grid = surface.control_grid.detach()
    if edge == "u0":
        points = grid[0]
    elif edge == "u1":
        points = grid[-1]
    elif edge == "v0":
        points = grid[:, 0]
    elif edge == "v1":
        points = grid[:, -1]
    else:
        raise ValueError(f"Unknown control edge: {edge!r}")
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_CONTROL_POLYGON,
        role=role,
        points=points,
        patch_ids=(int(patch_id),),
        parameter_edge=edge,
    )


def combine_ordered_patch_boundary(
    entities: list[ReviewGeometryEntity],
    *,
    entity_id: str,
    role: str,
    closed: bool,
    dedup_policy: str = "drop_duplicate_shared_endpoint",
) -> ReviewGeometryEntity:
    """Combine per-patch evaluated edges, in deterministic patch order, into one boundary.

    Adjacent patches are constructed so patch k's trailing sample equals patch
    (k+1)'s leading sample exactly; every entity after the first drops its
    first sample so the shared junction point is not duplicated.
    """
    torch = require_torch()
    if not entities:
        raise ValueError("combine_ordered_patch_boundary requires at least one entity.")
    if dedup_policy != "drop_duplicate_shared_endpoint":
        raise ValueError(f"Unknown shared-endpoint policy: {dedup_policy!r}")
    if any(entity.representation_kind != REPRESENTATION_EVALUATED_CURVE for entity in entities):
        raise ValueError("combine_ordered_patch_boundary only combines evaluated_curve entities.")
    combined = []
    patch_ids: list[int] = []
    for index, entity in enumerate(entities):
        points = _tensor_points(entity.points)
        combined.append(points if index == 0 else points[1:])
        patch_ids.extend(entity.patch_ids)
    points = torch.cat(combined, dim=0)
    if closed and int(points.shape[0]) > 1 and bool(torch.allclose(points[0], points[-1])):
        points = points[:-1]
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_EVALUATED_CURVE,
        role=role,
        points=points,
        patch_ids=tuple(patch_ids),
        closed=closed,
        sampling_policy=dedup_policy,
    )


def correspondence_chord_entity(
    points: Any,
    *,
    entity_id: str,
    role: str,
    patch_id: int,
) -> ReviewGeometryEntity:
    """A diagnostic straight-line evidence pairing -- not the evaluated support curve."""
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_CORRESPONDENCE_CHORD,
        role=role,
        points=points,
        patch_ids=(int(patch_id),),
    )


def observed_evidence_entity(
    points: Any,
    *,
    entity_id: str,
    role: str,
    source_component_id: int | None = None,
    source_loop_id: int | str | None = None,
    source_anchor_id: int | None = None,
    closed: bool = True,
    source_point_indices: tuple[int, ...] | None = None,
) -> ReviewGeometryEntity:
    return ReviewGeometryEntity(
        entity_id=entity_id,
        representation_kind=REPRESENTATION_OBSERVED_EVIDENCE,
        role=role,
        points=points,
        source_component_id=source_component_id,
        source_loop_id=source_loop_id,
        source_anchor_id=source_anchor_id,
        closed=closed,
        source_point_indices=source_point_indices,
    )


def fallback_curve_bundle_scale(curves: list[Any]) -> float:
    """Median positive consecutive-sample spacing of the curves being tested.

    This is a FALLBACK ONLY: it is resolution-dependent (finer sampling of the
    same physical curve shrinks this value), so it must not be the default
    tolerance basis for a real pipeline run. Callers with a stable geometric
    scale available (observed boundary/source-point spacing, local surface
    scale, component bbox scale) must pass ``tolerance_scale`` explicitly
    instead -- this exists so standalone/test callers without such a scale
    still get a deterministic number.
    """
    torch = require_torch()
    lengths = []
    for curve in curves:
        points = _tensor_points(curve)
        if int(points.shape[0]) > 1:
            lengths.append(torch.linalg.vector_norm(points[1:] - points[:-1], dim=1))
    if not lengths:
        return 1.0
    all_lengths = torch.cat(lengths)
    positive = all_lengths[all_lengths > 1e-12]
    return float(positive.median()) if int(positive.numel()) else 1.0


def _classify_interior_touch(
    interior_a: Any, interior_b: Any, tolerance: float, ambiguous_band: float
) -> tuple[str | None, float | None]:
    """Distinguish an isolated transversal crossing from a sustained overlapping run.

    A genuine single intersection's closest-approach point diverges quickly
    moving along either curve; two curves running alongside each other stay
    close at the neighboring sample too. When there are too few interior
    samples to check a neighbor at all, this returns ``near_touching_ambiguous``
    rather than guessing at a specific (and possibly wrong) defect kind --
    conservative under-classification, never a silent pass.
    """
    torch = require_torch()
    if int(interior_a.shape[0]) == 0 or int(interior_b.shape[0]) == 0:
        return None, None
    distances = torch.cdist(interior_a, interior_b)
    min_distance = float(distances.min())
    if min_distance > ambiguous_band:
        return None, min_distance
    if min_distance > tolerance:
        return CROSSING_NEAR_TOUCHING, min_distance
    flat_index = int(distances.argmin())
    na, nb = int(interior_a.shape[0]), int(interior_b.shape[0])
    i, j = divmod(flat_index, nb)
    candidates = []
    for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < na and 0 <= nj < nb:
            candidates.append(float(torch.linalg.vector_norm(interior_a[ni] - interior_b[nj])))
    if not candidates:
        return CROSSING_NEAR_TOUCHING, min_distance
    if min(candidates) <= tolerance * 2.0:
        return CROSSING_OVERLAPPING_SUPPORT_PATH, min_distance
    return CROSSING_TRANSVERSAL_INTERSECTION, min_distance


def classify_support_curve_pair(
    curve_a: Any,
    curve_b: Any,
    *,
    tolerance_scale: float,
    tolerance_fraction: float = 0.05,
    ambiguous_fraction: float = 0.2,
    expected_shared_point: Any | None = None,
    expected_shared_kind: str | None = None,
    endpoint_fraction: float = 0.15,
) -> dict[str, Any]:
    """Classify one pair of *actual evaluated* support curves for crossing.

    ``tolerance_scale`` MUST be a geometric scale independent of how finely
    these particular curves happen to be sampled (e.g. observed-boundary
    median spacing) -- ``tolerance_fraction``/``ambiguous_fraction`` only
    scale it further, they never substitute for it. ``expected_shared_point``/
    ``expected_shared_kind`` name a structurally known coincidence (e.g. a
    shared pole) so that intentional convergence is never reported as a
    defect; any other near-zero-distance touching is flagged, never silently
    accepted. A simple minimum-distance test cannot tell a transversal
    crossing from two curves running alongside each other, so a close-enough
    approach is further classified via :func:`_classify_interior_touch`; when
    even that cannot decide, the conservative ``near_touching_ambiguous``
    result is returned instead of guessing.
    """
    torch = require_torch()
    a, b = _tensor_points(curve_a), _tensor_points(curve_b)
    if int(a.shape[0]) < 2 or int(b.shape[0]) < 2:
        return {"classification": CROSSING_NOT_CHECKED, "reason": "curve_too_short", "min_distance": None}
    tolerance = max(float(tolerance_scale) * float(tolerance_fraction), 1e-9)
    ambiguous_band = max(float(tolerance_scale) * float(ambiguous_fraction), tolerance)

    distances = torch.cdist(a, b)
    flat_index = int(distances.argmin())
    b_count = int(b.shape[0])
    i, j = divmod(flat_index, b_count)
    min_distance = float(distances[i, j])
    a_count = int(a.shape[0])
    a_at_end = i == 0 or i == a_count - 1
    b_at_end = j == 0 or j == b_count - 1

    shared_ok = False
    if expected_shared_point is not None and a_at_end and b_at_end and min_distance <= tolerance:
        point = torch.as_tensor(expected_shared_point, dtype=a.dtype, device=a.device)
        if (
            float(torch.linalg.vector_norm(a[i] - point)) <= tolerance
            and float(torch.linalg.vector_norm(b[j] - point)) <= tolerance
        ):
            shared_ok = True

    if shared_ok:
        classification = CROSSING_VALID_SHARED_POLE if expected_shared_kind == "pole" else CROSSING_VALID_SHARED_ENDPOINT
        buffer_a = max(1, round(a_count * endpoint_fraction))
        buffer_b = max(1, round(b_count * endpoint_fraction))
        interior_a = a[buffer_a : a_count - buffer_a] if a_count > 2 * buffer_a else a[0:0]
        interior_b = b[buffer_b : b_count - buffer_b] if b_count > 2 * buffer_b else b[0:0]
        touch, interior_min = _classify_interior_touch(interior_a, interior_b, tolerance, ambiguous_band)
        if touch is not None:
            return {
                "classification": touch, "min_distance": min_distance, "interior_min_distance": interior_min,
                "closest_index_a": i, "closest_index_b": j,
            }
        return {"classification": classification, "min_distance": min_distance, "closest_index_a": i, "closest_index_b": j}

    touch, _ = _classify_interior_touch(a, b, tolerance, ambiguous_band)
    classification = CROSSING_NO_CROSSING if touch is None else touch
    return {"classification": classification, "min_distance": min_distance, "closest_index_a": i, "closest_index_b": j}


def detect_support_curve_crossings(
    curves: list[Any],
    *,
    tolerance_scale: float | None = None,
    tolerance_fraction: float = 0.05,
    ambiguous_fraction: float = 0.2,
    expected_shared_point: Any | None = None,
    expected_shared_kind: str | None = None,
    scope: str = CROSSING_SCOPE_REPRESENTATIVE_BUNDLE,
) -> dict[str, Any]:
    """Pairwise crossing check across a finite bundle of *actual evaluated* support curves.

    Control-polygon crossing is a different (weaker) diagnostic and must never
    substitute for this; callers should run this against
    ``evaluated_support_curves``, never ``support_control_polygons``. This is
    a REPRESENTATIVE, sampled check (``scope``), not a continuous NURBS
    non-intersection guarantee -- ``not_checked_categories`` names the related
    validity questions this function does not answer. If ``tolerance_scale``
    is omitted, :func:`fallback_curve_bundle_scale` derives one from the
    curves themselves (resolution-dependent -- convenience for standalone
    callers only, never the production path's choice).
    """
    if len(curves) < 2:
        return {
            "state": CROSSING_NOT_CHECKED,
            "reason": "fewer_than_two_curves",
            "scope": scope,
            "pairs": [],
            "has_invalid_crossing": False,
            "not_checked_categories": list(NOT_YET_CHECKED_CROSSING_CATEGORIES),
        }
    scale = float(tolerance_scale) if tolerance_scale is not None else fallback_curve_bundle_scale(curves)
    pairs = []
    invalid = False
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            result = classify_support_curve_pair(
                curves[i], curves[j], tolerance_scale=scale, tolerance_fraction=tolerance_fraction,
                ambiguous_fraction=ambiguous_fraction,
                expected_shared_point=expected_shared_point, expected_shared_kind=expected_shared_kind,
            )
            pairs.append({"curve_a": i, "curve_b": j, **result})
            if result["classification"] in CROSSING_INVALID_CLASSIFICATIONS:
                invalid = True
    return {
        "state": "checked",
        "scope": scope,
        "tolerance_scale": scale,
        "pairs": pairs,
        "has_invalid_crossing": invalid,
        "not_checked_categories": list(NOT_YET_CHECKED_CROSSING_CATEGORIES),
    }
