from __future__ import annotations

"""Worklog 99/100 -- pre-fit parametric domain validity for a Worklog 98
:class:`~osn_gs.surface.torch_latent_surface_tangent_frame_field.TangentFrameFieldComponent`.

Worklog 98 proves tangent-frame ORIENTATION coherence (no adjacent
transversal direction reversal, no holonomy inconsistency around a cycle).
That is necessary but not sufficient for a well-conditioned parameter
domain: the tree-integrated ``(u, v)`` map can still fold, singularize, or
stretch/compress extremely relative to the actual supported 3D spacing even
when every local orientation agrees. This module characterizes those
failure modes BEFORE any surface is fit, so a candidate patch-construction
architecture is never blamed for an input domain that was already broken.

Worklog 100 correction (two confounds identified in the original Worklog 99
version of this module, both fixed here):

1. Local Jacobians are now estimated from the component's OWN continuously
   supported source-graph adjacency (``tree_edges`` union the edges tested
   by ``holonomy_edges``) -- the exact same adjacency
   :func:`~osn_gs.surface.torch_latent_surface_tangent_frame_field.build_tangent_frame_field`
   already validated edge-by-edge for continuous support -- rather than a
   fresh UV-space kNN. A UV-space neighborhood can connect points that are
   NOT true source neighbors once the UV map is already distorted, which
   manufactures false folds instead of detecting real ones.

2. Local orientation is compared against the SYNCHRONIZED Worklog 98 frame
   (``e_u``, ``e_v`` at that same point; ``n_sync = normalize(e_u x e_v)``),
   never an independently-signed PCA/support normal. A single whole-chart
   flip of the entire parameter domain's orientation is gauge-equivalent
   (mathematically indistinguishable from swapping which side is "up"), not
   a real fold -- so the majority sign is first canonicalized as ONE global
   flip, and only orientation that reverses SPATIALLY relative to a point's
   immediate source-graph neighbors (after that canonicalization) is
   reported as a local fold.

Every check operates on the coherent component's own already-computed
``(u, v)`` -- nothing here repairs or reroutes through PCA, and nothing
here is informed by any downstream fit or held-out result.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# Fixed, structural constants (not tuned from any fit/held-out outcome):
DUPLICATE_UV_CELL_FRACTION = 1e-3  # relative to the component's own (u, v) extent
DUPLICATE_INCOMPATIBLE_POSITION_RATIO = 3.0  # multiple of local median 3D spacing
MIN_SOURCE_NEIGHBORS_FOR_JACOBIAN = 3  # minimum needed for a stable 2D LSQ fit


@dataclass(frozen=True)
class ParametricDomainValidityReport:
    valid: bool
    invalid_reasons: tuple[str, ...]
    u_extent: float
    v_extent: float
    duplicate_incompatible_count: int
    global_orientation_flip_applied: bool
    fold_fraction: float
    singular_fraction: float
    mean_condition_number: float | None
    max_condition_number: float | None
    stretch_ratio_p95: float | None
    area_distortion_p95: float | None
    shear_distortion_p95: float | None
    cycle_position_drift_p95_over_spacing: float | None


def _source_graph_adjacency(component: Any) -> dict[int, list[int]]:
    """All continuously supported source-graph edges the field already
    validated for this component -- the union of its spanning tree and the
    (non-tree) edges it tested for holonomy. This is exactly the same
    adjacency Worklog 98 built, never a fresh UV-space neighborhood."""

    adjacency: dict[int, set[int]] = {}

    def _add(a: int, b: int) -> None:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    for a, b in component.tree_edges:
        _add(a, b)
    for edge in component.holonomy_edges:
        _add(edge.node_a, edge.node_b)
    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


def _local_source_jacobian(
    component: Any, uv: Any, index: int, neighbor_indices: list[int],
) -> tuple[Any | None, Any | None]:
    """Least-squares local 2x2 map from source-tangent coordinates (in the
    point's OWN synchronized frame ``e_u``, ``e_v``) to UV-space deltas:
    for each source-graph neighbor j of point i,
    ``x_ij = [dot(p_j - p_i, e_u_i), dot(p_j - p_i, e_v_i)]`` versus
    ``y_ij = [u_j - u_i, v_j - v_i]``. Returns ``(J, singular_values)``, or
    ``(None, None)`` if too few continuously-supported source neighbors are
    available to support a 2D fit."""

    torch = require_torch()
    if len(neighbor_indices) < MIN_SOURCE_NEIGHBORS_FOR_JACOBIAN:
        return None, None

    neighbors = torch.tensor(neighbor_indices, dtype=torch.long, device=component.positions.device)
    delta_position = component.positions[neighbors] - component.positions[index]
    e_u_i = component.e_u[index]
    e_v_i = component.e_v[index]
    x = torch.stack([(delta_position * e_u_i).sum(dim=1), (delta_position * e_v_i).sum(dim=1)], dim=1)
    y = uv[neighbors] - uv[index]
    try:
        solution = torch.linalg.lstsq(x, y).solution  # (2, 2): source-tangent-coords -> uv
    except Exception:  # pragma: no cover - defensive
        return None, None
    jacobian = solution  # (2, 2)
    try:
        singular_values = torch.linalg.svdvals(jacobian)
    except Exception:  # pragma: no cover - defensive
        return jacobian, None
    return jacobian, singular_values


def assess_parametric_domain_validity(
    component: Any, uv: Any, median_spacing: float,
) -> ParametricDomainValidityReport:
    """Characterize one coherent component's already-computed ``(u, v)``
    map, sourcing local Jacobian neighbors from the component's own
    continuously-supported source-graph adjacency and comparing orientation
    against its synchronized ``e_u``/``e_v`` frame (never an independently
    signed PCA/support normal, never a fresh UV-space neighborhood)."""

    torch = require_torch()
    count = int(component.positions.shape[0])
    invalid_reasons: list[str] = []
    positions = component.positions

    u_extent = float((uv[:, 0].max() - uv[:, 0].min()).item())
    v_extent = float((uv[:, 1].max() - uv[:, 1].min()).item())
    if u_extent <= _EPS or v_extent <= _EPS:
        invalid_reasons.append("degenerate_uv_extent")

    # Duplicate/incompatible UV assignment: two points landing in the same
    # small UV cell but disagreeing geometrically beyond the local support
    # scale.
    cell_size_u = max(u_extent * DUPLICATE_UV_CELL_FRACTION, _EPS)
    cell_size_v = max(v_extent * DUPLICATE_UV_CELL_FRACTION, _EPS)
    cell_u = torch.round(uv[:, 0] / cell_size_u).to(torch.long)
    cell_v = torch.round(uv[:, 1] / cell_size_v).to(torch.long)
    duplicate_incompatible = 0
    cell_map: dict[tuple[int, int], int] = {}
    for index in range(count):
        key = (int(cell_u[index].item()), int(cell_v[index].item()))
        if key in cell_map:
            other = cell_map[key]
            distance = float((positions[index] - positions[other]).norm().item())
            if distance > DUPLICATE_INCOMPATIBLE_POSITION_RATIO * max(median_spacing, _EPS):
                duplicate_incompatible += 1
        else:
            cell_map[key] = index

    # Local source-tangent -> UV Jacobian at every point with enough
    # continuously-supported source-graph neighbors.
    adjacency = _source_graph_adjacency(component)
    determinants: dict[int, float] = {}
    condition_numbers: list[float] = []
    stretch_ratios: list[float] = []
    area_distortions: list[float] = []
    shear_distortions: list[float] = []
    singular_count = 0
    evaluated = 0
    for index in range(count):
        neighbor_indices = adjacency.get(index, [])
        jacobian, singular_values = _local_source_jacobian(component, uv, index, neighbor_indices)
        if jacobian is None:
            continue
        evaluated += 1
        determinants[index] = float(torch.linalg.det(jacobian).item())
        if singular_values is not None and int(singular_values.numel()) == 2:
            largest, smallest = float(singular_values[0].item()), float(singular_values[1].item())
            if smallest <= _EPS * max(largest, 1.0):
                singular_count += 1
            else:
                condition_numbers.append(largest / smallest)
                # Local area distortion: |det J| relative to the isotropic
                # scale implied by the singular values (1.0 = perfectly
                # conformal-area-preserving at this point's own scale).
                area_distortions.append(abs(largest * smallest))
                # Local angular/shear distortion: ratio of singular values
                # (1.0 = no shear; growing values = increasingly sheared).
                shear_distortions.append(largest / smallest)
            stretch_ratios.append(largest / max(median_spacing, _EPS))

    # Canonicalize a single chart-wide orientation flip BEFORE declaring any
    # local fold: a whole-component sign reversal of every determinant is
    # gauge-equivalent (equivalent to swapping u/v handedness once for the
    # entire chart), not a genuine fold.
    global_flip_applied = False
    if determinants:
        positive_count = sum(1 for value in determinants.values() if value > 0)
        negative_count = len(determinants) - positive_count
        if negative_count > positive_count:
            determinants = {index: -value for index, value in determinants.items()}
            global_flip_applied = True

    # A point is a genuine LOCAL fold only if its own (canonicalized) sign
    # disagrees with AT LEAST ONE of its own source-graph neighbors' signs
    # -- i.e. orientation changes SPATIALLY somewhere across that edge,
    # relative to the synchronized frame, not relative to any single
    # absolute reference. (A majority-vote rule would dilute exactly the
    # boundary edge that constitutes the fold, since both sides of a real
    # fold line are themselves locally self-consistent.)
    fold_count = 0
    for index, sign in determinants.items():
        own_positive = sign > 0
        for neighbor in adjacency.get(index, []):
            neighbor_sign = determinants.get(neighbor)
            if neighbor_sign is None:
                continue
            if (neighbor_sign > 0) != own_positive:
                fold_count += 1
                break

    fold_fraction = fold_count / evaluated if evaluated else 0.0
    singular_fraction = singular_count / evaluated if evaluated else 0.0
    mean_condition = sum(condition_numbers) / len(condition_numbers) if condition_numbers else None
    max_condition = max(condition_numbers) if condition_numbers else None
    stretch_p95 = (
        float(torch.quantile(torch.tensor(stretch_ratios), 0.95).item()) if stretch_ratios else None
    )
    area_p95 = (
        float(torch.quantile(torch.tensor(area_distortions), 0.95).item()) if area_distortions else None
    )
    shear_p95 = (
        float(torch.quantile(torch.tensor(shear_distortions), 0.95).item()) if shear_distortions else None
    )

    if fold_fraction > 0.0:
        invalid_reasons.append("uv_orientation_reversal_or_foldover")
    if singular_fraction > 0.0:
        invalid_reasons.append("local_jacobian_singularity")
    if duplicate_incompatible > 0:
        invalid_reasons.append("duplicate_uv_incompatible_geometry")

    return ParametricDomainValidityReport(
        valid=len(invalid_reasons) == 0,
        invalid_reasons=tuple(invalid_reasons),
        u_extent=u_extent, v_extent=v_extent,
        duplicate_incompatible_count=duplicate_incompatible,
        global_orientation_flip_applied=global_flip_applied,
        fold_fraction=fold_fraction, singular_fraction=singular_fraction,
        mean_condition_number=mean_condition, max_condition_number=max_condition,
        stretch_ratio_p95=stretch_p95,
        area_distortion_p95=area_p95, shear_distortion_p95=shear_p95,
        cycle_position_drift_p95_over_spacing=None,  # filled in by the caller from the field's holonomy edges
    )


def cycle_position_drift_p95(
    component: Any, median_spacing: float,
) -> float | None:
    """Position-based companion to Worklog 98's own orientation-based
    holonomy check: for every SUPPORTED cycle-closing edge already tested
    for orientation, additionally measure how far the tree-derived
    ``(u, v)`` at the far endpoint disagrees with what a direct chord-length
    step across that edge (from the near endpoint's own frame) would
    predict. Reported purely as a diagnostic drift magnitude, in units of
    local median 3D spacing. Unaffected by the Worklog 100 corrections
    above (already position-based, already uses the synchronized frame)."""

    torch = require_torch()
    drifts: list[float] = []
    for edge in component.holonomy_edges:
        a, b = edge.node_a, edge.node_b
        delta = component.positions[b] - component.positions[a]
        predicted_u = component.u[a] + (delta * component.e_u[a]).sum()
        predicted_v = component.v[a] + (delta * component.e_v[a]).sum()
        actual_u, actual_v = component.u[b], component.v[b]
        drift = torch.sqrt((predicted_u - actual_u) ** 2 + (predicted_v - actual_v) ** 2)
        drifts.append(float(drift.item()) / max(median_spacing, _EPS))
    if not drifts:
        return None
    return float(torch.quantile(torch.tensor(drifts), 0.95).item())
