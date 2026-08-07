from __future__ import annotations

"""Worklog 76: explicit `boundary_support_spacing` contract.

Worklog 72-74 measured a scale-domain mismatch in the dense region-owned
boundary-support certificate. The certificate searches for a continuation
candidate within `2.5 * scale`, but `scale` is the FULL-EVIDENCE sampling
spacing (median nearest-neighbour distance over every region-owned observed
point), while the objects actually being connected are BOUNDARY-SUPPORT
CANDIDATES -- a filtered, much sparser subset. Worklog 74 measured candidate
spacing at 2.08-3.72x full-evidence spacing, and the directionally-nearest
candidate for a failing half-line at a median of 3.89-4.79x full-evidence
spacing, i.e. routinely outside a 2.5x full-evidence radius but plausibly
inside a 2.5x *candidate* radius. That is a units error, not a threshold to
be tuned.

This module makes the three spacings explicit and semantically separate, and
supplies exactly three defensible estimators for the connectivity scale:

  * ``full_evidence_spacing``          -- the CURRENT production behaviour,
                                          unchanged, kept as the baseline.
  * ``region_boundary_support_spacing``-- one robust (median) nearest-neighbour
                                          spacing over the region's boundary-
                                          support candidates.
  * ``local_boundary_support_spacing`` -- a per-candidate robust local spacing
                                          (median of that candidate's own k
                                          nearest candidate distances), so a
                                          region with genuinely non-uniform
                                          boundary sampling is not forced onto
                                          a single global number.

`representative_spacing` is carried through untouched and stays REPORT-ONLY --
it is a third, independent quantity (worklog 32's per-representative
`mean_spacing`) and is never used as a connectivity scale here.

The distance multiplier (2.5) is NOT part of this contract and is not varied
per mode: the question this module exists to answer is which scale DOMAIN is
correct, not which constant produces loops.

`measure_edge_support_occupancy` is a pure DISCLOSURE metric for the main
safety risk of enlarging any connectivity radius -- an accepted edge that
spans observed empty space. It never accepts, repairs, or bridges anything.

MEASURED VERDICT (worklog 76, real baseline_compatible@2900, 7 regions):
both independent estimators were REJECTED for production use and the
production connectivity scale REMAINS `full_evidence_spacing`. They do fix
the units -- `no_candidate_within_local_scale` falls from 1108/1652 (67%) to
254 (region) and 21 (local, 1.3%) -- with no branch explosion and no proper
crossings. But they buy that continuity by connecting across observed empty
space: edges with an empty interior bin rise from 9/185 (4.9%) to 191/397
(48.1%) and 211/427 (49.4%), with the longest unsupported run reaching
75-92% of a single edge's length in every real region. That is gap bridging,
which is prohibited, and it purchases only 1 extra real closed loop whose
containment is still 99.78% interior-outside. Scale is therefore conclusively
NOT the bottleneck; the boundary-support candidate set is genuinely
non-adjacent. Do not re-open this as a threshold search -- see
`docs/worklogs/76_*.md`. The modes stay available because the three-way
spacing SEPARATION is the durable contract, but selecting an independent mode
in production requires new evidence that the gap-bridging above is resolved.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12

SPACING_MODE_FULL_EVIDENCE = "full_evidence_spacing"
SPACING_MODE_REGION_BOUNDARY_SUPPORT = "region_boundary_support_spacing"
SPACING_MODE_LOCAL_BOUNDARY_SUPPORT = "local_boundary_support_spacing"
SPACING_MODES = (
    SPACING_MODE_FULL_EVIDENCE,
    SPACING_MODE_REGION_BOUNDARY_SUPPORT,
    SPACING_MODE_LOCAL_BOUNDARY_SUPPORT,
)

# Neighbour count for the LOCAL estimator's robust per-candidate spacing. Small
# on purpose: it must describe this candidate's own boundary neighbourhood, and
# a median over 3 nearest candidates already rejects a single outlier without
# smearing in the far side of the region.
_LOCAL_NEIGHBOURS = 3


@dataclass(frozen=True)
class BoundarySupportSpacing:
    """The three spacings, kept semantically separate, plus the resolved
    per-candidate connectivity scale actually handed to the certificate."""

    mode: str
    full_evidence_spacing: float
    representative_spacing: float | None  # report-only, never a connectivity scale
    boundary_support_spacing: float | None  # region-level robust candidate spacing
    per_candidate_scale: tuple[float, ...]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.mode not in SPACING_MODES:
            raise ValueError(f"Unknown boundary support spacing mode: {self.mode!r}")


def _percentiles(values: Any) -> dict[str, float | None]:
    torch = require_torch()
    if int(values.numel()) == 0:
        return {"median": None, "p25": None, "p75": None, "p90": None, "max": None}
    return {
        "median": float(torch.quantile(values, 0.5)),
        "p25": float(torch.quantile(values, 0.25)),
        "p75": float(torch.quantile(values, 0.75)),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def candidate_nearest_neighbour_spacing(positions: Any) -> Any:
    """Per-candidate nearest-neighbour distance among the CANDIDATES only."""

    torch = require_torch()
    n = int(positions.shape[0])
    if n < 2:
        return torch.zeros((n,), dtype=positions.dtype, device=positions.device)
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    return distances.min(dim=1).values


def _local_robust_spacing(positions: Any, neighbours: int = _LOCAL_NEIGHBOURS) -> Any:
    """Per-candidate MEDIAN distance to its k nearest candidates -- robust to a
    single outlier neighbour in a way a bare nearest-neighbour distance is not."""

    torch = require_torch()
    n = int(positions.shape[0])
    if n < 2:
        return torch.zeros((n,), dtype=positions.dtype, device=positions.device)
    k = min(int(neighbours), n - 1)
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(k, dim=1, largest=False).values
    return nearest.median(dim=1).values


def resolve_boundary_support_spacing(
    mode: str,
    candidate_positions: Any,
    *,
    full_evidence_spacing: float,
    representative_spacing: float | None = None,
) -> BoundarySupportSpacing:
    """Resolve the per-candidate connectivity scale for ``mode``.

    Fail-safe by construction: if the boundary-support estimator degenerates
    (fewer than two candidates, or a non-positive robust spacing) the resolved
    scale falls back to ``full_evidence_spacing`` -- the current production
    value -- and says so in ``diagnostics``. A degenerate estimate must never
    silently produce a larger radius than the baseline.
    """

    torch = require_torch()
    if mode not in SPACING_MODES:
        raise ValueError(f"Unknown boundary support spacing mode: {mode!r}")
    n = int(candidate_positions.shape[0]) if candidate_positions is not None else 0
    baseline = float(full_evidence_spacing)

    if n == 0:
        return BoundarySupportSpacing(
            mode, baseline, representative_spacing, None, (), {"degenerate": "no_candidates"},
        )

    # float64 throughout: the resolved scale is compared against a Python-float
    # baseline, and a float32 round-trip would perturb `full_evidence` mode away
    # from the exact production value it is supposed to reproduce.
    candidate_positions = candidate_positions.to(torch.float64)
    nearest = candidate_nearest_neighbour_spacing(candidate_positions)
    region_spacing = float(nearest.median()) if n >= 2 else 0.0
    diagnostics: dict[str, Any] = {
        "candidate_count": n,
        "candidate_nn_spacing": _percentiles(nearest),
        "candidate_to_full_evidence_ratio": (
            _percentiles(nearest / max(baseline, _EPS)) if baseline > 0 else None
        ),
        "region_boundary_support_spacing": region_spacing if region_spacing > 0 else None,
        "fell_back_to_full_evidence": False,
    }

    if mode == SPACING_MODE_FULL_EVIDENCE:
        scale = torch.full((n,), baseline, dtype=candidate_positions.dtype, device=candidate_positions.device)
    elif mode == SPACING_MODE_REGION_BOUNDARY_SUPPORT:
        if region_spacing <= 0:
            diagnostics["fell_back_to_full_evidence"] = True
            region_spacing = baseline
        scale = torch.full((n,), region_spacing, dtype=candidate_positions.dtype, device=candidate_positions.device)
    else:  # SPACING_MODE_LOCAL_BOUNDARY_SUPPORT
        local = _local_robust_spacing(candidate_positions)
        degenerate = local <= 0
        if bool(degenerate.any()):
            diagnostics["fell_back_to_full_evidence"] = True
            diagnostics["locally_degenerate_candidates"] = int(degenerate.sum())
            local = torch.where(degenerate, torch.full_like(local, baseline), local)
        scale = local
        diagnostics["local_scale"] = _percentiles(scale)

    return BoundarySupportSpacing(
        mode=mode,
        full_evidence_spacing=baseline,
        representative_spacing=representative_spacing,
        boundary_support_spacing=(region_spacing if region_spacing > 0 else None),
        per_candidate_scale=tuple(float(x) for x in scale),
        diagnostics=diagnostics,
    )


def measure_edge_support_occupancy(
    edges: Sequence[tuple[int, int]],
    candidate_positions: Any,
    evidence_positions: Any,
    *,
    full_evidence_spacing: float,
) -> dict[str, Any]:
    """Disclosure-only: does each accepted edge actually run along observed
    evidence, or does it span empty space?

    Each edge is split into bins of ``full_evidence_spacing`` along its own
    axis. A bin is OCCUPIED when at least one region-owned evidence point
    projects into it within ``full_evidence_spacing`` of the edge axis. An edge
    with any empty interior bin spans a stretch of observed nothing -- exactly
    the failure mode that enlarging a connectivity radius risks introducing,
    and the reason this is measured rather than assumed.

    This function never modifies, accepts, or repairs an edge; it only counts.
    """

    torch = require_torch()
    spacing = max(float(full_evidence_spacing), _EPS)
    if not edges or int(evidence_positions.shape[0]) == 0:
        return {
            "edge_count": len(edges), "edges_with_empty_interior_bin": 0,
            "unsupported_edge_fraction": 0.0, "empty_bin_fraction": _percentiles(torch.zeros(0)),
            "max_unsupported_run_ratio": None, "measurement_only": True,
        }

    empty_fractions = []
    unsupported_runs = []
    unsupported_edges = 0
    for source, target in edges:
        a = candidate_positions[source]
        b = candidate_positions[target]
        axis = b - a
        length = float(axis.norm())
        if length <= _EPS:
            continue
        unit = axis / length
        offset = evidence_positions - a[None, :]
        projection = offset @ unit
        perpendicular = (offset - projection[:, None] * unit[None, :]).norm(dim=-1)
        bin_count = max(1, int(length / spacing))
        near_axis = perpendicular <= spacing
        inside = (projection >= 0.0) & (projection <= length) & near_axis
        if not bool(inside.any()):
            empty_fractions.append(1.0)
            unsupported_runs.append(1.0)
            unsupported_edges += 1
            continue
        bins = (projection[inside] / length * bin_count).clamp(0, bin_count - 1).long()
        occupied = torch.zeros((bin_count,), dtype=torch.bool, device=bins.device)
        occupied[bins] = True
        empty = int((~occupied).sum())
        empty_fractions.append(empty / bin_count)
        # Longest consecutive empty stretch, as a fraction of the edge -- a
        # single scattered empty bin is sampling noise; a long run is a gap.
        longest = run = 0
        for value in occupied.tolist():
            run = 0 if value else run + 1
            longest = max(longest, run)
        unsupported_runs.append(longest / bin_count)
        if empty > 0:
            unsupported_edges += 1

    empty_tensor = torch.tensor(empty_fractions, dtype=torch.float32) if empty_fractions else torch.zeros(0)
    run_tensor = torch.tensor(unsupported_runs, dtype=torch.float32) if unsupported_runs else torch.zeros(0)
    return {
        "edge_count": len(empty_fractions),
        "edges_with_empty_interior_bin": unsupported_edges,
        "unsupported_edge_fraction": (unsupported_edges / len(empty_fractions)) if empty_fractions else 0.0,
        "empty_bin_fraction": _percentiles(empty_tensor),
        "max_unsupported_run_ratio": (float(run_tensor.max()) if int(run_tensor.numel()) else None),
        "unsupported_run_ratio": _percentiles(run_tensor),
        "measurement_only": True,
    }
