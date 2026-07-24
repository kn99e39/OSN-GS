from __future__ import annotations

"""Surface-agnostic AABB sweep-and-prune broad phase.

Phase E prerequisite (docs/Urgent_Work/OSN_GS_Phase_E_Bounded_Candidate_Design.md
section 3, impl plan section 7). A small, dependency-free reimplementation of the
expanded-AABB sweep-and-prune pattern that ``torch_surface_candidate_graph.py``
(deprecated Stage 2 diagnostics) established for voxel leaves.

Deliberately does NOT import or rewire ``torch_surface_candidate_graph.py``:
migrating that frozen module onto a shared helper would risk regressing its
regression-locked diagnostics results and buys nothing since Stage 2 is slated
for removal. The ~40-line overlap is a temporary, regression-safe duplication.

This module knows nothing about continuation domains, NURBS, cameras, or
occlusion -- it operates purely on labeled axis-aligned boxes with per-item
scales, returns canonical ordered pairs, and preserves raw and scale-normalized
AABB distances plus the expand factor / threshold actually used.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class BroadPhasePair:
    """One canonical ordered pair whose expanded boxes overlap and whose raw
    AABB distance is within ``expand_factor * max(scale_a, scale_b)``."""

    label_a: str
    label_b: str
    aabb_distance: float
    scale_normalized_aabb_distance: float
    scale_a: float
    scale_b: float
    expand_factor: float
    threshold: float

    def payload(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "aabb_distance": self.aabb_distance,
            "scale_normalized_aabb_distance": self.scale_normalized_aabb_distance,
            "scale_a": self.scale_a,
            "scale_b": self.scale_b,
            "expand_factor": self.expand_factor,
            "threshold": self.threshold,
        }


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _aabb_distance(aabb_min_a: Any, aabb_max_a: Any, aabb_min_b: Any, aabb_max_b: Any) -> float:
    torch = require_torch()
    gap = torch.maximum(aabb_min_a - aabb_max_b, aabb_min_b - aabb_max_a).clamp_min(0)
    return float(torch.linalg.norm(gap))


def sweep_and_prune_pairs(
    labels: Sequence[str],
    aabb_min: Any,
    aabb_max: Any,
    scales: Sequence[float],
    *,
    expand_factor: float = 1.0,
    tol: float = 1e-9,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> list[BroadPhasePair]:
    """Deterministic sweep-and-prune over labeled AABBs.

    ``aabb_min``/``aabb_max`` are ``(N, 3)`` tensors; ``scales`` is a length-N
    per-item scale used both to expand the boxes (``aabb +/- expand_factor *
    scale``) for the broad-phase overlap test and to normalize the surviving
    raw AABB distance. A pair survives iff their expanded boxes overlap on all
    three axes AND the raw ``aabb_distance <= expand_factor * max(scale_a,
    scale_b) + tol``.

    Output pairs are canonical (``label_a < label_b``) and sorted by
    ``(label_a, label_b)`` so the result is fully deterministic. ``excluded_pairs``
    (canonical tuples) are dropped -- used by callers to suppress self/duplicate
    source pairings before the exact distance check.
    """

    torch = require_torch()
    if not (len(labels) == int(aabb_min.shape[0]) == int(aabb_max.shape[0]) == len(scales)):
        raise ValueError("labels, aabb_min, aabb_max, scales must have matching length")
    if not (math.isfinite(float(expand_factor)) and expand_factor >= 0.0):
        raise ValueError(f"expand_factor must be finite and non-negative, got {expand_factor!r}")

    excluded = excluded_pairs or set()
    n = len(labels)
    if n < 2:
        return []

    scale_tensor = torch.as_tensor([float(s) for s in scales], dtype=aabb_min.dtype, device=aabb_min.device)
    safe_scale = scale_tensor.clamp_min(tol)
    radius = float(expand_factor) * safe_scale
    expanded_min = aabb_min - radius[:, None]
    expanded_max = aabb_max + radius[:, None]

    entries = list(range(n))
    entries.sort(key=lambda i: (float(expanded_min[i, 0]), labels[i]))

    pairs: list[BroadPhasePair] = []
    seen: set[tuple[str, str]] = set()
    active: list[int] = []
    for current in entries:
        current_min_x = float(expanded_min[current, 0])
        active = [other for other in active if float(expanded_max[other, 0]) >= current_min_x - tol]
        for other in active:
            # Broad-phase reject on the two non-sweep axes.
            if any(
                float(expanded_max[other, axis]) < float(expanded_min[current, axis]) - tol
                or float(expanded_max[current, axis]) < float(expanded_min[other, axis]) - tol
                for axis in (1, 2)
            ):
                continue
            label_pair = _canonical_pair(labels[other], labels[current])
            if label_pair[0] == label_pair[1] or label_pair in seen or label_pair in excluded:
                continue
            distance = _aabb_distance(
                aabb_min[other], aabb_max[other], aabb_min[current], aabb_max[current]
            )
            scale_a, scale_b = float(safe_scale[other]), float(safe_scale[current])
            pair_scale = max(scale_a, scale_b, tol)
            threshold = float(expand_factor) * pair_scale
            if distance > threshold + tol:
                continue
            seen.add(label_pair)
            # Order the recorded scales to match label ordering.
            scale_by_label = {labels[other]: scale_a, labels[current]: scale_b}
            pairs.append(
                BroadPhasePair(
                    label_a=label_pair[0],
                    label_b=label_pair[1],
                    aabb_distance=distance,
                    scale_normalized_aabb_distance=distance / pair_scale,
                    scale_a=scale_by_label[label_pair[0]],
                    scale_b=scale_by_label[label_pair[1]],
                    expand_factor=float(expand_factor),
                    threshold=threshold,
                )
            )
        active.append(current)

    pairs.sort(key=lambda pair: (pair.label_a, pair.label_b))
    return pairs
