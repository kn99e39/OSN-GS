from __future__ import annotations

"""Worklog 68: adjacency-aware LOCAL orientation-folding check.

`torch_parametric_diagnostics.py::compute_orientation_consistency` is a
FLAT/UNORDERED single-reference check: it builds ONE self-consistent
reference direction and reports each sample's dot product against it. That
deliberately conflates two very different things when applied to a regular
UV sample grid:

  - GLOBAL reversal: the whole patch's arbitrary reference direction (an
    eigenvector/cross-product sign is never physically meaningful by
    itself) happens to point one way rather than the other. Not a defect --
    flipping every sample's sign at once changes nothing about the surface.
  - LOCAL folding: a sample's normal disagrees with its immediate NEIGHBOR
    in the grid. This the single-reference check can equally under- or
    over-report, because a globally "mostly reversed" patch can still be
    perfectly smooth locally (every neighbor pair agrees, just all pointing
    the "wrong" way relative to the arbitrary global reference), while a
    locally folding patch can coincidentally still align with the global
    reference on average.

This module owns exactly the second (LOCAL, adjacency-aware) property, as
its own module per `torch_parametric_diagnostics.py`'s own documented
convention ("Topology-specific orientation-adjacency logic... is NOT this
module's job -- each owns its own wrapper"). It requires a REGULAR
``resolution x resolution`` UV sample grid (the same one every caller in
this codebase already samples for Jacobian diagnostics) and checks only
4-connected neighbor sign agreement -- never a global reference.
"""

from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-8


def compute_local_orientation_folding(normals: Any, resolution: int, *, eps: float = _EPS) -> dict[str, Any]:
    """``normals`` is ``(resolution*resolution, 3)``, row-major over a
    regular UV grid (same layout `surface.evaluate_with_derivatives` +
    ``torch.cross(deriv_u, deriv_v)`` produces). Returns 4-connected
    neighbor sign-agreement diagnostics -- a LOCAL, adjacency-only property,
    never compared against any single global reference direction.
    """

    torch = require_torch()
    grid = normals.reshape(resolution, resolution, 3)
    area = grid.norm(dim=-1)
    unit = grid / area.clamp_min(eps).unsqueeze(-1)

    dot_u = (unit[:-1, :, :] * unit[1:, :, :]).sum(dim=-1)  # neighbors along the u axis
    dot_v = (unit[:, :-1, :] * unit[:, 1:, :]).sum(dim=-1)  # neighbors along the v axis
    all_dots = torch.cat((dot_u.reshape(-1), dot_v.reshape(-1)))

    fold_mask = all_dots < 0.0
    fold_count = int(fold_mask.sum())
    total_pairs = int(all_dots.numel())

    return {
        "local_fold_count": fold_count,
        "local_adjacent_pair_count": total_pairs,
        "local_fold_fraction": (fold_count / total_pairs) if total_pairs else 0.0,
        "local_adjacent_dot_min": float(all_dots.min()) if total_pairs else None,
        "local_adjacent_dot_mean": float(all_dots.mean()) if total_pairs else None,
    }
