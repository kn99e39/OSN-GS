from __future__ import annotations

"""W174 pre-batch probe: can construction-native zero-set triangles be recovered?

DIAGNOSTIC ONLY.  This script does not perform the W174 analysis.  It answers
one question before that batch is approved: does a deterministic marching-cubes
re-extraction from the FROZEN stored corner values reproduce geometry that is
consistent with the frozen W154 zero-surface samples, cell for cell?

Nothing is written into any historical output directory and no frozen artifact
is modified.  The probe reads the W172 selected tabletop rows only.

The comparison is deliberately conservative:

  * Triangles come from the same 8 corner scalars the W154 extractor already
    stored per cell, so cell ownership is exact by construction and no
    distance, radius, normal or threshold matching is used anywhere.
  * The stored `world_xyz` is the mean of the cube-edge zero crossings
    (surface-nets style), NOT a marching-cubes vertex, so the two are not
    expected to be bitwise equal.  Agreement is therefore reported as a
    distribution in units of h, alongside the exact containment test that
    every re-extracted triangle lies inside its owning cell.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from devtools.demo import worklog_172_fragmentation_gate_vs_structural_fit_audit as w172
from devtools.demo import worklog_173_tabletop_multi_loop_domain_attribution as w173

w171 = w172.w171
CASE = "real_tabletop_coherent"

# Corner ordering used by the frozen W154 extractor, reproduced verbatim from
# osn_gs/surface/torch_gaussian_region_owned_tsdf.py so the probe cannot drift
# from the historical convention.
CORNER_OFFSETS = np.asarray(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
    dtype=np.int64,
)


def load_selected_corner_values():
    """Read the frozen corner scalars for exactly the W172 selected rows.

    Row identity is re-derived through the same selection the historical batch
    used, then checked against the stored selected-support arrays, so a silent
    reordering upstream would fail here rather than corrupt the probe.
    """
    case_dir = w172.OUT / "case_artifacts" / CASE
    stored = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    spec = stored["baseline"]["selection"]
    source = w171.DEFAULT_W154 / "candidate_f_tsdf_surface_samples.npz"

    with np.load(source) as data:
        xyz = data["world_xyz"]
    with np.load(w171.DEFAULT_W154 / "candidate_f_region_owned_support.npz") as data:
        owned = data["owned_region_id"]
        mask = data["accepted_mask"].astype(bool) & (owned == spec["prior_region_id"])
    mask &= (np.all(xyz >= np.asarray(spec["aabb_min"]), axis=1)
             & np.all(xyz <= np.asarray(spec["aabb_max"]), axis=1))
    complete_indices = np.flatnonzero(mask)
    del xyz, owned, mask

    with np.load(case_dir / "selected_support.npz") as data:
        row_indices = data["complete_row_indices"].copy()
        selected_cells = data["cell_indices"].copy()
        selected_xyz = data["world_xyz"].copy()
        selected_keys = data["source_cell_keys"].copy()

    selected_indices = complete_indices[row_indices]
    with np.load(source) as data:
        corner_values = data["corner_values"][selected_indices].copy()
        check_cells = data["cell_indices"][selected_indices].copy()
        check_keys = data["source_cell_keys"][selected_indices].copy()

    # Frozen-identity guard: the rows we just pulled must be the W172 rows.
    assert np.array_equal(check_cells, selected_cells), "cell_indices drift"
    assert np.array_equal(check_keys, selected_keys), "source_cell_keys drift"
    assert len(selected_indices) == 15189, len(selected_indices)
    return corner_values, selected_cells, selected_xyz, float(stored["baseline"]["support"]["h"])


def reextract_cell_triangles(corner_values, cells, h):
    """Marching-cubes one authoritative cell at a time, from stored corners.

    Each cell is fed to skimage as its own 2x2x2 block, so a triangle can only
    ever be attributed to the cell whose scalars produced it.  This is why no
    spatial matching is needed: ownership is structural, not geometric.
    """
    from skimage.measure import marching_cubes

    grid_of = np.zeros((2, 2, 2), dtype=np.float64)
    per_cell_triangles = np.zeros(len(cells), dtype=np.int64)
    centroid_offsets = np.full((len(cells), 3), np.nan, dtype=np.float64)
    containment_violations = 0
    degenerate_cells = 0

    for row in range(len(cells)):
        values = corner_values[row].astype(np.float64)
        for corner, (dx, dy, dz) in enumerate(CORNER_OFFSETS):
            grid_of[dx, dy, dz] = values[corner]
        try:
            verts, tris, _n, _v = marching_cubes(
                grid_of, level=0.0, step_size=1, method="lewiner", allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            # A cell whose scalars touch zero only at a corner/edge can fail to
            # produce a surface.  Recorded, never silently dropped.
            degenerate_cells += 1
            continue
        if tris.shape[0] == 0:
            degenerate_cells += 1
            continue
        per_cell_triangles[row] = int(tris.shape[0])
        corners = verts[tris]
        if not np.all((corners >= -1e-9) & (corners <= 1.0 + 1e-9)):
            containment_violations += 1
        # Local unit-cell centroid -> world, using the frozen cell convention
        # (cell index + local offset + 0.5) * h from the W154 extractor.
        local_centroid = corners.reshape(-1, 3).mean(axis=0)
        centroid_offsets[row] = (cells[row].astype(np.float64) + local_centroid + 0.5) * h

    return per_cell_triangles, centroid_offsets, containment_violations, degenerate_cells


def distribution(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def main():
    corner_values, cells, stored_xyz, h = load_selected_corner_values()

    counts, centroids, violations, degenerate = reextract_cell_triangles(corner_values, cells, h)
    produced = counts > 0
    offset = np.linalg.norm(centroids - stored_xyz.astype(np.float64), axis=1)

    report = {
        "status": "FEASIBILITY_PROBE_ONLY_NO_W174_ANALYSIS",
        "case": CASE,
        "h": h,
        "selected_cells": int(len(cells)),
        "frozen_inputs_modified": False,
        "reextraction_source": "stored corner_values of the frozen W154 samples; TSDF field not reloaded",
        "ownership_rule": "one authoritative cell decoded into its own 2x2x2 block; triangles cannot cross cells",
        "no_distance_or_threshold_matching": True,
        "cells_producing_triangles": int(produced.sum()),
        "cells_without_triangles": int(degenerate),
        "triangle_total": int(counts.sum()),
        "per_cell_triangle_count": distribution(counts[produced]),
        "unit_cell_containment_violations": int(violations),
        "centroid_offset_world": distribution(offset),
        "centroid_offset_in_h": distribution(offset / h),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
