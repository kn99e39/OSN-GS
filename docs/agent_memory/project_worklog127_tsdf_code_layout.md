---
name: project-worklog127-tsdf-code-layout
description: WL127 TSDF code family layout, the construction/attribution isolation rule, and the closure + caching gotchas
metadata:
  type: project
---

`scripts/devtools/evidence_bounded_tsdf/` is a deliberately ISOLATED module family:

- CONSTRUCTION half (`scale`, `field`, `extraction`, `mesh_ops`, `synthetic`) must import NOTHING
  historical (topology/boundary/region/chart/KNN/NURBS/Trust/occluded). That isolation IS the
  control experiment; `tests/test_evidence_bounded_projective_tsdf.py` enforces it with AST checks.
  `field.project_world_points` is re-implemented rather than imported and is asserted BITWISE equal
  to the frozen `observed_occluded.shared.project_queries`.
- `attribution.py` is the ONLY module allowed to read historical quantities, and only after the
  mesh exists. `scripts/devtools/evidence_bounded_tsdf_stages.py` holds the baseline replay/exports.

Gotchas worth remembering:

- **skimage's `marching_cubes(mask=...)` checks ONE cell corner, not all eight** (measured). The
  8-corner contract is implemented as sentinel-fill + discard-triangles-from-ineligible-cells, which
  is provably identical because MC is per-cell; a regression test runs opposite sentinels and
  compares kept triangles.
- **Closure enumeration must test only the PREVIOUS round's new voxels' shell**, not the whole
  field's shell. Rejections are permanent, so it reaches the same fixed point -- verified bitwise
  identical and 22.7x faster, and reproduced the exact same 76,720,314 authoritative voxels at full
  scale. A regression test pins the equivalence.
- `accumulate_image_space_pairs` needs **(H, W)** representative maps, not flattened ones.
- The driver has a `--cache` directory (field / mesh / evidence / raycast `.npz`). It stores ONLY
  results, never a parameter, and refuses to load a field built at a different h/mu. Without it a
  rerun costs ~3.5 h; with it, ~48 min.
- The historical replay's exact-KNN `build_candidate_graph` needs GPU headroom; the driver parks the
  field, mesh tensors and mesh depth maps on CPU around it.
