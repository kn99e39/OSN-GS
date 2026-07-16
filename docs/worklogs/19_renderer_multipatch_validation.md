# 19. Renderer Multi-Patch Validation

Date: 2026-07-15

## Scope

Validated the already committed WebRenderer revision 88477a8 without touching the
active baseline notebook or the gaussian-splatting project.

## Confirmed implementation

- NurbsGeometry.buildGeometry() consumes every valid entry in patches[] and
  does not duplicate the top-level primary patch when the array exists.
- Each rendered patch receives a deterministic color variation.
- Surface and iso-line vertices are assembled per patch, so line segments do
  not join across patch boundaries.
- Geometry bounds cover all valid patches and are used by the existing
  camera-reset path.
- Flattened control grids, malformed patches, patch IDs, and skipped-patch
  reporting are covered by tests/nurbs_geometry_smoke_test.js.

## Verification

- Static code review confirmed the behavior in WebRenderer/util/NurbsGeometry.js
  and WebRenderer/main.js.
- The repository includes a Node smoke test for single-patch, multi-patch,
  flattened-grid, and malformed-patch cases.
- Runtime execution was not possible because node is not installed on this
  machine. No browser or WebGPU runtime test was attempted.

## Remaining Priority 0 work

- Patch isolate/toggle UI.
- Gaussian coloring by assigned patch ID.
- Independent toggles for sampled surface points, U/V iso-lines, control grid,
  and diagnostic curves.
- Python/JavaScript numerical NURBS parity test.
- Export provenance fields and artifact-to-run linkage.
- Perspective/orthographic comparison control.

## TODO update

Removed only the confirmed multi-patch rendering, deterministic patch-color,
cross-patch iso-line, and all-patch bounds items from TODO.md.
