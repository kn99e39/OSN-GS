# 25. Renderer Local-Test Handoff

Use a WebGPU-capable local machine. Do not touch the active notebook or gaussian-splatting.

## Input

Serve WebRenderer through HTTP, not file://. Load matching point_cloud.ply and nurbs_surface.json from a synthetic run with two separated patches (crease is suitable). Record renderer revision, artifact path, run command/config, and seed.

## Tests and pass criteria

1. Run from WebRenderer: node --check util/NurbsGeometry.js; node --check main.js; node tests/nurbs_geometry_smoke_test.js. PASS: all exit 0.
2. Browser NURBS Surface and NURBS Curves: every valid patches[] entry appears, patch colors differ deterministically, reset camera contains all patches, and U/V lines end at their own patch. PASS: no console errors and both patches visible.
3. Diagnostics: check patch isolate, Gaussian patch-ID color, independent sampled-surface/U/V/control-grid/base-curve toggles, and perspective/orthographic comparison. Report absent controls as NOT IMPLEMENTED, not fitting failure.
4. Parity: for each patch evaluate Python and JS at four corners plus five fixed interior UVs. PASS: max Euclidean error <= 1e-6. On failure record patch id, UV, positions, degrees, shape, and knots.
5. Provenance: JSON must contain source path, CLI/config, seed, timestamp, and file hash; screenshot and validation report must identify the same hash. Missing fields are NOT IMPLEMENTED.

Return a PASS/FAIL/NOT IMPLEMENTED table with console errors and screenshots. Do not call a renderer image a NURBS collapse until all checks pass.

## TODO update

Moved renderer Priority 0 testing out of TODO.md into this handoff.
