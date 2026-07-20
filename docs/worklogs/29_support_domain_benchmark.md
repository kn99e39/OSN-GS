# 29. Support-Domain Constructor Benchmark

Date: 2026-07-15

## Work

Added deterministic triangle, U-shape, crescent, and planar-hole (annulus) scenes. Each has an analytic GT support predicate and samples Gaussian centers only inside it.

The benchmark now compares GT and trim-respecting generated support on a common XY grid: coverage, unsupported/uncovered, precision/recall/IoU, component/hole/Euler topology mismatch, and boundary Chamfer/Hausdorff. It exports shared-XY support JSON/SVG plus per-patch UV occupancy/trim-mask JSON/SVG, and report.json records all artifact paths.

## Verification

CPU runs of all four new scenes passed with finite results. A planar-hole renderer export included uv_support and diagnostic artifact paths.

## Remaining risks

Raster metrics are resolution-dependent diagnostics; existing global/local point-spacing metrics remain available during support calibration.
