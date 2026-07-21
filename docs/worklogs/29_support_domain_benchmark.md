# 29. Support-Domain Constructor Benchmark

날짜: 2026-07-15

## 작업

결정론적인 triangle, U-shape, crescent, planar-hole(annulus) scene을 추가했다. 각 scene은 analytic GT support predicate를 가지며 Gaussian center는 해당 영역 내부에서만 sample한다.

benchmark는 공통 XY grid에서 GT support와 trim을 반영한 generated support를 비교한다. coverage, unsupported/uncovered, precision/recall/IoU, component/hole/Euler topology mismatch, boundary Chamfer/Hausdorff를 측정한다. shared-XY support JSON/SVG와 patch별 UV occupancy/trim-mask JSON/SVG를 export하고 report.json에는 모든 artifact path를 기록한다.

## 검증

새 네 scene을 모두 CPU에서 실행해 finite result를 확인했다. planar-hole renderer export에는 uv_support와 diagnostic artifact path가 포함됐다.

## 남은 위험

raster metric은 resolution-dependent diagnostic이다. support calibration 동안 기존 global/local point-spacing metric도 함께 사용할 수 있다.
