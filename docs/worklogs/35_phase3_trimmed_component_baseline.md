# 35. Phase 3 Trimmed Component 정확성 기준선

날짜: 2026-07-20

## 수행 내용

- Final Boundary governing plan을 기준으로 진행 중인 Phase 3 component-level trimmed NURBS baseline을 검증했다.
- legacy 및 voxel_patch_stage1 construction path를 변경하지 않고 기존 IDW seed, regularized LSQ, foot-point correction fitter를 재사용했다.
- support-domain 평가가 patch-union 평가와 동일한 extent-adaptive trim-aware rasterization을 사용하도록 수정했다. output cell마다 sample 하나만 사용하던 방식은 trimmed surface를 조각내어 수천 개의 가짜 hole을 보고했다.
- Phase 2와 동일하게 20셀 이하 loop는 tiny diagnostic artifact로 표시하되 significant topology hole로 보고하지 않는다.

## 결과

points=600, seed=0 기준:

- plane: component/patch 1개, support IoU 0.981, significant hole 0개, Jacobian degeneracy 없음.
- sine: component/patch 1개, support IoU 0.981, significant hole 0개, Jacobian degeneracy 없음.
- planar_hole: component 1개와 geometry patch 1개, significant hole 1개, uncovered support 0, support IoU 0.916, active-active seam 없음.
- crease와 close_parallel_sheets는 ARI 1.0으로 두 component/patch를 유지했다.
- Phase 3 fitter test와 ground-truth NURBS test를 모두 통과했다(10개 test).

## 평가

이 결과는 필수 correctness baseline이다. control grid는 hole을 가로지를 수 있지만 trim mask가 render/evaluation support에서 hole 영역을 제외한다. 구현은 benchmark 전용이며 trainer나 ADC 동작을 변경하지 않는다.

## 남은 위험

- density_gradient는 support IoU 0.772, uncovered fraction 0.093과 함께 significant support gap 하나를 계속 보고한다. 이는 planar-hole trim contract의 문제가 아니라 sparse-support calibration과 inactive input leaf에서 발생한다.
- 20셀 significant-hole threshold는 기본 128x128 support raster에 맞춰 조정된 값이며 resolution 변경 시 재검증해야 한다.
- Phase 4 boundary-conforming chart는 별도의 gated phase로 남아 있다.
