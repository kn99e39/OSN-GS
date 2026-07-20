# 10. NURBS Derivatives and Foot-Point UV Binding (Phase A)

날짜: 2026-07-11

## 작업

- `TorchNURBSSurface`에 해석적 1차 도함수 평가를 추가했다. Cox-de Boor recursion이 degree-(p-1) basis를 함께 반환하도록 확장하고, 표준 B-spline 도함수 공식과 rational quotient rule로 `evaluate_with_derivatives()`와 `normals()`를 구현했다.
- `project_torch_points_to_nurbs()` foot-point projection(point inversion)을 구현했다. 조밀 UV 평가 grid에서 최근접 샘플로 초기화한 뒤 damped Gauss-Newton으로 정밀화하며, 잔차가 줄지 않으면 grid 초기값을 유지하므로 결과가 초기화보다 나빠지지 않는다. 전 과정이 `no_grad`로 동작하고 chunk 단위로 처리된다.
- `maintain_surface_from_certain()`에 `refresh_uv` 단계를 추가했다. patch 품질 검사 전에 certain Gaussian의 UV binding을 foot-point projection으로 재계산하므로, 품질 잔차가 "낡은 UV anchor까지의 거리"가 아니라 실제 point-to-surface 거리를 측정한다.
- local correction으로 추가되는 새 patch의 Gaussian binding도 PCA 평면 투영 대신 foot-point projection을 사용한다.
- config에 `surface_projection_iterations`를 추가했고 기존 `surface_projection_chunk_size`를 실제로 사용한다. maintenance 로그에 `uv_refreshed` 카운트를 출력한다.

## 결과

- 도함수는 float64 중심차분과 `atol=1e-4`에서 일치한다. 평면 patch의 normal은 z축과 정확히 정렬된다.
- 표면 위 점의 foot-point projection 잔차는 1e-5 이하다. Gauss-Newton은 어떤 점에서도 grid 초기화보다 잔차를 키우지 않는다.
- UV refresh 후 patch 최대 잔차 비율은 stale UV 기준 잔차보다 항상 작거나 같다 (회귀 테스트로 고정).

## 평가

Architecture가 요구하는 Gaussian-surface binding이 이제 실제 표면 최근접점 기반으로 유지된다. surface loss와 patch 품질 검사가 같은 정직한 대응점을 공유하므로, "표면이 틀렸는지 binding이 낡았는지" 구분 불가능하던 문제가 해소됐다. 도함수 평가는 Phase B의 parameter correction과 이후 curvature 정규화, normal binding의 기반이 된다.

## 검증

- 신규 `tests/test_nurbs_surface.py` 6개 테스트 (도함수 FD 대조, degree-0 축 처리, 평면 normal, on-surface 왕복, GN 비회귀, maintenance refresh 회귀).
- 기존 smoke/regression 15개 포함 전체 테스트 통과.

## 남은 위험

- Foot-point projection은 patch 단위 최근접이며 patch 간 재할당은 하지 않는다. 잘못된 patch에 binding된 Gaussian은 여전히 local correction 경로로만 회복된다.
- Gauss-Newton step은 ±0.25로 clamp되어 매우 길쭉한 patch에서는 수렴이 느릴 수 있다 (`surface_projection_iterations`로 조절).
