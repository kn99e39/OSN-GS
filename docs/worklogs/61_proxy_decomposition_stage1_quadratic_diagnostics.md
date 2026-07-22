# Worklog 61: Proxy-Based Surface Decomposition Stage 1 Quadratic 진단

날짜: 2026-07-22

상태: **Stage 1 완료. Diagnostics-only, production 변경 없음. Stage 2는 사용자 승인 대기.**

## 목표

Local quadratic height-field proxy가 smooth curved 연결 후보와 crease/parallel/disconnected negative control을 구분할 근거를 제공하는지 검증했다. 이 단계에서는 candidate graph, component merge, threshold/admissibility를 구현하지 않았다.

## 변경 파일

- `osn_gs/surface/torch_surface_proxy.py`
- `scripts/devtools/analyze_surface_proxy.py`
- `tests/test_surface_proxy.py`
- `artifacts/proxy_decomposition_stage1.json`

## 구현 내용

- PCA local frame에서 tangent 좌표와 normal 높이를 동일한 isotropic support scale로 정규화했다.
- `z = ax^2 + bxy + cy^2 + dx + ey + f`를 regularized weighted LSQ로 fitting했다.
- proxy별 world/normalized RMS·max residual, plane residual, condition number, planarity, curvature proxy, residual concentration, validity 사유를 기록했다.
- pair merge error는 서로 다른 child RMS를 직접 빼지 않고, child world SSE를 합산한 뒤 merged support scale로 한 번만 정규화했다.
- merge score를 만들지 않고 proxy error, support gap, normal variation, layer separation을 독립 diagnostics로 유지했다.
- support gap은 전체 point 분포에 지배되지 않도록 symmetric nearest-support 2% quantile을 사용했다.
- layer separation은 `normal_offset / tangent_distance`를 양쪽 proxy에서 계산해, 곡률에 따른 normal offset과 실제 parallel-layer 방향을 분리했다.

설정값: `regularization=1e-6`, `support_gap_quantile=0.02`, 계산 dtype `float64`.

## 정량 결과

아래 값은 `count=600`, `seed=0` 실제 voxel leaf pair 범주의 중앙값이다.

| 범주 | pair 수 | normalized error increase | merged normalized RMS | support gap / spacing | normal angle | layer score |
|---|---:|---:|---:|---:|---:|---:|
| curved_annulus missing cross-component | 4 | 3.96e-5 | 0.00643 | 4.19 | 16.55° | 0.108 |
| curved_annulus existing merged | 14 | 3.48e-5 | 0.00716 | 1.99 | 16.13° | 0.096 |
| mild_curved_sheet merged | 12 | 2.32e-8 | 0.000184 | 1.92 | 11.90° | 0.099 |
| crease rejected | 4 | 1.30e-3 | 0.0361 | 22.43 | 48.46° | 0.293 |
| parallel layers rejected | 4 | 2.11e-2 | 0.145 | 1.98 | 0.00° | 1.97 |
| density_gradient merged | 4 | 8.13e-4 | 0.0325 | 2.33 | 15.18° | 0.116 |

Analytic controls:

- disconnected coplanar pair: quadratic error 0이지만 support gap/spacing 18.0으로 분리된다.
- high-curvature smooth pair: normal angle 58.9°지만 merged RMS 8.95e-7, quadratic/plane RMS ratio 3.46e-6으로 매끄러운 단일 proxy가 성립한다.

## 평가

1. **Quadratic proxy는 `curved_annulus` 누락 pair에 유효하다.** 누락 pair의 error/RMS가 같은 장면의 기존 merged pair와 같은 규모이며 crease·parallel보다 작다.
2. **Parallel layer는 명확히 분리된다.** merged quadratic RMS가 가장 크고 quadratic/plane RMS ratio가 약 0.99라 quadratic이 두 층을 하나의 surface로 설명하지 못하며 layer score도 1.97로 다른 실제 smooth 범주보다 높다.
3. **Disconnected coplanar support는 proxy error로 분리할 수 없다.** support gap이 반드시 독립 신호여야 한다.
4. **단일 proxy-error threshold는 부족하다.** 정상 `density_gradient` pair의 error가 crease 범위와 일부 겹친다.
5. **단일 normal-angle threshold도 부족하다.** high-curvature smooth control이 crease보다 큰 normal angle을 보이면서 quadratic proxy에는 거의 정확히 맞는다.
6. Residual concentration과 condition number는 이번 fixture에서 결정적인 분리 신호가 아니었지만 invalid/수치 불안정 감시용으로 유지할 가치가 있다.

## 검증

- `python -B -m unittest tests.test_surface_proxy -v`: 10개 통과.
- 전체 suite: **171 passed, 1 skipped**.
- Stage 1 artifact를 동일 입력으로 두 번 생성한 SHA-256이 일치했다: `15E0EDD5501507543F1E501001F2D9701A24441982F9ABFE54A9F6319279A882`.

## 결론

Stage 2 spatial candidate graph로 진행할 기술적 근거는 충분하다. 특히 현재 face adjacency가 누락한 `curved_annulus` cross-component pair가 기존 smooth merged pair와 비슷한 quadratic distortion을 보였다.

다만 Stage 2/3에서 하나의 weighted score나 threshold를 지금 확정하면 안 된다. Candidate graph는 recall과 계산량만 담당하고, admissibility는 proxy error·support gap·normal variation·layer direction을 독립적으로 기록한 뒤 실제 scene sweep에서 결정해야 한다.

## 남은 위험 및 승인 요청

- Quadratic height field는 local proxy이며 large/vertical/multi-valued component의 최종 표현이 아니다.
- Agglomeration이 커질수록 하나의 region frame에서 quadratic validity가 떨어질 수 있다.
- `mild_curved_sheet`의 spurious annulus는 Phase 2 문제라 Stage 2 candidate graph로 자동 해결되지 않는다.
- outer-boundary conformance도 이번 범위 밖이다.
- 문서 지침에 따라 여기서 멈춘다. **Stage 2 candidate graph 구현은 사용자 승인 후에만 진행한다.**
