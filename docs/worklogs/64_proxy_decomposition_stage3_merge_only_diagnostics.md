# Proxy-Based Surface Decomposition Stage 3 Merge-Only 진단

날짜: 2026-07-22

상태: **Stage 3 diagnostics-only prototype 구현 및 검증 완료. 광범위 feasibility gate 실패. Stage 4 진행 금지.**

## 목표

Stage 2 spatial candidate graph 위에서 local quadratic proxy, support gap, layer consistency를 독립 gate로 사용하는 deterministic merge-only agglomeration이 `curved_annulus` 연결 복원과 crease/parallel/disconnected 보호를 동시에 달성하는지 검증했다.

## 변경 파일

- `osn_gs/surface/torch_surface_decomposition.py`
- `scripts/devtools/analyze_surface_decomposition.py`
- `tests/test_surface_decomposition.py`
- `artifacts/proxy_decomposition_stage3.json`
- `artifacts/proxy_decomposition_stage3_production_benchmark.json/report.json`

Production `build_surface_components`, Phase 2 boundary/topology, NURBS fitting 파일은 수정하지 않았다.

## 구현 내용

- Atomic adaptive leaf 하나를 초기 region 하나로 사용한다.
- Stage 2 candidate leaf edge의 union을 region adjacency provenance로 유지한다.
- 모든 pair에서 child/merged proxy, normalized RMS/error increase, condition, symmetric support gap/spacing, layer direction, normal variation, residual concentration, support scale, point count/support mass를 먼저 계산한다.
- Ordered gate는 `invalid_proxy -> insufficient_support -> disconnected_support -> multi_layer_inconsistency -> excessive_proxy_distortion -> excessive_error_increase` 순서다. 모든 gate raw 값과 pass/fail을 보존하며 short-circuit하지 않는다.
- Multi-layer gate는 layer direction, merged/child RMS ratio, 그리고 normalized error increase 또는 diffuse residual concentration의 명시적 boolean conjunction이다. Weighted scalar score는 없다.
- Priority는 admissibility와 분리된 단일 `merged_normalized_quadratic_rms` 오름차순이다. 동률은 canonical member leaf ID로 결정한다.
- Heap entry는 active region과 current adjacency를 확인해 stale entry를 무효화한다. Merge 후 영향받는 neighbor pair만 전체 diagnostics를 다시 계산한다.
- Final region의 member leaf IDs, leaf-to-region map, 전체 pair evaluation, merge history, stale count를 저장한다.
- Runtime entry는 scene name, GT topology, GT component count를 받지 않는다. GT label은 runner에서 결과 생성 후 purity 평가에만 쓴다.

## 사용한 provisional 기본 설정

| config | 기본값 |
|---|---:|
| `max_normalized_proxy_rms` | 0.1 |
| `max_normalized_error_increase` | 0.01 |
| `max_support_gap_over_spacing` | 4.0 |
| `max_layer_separation` | 0.2 |
| `min_layer_rms_ratio` | 1.0 |
| `min_layer_normalized_error_increase` | 0.001 |
| `max_layer_residual_concentration` | 2.0 |
| `minimum_support` | 6 |
| `proxy_regularization` | 1e-6 |
| `candidate_radius_factor` | 0.25 |
| `candidate_max_neighbors` | 0 |
| `support_gap_quantile` | 0.02 |
| `max_proxy_condition_number` | 1e10 |

이는 production default가 아니며 Stage 3 artifact용 provisional config다.

## Threshold sweep 범위

- RMS: 0.05, 0.075, 0.1, 0.125
- error increase: 0.003, 0.006, 0.01, 0.02
- gap/spacing: 2, 3, 3.5, 4, 5, 6, 8
- layer separation: 0.1, 0.15, 0.2, 0.25, 0.3, 0.5
- layer RMS ratio: 0.5, 1, 2, 5
- layer error evidence: 0.0005, 0.001, 0.002
- layer residual concentration: 1.5, 2, 2.5, 3
- minimum support: 6, 10, 20
- regularization: 1e-8, 1e-6, 1e-4
- candidate radius: 0, 0.1, 0.25, 0.5
- candidate max neighbors: 0, 4, 8
- support quantile: 0.01, 0.02, 0.05, 0.1
- condition limit: 1e3, 1e6, 1e10

Seed-0 compact suite를 동시에 통과한 독립 구간은 RMS 0.1–0.125, error 0.01–0.02, gap 3.5–4.0, layer 0.15–0.25, support quantile 0.02였다. 이 구간은 아래 seed/distance 실패를 해결하지 못한다.

## 기본 fixture 정량 결과

| case | initial/edges | final | merges | evaluations | stale | 주요 reject |
|---|---:|---:|---:|---:|---:|---|
| curved_annulus | 12/39 | 1 | 11 | 83 | 26 | gap 46 |
| crease | 8/28 | 2 | 6 | 49 | 12 | gap 25, layer 6 |
| close_parallel_sheets | 8/28 | 2 | 6 | 49 | 28 | gap 7, layer 8 |
| density_gradient | 5/10 | 1 | 4 | 16 | 7 | gap 5 |
| disconnected gap 0.1 | 10/22 | 2 | 8 | 50 | 17 | gap 25 |
| high-curvature smooth | 8/28 | 1 | 7 | 49 | 27 | gap 15 |

Plane 및 planar annulus 4종도 모두 final region 1을 유지했다. Production `curved_annulus`는 실제 benchmark에서 기존 2 components (`disk_like`, `complex`) 그대로이며 diagnostics 결과가 production membership을 바꾸지 않았다.

## 안정성 및 결정성

- `curved_annulus` x/y/z 30/60/90도 회전 9종: 모두 region 1.
- point count 300/600/1200: 모두 region 1.
- adaptive voxel max count 75/150/300: 모두 region 1.
- density fraction 0/0.5/0.7/0.9: 모두 region 1.
- parallel gap 0.03/0.06/0.12/0.24/0.48: 모두 두 layer 유지.
- 5 seeds: curved 5/5, crease 5/5, parallel 5/5, density-gradient 4/5, disconnected gap 0.1은 2/5 성공.
- 동일 입력 repeat와 reversed candidate ordering의 result hash가 모두 `6a2ae0229dc5678c5eff83b1cb3800614510388a9ccbd42a3ff33838d792454b`로 일치했다.
- 전체 artifact 반복 생성 SHA-256도 `950A2576996C29A1451AA859131154304366BEC1409FBC5FA88C517CC454F00E`로 byte-identical했다.

## 구조적 실패

- `mild_curved_sheet` seed 0은 gap/spacing 4.165에서 2 regions로 과분할됐다.
- `density_gradient` seed 2는 gap/spacing 5.163 때문에 2 regions로 과분할됐다.
- Disconnected gap 0.02/0.05는 한 region으로 오병합됐다.
- Disconnected gap 0.1도 seeds 1/2/4에서 오병합됐다. Seed 1의 최초 false merge gap/spacing은 3.496이었다.
- 따라서 density seed 2를 연결하려면 gap threshold가 5.163보다 커야 하지만 disconnected seed 1을 차단하려면 3.496보다 작아야 한다. 현재 signal set에는 scene-independent 공통 threshold가 없다.

이는 tuning 부족이 아니라 density/sampling gap과 sub-spacing disconnected gap이 quadratic proxy와 normalized support gap에서 겹치는 구조적 한계다. Scene/seed selector, fallback, retry, GT 분기, weighted score로 숨기지 않았다.

## 검증

- Stage 3 집중 unittest: 9 passed.
- `pytest 9.1.1` 설치 후 집중 pytest: 9 passed, 8 subtests passed.
- 전체 pytest: 190 passed, 1 skipped, 8 subtests passed. 기존 `torch_nurbs.py:433`의 requires-grad tensor scalar 변환 warning 1건 외 오류는 없다.
- 전체 unittest: Ran 191 tests, 1 skipped.
- 실제 current pipeline benchmark: `osn-gs benchmark --constructor boundary_first` 성공.
- Artifact JSON `allow_nan=False` 직렬화 성공.

## 성공/실패 및 채택 판단

- 기본 seed-0 5개 목표는 달성했다: curved 복원, crease/parallel/disconnected 0.1 보호, planar/density 회귀 없음.
- Orientation, point count, leaf resolution, parallel distance에는 안정적이었다.
- 그러나 seed 및 disconnected distance sweep에서 필수 robustness를 충족하지 못했다.
- **Stage 3 methodology feasibility는 광범위 gate에서 실패했다. Production 및 Stage 4 integration은 기각한다.**

## Production 변경 여부와 다음 승인

Production 변경 없음. `build_surface_components`, Phase 2, chart/NURBS 경로와 기본값은 그대로다.

Stage 3에서 멈춘다. 현재 signal conflict를 해결할 별도 방법론 검토와 사용자 승인이 없으면 Stage 4로 진행하지 않는다.
