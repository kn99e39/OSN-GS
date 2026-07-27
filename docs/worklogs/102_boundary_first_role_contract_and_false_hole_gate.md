# Worklog 102 — Boundary-first 역할 계약 및 false-hole support gate

## 상태

진행 중. 기본 constructor dispatcher 및 production 경로는 변경하지 않았다.

## 수행 내용

- visible builder의 구성 계약을 `outer_boundary + interior_boundary` 또는 `outer_boundary + observed interior_anchor`로 통일했다.
  - loop 수와 topology는 관측 증거를 해석하는 진단 정보일 뿐, 별도 surface-construction methodology를 선택하지 않는다.
  - 두 역할 조합은 공통 `_materialize_boundary_role_network()` 진입점으로 들어간다.
- hole이 없는 plane, sine, triangle도 관측된 outer boundary와 interior anchor가 유효하면 동일한 Boundary-first 계약으로 구성한다.
- KDE가 만든 작은 enclosed hole은 `tiny_artifact`로 보고만 하던 기존 처리를 보완했다.
  - 유의미한 hole 면적 기준에 미달한 enclosed cell만 `refined_mask`에 제한적으로 복원한다.
  - 실제 material hole은 복원하지 않아 `interior_boundary` provenance를 보존한다.
  - sine fixture의 false hole 3개는 진단에 남고, anchor ray support 검증에는 false negative를 만들지 않는다.
- central-cap payload의 topology 명칭을 제거하고 `shared_observed_anchor_fan` materialization provenance로 교체했다.

## 결과

- sine: `interior_support_crosses_unobserved_region` 거부에서 `constructed`로 전환됐다.
- sine anchor ray support coverage: `0.9817 → 0.9997`.
- source-boundary 거리 진단에서는 sine의 median/local-spacing 비율이 `2.826`으로 측정되어, 외곽선 충실도 gate의 우선 검토 대상으로 유지한다. 이 수치는 아직 자동 차단 임계값이 아니다.

## 검증

다음 관련 회귀가 통과했다.

```text
21 tests OK
- tests.test_component_boundary
- tests.test_boundary_first_visible_builder
- tests.test_boundary_central_cap
- tests.test_boundary_first_support_pipeline
- tests.test_boundary_surface_quality
- tests.test_boundary_first_support_runner
```

## 남은 위험과 다음 작업

- 현재 outer contour는 raw point cloud에 대한 source-boundary fidelity 수치를 payload/review gate로 아직 강제하지 않는다.
- multi-loop planar domain decomposition, curvature/normal regularity, full-regression 해소 전에는 default dispatcher와 production integration을 변경하지 않는다.
- 이 국소 Boundary-first 구현의 다음 목표는 source-boundary fidelity payload와 deterministic review gate이며, 현재 진행률은 약 68%다.