# Worklog 103 — Boundary-first source-boundary fidelity payload

## 상태

진행 중. 이번 변경도 isolated Boundary-first 경로에 한정하며 default dispatcher와 production integration은 변경하지 않았다.

## 수행 내용

- `torch_boundary_source_fidelity.py`를 추가했다.
  - ordered observed boundary sample에서 같은 component raw point까지의 minimum/median/mean/maximum 거리를 계산한다.
  - component의 median nearest-neighbor spacing으로 median/max 거리를 정규화한다.
  - 닫힌 contour의 중복 종점은 한 번만 계수하며 결과는 deterministic payload로 보존한다.
- visible builder가 `source_boundary_fidelity`를 provenance에 기록하도록 연결했다.
- 이 값은 point cloud가 연속 경계 위에 정확히 존재해야 한다는 가정을 하지 않는 review metric이다. 현재 eligibility나 construction state를 자동으로 바꾸지 않는다.
- 새 artifact `artifacts/boundary_first_support_review_20260727_v5_role_fidelity/`를 생성했다.

## 결과

15-scene isolated review (`--max-source-point-rms 0.1`) 결과:

```text
constructed      12
review_required   1
unsupported       2
```

- sine은 `constructed`, `normalized_median_distance = 2.826`이다.
- `planar_hole_density_gradient`는 source surface RMS gate에서 `review_required`다.
- close parallel sheets는 recovery merge 없이 각각의 raw component가 독립 역할 계약으로 기록된다.

## 검증

```text
22 tests OK
- tests.test_boundary_source_fidelity
- tests.test_component_boundary
- tests.test_boundary_first_visible_builder
- tests.test_boundary_central_cap
- tests.test_boundary_first_support_pipeline
- tests.test_boundary_surface_quality
- tests.test_boundary_first_support_runner
```

## 남은 위험과 다음 작업

- boundary fidelity 수치의 허용 임계값은 아직 확정하지 않았다. 우선 boundary resolution, KDE threshold, point density 변화에 대한 deterministic sweep이 필요하다.
- multi-loop planar-domain decomposition과 curvature/normal regularity도 미완료다.
- default dispatcher와 production integration은 위 review gates가 정리되고 별도 승인을 받기 전까지 변경하지 않는다.
- 이 국소 구현의 진행률은 약 76%다.