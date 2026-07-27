# Worklog 106 — Boundary-first resolution/tolerance deterministic sweep

## 상태

진행 중. resolution 96과 1-cell raster tolerance가 특정 fixture에만 맞춘 값이 아닌지 positive/negative control sweep으로 확인했다.

## Sweep

각 scene에 대해 point count `400`, `600`과 seed `0`, `1`, `2`를 실행했다.

| scene | 조합 수 | 기대 상태 | 결과 |
| --- | ---: | --- | --- |
| sine | 6 | constructed | 6/6 constructed |
| curved_annulus | 6 | constructed | 6/6 constructed |
| u_shape | 6 | unsupported | 6/6 unsupported |

U-shape의 거부 사유는 seed/밀도에 따라 `outer_boundary_ambiguous` 또는 `interior_support_crosses_unobserved_region`이지만, 어느 경우에도 중앙 fan으로 조용히 채우지 않는다.

## 검증

`tests.test_boundary_first_support_pipeline`에 above sweep을 고정했고 3 tests가 통과했다.

## 해석

- 1-cell tolerance는 sine의 고해상도 raster quantization gap을 보정한다.
- tolerance가 concave U-shape의 실제 비관측 영역을 통과시키지는 않는다.
- 이 근거는 isolated review 경로에만 적용한다. default dispatcher/production integration 변경의 근거가 아니다.

## 다음 작업

- multi-loop planar-domain decomposition은 여전히 materialization하지 않고 review evidence로 유지한다.
- source-boundary fidelity의 threshold proposal과 artifact 비교를 완료한 뒤에만 feature-gated dispatcher 연결 여부를 사용자에게 보고한다.
- 현재 국소 구현 진행률은 약 90%다.