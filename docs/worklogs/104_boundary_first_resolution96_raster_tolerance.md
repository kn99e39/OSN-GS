# Worklog 104 — Boundary-first resolution 96 및 raster tolerance 검증

## 상태

진행 중. 변경은 isolated Boundary-first support review 경로에만 적용했다. default constructor dispatcher 및 production integration은 변경하지 않았다.

## 수행 내용

- isolated `construct_boundary_first_support()`의 observed-support raster 기본 resolution을 64에서 96으로 올렸다.
- runner에 `--boundary-resolution` 옵션을 추가해 review artifact가 사용하는 해상도를 명시했다.
- anchor-to-boundary support ray 검사는 raster 양자화 오차만 허용하도록 `1` cell dilation tolerance를 명시했다.
  - payload에 `anchor_ray_support_tolerance_cells = 1`을 남긴다.
  - 실제 concave void를 채우는 fallback은 아니다.

## 측정 결과

새 artifact: `artifacts/boundary_first_support_review_20260727_v6_resolution96/`

```text
constructed      12
review_required   1
unsupported       2
```

| scene | 상태 | normalized median boundary distance |
| --- | --- | ---: |
| sine | constructed | 2.826 → 2.379 |
| curved_annulus | constructed | 2.491 → 2.113 |
| u_shape | unsupported | 1.593 |
| planar_hole_density_gradient | review_required | 2.720 |

sine의 source surface RMS는 `0.07184`로 0.1 review gate를 통과했다. U-shape는 tolerance 후에도 coverage가 64/96에서 각각 약 `0.799/0.783`으로 충분히 낮아 계속 `interior_support_crosses_unobserved_region`으로 거부된다.

## 검증

```text
22 tests OK
```

source fidelity, component boundary, role builder, central cap, pipeline, surface quality, runner 회귀를 함께 실행했다.

## 남은 위험과 다음 작업

- `1` cell tolerance의 scene/point-density sweep을 더 넓혀야 한다.
- source-boundary fidelity의 자동 차단 임계값은 아직 정하지 않았다.
- multi-loop planar-domain decomposition, curvature/normal regularity와 전체 regression은 계속 남아 있다.
- 현재 국소 구현 진행률은 약 82%다.