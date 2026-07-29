# Worklog 123 — Canonical Tangent Frame Invariance Repair

## 작업 내용

Gaussian-only Visible Surface Construction의 local support-termination sector 기준축을 수정했다.

- local[0] 및 입력 순서 기반 sector 기준축을 제거했다.
- stable region의 accepted local topology에서 seed와 tangent frame을 구성하고, normal/tangent을 topology 경로로 transport한다.
- tangent major/minor가 불안정한 경우 accepted-neighborhood의 sign-invariant structure tensor를 사용한다.
- sector occupancy에는 angular-margin sharing을 적용했다.
- candidate 방향은 sector array index가 아닌 실제 circular support gap의 중심으로 계산한다.
- wrap-around missing run을 하나로 정규화했고, raw run index를 half-edge ID에 사용하지 않는다.
- 분리된 여러 region이 동시에 materialize되어도 단일 완성 surface처럼 constructed를 주장하지 않도록 상태 정책을 보정했다.

## 수정 전 결과

기존 tests/test_visible_surface_construction.py에서 curved-sheet 변환 variant가 constructed에서 review_required로 바뀌었다. 원인은 support-termination이 첫 local direction을 sector axis로 사용해 동일한 geometry의 occupied sector/run과 directed boundary graph가 달라진 것이었다.

## 수정 후 invariance matrix

| 입력 | 상태 | materialized NURBS | stable region / boundary membership |
| --- | --- | --- | --- |
| clean plane 원본 | constructed | 1 | 유지 |
| smooth curved sheet 원본 | constructed | 1 | 유지 |
| 3개 rigid rotation | constructed | 1 | 유지 |
| uniform scale 0.31 / 4.2 | constructed | 1 | 유지 |
| reverse 및 deterministic shuffle | constructed | 1 | stable ID 기준 유지 |
| covariance sign-equivalent covariance | constructed | 1 | 유지 |

## 평가

tests/test_visible_surface_construction.py와 신규 tests/test_visible_surface_construction_invariance.py는 통과했다. close parallel sheets와 perpendicular surfaces control은 단일 constructed 상태로 오인되지 않으며, accepted topology 외 edge를 frame transport 또는 ordering에 사용하지 않는다.

## 남은 위험

이 수정은 fixture 범위의 Gaussian-only visible surface construction에 한정된다. arbitrary multi-hole, T-junction, crossing sheet, trained-scene coverage 및 production dispatcher는 여전히 범위 밖이다.

