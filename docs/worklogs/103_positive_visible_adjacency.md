# Worklog 103 — 양성 관측 지지 가시 인접성 (Positive Visible Adjacency)

## 상태

**완료 — 실측 있음. 실제 scene에서 명확히 혼재된(mixed) 결과이며, 정직하게 보고한다.** Worklog 102의 "완전한 관측-가시 evidence에서 시작 → 명시적 증거가 있을 때만 CUT" 원칙 자체는 유지하되, 그 원칙이 적용된 그래프를 교체했다: "spatial kNN에서 모순이 없으면 연결 유지"(WL102)가 아니라 **"spatial kNN은 후보 관계 생성기일 뿐, 양성(positive) 관측 증거가 있어야만 가시 인접성이 성립한다"**로 뒤집었다. WL102가 92.69%까지 percolation한 원인은 정확히 이 지점이었다 — spatial edge 5,132,180개 중 25.3%(1,295,809개)만 실제로 어떤 카메라에 co-observe되었고, 나머지 74.7%는 "모순 없음"이라는 이유만으로 기본 연결 상태를 유지했다.

이번 배치의 실측 결과: **최대 component 비율이 92.69% → 10.50%로 극적으로 낮아졌고, 테이블/패티오가 실제로 서로 분리된 별도 component가 되었다**(직접 시각 확인). 그러나 그 대가로 **전체 surfel의 63.4%가 고립된 singleton component가 되었고**, component 수는 13,585개(WL102) → 768,829개로 급증했다. Percolation은 해결됐지만 그 반대쪽 실패 모드(과도한 단절)로 진자가 크게 넘어갔다 — 두 실패 모두 정직하게 보고하며, 어느 쪽도 architecture 채택으로 해석하지 않는다.

## 아키텍처

```
학습된 2DGS surfel
    -> spatial kNN candidate graph (WL96-102와 완전히 동일, 재사용) — 이제 "후보 관계 생성기"일 뿐
    -> per-edge 관계 상태 판정 (7-state, 상호 배타):
         POSITIVE_VISIBLE_CONTINUATION      -- 양성 관측 지지 + 기하 검사 통과
         CUT_KNOWN_FREE_SPACE               -- 유일한 모순이 free space
         CUT_OCCLUDED_DOMAIN                -- 유일한 모순이 occlusion
         CUT_VISIBLE_GEOMETRIC_DISCONTINUITY -- 양성 지지를 받았으나 shape-operator residual 실패 (WL98/102 재사용)
         CUT_POSITIONAL_SHEET_SEPARATION    -- 양성 지지를 받았으나 법선-방향 offset 실패 (WL102의 수정된 부호 공식 재사용)
         UNRESOLVED_OBSERVATION_CONFLICT    -- 같은 edge에 양성과 모순이 동시에 존재
         UNKNOWN_NO_POSITIVE_OBSERVATION    -- 어떤 카메라도 co-observe하지 않았거나, co-observe했지만 깨끗한 신호를 얻지 못함
    -> POSITIVE_VISIBLE_CONTINUATION 만으로 그래프를 구성 -> connected components = Visible Surface Components
```

핵심 반전: "모순 없음 => 연결"(WL102)이 아니라 **"양성 관측 지지 => 연결 가능"**. Spatial 근접성은 후보 생성에만 쓰이고, 그 자체로 topology가 되지 않는다.

## 1. 재사용 — 새 시스템을 만들지 않았다

새 모듈 `osn_gs/surface/torch_positive_visible_adjacency.py`는 WL102의 `_per_view_status_codes`, `_project_to_camera`(둘 다 canonical Phase-C `_project_points`를 감싼 벡터화 함수)를 **직접 import**해서 재사용한다 — 재구현하지 않았다. WL98/102의 shape-operator/residual 기계(`_fit_shape_operators`, `_predicted_delta_n_t`, `_tangent_plane_components`, `_knn`)도 그대로 import했다. `torch_observation_evidence.py`와 `torch_maximal_visible_connectivity.py`(WL102)는 이번 배치에서 **한 줄도 수정하지 않았다** — WL102는 review export의 baseline 비교(뷰 H)로 그대로 재실행된다.

WL102가 이미 수정한 위치-오프셋 부호 버그(§13 지시사항)도 이 모듈에서 처음부터 올바른(signed) 버전으로 작성했다 — WL98/99/100은 여전히 손대지 않았다.

## 2. 양성 시야-연속성의 정확한 정의 (지시 §4)

두 candidate 끝점이 **같은 카메라 시야에서 동시에 `on_observed_surface`**여야 co-observation 후보가 된다. 그 다음 WL102가 이미 검증한 **RANGE 기반** 화면-공간 걷기(직선 3D chord도, 선형 depth 보간도 아님 — 두 방식 모두 지시가 명시적으로 금지한 실패 모드와 구조적으로 동일함을 WL102가 이미 증명했다)를 그대로 재사용한다: 3개 interior 샘플 전부가 `[min(depth_i,depth_j) - edge_length - epsilon, max(depth_i,depth_j) + edge_length + epsilon]` 범위 안에 있고 **전부 valid**해야 "양성"(POSITIVE)이다. 범위를 벗어나면 free/occluded, 하나라도 invalid(카메라가 그 지점에 대해 아무 데이터도 없음)면 양성이 아니다 — "모순 없음"과 "양성 지지 있음"을 여기서 명확히 구분한다: WL102는 invalid 샘플을 "모순 아님 = 통과"로 취급했지만, 이 모듈은 invalid 샘플이 하나라도 있으면 그 시야에서는 양성이 아니다(다른 시야가 깨끗하면 여전히 양성일 수 있다).

## 3. 다중 시야 취합 — 퍼센트 threshold 없음 (지시 §6)

`any_positive`(≥1개 시야에서 양성) / `any_free` / `any_occluded`를 시야별로 OR 취합한다. 과반수·비율 파라미터는 어디에도 없다.

- `any_positive & ~(any_free|any_occluded)` → 기하 검사 대상 (POSITIVE 후보)
- `(any_free|any_occluded) & ~any_positive` → CUT_KNOWN_FREE_SPACE / CUT_OCCLUDED_DOMAIN
- `any_positive & (any_free|any_occluded)` → UNRESOLVED_OBSERVATION_CONFLICT (양쪽 다 있으면 침묵으로 양성 처리하지 않는다)
- 둘 다 없음 → UNKNOWN_NO_POSITIVE_OBSERVATION

## 4. 기하 불연속/위치 분리 게이트는 이제 2차 필터다 (지시 §7)

WL98/100/102의 shape-operator residual과 법선-방향 offset 검사를 그대로 재사용하되, 적용 대상이 다르다 — WL102는 **모든** spatial edge에 적용했지만, 이 모듈은 **이미 양성 관측 지지를 받은 edge에만** 적용한다. 법선 방향 차이 자체는 여전히 cut 근거가 아니다(곡면은 여전히 하나의 component로 남을 수 있어야 한다).

## 5. 합성 fixture — 10개 계약, 전부 재사용 가능한 fixture 위에 신규 작성

`tests/test_positive_visible_adjacency.py`, 14 tests, 전부 통과. WL102의 `test_maximal_visible_connectivity.py`에서 `_Orientation`/`_lookat_world_view`/`_build_camera`/`_grid`/`_flat_orientation`/`_cylinder_band`/`_wall_pair_with_gap`/`_wall_gap_wall_camera_evidence`/`_procedural_evidence`/`_LOCAL_CONFIG`를 그대로 import해서 재사용했다(제2의 fixture 시스템을 만들지 않음). 새로 작성한 것은 `_splatted_evidence` — 점군을 자신의 정확한 projected depth로 2-pass(자기 픽셀 우선 기록 → 남은 픽셀만 dilation으로 채움) splatting하는 helper로, CPU fallback 렌더러의 blend 근사 없이 "카메라가 실제로 완전히, 모순 없이 관측했다"는 정밀한 fixture를 구성하기 위해 필요했다.

| # | 계약 | 결과 |
|---|------|------|
| A | 평면 + 완전 양성 관측 → 1 component | PASS |
| B | 곡면(원통) + 완전 양성 관측 → 1 component, occlusion/free-space 오판 없음 | PASS |
| C | 벽+occluder+벽 → 2 component, `CUT_OCCLUDED_DOMAIN` 발동 | PASS |
| D | 곡면 + 가려진 gap → 2 component 유지 | PASS |
| E | known free-space gap → 2 component, `CUT_KNOWN_FREE_SPACE` 발동 | PASS |
| F | 양성 관측을 받았지만 법선-방향 offset이 지배적인 두 판 → `CUT_POSITIONAL_SHEET_SEPARATION` 발동, 분리 | PASS |
| G | 실제 sharp crease + 양쪽 다 양성 관측 → 보존(`CUT_VISIBLE_GEOMETRIC_DISCONTINUITY`) | PASS |
| H | 카메라가 전혀 없음(공간적으로 가까운 pair라도) → 어떤 edge도 POSITIVE가 되지 않고, 전부 singleton | PASS — WL102라면 동일 입력이 1 component가 됐을 것 |
| I | 한 시야는 양성, 다른 시야는 아예 관측 불가(반대 방향 카메라) → 그래도 연결 유지 | PASS — 관측 부재는 모순이 아님 |
| J | 한 시야는 양성, 다른 시야는 진짜 모순(occluder) → `UNRESOLVED_OBSERVATION_CONFLICT`, 연결 안 됨 | PASS |

Coverage/determinism/empty-input 계약(모든 surfel이 정확히 하나의 owner를 가짐, 반복 실행 시 동일 결과, 빈 입력에서도 정합성 유지)도 별도로 확인했다.

## 6. 실제 scene 진단 (동일 체크포인트: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`, 1,190,469 surfel, train camera 161개)

| 항목 | 값 |
|---|---|
| spatial candidate edge | 5,132,180 |
| co-observation을 받은 edge (≥1개 카메라) | 1,295,809 (25.3%) — WL102의 `observation_evaluated_edge_count`와 정확히 일치 |
| 기하 검사 전 "양성" edge | 1,274,706 |
| `POSITIVE_VISIBLE_CONTINUATION` (최종 채택) | 1,043,908 (20.3%) |
| `CUT_KNOWN_FREE_SPACE` | 1,616 |
| `CUT_OCCLUDED_DOMAIN` | 1,380 |
| `CUT_VISIBLE_GEOMETRIC_DISCONTINUITY` | 77,496 |
| `CUT_POSITIONAL_SHEET_SEPARATION` | 153,302 |
| `UNRESOLVED_OBSERVATION_CONFLICT` | 18,107 |
| `UNKNOWN_NO_POSITIVE_OBSERVATION` | 3,836,371 (74.7%) |
| 최종 visible component 수 | 768,829 |
| 최대 component 비율 | **10.50%** (124,984 surfel) |
| component 크기 (min/median/mean/p95/max) | 1 / 1 / 1.548 / 1 / 124,984 |
| singleton surfel 비율 | **63.4%** (754,988) |
| coverage identity | True (모든 surfel이 정확히 1개 owner) |

WL102(H, 동일 관측 evidence로 재실행한 baseline)는 이번 실행에서도 largest=92.69%, 13,585 components로 재현되어 교차 검증됐다.

## 7. 테이블 / 패티오 / hedge-배경 정성 확인 (지시 §11, PNG 리뷰)

- **테이블**: `POSITIVE_OBSERVATION_VISIBLE_COMPONENTS` 뷰에서 테이블 상판이 단일 파란색(하나의 component)으로 뚜렷하게 나타나며, WL102 baseline(`WORKLOG102_MAXIMAL_SPATIAL_BASELINE`, 거의 균일한 붉은색 하나)과 달리 **주변 패티오 바닥(별도의 붉은-살구색 단일 component)과 명확히 분리**된다 — WL97-100이 지켰던 테이블 독립성이 이번에는 관측-지지 방식으로 복원됐다.
- **패티오**: 바닥 전체가 대체로 하나의 큰(아마도 최대) component로 남아 있다 — 넓고 평평하고 대부분의 카메라에서 반복적으로 깨끗하게 관측되는 표면이라 양성 지지가 조밀하게 형성된 것으로 보인다.
- **hedge/배경**: `POSITIVE_OBSERVATION_VISIBLE_COMPONENTS` 뷰에서 배경 식생 영역은 균일한 무지개색 스페클(수많은 서로 다른 극소 component)로 나타난다 — WL102처럼 하나의 거대 component로 뭉치지도 않지만, 동시에 실제 식생이 이루는 표면 연속성도 대부분 보존하지 못한다(잎/가지의 미세 구조가 다중 뷰에서 "완전히 깨끗한" 3-샘플 판정을 통과하기 어려운 것으로 추정).
- `OCCLUDED_VISIBLE_TERMINATIONS`/`KNOWN_FREE_SPACE_TERMINATIONS` 뷰는 여전히 드물게(파란/붉은 점 산발) 나타나며, 대부분 hedge 경계 부근에 몰려 있다 — 즉 **연결이 끊기는 대부분의 위치는 "양성 occlusion 증거가 있어서"가 아니라 "애초에 양성 관측이 성립하지 않아서"(UNKNOWN)이며, 이는 회색조 `UNKNOWN_SPATIAL_RELATIONS` 뷰가 장면 전반에 옅게 퍼져 있는 것과 일치한다.

## 8. 결론

**혼재된(mixed) 결과다.** 지시 완결 조건 질문("spatial 근접성만으로가 아니라 실제 관측된 가시 evidence로 양성 지지되는 국소 관계로만 가시 topology를 구성하면, OSN-GS가 scene 전체 percolation 없이 최대한의 관측-가시 표면 연결성을 보존할 수 있는가?")에 대해:

- **Percolation은 해결됐다.** 92.69% → 10.50%, 테이블/패티오가 실제로 분리됐다. 이는 이 배치가 노린 정확히 그 문제에 대한 명확한 성공이다.
- **그러나 "최대한의 연결성 보존"은 달성하지 못했다.** 전체 surfel의 63.4%가 고립된 singleton이 됐고, 특히 hedge/배경처럼 국소 co-observation 밀도가 낮은 영역에서 원래 연속적인 표면조차 심하게 단절됐다. Spatial kNN 후보 5,132,180개 중 겨우 25.3%만 어떤 카메라에도 co-observe되지 않았다는 사실 자체가, "positive-only" 원칙을 이 evidence 밀도 위에 곧이곧대로 적용하면 구조적으로 과소-연결이 불가피함을 보여준다.
- **어느 쪽 architecture도 채택하지 않는다.** WL102(모순 없으면 연결)와 WL103(양성 지지 없으면 미연결)은 같은 축의 양 극단이며, 둘 다 실제 scene에서 원하는 결과를 주지 못했다. 다음 단계가 있다면 이분법(양성/모순만)이 아니라 co-observation 밀도 자체를 늘리는 방향(더 많은 카메라 재사용, 화면-공간 corridor 판정의 관용도 조정 등)이거나, 두 극단 사이의 원칙적인 중간 지점을 찾는 방향이어야 하지만, 이번 배치에서는 그 판단을 내리지 않는다.

## 9. 참고

- 새 모듈: `osn_gs/surface/torch_positive_visible_adjacency.py`
- 테스트: `tests/test_positive_visible_adjacency.py` (14 tests)
- Export 스크립트: `scripts/devtools/positive_visible_adjacency_export.py`
- 결과: `output/osn_gs_positive_visible_adjacency/` (8개 뷰 PLY/PPM/JSON), PNG 미리보기: `output/osn_gs_positive_visible_adjacency/preview_png/`
- 전체 리포트: `output/osn_gs_positive_visible_adjacency/positive_visible_adjacency_report.json`
- 전체 회귀 테스트: `1198 passed, 1 skipped, 1 warning, 18 subtests passed`(WL102의 1184 + 신규 14)
- 관련: [[project_maximal_visible_surface_components]] (WL102, 이번 배치가 교체한 그래프의 이전 버전)
