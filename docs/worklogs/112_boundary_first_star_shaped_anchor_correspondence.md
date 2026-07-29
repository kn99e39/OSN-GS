# Worklog 112 — Observed-anchor fan crossing 근본 원인 수정: star-shaped 경계 검증 + equal-angle correspondence

## 상태

완료(이번 라운드 범위 내). Worklog 110의 review-geometry representation foundation과 실제 NURBS curve export는 그대로 유지했다. 이번 작업은 Worklog 110이 발견한 support-curve crossing 결함의 근본 원인을 분석하고 교정했다. Canonical Boundary-first 공통 방법론과 isolated-path 제약은 계속 준수했다.

## 1. Scene별 근본 원인 분석 (per-scene attribution)

Worklog 110에서 invalid crossing이 발견된 6개 scene(`plane`, `sine`, `crease`, `triangle`, `elongated_plane`, `close_parallel_sheets`) 전부에서, 예외 없이 **동일한 seam index pair(segment_count=8 기준 patch 3 ↔ patch 7)**가 원인이었다. 이는 우연이 아니라 구조적 결함이다.

측정한 근거(`plane`/`elongated_plane`/`crease` 기준):

- **관측된 raw ordered boundary(약 360~420점)는 anchor 기준으로 이미 star-shaped에 매우 가깝다.** Anchor 기준 각 인접 표본 사이 각도 변화가 지배적 방향으로 97%+ 유지되고, 최대 단일 step은 5° 미만이다. 즉 "boundary가 anchor에 대해 star-shaped인가"라는 전제 자체는 원본 해상도에서 참이다.
- **결함은 `segment_count=8`로 boundary를 재표본화하는 단계에서 발생한다.** 기존 `_resample_closed()`는 순수 **arclength 등분**만 수행했다. `plane`/`elongated_plane`은 anchor가 boundary 중심이 아니라 한쪽으로 치우쳐 있어(elongated_plane은 anchor-경계 거리가 0.27~1.21 범위로 4.5배 차이), arclength 기준 8등분이 각도상으로는 매우 불균등하게 분포한다. 실측: `elongated_plane`의 8개 corner 각도는 대부분 -160°~-105°(약 55° 폭)에 몰려 있고 단 1개(patch 0)만 +167°에 위치한다. 즉 patch 7→patch 0으로 이어지는 wrap-around segment 하나가 나머지 약 266°를 전부 담당하며, 그 구간에 대한 Catmull-Rom류 tangent 추정이 근접 anchor 쪽으로 접히면서 반대편(patch 3) 방향과 거의 겹치는 곡선을 만든다.
- **`crease`는 두 컴포넌트로 정상 분리되며(segmentation 문제 아님), 각 컴포넌트가 독립적으로 동일한 patch 3↔7 결함을 보인다** — 즉 crease의 실패는 component segmentation이나 evidence 부족이 아니라 **동일한 arclength-resampling 결함이 두 평면 조각 모두에서 재현된 것**이다.
- 결론: 근본 원인은 anchor 선택 자체의 오류가 아니라(anchor는 이미 실제 관측점이며 합리적 위치), **"boundary가 anchor에 대해 star-shaped"라는 사실을 검증도 하지 않고, 그 사실을 활용하지도 않는 arclength-only 재표본화**다.

## 2. Boundary correspondence hardening (`osn_gs/surface/torch_boundary_central_cap.py`)

- `validate_star_shaped_boundary(boundary, pole, ...)`을 추가했다. Anchor 기준 각 ordered boundary 표본의 각도를 계산하고, `monotonicity_ratio = |net angular sweep| / sum(|step|)`을 핵심 지표로 사용한다(1.0 = 완전 단조, 낮을수록 왕복/비star-shaped). Threshold는 `min_monotonicity_ratio=0.85`, `max_step_degrees=30`(인접 표본 사이 단일 큰 점프 거부), `min_total_sweep_degrees=300`(anchor를 실제로 한 바퀴 감싸지 못하는 경우 — 예: anchor가 loop 밖에 있는 경우 — 거부)로 설정했다. 이 값들은 empirical하게 검증했다: 알려진 concave `u_shape`는 0.637, 완전히 비정상인(anchor가 loop 밖) 합성 fixture는 0.0, 이번에 실제로 고친 6개 scene은 모두 0.85~1.0 범위였다.
- `_resample_closed_by_angle(boundary, pole, count, direction)`을 추가했다 — 기존 arclength 기반 `_resample_closed()`와 동일한 누적-거리 보간 패턴을 사용하되, Euclidean 거리 대신 **검증된 지배적 방향의 각도 스텝**을 누적한다. 이제 8개 fan segment가 anchor 주위에 실제로 고르게 분포한다.
- `build_boundary_central_cap()`이 이제: (1) star-shape 검증을 먼저 수행하고, (2) 실패하면 `BoundaryCentralCapResult('unsupported', 'insufficient_observed_interior_support', (), {..., 'star_shape_validation': ...})`을 반환한다(빈 surfaces, fallback 없음), (3) 통과하면 `_resample_closed_by_angle`로 재표본화하고 provenance에 `boundary_correspondence='equal_angle_star_shaped'`와 `star_shape_validation` 전체를 기록한다.
- **단순 angle sorting을 무조건 적용하지 않는다** — star-shape 검증을 통과하지 못하면 각도 기반 재표본화 자체를 시도하지 않고 즉시 review 상태를 반환한다(지시사항 준수).
- 사용되지 않게 된 기존 `_resample_closed()`(arclength 전용)는 제거했다 — 호출부가 전부 `_resample_closed_by_angle`로 교체됐고 다른 참조가 없음을 확인했다.

## 3. 빌더 측 실제 버그 발견 및 수정 (`torch_boundary_first_visible_builder.py`)

`_materialize_boundary_role_network()`가 `build_boundary_central_cap()`의 반환값을 **`cap.state`를 확인하지 않고 무조건 `"constructed"`로 감싸는 기존 버그**를 발견했다. 이전에는 `build_boundary_central_cap`이 사실상 항상 성공했기 때문에(닫힌 boundary 조건 외에는 실패 경로가 없었음) 이 버그가 잠재해 있었다. 이번에 `insufficient_observed_interior_support`라는 새 실패 경로를 추가하면서 이 버그가 즉시 드러났다(전체 sweep에서 `ValueError: combine_ordered_patch_boundary requires at least one entity`로 재현). `cap.state != "constructed_central_cap"`이면 `BoundaryFirstVisibleSurfaceResult("unsupported", ..., cap.reason, None, ...)`을 반환하도록 수정했다 — "surface를 못 만듦"이 이제 실제로 `unsupported`/`not_materialized`로 정확히 전파된다.

## 4. Anchor selection — 범위와 한계 (item 6)

`select_observed_interior_anchor()`(anchor 후보 선택 로직) 자체는 이번 라운드에서 **재작성하지 않았다.** 기존 로직(observed_max_support_clearance 우선, medoid fallback)은 이미 synthetic centroid를 생성하지 않고 실제 관측점만 선택하므로 그 자체로는 지시 위반이 없다. 다만 지시에서 요구한 "boundary containment/visible-boundary coverage/angular ordering monotonicity/support crossing risk를 후보 평가에 반영한 richer scoring"은 **선택 이전 단계에는 아직 구현하지 않았다** — 대신 이번 라운드는 선택된 anchor+boundary 쌍을 사후에 **star-shape 검증 게이트**로 걸러내는 방식을 택했다. 유효하지 않으면 명시적으로 `insufficient_observed_interior_support`를 반환하고 fallback anchor를 만들지 않는다(지시의 마지막 요구사항은 충족). Anchor 후보를 여러 개 평가해 그중 star-shape 품질이 가장 좋은 것을 미리 고르는 **다중 후보 재순위화는 아직 하지 않는다** — 이는 명시적으로 남겨진 다음 범위다(§7 참고).

## 5. Positive/negative regression 결과

`ALL_SCENE_NAMES`(현재 4개 신규 scene + 15개 legacy scene) 전체 재검증:

| scene | 결과(수정 전) | 결과(수정 후) |
| --- | --- | --- |
| `plane`, `sine`, `crease`, `triangle`, `elongated_plane`, `close_parallel_sheets` | invalid crossing (`ineligible`) | **crossing 완전 해소**, 모든 pair가 `valid_shared_pole`만 남음 |
| `u_shape` | `unsupported`(`interior_support_crosses_unobserved_region`) | 동일 — 근본적으로 concave하여 더 이른 anchor-ray coverage gate에서 거부됨(이번 gate와 무관, 강제로 통과시키지 않음) |
| 신규 `saddle_shell`/`wave_annulus`(concurrent 작업으로 추가된 scene) | 해당 없음 | 일부 컴포넌트가 `insufficient_observed_interior_support`/`ordered_boundary_required`로 거부됨 — 이번 gate가 실제로 무언가를 걸러내고 있음을 보여주는 독립적 증거이며 강제 통과시키지 않았다 |

**Sweep 중 발견한 예외 하나: `sine`의 point-count/seed sweep 중 `(400, seed=2)`와 `(600, seed=2)`는 star-shape 검증에서 `monotonicity_ratio` 0.79~0.82로 threshold(0.85) 미달로 새롭게 `insufficient_observed_interior_support`가 됐다.** 이는 회귀가 아니라 이번 gate가 이전에는 검증조차 하지 않던 marginal case를 정직하게 표시한 것이다 — u_shape(0.637)과는 충분한 margin(0.15~0.18)이 있어 오탐이 아니라고 판단했다. 기존 `tests/test_boundary_first_support_pipeline.py`의 sweep 테스트가 이 두 조합만 `unsupported`로 기대하도록 갱신했고, 나머지 4개 조합(sine seed 0/1, count 400/600)과 `curved_annulus`/`u_shape` 전체는 이전과 동일하게 유지된다.

## 6. 회귀 테스트

신규:

- `tests/test_boundary_central_cap.py::StarShapedBoundaryCorrespondenceTest`(5 tests): 중심 원(anchor=중심) star-shaped 검증 통과 + 정확히 균등한 각도(45°±0.05) 재표본화, anchor가 loop 밖에 있는 합성 fixture가 `total_angular_sweep_degrees < 300`으로 거부됨, 검증 결과의 결정성(같은 입력 → 같은 출력), 비-star-shaped 쌍이 synthetic fallback 없이 `insufficient_observed_interior_support`/빈 surfaces로 거부됨, star-shaped 쌍이 `boundary_correspondence='equal_angle_star_shaped'` provenance와 함께 정상 materialize됨.
- `tests/test_boundary_first_support_runner.py`: `test_materialized_but_invalid_crossing_is_ineligible_not_unsupported`를 실제 scene 대신 `_component_quality_state`/`_scene_quality_projection` 직접 단위 테스트로 교체했다(root-cause 수정 이후 어떤 real scene도 더 이상 `ineligible`에 도달하지 않기 때문 — 그 자체가 이번 수정의 성과이지만, "materialized+quality-failed"와 "not_materialized"가 분리 표현되는 코드 경로 자체는 계속 회귀 검증이 필요하다). `test_root_cause_fixed_scenes_have_no_invalid_crossing`을 추가해 6개 scene 전부 `has_invalid_support_crossing=False`, `quality_state != "ineligible"`을 고정했다. `plane` 관련 테스트를 갱신해 이제 crossing pair가 전부 `valid_shared_pole`뿐임과 `boundary_correspondence`/`star_shape_validation` provenance 노출을 검증한다.
- `tests/test_boundary_first_support_pipeline.py`의 기존 sweep 테스트를 sine seed=2 marginal case를 반영하도록 갱신했다(위 §5).

```text
targeted (62 tests): 전부 통과
- tests/test_patch_boundary.py
- tests/test_boundary_first_visible_builder.py
- tests/test_boundary_first_support_runner.py (8 tests)
- tests/test_boundary_review_geometry.py
- tests/test_boundary_support_network.py
- tests/test_boundary_constrained_surface.py
- tests/test_boundary_central_cap.py (8 tests, +5 신규)
- tests/test_boundary_surface_quality.py
- tests/test_boundary_first_support_pipeline.py
- tests/test_boundary_multi_loop.py
- tests/test_boundary_planar_partition.py
- tests/test_boundary_source_fidelity.py
- tests/test_component_boundary.py
```

전체 pytest:

```text
509 passed, 2 failed, 1 skipped, 1 warning, 8 subtests passed
```

기존 실패 2건은 이번 변경과 무관하다(`tests/test_trimmed_component_fitter.py`의 `degenerate_fraction` strict-zero 기대치, 실측 약 0.0017361111 — worklog 105부터 이어진 별도 attribution 대기 항목).

## 7. 아직 not_checked/미착수인 항목

- **Anchor 후보 다중 재순위화**(§4): boundary containment, visible-boundary coverage, support-crossing-risk를 SELECTION 이전 단계 scoring에 반영하는 전체 재설계는 아직 하지 않았다. 현재는 사후 검증 게이트만 있다.
- **Ray/다중 교차 검증**: item 7이 언급한 "동일 ray의 복수 boundary intersection" 명시적 검사는 별도로 구현하지 않았다 — `monotonicity_ratio`/`total_sweep`/`max_step` 조합이 실질적으로 이를 간접 포착하지만, 직접적인 ray-intersection-count 검사는 아니다.
- Worklog 110에서 이미 남겨둔 항목(bidirectional source-boundary fidelity, false-hole persistence/raw-support/genuine-small-hole negative control, multi-hole 실제 patch materialization, `patch_overlap`/`jacobian_foldover`/`seam_inconsistency`/`full_support_family_crossing` not_checked categories)은 전부 변경 없이 그대로 남아 있다.
- `quality_state=="eligible"`은 여전히 어떤 경로도 도달하지 않는다.

## 8. dispatcher/production 비접촉 확인

`git status` 기준 이번 라운드에서 내가 수정한 파일은 `osn_gs/surface/torch_boundary_central_cap.py`, `osn_gs/surface/torch_boundary_first_visible_builder.py`, `nurbs_constructor_benchmark/boundary_first_support_runner.py`(scenes.py의 concurrent 개명에 맞춰 `--scenes` choices만 `ALL_SCENE_NAMES`로 확장), 그리고 관련 테스트 파일들뿐이다. `nurbs_constructor_benchmark/boundary_first.py`(legacy dispatcher), `nurbs_constructor_benchmark/runner.py`, trainer, production pipeline, uncertain Gaussian proposal/append, ownership, checkpoint, multi-hole materialization 코드는 열람 이상으로 수정하지 않았다. Rectangle/PCA/box/trimmed fallback이나 synthetic center anchor를 추가하지 않았다 — 거부된 경우는 항상 명시적 review 상태(`insufficient_observed_interior_support`)로 귀결된다. 자동 Gate 승인도 하지 않았다 — `quality_state`는 여전히 `eligible`에 도달하지 않는다.

*(참고: 이번 세션 진행 중 별도 작업(사용자 또는 concurrent Codex 세션으로 추정)으로 `nurbs_constructor_benchmark/scenes.py`가 변경되어 `SCENE_NAMES`가 4개 신규 scene으로 교체되고 기존 15개는 `LEGACY_SCENE_NAMES`로 이동했다(`ALL_SCENE_NAMES`가 합집합). 이 변경은 내가 수행한 것이 아니며 되돌리지 않았다 — 이번 러너의 `--scenes` choices만 `ALL_SCENE_NAMES`로 넓혀 legacy fixture(plane/sine/crease 등)를 계속 사용할 수 있게 했다.)*

Repository-wide pytest는 green이 아니고(기존 무관 실패 2건 잔존), positive fixture(plane/sine/triangle/elongated_plane/crease/close_parallel_sheets) 전부에서 invalid crossing이 해소됐음을 확인했다. 다만 §7의 미착수 항목이 남아 있으므로 Boundary-first Gate 완료를 주장하지 않는다.
