# Worklog 4 — Boundary-first review payload geometry semantics 교정 및 support crossing gate

## 상태

진행 중. Worklog 109(planar-partition evidence foundation)와 isolated review-layer foundation(exporter 작업, 지난 세션)은 그대로 유지한다. 이번 작업은 fidelity/crossing gate를 canonical로 승격하기 전에 review payload의 geometry semantics를 먼저 교정하고, support crossing 검출 foundation을 추가한 것이다. 이번 작업부터는 사용자 지시에 따라 새 worklog 번호를 사용한다(109에 append하지 않음).

## 문제 진단

지난 세션의 exporter는 `support_curves`/`reconstructed_outer_boundary`/`reconstructed_inner_boundary`를 patch의 raw `control_grid` row/column을 그대로 복사해 만들었다. 이는 **control polygon 관점의 lossless export**일 뿐, 실제 NURBS 곡선(evaluated curve) 관점의 lossless export가 아니다. `degree_v=1`(radial) 축에서는 control point가 우연히 curve 위에 정확히 존재하지만, `degree_u=3`(circumferential/outer-inner boundary, observed-anchor fan의 outer edge) 축에서는 두 내부 control point가 실제 곡선 위에 있지 않다. 특히 observed-anchor fan의 pole→corner 2점 데이터는 진짜 support curve가 아니라 대응(correspondence) 진단용 chord였다.

## 수행 내용

### 1. 신규 모듈 `osn_gs/surface/torch_boundary_review_geometry.py`

- 5개 representation kind를 명시적으로 구분한다: `observed_evidence_points`, `resampled_observed_evidence`, `control_polygon`, `correspondence_chord`, `evaluated_curve`.
- `ReviewGeometryEntity` dataclass: entity_id, representation_kind, role, source_component_id, source_loop_id/source_anchor_id, patch_ids, coordinate_space, ordered, closed, orientation, source_point_indices, correction_applied/reason, parameter_edge/direction/samples, sampling_policy, points. `payload()`로 JSON 직렬화.
- `evaluate_iso_edge()` / `evaluate_interior_iso_curve()`: 항상 `surface.evaluate()`(실제 Cox-de Boor NURBS evaluator)를 통해 iso-parametric edge/curve를 표본화한다. Raw `control_grid`를 절대 읽지 않는다.
- `control_polygon_entity()`: raw control-net row/column을 별도 representation으로 명시(“이것은 fitting data이지 curve가 아니다”).
- `combine_ordered_patch_boundary()`: patch 순서대로 evaluated edge를 이어붙이며 인접 patch의 공유 접합점을 정확히 1회만 남기고, `closed=True`인 경우 마지막 wrap-around 중복점도 제거한다. `control_polygon` entity를 넣으면 `ValueError`.
- `correspondence_chord_entity()` / `observed_evidence_entity()`: 각각 diagnostic chord와 원본 관측 evidence를 별도 representation으로 생성.
- Support crossing: `classify_support_curve_pair()` / `detect_support_curve_crossings()`. Scale은 curve 자체의 median 인접 표본 간격에서 유도한다. `expected_shared_point`/`expected_shared_kind`(pole/boundary_endpoint)가 주어지고 실제로 끝점이 그 지점과 일치할 때만 `valid_shared_pole`/`valid_shared_boundary_endpoint`로 분류하며, 그 경우에도 각 곡선의 끝점 buffer를 제외한 **interior** 구간을 별도로 재검사해 우연한 추가 교차를 놓치지 않는다. 그 외 근접은 `invalid_interior_crossing`(≤ tolerance) 또는 `near_touching_ambiguous`(≤ ambiguous band)로, curve가 2개 미만이면 `not_checked`로 분류한다.

### 2. `osn_gs/surface/torch_boundary_first_visible_builder.py`

- `observed_outer_boundary`/`observed_inner_boundary`/`observed_interior_anchor` provenance 값을 raw point list에서 `observed_evidence_entity(...).payload()`(entity dict)로 교체했다. `unsupported`(interior_support_crosses_unobserved_region)로 거부된 anchor도 evidence를 그대로 보존한다.

### 3. `nurbs_constructor_benchmark/boundary_first_support_runner.py`

- Closed inner/outer support-curve-network(annulus류) 경로: patch당 `support_control_polygons`(raw u=0 control row) + `evaluated_support_curves`(실제 `S(0,v)` 표본, seam과 동일 — patch k의 u=0 edge와 patch k-1의 u=1 edge는 clamped 공통 control column 때문에 항상 정확히 같음). `outer_boundary_control_polygons`/`inner_boundary_control_polygons`(patch별 cubic Bezier control polygon, `v=1`/`v=0`)와 실제 `evaluate_iso_edge`로 얻은 patch별 edge를 `combine_ordered_patch_boundary`로 이어 붙인 `reconstructed_outer_boundary`/`reconstructed_inner_boundary`(닫힌 evaluated curve, patch 순서, 접합점 중복 제거).
- Observed-anchor fan 경로: pole↔corner 2점 데이터를 `support_correspondence_chords`로 분리했다. 실제 support curve는 각 patch의 `u=0.5`(patch 내부) 고정 iso-curve를 pole(v=0)에서 curved outer edge(v=1)까지 평가해 `evaluated_support_curves`에 넣는다. `reconstructed_outer_boundary`는 patch별 실제 outer edge(`v=1`)를 evaluate 후 결합한다. Pole 관련 provenance(`has_central_pole`, `singularity_kind`, `pole_aware_regularity_contract`, `parameter_direction`, `pole_point`)를 `pole_metadata`로 보존한다.
- `support_crossing`: annulus류는 `evaluated_support_curves` 전체 쌍 검사(공유점 기대 없음), fan류는 `expected_shared_point=pole`, `expected_shared_kind="pole"`로 검사한다.
- `patch_seams`(→`patch_boundaries`)는 이제 annulus류는 `evaluated_support_curves`, fan류는 `support_correspondence_chords`를 seam 소스로 사용하고, `representation_kind`를 payload에 함께 기록한다.
- `evaluate_scene()`의 `fidelity_gate`에 `has_invalid_support_crossing`을 추가했고, `gate_pass = rms_pass and not has_invalid_crossing`으로 강화했다. 즉 invalid interior crossing이 있으면 `record.state`가 `constructed`가 될 수 없다(요구사항 “invalid crossing이면 eligible로 처리하지 마라”를 실제로 강제).
- `renderer_payload()`(`boundary_first.py`)에는 keyword-only `boundary_first_review` 매개변수만 추가했다(기본 `None`). Legacy dispatcher(`construct_boundary_first`) 호출부와 `runner.py`, `tests/test_patch_boundary.py`는 인자를 넘기지 않으므로 동작이 그대로다.

## 실제로 발견한 결함 — support crossing gate가 처음으로 실동작

새 crossing gate를 15개 benchmark scene 전체(curve_count=8, 기존 review 설정)에 적용한 결과:

| scene | invalid crossing |
| --- | --- |
| plane, sine, crease, triangle, elongated_plane, close_parallel_sheets | **있음** |
| density_gradient, u_shape | not_checked(unsupported라 surface 자체가 없음) |
| crescent, planar_hole, planar_hole_offcenter, planar_hole_elliptical, curved_annulus, mild_curved_sheet | 없음 |
| planar_hole_density_gradient | 없음(review_required는 기존 RMS gate 때문) |

`plane` scene(point=600, seed=0, curve_count=8)에서 segment index 3과 7의 실제 evaluated 내부 support curve가 공유 pole 이외의 지점에서도 서로 0.003~0.018 단위로 거의 완전히 겹친다(같은 pair의 다른 27개 조합은 모두 `valid_shared_pole`만 검출). 이는 anchor(pole)가 기하학적 중심이 아니라 outer boundary 근처에 위치하고, outer boundary가 arclength 기준으로 등분되기 때문에 발생하는 실제 구조적 결함이다 — 이번 작업 전에는 검사 자체가 없어 조용히 `constructed`로 export되고 있었다. 이번 세션은 이 결함의 **근본 수정(anchor 선택/재표본화 방식 개선)을 범위에 포함하지 않는다.** crossing gate가 올바르게 이를 탐지해 `review_required`로 전환시켰다는 것만 이번 범위의 성과다.

## 회귀 테스트

새/갱신 테스트:

- `tests/test_boundary_review_geometry.py`(신규, 10 tests): cubic control polygon vs evaluated curve가 다름(중간점이 raw control point와 다름, 끝점만 일치), 선형(degree=1) 축에서는 evaluate가 control point와 정확히 일치, `combine_ordered_patch_boundary`의 접합점 중복 제거(개방/폐곡선 wrap-around 포함) 및 control-polygon 입력 거부, `valid_shared_pole`/`valid_shared_boundary_endpoint`/`invalid_interior_crossing`/`near_touching_ambiguous`/`no_crossing`/`not_checked` 6개 분류 전부.
- `tests/test_boundary_first_visible_builder.py`: `observed_outer_boundary`/`observed_inner_boundary`가 이제 `representation_kind="observed_evidence_points"` entity임을 검증.
- `tests/test_boundary_first_support_runner.py`(재작성): schema_version 확인, control-polygon/evaluated-curve/chord가 별도 필드·별도 개수로 export됨, `reconstructed_outer/inner_boundary`가 닫힌 evaluated curve로 patch_count*4개 표본을 가짐, pole_metadata 존재, `plane` scene의 crossing 발견(3↔7만 invalid, 나머지 27쌍은 valid_shared_pole)을 고정 fixture로 회귀에 반영, unsupported(u_shape) 결과도 observed evidence를 보존하고 crossing은 `not_checked`, legacy `patch_boundaries`가 실제로 채워짐(`representation_kind` 포함).
- `tests/test_boundary_multi_loop.py`, `tests/test_boundary_planar_partition.py`는 수정 없이 그대로 통과 — multi-hole은 계속 `review_required`/`partition_materialization_required`이며 이번 작업이 건드리지 않았다.

```text
targeted (52 tests): 전부 통과
- tests/test_patch_boundary.py
- tests/test_boundary_first_visible_builder.py
- tests/test_boundary_first_support_runner.py
- tests/test_boundary_review_geometry.py (신규)
- tests/test_boundary_support_network.py
- tests/test_boundary_constrained_surface.py
- tests/test_boundary_central_cap.py
- tests/test_boundary_surface_quality.py
- tests/test_boundary_first_support_pipeline.py
- tests/test_boundary_multi_loop.py
- tests/test_boundary_planar_partition.py
- tests/test_boundary_source_fidelity.py
- tests/test_component_boundary.py
```

전체 pytest:

```text
495 passed, 2 failed, 1 skipped, 1 warning, 8 subtests passed
```

기존 실패 2건은 이번 변경과 무관하다(`tests/test_trimmed_component_fitter.py`의 `degenerate_fraction` strict-zero 기대치, 실측 약 0.0017361111 — worklog 105에서부터 이어진 별도 attribution 대기 항목).

## dispatcher/production 비접촉 확인

`git status`/`git diff` 기준으로 이번 세션이 수정한 기존 추적 파일은 `nurbs_constructor_benchmark/boundary_first.py` 단 하나이며, 변경은 `renderer_payload()`에 keyword-only 옵션 인자 1개를 추가한 것뿐이다. `construct_boundary_first`(legacy dispatcher), `BoundaryFirstState`, trainer, production pipeline, uncertain Gaussian proposal/append/ownership/checkpoint, multi-hole 실제 patch materialization 코드는 어느 것도 열람 이상으로 수정하지 않았다. 자동 Gate 승인도 수행하지 않았다 — `quality_state`는 여전히 항상 `review_required`이고, 이번에 추가한 `has_invalid_support_crossing` 게이트는 `constructed`를 더 엄격하게 만들 뿐 새로운 “eligible” 상태를 만들지 않는다.

## Not-checked로 남긴 항목 (범위 밖)

- **Bidirectional source-boundary fidelity**: 이번 단계는 옵션(“구현할 수 있다면”)이었고, 기존 `torch_boundary_source_fidelity.py`(관측 경계 → raw 소스점 단방향)를 그대로 유지했다. 역방향(재구성 곡선 → 관측 경계)은 여전히 미구현이며 `quality_state`는 계속 `review_required`다.
- **False-hole persistence/raw-support/genuine-small-hole negative control**: 미착수, 기존 area-ratio 보정을 canonical로 승격하지 않았다.
- **Multi-hole 실제 patch materialization**: 미착수, review-only 유지.
- **plane/sine/crease/triangle/elongated_plane/close_parallel_sheets의 crossing 근본 원인(anchor 선택/재표본화) 수정**: 이번 범위 밖. 다음 단계에서 anchor-fan 재표본화 방식(예: pole 기준 각도 균등 분할) 검토가 필요하다.

## 다음 작업

1. crossing gate가 발견한 6개 scene의 근본 원인(비-중심 anchor + arclength 재표본화) 검토 여부를 사용자에게 보고하고 방향을 확인한다.
2. Bidirectional source-boundary fidelity(양방향 거리) 구현.
3. False-hole persistence/raw-support/genuine-small-hole negative control 구현.
4. 위 항목이 정리된 뒤에만 quality_state의 `eligible` 상태 도입 여부를 사용자와 논의한다.

Repository-wide pytest가 green이 아니고(기존 무관 실패 2건 잔존), 이번 작업으로 Boundary-first Gate 완료를 주장하지 않는다.

## 2026-07-27 후속 — 상태 계약/seam semantic/crossing 정책 교정 (append-only)

사용자가 review payload의 geometry semantics 자체를 더 교정하라고 지시했다. crossing/fidelity gate를 canonical로 승격하기 전에 처리해야 하는 교정이므로 기존 Worklog 4에 이어서 기록한다. anchor/support construction의 근본 수정(scene별 attribution, anchor 선택 hardening, boundary correspondence hardening)은 별도 Worklog 5로 분리한다.

### 1. materialization_state / quality_state 분리 (`boundary_first_support_runner.py`)

- `evaluate_scene()`이 이제 scene 및 component 수준에서 canonical `materialization_state`(`materialized`/`not_materialized`)와 `quality_state`(`unsupported`/`review_required`/`ineligible`, `eligible`는 vocabulary에만 존재하고 아직 어떤 경로도 도달하지 않음) + `quality_reason`을 계산한다. `_component_quality_state()`가 component 단위로, `_scene_quality_projection()`이 scene 단위로 집계한다.
- 기존 `record["state"]`(`constructed`/`review_required`/`unsupported`)는 **compatibility projection으로만** 유지한다 — 하위 호환을 위한 필드이며 canonical이 아님을 코드 주석에 명시했다.
- Invalid support crossing이 검출된 component는 `materialization_state="materialized"` + `quality_state="ineligible"` + `quality_reason="invalid_support_crossing"`으로 표현된다 — "surface를 못 만듦"과 "surface는 만들었지만 품질 검증 실패"가 이제 서로 다른 축의 다른 값이다.
- RMS fidelity gate 초과는 component별로 `quality_state="review_required"`, `quality_reason="source_point_rms_exceeds_threshold"`로 반영되며, 이미 `ineligible`인 component는 덮어쓰지 않는다(crossing이 더 강한 결함).
- `quality_state=="eligible"`은 vocabulary에는 존재하지만 어떤 경로도 아직 도달하지 못한다 — crossing 검사가 대표 표본(bundle)일 뿐 연속 보장이 아니고, bidirectional fidelity/false-hole hardening이 미구현이기 때문이다. 이 라벨은 isolated review 진단 전용이며 production Gate 승인과 무관함을 docstring에 명시했다.

### 2. Fan seam semantic 교정 (`_anchor_fan_review_layers`)

- 이전에는 pole↔corner 2점 correspondence chord를 그대로 `patch_seams`(→`patch_boundaries`)의 seam으로 사용했다. 이제는 별도로 `evaluate_iso_edge(surface, "u0", samples=5, ...)`를 호출해 **독립적으로 평가한** `evaluated_curve` seam entity를 만들고, 이것만 `patch_boundaries` export에 사용한다.
- Pole↔corner 2점 데이터는 `support_correspondence_chords`에만 남는다. 두 표현이 기하학적으로 같은 직선(degree_v=1이라 정확히 일치)이어도 별도 entity로 유지한다 — 수치가 같다고 semantic entity를 합치지 않는다는 지시를 반영했다.
- `_seam_payloads()`가 이제 `edge_a`("u1", 인접 patch 쪽), `edge_b`(entity의 `parameter_edge`, 항상 "u0"), `parameter_direction`, `same_orientation`(현재 구성상 항상 `True` — patch k의 u0 열과 patch k-1의 u1 열이 동일 control column을 그대로 재사용하기 때문), `shared_endpoint_policy`, `parameter_samples`를 추가로 보존한다.

### 3. Crossing gate 범위 제한과 대표 bundle (`torch_boundary_review_geometry.py`, `_anchor_fan_review_layers`)

- `detect_support_curve_crossings()`가 이제 `scope="representative_support_curve_bundle_crossing"`과 `not_checked_categories=["patch_overlap","jacobian_foldover","seam_inconsistency","full_support_family_crossing"]`를 항상 payload에 포함한다 — 이 gate가 patch 내부 self-overlap, seam tangent/Jacobian 일관성, 연속 non-intersection 증명 중 어느 것도 검사하지 않음을 명시한다.
- Fan 경로의 `evaluated_support_curves`는 이제 patch당 단일 `u=0.5` 곡선이 아니라, **configurable** `interior_support_curve_fractions`(기본 `(0.25, 0.5, 0.75)`, CLI `--interior-support-curve-fractions`)의 각 값마다 실제 iso-curve를 평가한 bundle이다(8 patch × 3 = 24 curves). 하드코딩된 canonical 상수가 아니라 기본값이 있는 설정으로 구현했다.
- Bundle 확장 결과 이전에는 patch 3/7만 비교하던 것이 이제 patch 3/7의 세 fraction 조합까지 포함해 더 넓은 interior 영역을 커버한다.

### 4. Crossing tolerance/sampling 분리 (`torch_boundary_review_geometry.py`)

- 이전 `scale`(curve 자신의 인접 표본 간격 median)을 tolerance 기준으로 쓰던 방식을 제거했다. `classify_support_curve_pair`/`detect_support_curve_crossings`는 이제 `tolerance_scale`을 **명시적으로 받는다.** 러너는 `_stable_tolerance_scale()`로 이미 계산돼 있는 `source_boundary_fidelity.local_spacing_median`(관측 raw point의 median nearest-neighbor 간격)을 사용한다 — curve 표본 해상도와 무관한 안정적 기하 척도다. `tolerance_scale`을 안 주면 예전 방식(`fallback_curve_bundle_scale`, resolution-dependent)으로 fallback하되, 이는 standalone/테스트 편의용이며 프로덕션 경로 기본값이 아니라고 docstring에 명시했다.
- 분류를 6개로 재정의했다: `valid_shared_pole`, `valid_shared_boundary_endpoint`, `transversal_intersection`(단일 교차점), `overlapping_support_path`(구간 전체가 겹치는 병렬 주행 — `plane` scene 최초 발견의 실제 유형), `near_touching_ambiguous`, `no_crossing`, `not_checked`. 단순 최소거리만으로는 transversal과 overlapping을 구분할 수 없어, 최근접점의 인접 표본까지 함께 확인하는 `_classify_interior_touch()`를 추가했다 — 인접 표본을 확인할 해상도가 부족하면(예: 표본 1개) 추측하지 않고 `near_touching_ambiguous`로 보수적으로 반환한다.
- `has_invalid_crossing`은 `transversal_intersection`과 `overlapping_support_path` 두 분류에만 True다. `near_touching_ambiguous`는 non-blocking이다.
- Resolution-invariance 테스트(`test_explicit_tolerance_scale_is_resolution_independent`)로 동일 curve pair를 8/16/32/64 표본에서 재평가해도 (고정된 `tolerance_scale`을 쓰면) 분류가 바뀌지 않음을 확인했다.

### 재검증한 15-scene sweep 결과 (tolerance 분리 + bundle 적용 후)

| scene | invalid crossing (item 1-4 이후) | 비고 |
| --- | --- | --- |
| crease, elongated_plane | **있음** (`ineligible`) | 근본 원인 조사 필요 — Worklog 5 대상 |
| plane, sine, triangle, close_parallel_sheets | **없음** (이전 라운드에서는 invalid였음) | 안정적 tolerance_scale 적용만으로 해소됨 |
| density_gradient, u_shape | not_checked(unsupported) | 변화 없음 |
| crescent, planar_hole, planar_hole_offcenter, planar_hole_elliptical, curved_annulus, mild_curved_sheet, planar_hole_density_gradient | 없음 | 변화 없음 |

이전 라운드(Worklog 4 최초 기록)에서 invalid로 보고했던 6개 scene 중 4개(`plane`, `sine`, `triangle`, `close_parallel_sheets`)는 **curve 표본 해상도에 근거한 부정확한 tolerance**가 원인이었고, 실제 관측 point 밀도 기준으로 재계산하니 `plane`의 patch 3/7 조합은 `near_touching_ambiguous`로 재분류됐다(거리 자체는 이전과 동일, 판정 기준만 교정됨). `crease`와 `elongated_plane`은 여전히 `ineligible`이며 이번 라운드로는 해소되지 않았다 — 사용자가 지정한 대로 이 둘은 근본 원인(anchor 선택, boundary correspondence, 또는 component segmentation) 조사가 필요한 대상으로 남겨두고, 억지로 anchor fan을 통과시키지 않았다.

### 회귀

새/갱신 테스트:

- `tests/test_boundary_review_geometry.py`: `CROSSING_INVALID_INTERIOR`를 제거하고 `CROSSING_TRANSVERSAL_INTERSECTION`/`CROSSING_OVERLAPPING_SUPPORT_PATH`로 교체, `overlapping_support_path` 신규 테스트, 해상도 불변성 테스트, "판별 불가능하면 ambiguous" 테스트 추가(총 14 tests).
- `tests/test_boundary_first_support_runner.py`: `materialization_state`/`quality_state`/`quality_reason` canonical 필드 검증, fan seam이 `evaluated_curve`(not chord)로 export됨 검증, bundle 적용 후 `evaluated_support_curves` 개수(`patch_count*3`) 검증, `plane`의 재분류(더 이상 invalid 아님, ambiguous 1건 이상) 검증, `elongated_plane`으로 `materialized + ineligible` 경로를 고정 fixture로 회귀에 반영, unsupported(u_shape)의 `materialization_state="not_materialized"`/`quality_state="unsupported"` 검증(총 6 tests).

```text
targeted (56 tests): 전부 통과
- tests/test_patch_boundary.py
- tests/test_boundary_first_visible_builder.py
- tests/test_boundary_first_support_runner.py (6 tests)
- tests/test_boundary_review_geometry.py (14 tests)
- tests/test_boundary_support_network.py
- tests/test_boundary_constrained_surface.py
- tests/test_boundary_central_cap.py
- tests/test_boundary_surface_quality.py
- tests/test_boundary_first_support_pipeline.py
- tests/test_boundary_multi_loop.py
- tests/test_boundary_planar_partition.py
- tests/test_boundary_source_fidelity.py
- tests/test_component_boundary.py
```

전체 pytest:

```text
499 passed, 2 failed, 1 skipped, 1 warning, 8 subtests passed
```

기존 실패 2건은 이번 변경과 무관하다(`tests/test_trimmed_component_fitter.py`의 `degenerate_fraction` strict-zero 기대치, 실측 약 0.0017361111).

### 아직 not_checked인 항목 (범위 밖, 변화 없음)

- `patch_overlap`, `jacobian_foldover`, `seam_inconsistency`, `full_support_family_crossing` — crossing gate의 `not_checked_categories`로 명시적으로 유지.
- Bidirectional source-boundary fidelity, false-hole persistence/raw-support/genuine-small-hole negative control, multi-hole 실제 patch materialization — 모두 미착수.
- `quality_state=="eligible"`은 여전히 어떤 경로도 도달하지 않는다.

### dispatcher/production 비접촉 확인

`git status` 기준 이번 라운드에서 수정한 기존 추적 파일은 없다(`torch_boundary_review_geometry.py`, `boundary_first_support_runner.py`, `tests/test_boundary_review_geometry.py`는 이미 이번 세션 이전 라운드에서 untracked/modified 상태였던 파일이다). `boundary_first.py`(legacy dispatcher/`renderer_payload`)는 이번 라운드에서 추가로 수정하지 않았다. Default dispatcher, trainer, production pipeline, uncertain Gaussian proposal/append, ownership, checkpoint, multi-hole materialization 중 어느 것도 건드리지 않았다. Rectangle/PCA/box/trimmed fallback이나 synthetic center anchor를 추가하지 않았다. 자동 Gate 승인도 하지 않았다 — `quality_state=="eligible"`은 여전히 도달 불가능한 vocabulary 값이다.

### 다음 작업 (Worklog 5 대상)

anchor/support construction의 독립적인 근본 수정이 완료되면 새 Worklog 5을 사용한다(이 문서에는 더 이상 append하지 않는다).

1. `crease`/`elongated_plane`(과 필요시 다른 scene)의 crossing 실패를 독립적으로 attribution한다: anchor clearance/boundary containment/angular coverage/correspondence ordering/component 근거를 각각 확인한다.
2. Anchor 후보 선택을 observed evidence 기반으로 hardening한다(synthetic centroid 금지).
3. Boundary correspondence(현재 순수 arclength 등분)를 anchor 기준 star-shaped 검증 없이 각도 정렬에 의존하지 않도록 hardening한다.
4. `plane`/`triangle`/`elongated_plane`/`sine`을 positive control로, `crease`/`close_parallel_sheets`를 억지로 통과시키지 않는 것을 목표로 positive/negative regression을 추가한다.

Repository-wide pytest가 green이 아니고, positive fixture에 invalid crossing이 남아 있으므로(현재 `elongated_plane`) 이번 라운드로 Boundary-first Gate 완료를 주장하지 않는다.
