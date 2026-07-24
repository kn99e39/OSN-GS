# Worklog 81: Phase D — Parametric Continuation Domain 구현 및 Gate D 보고

날짜: 2026-07-23

상태: **Phase D 구현·검증 완료. 승인 게이트 D 보고. Phase E candidate 생성, Phase F NURBS fitting, Phase C evidence 실제 결합, production integration 모두 미착수.**

## 1. 배경

사용자가 Phase D의 큰 방향과 설계 Revision 3(`docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`)을 승인하고 구현 착수를 지시했다. 이 문서는 그 구현 결과와 마스터 플랜(`docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §6)이 요구하는 승인 게이트 D — "continuation strip의 방향·크기·결정성을 보고하고 멈춘다" — 를 보고한다.

## 2. Prerequisite 리팩터

설계 §9가 요구한 대로, 새 모듈 `osn_gs/surface/torch_parametric_diagnostics.py`를 만들어 `torch_annulus_chart.py`의 `_jacobian_diagnostics`가 뒤섞고 있던 두 책임을 분리했다.

- `compute_parametric_jacobian_metrics(deriv_a, deriv_b, eps, scale)`: `J^T J`의 closed-form eigenvalue 기반 진짜 singular-value/condition 계산. `TorchNURBSSurface` 대신 raw `(deriv_a, deriv_b)` 텐서를 받는다.
- `compute_orientation_consistency(normals, valid_mask, eps)`: 단일 self-consistent reference 기반 orientation 검사(annulus가 기존에 쓰던 것과 동일한 flat/adjacency-비의존 알고리즘, 그대로 재사용).

`torch_annulus_chart.py`의 `_jacobian_diagnostics`는 이제 이 두 헬퍼를 호출하도록 리팩터했다(필드 이름/의미는 100% 동일하게 유지). `_orientation_holonomy`(cross-slice ring topology 체크)는 손대지 않았다 — 설계가 명시한 대로 topology별 adjacency 로직은 각자 소유한다.

**회귀 검증**: `tests/test_annulus_chart.py` 48개 전부 통과(byte-identical 요구 없이, 상태 분류/report field/수치 tolerance 기준 — 실제로는 수식을 그대로 옮겼으므로 수치도 동일하다).

## 3. 구현

새 모듈 `osn_gs/surface/torch_continuation_domain.py`:

- `ContinuationDomain` dataclass, `STATE_VALID`/`STATE_DEGENERATE`/`STATE_REJECTED`, `ContinuationDomainBuildError` 예외 타입.
- `build_continuation_domain(surface, boundary, *, expected_patch_id=None, extent_multiplier=1.0, local_surface_scale=None, arclength_epsilon=1e-6, t_count=5, jacobian_eps=1e-8, second_order_growth_threshold=0.5)`: entry point.
- `interpolate_boundary_arclength(boundary, s_query)`: piecewise-linear 보조 함수.
- 내부 헬퍼: closed-loop 중복 종료점 stripping(§2.1), world-arclength `s_world`/`boundary_length`(§4.1), world-arclength 기준 유한차분(개방/폐쇄 periodic 포함), world-space outward-direction solver(§4.2), UV 투영 second-order diagnostic(§4.3), `local_surface_scale` canonical aggregate(§4.5.1), strip-adjacency orientation flip count(§9 prerequisite 2 — annulus와 별도로 이 모듈이 직접 소유).

`osn_gs/surface/__init__.py`에 `ContinuationDomain`/`ContinuationDomainBuildError`/`build_continuation_domain`을 entry point + result/예외 타입만 export했다(house 관례).

### 구현 중 발견해 수정한 설계-구현 간극

설계 §2.3은 "Boundary 전체에서 tangent/outward direction을 단 하나도 만들 수 없음(모든 샘플이 degenerate)"을 `ContinuationDomainBuildError` 트리거로 명시했다. 최초 구현은 이 경우를 빠뜨리고 `state=rejected`인 완전한 `ContinuationDomain`을 반환하도록 잘못 처리하고 있었다 — 테스트(전체 방향이 degenerate한 fixture) 작성 중 이 간극을 발견해, `direction_valid_mask`가 모든 샘플에서 `False`이면 grid 구성 전에 즉시 `ContinuationDomainBuildError`를 던지도록 수정했다. 설계 문서 자체는 변경하지 않았다 — 구현이 설계를 그대로 따르도록 고쳤다.

## 4. 승인 게이트 D 보고 (설계 문서 §10 표, 실제 수치)

| 검증 항목 | Fixture | 결과 |
|---|---|---|
| UV axis swap/scale/skew, loop reversal에 대한 world-space invariance | `test_uv_axis_swap_world_space_invariance`, `test_uv_scale_skew_world_space_invariance`, `test_reversed_parameter_direction_same_world_geometry` | PASS — outward 방향이 `atol=1e-6` 이내로 동일 |
| Plane | `test_planar_boundary_outward_direction_and_zero_second_order_growth`(degree=1 평면) | PASS — outward가 정확히 `(0,-1,0)`, `second_order_growth_ratio_max < 1e-8` |
| Curved | `test_smoothly_curved_boundary_bounded_second_order_growth` | PASS — `second_order_growth_ratio_max < 0.5` |
| Orthogonal/Oblique — normal/facing hard gate 없음 policy 확인 | `test_orthogonal_and_oblique_boundaries_not_rejected_on_facing` | PASS — 서로 직교하는 두 patch 모두 독립적으로 domain 생성됨 |
| Degenerate Jacobian/normal/tangent — NaN 없이 mask로 표현 | `test_degenerate_direction_uses_mask_not_nan` | PASS — `direction_valid_mask` 일부 `False`, `state=degenerate`, NaN 없음(zero vector로 대체) |
| Closed boundary closing segment가 `boundary_length`에 정확히 반영됨 | `test_closed_boundary_closing_segment_in_boundary_length` | PASS |
| 최소 sample 수 계약 충족 | `test_minimum_sample_count_open_and_closed`(open 3개, closed 4개 unique) | PASS |
| 인접 duplicate/zero-length segment가 `ValueError`로 거부됨 | `test_open_boundary_adjacent_duplicate_raises_value_error`, `test_closed_boundary_closing_segment_zero_length_raises_value_error` | PASS |
| `reconciled_internal`이 `ValueError`로 거부됨 | `test_reconciled_internal_boundary_raises_value_error`, `test_annular_seam_rejected_observed_edges_accepted` | PASS |
| `local_surface_scale` 자동 도출 실패가 `ContinuationDomainBuildError`로 구분됨 | `test_local_surface_scale_derivation_failure_raises_build_error` | PASS |
| `state`가 `occluded_candidate`/`validated chart`로 승격되지 않음 | `test_state_never_promoted_beyond_continuation_states` + 전체 fixture 공통 | PASS |
| `local_surface_scale`/`continuation_extent`가 단일 probe에 비정상 종속되지 않음 | `test_local_surface_scale_not_dominated_by_single_probe` | PASS — inner probe distance를 50배 늘려도 `local_surface_scale` 비율 10배 미만 |
| Second-order diagnostic이 `second_order_*` 명칭으로 보고되고 intrinsic curvature로 오인되지 않음 | 전체 fixture 공통(`uncertainty` dict 필드명 확인) + `test_high_curvature_fold_over_flags_excessive_growth` | PASS |

추가로 확인한 항목(설계 문서 §7에 있었으나 위 표에 없던 것): Rotated plane invariance, corner/endpoint 불변식(`world[:,0,:] == boundary.world`), `t_world`가 실제 측정 거리와 일치 — 전부 PASS.

### 결정성(determinism)

동일 입력(surface, boundary, 파라미터)에 대해 `build_continuation_domain`은 무작위성이 전혀 없는 순수 함수다(torch 연산은 전부 결정적 closed-form 대수 연산이며, GPU 비결정적 reduction에 의존하는 연산 없음). `boundary_resampling_density_invariance` 테스트가 서로 다른 샘플링 밀도에서도 같은 물리적 위치의 outward 방향과 `boundary_length`가 일치함을 확인했다.

## 5. 회귀 검증

- 신규 파일만(`tests/test_continuation_domain.py`): `22 passed`
- `tests/test_annulus_chart.py`(prerequisite 리팩터 대상): `48 passed`(회귀 없음)
- 전체 pytest suite: `260 passed, 1 skipped, 8 subtests passed`(Gate C 최종 승인 시점 `238 passed`에서 정확히 +22)
- 전체 unittest: `261 tests, OK (skipped=1)`
- `git diff --check`: 통과(줄바꿈 정규화 경고만, 무관)
- Production(`torch_pipeline.py`/`torch_trainer.py`) 미변경. Phase C evidence(`torch_observation_evidence.py`) 미호출. `osn_gs/core/` 어디에도 이번 신규 모듈을 import하지 않는다.

## 6. 남은 제한 (설계 문서가 이미 명시한 것들, 변경 없음)

- Self-intersection, 독립적 visible-surface-penetration 검사는 하지 않는다(`validity["self_intersection_checked"] = False` 등으로 명시). Phase F 몫이다.
- Boundary pairing, overlap, 공동 empty-region 지지 여부는 검사하지 않는다. Phase E 몫이다.
- Multi-scale strip은 구현하지 않았다.
- `extent_multiplier`/`second_order_growth_threshold`/`arclength_epsilon`의 기본값은 이 구현에서 잠정값(1.0/0.5/1e-6)을 그대로 썼다 — 설계 문서가 명시한 대로 확정값이 아니며, 향후 벤치마크로 재검증이 필요하다.
- `expected_patch_id` self-consistency 체크만으로는 잘못된 surface/boundary 조합을 완전히 막지 못한다는 설계상의 한계는 그대로 남아 있다.
- `patch_id -> TorchNURBSSurface` 매핑 관례는 여전히 caller(향후 benchmark 오케스트레이터) 책임이며, 이번 구현에는 그런 오케스트레이터를 만들지 않았다 — 테스트는 각자 자신의 surface/boundary 쌍을 직접 구성한다.

## 7. 중단 및 다음 승인

계획대로 Phase D 구현과 Gate D 보고까지 완료하고 멈춘다. 다음 승인 요청은 **Phase E — High-Recall Bounded Candidate Builder**로 한정하며, 별도 사용자 승인 없이는 시작하지 않는다. Phase C evidence 실제 결합, Phase F NURBS fitting, production integration도 마찬가지다.
