# Worklog 80: Phase D — Parametric Continuation Domain 설계

날짜: 2026-07-23

상태: **설계 Revision 3. 구현 미착수, 미승인. Phase C evidence 결합, Phase E candidate 생성, production integration 모두 미착수.**

---

> ## IMPORTANT — SUPERSEDED DESIGN SECTIONS
>
> 이 worklog의 §1–6은 revision 1 당시의 historical record이며
> 현재 구현 계약이 아니다.
>
> 현재 canonical Phase D 계약은:
> - 이 파일의 revision 2 이후 절(§7, §8)
> - `docs/Urgent_Work/OSN_GS_Phase_D_Continuation_Domain_Design.md`
>
> 를 따른다.
>
> UV-space perpendicular, continuous analytic ContinuationDomain,
> second-order canonical position, candidate 상태명, byte-identical
> regression 요구는 모두 폐기됐다.
>
> 추가로(revision 3, §8): closed-loop arclength/`boundary_length` 계약,
> `local_surface_scale`의 canonical 집계 공식, `ContinuationDomainBuildError`를
> 이용한 pre-grid 실패 구분, second-order diagnostic 명칭(`second_order_*`)도
> §7 시점 이후 다시 한 번 개정됐다 — §7만 읽고 최신이라고 오해하지 말 것.
> 항상 `OSN_GS_Phase_D_Continuation_Domain_Design.md` 자체를 canonical
> source로 확인한다.

---

## 1. 배경

사용자가 Gate C를 최종 승인했다(`docs/worklogs/79_observation_evidence_phase_c_gate_c_round2.md`의 보완 결과 포함). Non-blocking note 2건(duplicate camera 순서 교환 시 fingerprint 동일 — 문제로 보지 않음; synthetic per-view payload 기반 순수 aggregation truth-table 테스트는 향후 추가 권장)을 남겼다.

이어서 사용자는 Phase D의 **구현이 아니라 설계**를 요청했다: `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §6(Phase D)을 기준으로 입력/출력 계약, construction 방법, validity 조건, boundary pairing 관계, fixture/test 계획, 대안 비교, 권장 최소 구현 범위를 상세 설계하라는 지시였다. 코드 구현은 명시적으로 금지됐고, 존재하지 않는 API를 이미 구현된 것처럼 가정하지 말고 실제 코드를 먼저 감사하라는 지시가 있었다.

## 2. 착수 전 코드 감사

설계에 앞서 두 개의 조사를 병렬로 수행했다(Explore agent, 코드 변경 없음).

1. `osn_gs/surface/torch_nurbs.py` 전체 API — `TorchNURBSSurface`의 정확한 필드, `evaluate_with_derivatives`/`evaluate_with_second_derivatives`의 정확한 signature와 반환 형식, `boundary_control_indices`/`SharedBoundaryConstraint`/`fit_coupled_patch_graph_lsq`, knot vector 관례, 그리고 `predict_torch_occlusion_curves`/`build_torch_surface`/`sample_torch_occluded_surface`/`fit_torch_base_curves`/`pca_parameterize_points` 등이 실제로 살아있는 코드인지 pre-reset legacy인지 판정.
2. `osn_gs/surface/torch_annulus_chart.py`의 기존 diagnostics(`_jacobian_diagnostics`, `_orientation_holonomy`, `_parameter_quality`, `_boundary_conformance`, `_measure_seams`)와 두 개의 과거(pre-reset) 확장 설계 문서(`OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md`, `OSN_GS_Final_Boundary_First_NURBS_Direction.md`)에 이미 남아있는 재사용 가능한 아이디어.

직접 코드를 다시 읽어 핵심 부분(`evaluate_with_second_derivatives`의 실제 rational quotient-rule 구현, `_clamped_knot_vector`/`_bspline_basis_pair`의 domain-clamping 로직, `_jacobian_diagnostics`의 `J^T J` closed-form eigenvalue 공식)을 직접 검증했다.

핵심 발견:

- Knot vector는 `[0,1]`에 하드코딩된 clamped uniform B-spline이며, `[0,1]`을 넘어서는 domain 확장 헬퍼는 어디에도 없다. Span-detection 로직이 폐구간을 전제하므로 안전하게 확장할 수 없다.
- `PatchBoundarySegment`는 `patch_id`(정수)만 갖고 실제 `TorchNURBSSurface` 참조는 갖지 않는다 — Phase D는 caller가 `patch_id -> surface` 매핑을 별도로 제공해야 한다.
- `predict_torch_occlusion_curves`/`build_torch_surface`/`sample_torch_occluded_surface`는 자체 docstring에 "Stage 2용 legacy helper"로 명시돼 있고 production/test 호출부가 전혀 없다 — pre-reset의 단일 방향 global occlusion 추정 방식 그대로다. Phase D는 이 셋을 재사용하지 않는다.
- `torch_annulus_chart.py`의 `_jacobian_diagnostics`는 `‖Su×Sv‖`가 아니라 `J^T J`의 진짜 closed-form singular value를 계산하지만, 입력이 `TorchNURBSSurface` 객체이고 내부에서 그 객체의 `evaluate_with_derivatives`를 직접 호출한다 — Phase D의 continuation domain은 `TorchNURBSSurface`가 아니므로 그대로 재사용할 수 없다(prerequisite로 별도 명시).
- 과거 확장 설계 문서 2개는 구체적 수식(knot insertion, control-point extrapolation 공식 등)이 없고 구조적 아이디어(`S_ext(s,t)`/`C_ext(s,t)` 분리, local frame 개념)만 갖고 있었다 — 이 구조적 아이디어는 premise-independent(global component 선행복구라는 폐기된 전제와 무관)이므로 재사용했다.

## 3. 설계 결과 요약

전체 설계는 `docs/Urgent_Work/OSN_GS_Phase_D_Continuation_Domain_Design.md`에 기록했다. 핵심 결정만 요약한다.

### Outward 방향 — 축-비의존 general formula

사각형 patch edge(`u0/u1/v0/v1`)라면 `±S_u`/`±S_v` shortcut으로 충분하지만, trimmed loop boundary는 축에 정렬돼 있지 않다. 최소자승으로 `tangent_world ≈ a*S_u + b*S_v`를 풀어 UV-space tangent `(a,b)`를 구하고, 두 perpendicular 후보 중 `inner_uv`(Phase A가 이미 계산해 둔 값, 재추정하지 않음) 반대쪽을 outward로 선택하는 general formula를 설계했다. 사각형 edge에서는 이 general formula가 축 정렬 shortcut과 수치적으로 일치해야 한다는 것을 fixture invariant로 고정했다.

### Continuation은 `TorchNURBSSurface`가 아니라 별도 경량 객체

`[0,1]`을 넘는 knot vector 확장은 새로운 low-level B-spline 수학(knot insertion 등)이 필요해 구현 복잡도/위험이 높다는 것을 §0 감사로 확인했다. 대신 `ContinuationDomain`을 boundary의 analytic `S, S_u, S_v, S_uu, S_uv, S_vv`로부터 매번 닫힌형(Taylor/Hermite류)으로 평가되는 독립 객체로 설계했다 — 1차항(canonical baseline)은 `world(s,t) = boundary_world(s) + t*outward_tangent_world(s)`, 2차항(curvature-aware, uncertainty 전용)은 `+ 0.5*t^2*outward_second_derivative_world(s)`. `t`는 world-length 단위이며, `outward_tangent_world`를 unit vector로 정규화해서 얻는다.

대안 두 가지(control-net extrapolation, local fitted proxy)와 비교했다 — control-net 방식은 정확도/NURBS 일관성은 최고지만 구현 복잡도·위험이 크고(§0에서 확인한 knot 확장 헬퍼 부재), local fitted proxy는 이미 analytic하게 정확한 함수를 다시 근사 fit하는 순환적 정보 손실이라 어느 단계에서도 채택하지 않기로 했다.

### Validity 조건과 self-intersection의 범위 밖 처리

마스터 플랜(§8 Phase F)이 self-intersection 진단을 명시적으로 Phase F 소유로 두고 있으므로, Phase D는 Jacobian collapse/curvature-growth 지표만 로컬 proxy로 제공하고 전체 pairwise 자기교차 검사는 구현하지 않기로 했다 — 결과에 `self_intersection_checked: false`를 명시해 후속 phase가 "이미 검증됐다"고 오해하지 않게 했다. Source-visible-surface 역침범도 동일하게 curvature-growth 지표로 근사하고 독립적인 nearest-point 검사는 미룬다.

Invalid domain은 폐기하지 않고 `state`(`candidate|degenerate|rejected`)와 `reason`을 함께 항상 반환한다 — `PatchReconciliationResult`가 실패한 pair도 보존하는 기존 관례를 그대로 따랐다.

### Boundary pairing은 Phase E 소유, Phase D는 하지 않음

Phase D는 boundary 하나당 독립된 `ContinuationDomain` 하나만 만든다. Phase E가 필요로 할 최소 interface(`world` 샘플, `aabb_min`/`aabb_max`, `source_patch_id`/`source_boundary_id`)만 미리 제공한다. `aabb_min`/`aabb_max` 필드명은 `torch_voxel_hierarchy.VoxelNode`의 기존 AABB 관례와 통일했다.

### Phase C evidence와의 interface (호출하지 않음)

`ContinuationDomain.world.reshape(-1, 3)`가 `classify_world_samples(evidence, world_points)`가 받는 `(N,3)` 형태와 이미 일치한다는 것만 문서화했다. Phase D 코드 자체는 `torch_observation_evidence` 모듈을 import하지 않는다.

## 4. 식별한 Prerequisite

구현 착수 전 필요한 순수 리팩터(무동작-변경) 하나를 발견했다: `torch_annulus_chart.py`의 `_jacobian_diagnostics` 핵심 계산(`J^T J` closed-form eigenvalue, self-consistent orientation reference)을 `TorchNURBSSurface`가 아니라 raw `(deriv_a, deriv_b)` 텐서를 받는 surface-agnostic 헬퍼로 추출해야 한다. 이 리팩터 없이는 Phase D의 validity 조건을 코드 중복 없이 구현할 수 없다. 리팩터는 기존 annulus 테스트가 byte-identical 결과를 내는지로 검증해야 한다.

## 5. 문서 갱신

- `docs/Urgent_Work/OSN_GS_Phase_D_Continuation_Domain_Design.md`(신규): 요청받은 8개 섹션 전체(입력 계약, 출력 계약, construction 방법, validity 조건, boundary pairing, fixture/test 계획 12건, 대안 비교, 권장 최소 구현 범위) + 승인 게이트 D 보고 표 초안(구현 후 채울 빈 표).
- `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`: 최상위 상태 줄과 §11(승인 상태) 갱신 — Gate C 최종 승인 기록, Phase D를 "설계 완료, 구현 미승인"으로 표시, §6(Phase D) 작업 목록에 prerequisite 리팩터와 `reconciled_internal` 제외 규칙을 명시적으로 추가.
- `docs/README.md`: 요약 항목 추가(다음 절에서 이어짐).

## 6. 남은 위험과 다음 승인

- 이 설계 문서 자체가 사용자 승인 대상이다 — 승인 전에는 어떤 코드도 작성하지 않는다.
- 설계에서 다룬 general outward-direction formula(최소자승 기반)는 실제 구현 시 annulus의 내부-극점 근방처럼 `S_u`/`S_v`가 거의 평행해지는 퇴화 케이스에서 최소자승 시스템 자체가 ill-conditioned해질 수 있다 — 설계 문서 §5의 Jacobian collapse 조건이 이 케이스를 감지하도록 의도했지만, 실제 구현 후 fixture로 재확인이 필요하다.
- Prerequisite 리팩터의 정확한 위치(`osn_gs/surface/torch_jacobian_diagnostics.py` 신설 여부 등)는 구현 착수 시점에 최종 결정한다.

계획대로 설계만 완료하고 멈춘다. 다음 승인 요청은 **Phase D 구현 착수**(이 설계 문서의 승인을 전제)로 한정한다.

## 7. 사용자 검토 후 설계 수정 (revision 2)

사용자가 Phase D의 큰 방향(마스터 플랜 §6 역할 제한: "Visible NURBS boundary → sampled continuation strip → validity/uncertainty metadata"만 하고 최종 NURBS chart는 만들지 않음)은 승인했으나, §1-10 설계 문서 상태로는 구현 착수를 승인하지 않고 11개 항목의 교정을 요구했다. 새 문서를 만들지 않고 기존 두 파일(`OSN_GS_Phase_D_Continuation_Domain_Design.md`, `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`)만 수정했다.

### 반영한 핵심 교정

1. **`ContinuationDomain`을 sampled domain으로 제한**: 임의 `(s,t)`를 analytic closed-form으로 재평가하는 `evaluate()`/`evaluate_with_derivatives()` 계약을 제거했다. `PatchBoundarySegment`가 연속 함수가 아니라 ordered samples만 갖는다는 사실과 대칭을 맞춰, sampled grid(`world`/`s_world`/`t_world`/`tangent_s`/`tangent_t`/`normal`)를 canonical source of truth로 고정했다. 임의 `s` 조회가 필요하면 `interpolate_boundary_arclength()`라는 별도의 piecewise-linear 보조 함수만 제공한다(analytic이 아님을 명시).
2. **`s`의 단위 고정**: sample index/normalized parameter/arclength 혼용을 없애고 `s = boundary를 따른 cumulative world arclength`, `t = world-space continuation distance`로 확정했다. 모든 `tangent_s`/`d(outward)/ds`는 world-arclength 간격으로 정규화해 sampling density 변화에 안정적이도록 했다.
3. **Outward 방향을 world-space에서 재정의**: 이전 설계의 UV-space perpendicular 최소자승 공식(`(b,-a)` 후보 선택)을 폐기하고, `N = normalize(S_u x S_v)`, `C = normalize(N x T)`(world tangent `T`는 world-arclength 기준), `inner_world` 방향과의 내적으로 부호를 선택하는 순수 world-space 공식으로 교체했다. UV parameterization의 skew/scale에 영향받지 않는다 — 이 invariant를 새 fixture(UV scale/skew)로 명시적으로 검증하도록 §7에 추가했다.
4. **UV 방향 벡터의 world-metric normalization**: Second-derivative diagnostic이 필요할 때만 world outward 방향을 UV로 투영하는 최소자승(`q = q_raw / ||J q_raw||`, `||Jq||=1` 정규화)을 별도 단계(§4.3)로 분리했다 — outward 방향 결정 자체에는 더 이상 이 최소자승이 쓰이지 않는다.
5. **Canonical geometry는 first-order only**: `position_order`/`curvature_aware` position variant 필드를 제거했다. `world`는 항상 1차 근사이며, 2차 정보는 `curvature_displacement_at_t_max`/`curvature_growth_ratio`라는 uncertainty-전용 스칼라로만 존재한다.
6. **Inner-probe distance와 continuation extent 분리**: `inner_distance_median`을 곧바로 `t_max`로 쓰던 것을 폐기하고, `inner_probe_distance`(부호 선택 전용) / `local_surface_scale`(boundary spacing, inner-isocurve distance, control-net local edge length 등 여러 finite-positive 후보의 median aggregate) / `continuation_extent`(`= extent_multiplier * local_surface_scale`)로 3분했다. `extent_multiplier`의 정확한 기본값은 고정하지 않고 벤치마크로 검증할 configurable 값으로 남겼다.
7. **상태 이름 변경**: `candidate`/`degenerate`/`rejected`를 `valid`/`degenerate`/`rejected`로 바꿔 Phase E의 `occluded-region candidate`와의 혼동을 없앴다. `valid`가 "occluded surface로 승인됨"을 의미하지 않는다는 것을 docstring에 명시했다.
8. **입력 오류와 결과 품질 문제 분리**: `reconciled_internal` boundary, `expected_patch_id` 불일치, shape 불일치, 최소 sample 수 미달은 전부 `ValueError`를 즉시 발생시키는 입력 계약 위반으로 통일했다. Jacobian collapse/orientation 비일관성/curvature growth 등은 예외 없이 `state=degenerate` + `reason`으로 반환한다. Degenerate normal/direction은 NaN 대신 zero vector + `normal_valid_mask`/`direction_valid_mask`/`sample_valid_mask`로 표현하도록 바꿨다.
9. **최소 boundary sample 계약 추가**: open boundary 최소 3개, closed boundary 최소 4개 unique sample을 입력 계약에 명시하고, 이 기준이 충분한지 확인하는 fixture를 추가했다.
10. **Diagnostics helper를 두 책임으로 분리**: 기존에 하나의 헬퍼로 뭉뚱그렸던 prerequisite를 `compute_parametric_jacobian_metrics`(singular value/condition, annulus와 공유)와 `compute_orientation_consistency`(orientation flip, topology별 wrapper가 각자 소유 — annulus의 순환 topology와 continuation strip의 비순환 topology는 일관성 검사 형태가 다르므로)로 나눴다. 리팩터 회귀 기준도 byte-identical에서 "상태 분류/report field/수치 tolerance 동일"로 완화했다(이 편이 실제로 검증 가능한 기준이라는 사용자 지적을 반영).
11. **Implementation Plan 문서 정합성**: Phase B 상태를 "Gate B 검토 대기"에서 "완료, Gate B 승인"으로, Phase C 상태를 "사용자 최종 재검토 대기"에서 "Gate C 최종 승인"으로 수정했다. Phase C의 per-view 5-state와 aggregate 5-state를 최신 enum(`on_observed_surface`/`conflicting_evidence` 포함)으로 명시적으로 분리해 기록했다. §11의 승인 이력 서술을 표 형태의 canonical 현재 상태 + worklog 링크 중심으로 압축했다.

### 반영하지 않거나 구현하지 않은 것(이번 작업 범위 밖, 지시대로)

- Phase D 코드 구현.
- Diagnostics helper 실제 리팩터.
- Phase C evidence 호출, Phase E pairing/candidate 생성, Phase F NURBS fitting.
- Production integration.

### 남은 prerequisite (변경 없음)

- `compute_parametric_jacobian_metrics`/`compute_orientation_consistency` 리팩터는 여전히 구현 착수 전 필요하다(§9).
- `patch_id -> TorchNURBSSurface` 매핑 관례는 여전히 caller 책임이며 아직 확정되지 않았다.
- `expected_patch_id` 자기일관성 체크만으로는 잘못된 surface/boundary 조합을 완전히 막을 수 없다는 한계를 설계 문서 §2에 명시했다.

계획대로 문서 두 개(`OSN_GS_Phase_D_Continuation_Domain_Design.md`, `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`)만 수정하고 멈춘다. 새 worklog는 만들지 않았다. Phase D 코드 구현, Phase C evidence 실제 결합, Phase E candidate 생성, Phase F NURBS fitting, production integration은 모두 별도 승인 전까지 시작하지 않는다.

## 8. 사용자 검토 후 설계 수정 (revision 3)

사용자가 Phase D의 큰 방향과 revision 2의 핵심 수학 계약(world-space outward 공식, sampled-grid source of truth, first-order canonical geometry, 상태 이름 `valid/degenerate/rejected` 등)을 승인했다. 다만 구현 착수 전 마지막으로 문서 5개 보완 항목(+ worklog banner, impl plan 상태 갱신)을 요구했다. 새 문서를 만들지 않고 기존 세 파일(`OSN_GS_Phase_D_Continuation_Domain_Design.md`, `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`, 이 worklog)만 수정했다.

### 반영한 교정

1. **Closed boundary arclength 계약 완성**: `s_world[i]`는 "첫 sample부터 i번째 sample까지의 누적 길이"만 의미하도록 명확히 하고, closing segment 길이는 `s_world`에 섞지 않고 신규 필드 `boundary_length`(open: `s_world[-1]`, closed: `s_world[-1] + closing segment 길이`)에만 반영하도록 분리했다. Closed boundary의 periodic central difference는 `boundary_length`를 이용해 wrap-around 이웃의 "periodic s"를 오프셋하는 방식으로 명시했다(`previous의 periodic s = s_world[-1] - boundary_length`, `next의 periodic s = boundary_length`(wrap 시)). Canonical representation은 duplicate closing endpoint를 저장하지 않는 ordered unique samples로 통일했다 — `torch_patch_boundary.py`의 `_canonicalize_closed_loop`가 실제로 마지막 샘플을 첫 샘플과 중복시킨다는 사실을 확인하고(`ordered + [ordered[0]]`), `build_continuation_domain`이 이 중복을 정규화 단계에서 먼저 제거한 뒤 자신의 unique-sample 표현을 구성하도록 §2.1을 신설했다.
2. **Adjacent duplicate/zero-length segment 검증**: 전체 unique sample 수 확인만으로는 부족하다는 지적을 반영해, 인접 샘플 간 world distance가 `arclength_epsilon`(신규 파라미터, 기본값 `1e-6`, scale-aware 값을 caller가 넘길 수도 있음) 이하이면 `ValueError`를 던지도록 §2.2에 추가했다(open은 모든 adjacent pair, closed는 closing segment도 포함). 자동 deduplication/repair는 하지 않는다. 단순 `eps` clamp로 잘못된 tangent를 만드는 대신 명시적 reject로 처리한다.
3. **Pre-grid build 실패와 사후 품질 문제 분리**: 신규 예외 타입 `ContinuationDomainBuildError(RuntimeError)`를 도입해 3단계로 분리했다 — (a) 입력 계약 위반(`ValueError`, §2.2: reconciled_internal, patch_id 불일치, shape 불일치, sample 부족, adjacent duplicate, 명시적 scale/multiplier가 non-finite/`<=0`), (b) grid 자체를 구성할 수 없는 필수 실패(`ContinuationDomainBuildError`, §2.3: automatic scale derivation 실패, surface evaluation 전체 실패, tangent/outward를 전혀 만들 수 없음, finite grid/AABB 구성 불가), (c) grid는 구성됐지만 부분적 품질 문제(`state=degenerate`/`rejected`인 `ContinuationDomain` 반환, §2.4/§5). `local_surface_scale` 자동 도출 실패가 `state=rejected`인 완전한 객체를 반환하는 것처럼 읽히던 이전 문서의 모호함을 해소했다 — 이제는 grid/AABB가 실제로 만들어지지 않았으면 `ContinuationDomain` 객체 자체가 없다.
4. **`local_surface_scale` canonical 공식 확정**: 구현 시점에 임의로 집계 방식을 정하지 않도록 정확한 공식을 고정했다(§4.5.1) — `L_boundary`(양의 인접 boundary segment 길이의 median, closed는 closing segment 포함), `L_inner`(양의 `‖inner_world - boundary_world‖`의 median), `L_control`(source NURBS control_grid의 u/v 방향 양의 인접 control-point edge 길이의 median)을 각각 하나의 scalar로 축약한 뒤, `valid_scales = {L_boundary, L_inner, L_control}` 중 finite·positive만 모아 `local_surface_scale = median(valid_scales)`. `len(valid_scales) < 2`이면 `ContinuationDomainBuildError`. Caller가 명시적으로 `local_surface_scale`을 제공하면 이 자동 집계를 건너뛰되 explicit 값은 finite·positive여야 한다(§2.2). Inner probe 단독 비의존/boundary density 안정성/control grid resolution 안정성 invariant를 §7 fixture에 명시했다.
5. **Second-order diagnostic 명칭·의미 정제**: `curvature_growth_ratio`/`curvature_displacement_at_t_max`를 각각 `second_order_growth_ratio`/`second_order_displacement_at_extent`로 개명했다. §4.3 상단에 "이 값은 surface의 intrinsic curvature(mean/Gaussian/normal/geodesic) 추정치가 아니라 world-metric normalized 하나의 outward 방향에 대한 directional second-order continuation diagnostic일 뿐"이라는 문구를 명시적으로 추가해, 논문에서 쓰는 surface curvature와 동일한 것처럼 오인되지 않도록 했다.
6. **Worklog 80 SUPERSEDED 표시**: 이 worklog 최상단 상태 줄 바로 아래에 지시받은 문구 그대로 banner를 추가해, §1-6(revision 1)이 historical record이며 canonical 계약은 이 문서의 revision 2/3 절과 설계 문서 본체를 따라야 함을 명시했다. §7까지만 읽고 최신으로 오인하지 않도록, revision 3(§8)이 §7의 일부 내용(예: 8번 항목의 `curvature_growth_ratio` 명칭, 6번 항목의 `local_surface_scale` 도출-실패 처리)도 다시 개정했다는 점을 banner에 덧붙였다.
7. **Implementation Plan 상태 갱신**: 최상위 상태 줄과 §11 표를 "Phase D: 방향 승인, Design Revision 3, 구현 미승인"으로 갱신했다. §6(Phase D) 요약에 `boundary_length`, adjacent-duplicate/zero-length segment 거부, `local_surface_scale`의 canonical 집계 공식, pre-grid build 실패가 `ContinuationDomainBuildError`라는 점, second-order diagnostic이 intrinsic curvature가 아니라는 점을 모두 반영했다.

### 반영하지 않거나 구현하지 않은 것(이번 작업 범위 밖, 지시대로)

- Phase D 코드 구현.
- Diagnostics helper 실제 리팩터.
- Phase C evidence 호출, Phase E pairing/candidate 생성, Phase F NURBS fitting.
- Production integration.

### 남은 prerequisite (변경 없음)

- `compute_parametric_jacobian_metrics`/`compute_orientation_consistency` 리팩터는 여전히 구현 착수 전 필요하다(§9).
- `patch_id -> TorchNURBSSurface` 매핑 관례는 여전히 caller 책임이며 아직 확정되지 않았다.
- `arclength_epsilon`의 정확한 기본값/scale-aware 도출 방식은 이 문서에서 고정하지 않았다 — 구현 착수 시점에 확정한다.

계획대로 문서 세 개(`OSN_GS_Phase_D_Continuation_Domain_Design.md`, `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`, 이 worklog)만 수정하고 멈춘다. 새 worklog나 새 설계 문서는 만들지 않았다. Phase D 코드 구현, diagnostics helper 리팩터, Phase C evidence 호출, Phase E/F 작업, production integration은 모두 사용자 승인 전까지 시작하지 않는다.
