# OSN-GS Boundary-Conditioned Occluded Surface Implementation Plan

상태: **ACTIVE — Phase A–F 구현·검증 완료, Gate B/C/D/E/F 승인 완료, Phase F.1 보완 구현 진행 중, Gate F.1 승인 대기. Phase G 이후 미승인.**

상위 방향 문서: `OSN_GS_Direction_Reset_Plan.md`

이 문서는 global visible-surface decomposition 중심의 remediation을 중단하고, local visible NURBS boundary에서 bounded occluded surface를 구성하는 새 방법론의 active implementation gate다. 사용자는 2026-07-23 Phase A–B를 승인했고, Gate B 검토 후 Phase C만 추가 승인했으며, Gate D도 승인했다. 세 범위 모두 구현·검증을 완료했다. Phase E 이후는 별도 승인 전까지 구현하지 않고 production 기본값, main training path, 기존 Phase 1 membership을 변경하지 않는다.

## 0. 고정 계약

- Certain Gaussian은 image loss로만 움직이며 NURBS gradient의 영향을 받지 않는다.
- 데이터 흐름은 `certain Gaussian -> visible NURBS -> occluded NURBS -> uncertain Gaussian proposal`의 단방향이다.
- Voxel은 local partition/neighborhood evidence이며 occluded area의 정답이 아니다.
- Global component correctness를 새 prototype의 선행 성공조건으로 두지 않는다.
- Facing, normal similarity, tangent alignment, curvature agreement는 soft evidence다.
- 초기 범위는 two-sided 또는 multi-sided support가 있는 finite bounded region이다.
- Unsupported one-sided extrapolation, true object-boundary classifier, GT runtime 분기는 범위 밖이다.
- Independent fit 후 control point를 덮어쓰는 seam 봉합은 금지한다.
- 모든 새 기능은 isolated benchmark에서 시작하고 production model에는 연결하지 않는다.

## 1. Branch와 문서 관계

### 유지

- Adaptive voxel hierarchy와 local visible patch bootstrap
- Phase 2 support/boundary estimator
- Trimmed component 및 annulus chart fitter
- Step 5-A coupled shared-boundary fit과 Jacobian/seam diagnostics
- Existing Gaussian/NURBS binding, renderer, checkpoint/export infrastructure

### Deprecated diagnostics

- `torch_surface_proxy.py`
- `torch_surface_candidate_graph.py`
- `torch_surface_decomposition.py`
- `torch_gaussian_support_continuity.py`
- Proxy Stage 0–3/3-R script, test, artifact

이 코드는 실패 근거와 재현성 보존을 위해 남기되 production component, continuation candidate, occluded chart에 import하지 않는다.

### 승인 후 문서 migration

- `OSN_GS_Direction_Reset_Plan.md`: 최상위 methodology
- 이 문서: active implementation gate
- 기존 Final Boundary-First 문서: Phase 0–5 구현 배경 및 보존 계약
- 기존 Phase 5 문서: Step 5-A 완료 기록과 pre-reset blocker 기록
- `docs/architecture.md`, `TODO.md`, `docs/README.md`: 새 active flow에 맞춰 후속 정리

## 2. 목표 데이터 모델

### `VisiblePatchRecord`

- `patch_id`
- `surface`
- Gaussian indices와 UV binding
- component/leaf/chart provenance
- support mask와 local scale
- fit/Jacobian confidence

### `PatchBoundarySegment`

- stable `boundary_id`, `patch_id`
- ordered UV/world samples 또는 parametric curve
- endpoint와 orientation
- boundary tangent, inward surface tangent, normal
- adjacent inner isocurve
- source support/loop/chart edge provenance
- confidence
- 상태: `unclassified`, `reconciled_internal`, `unsupported`, `extension_candidate`

### `PatchBoundaryGraph`

- adjacent patch/boundary pair
- endpoint correspondence
- same/reversed parameter direction
- normalized gap와 overlap
- tangent/normal/derivative evidence
- reconciliation fit 및 post-fit validity

### `ObservationEvidence`

- camera pose/projection/intrinsics
- 명시적인 depth convention
- coverage/alpha 또는 유효 depth mask
- world point의 per-view projection
- known-free-space / behind-observation / unobserved query
- evidence가 생성된 Gaussian/NURBS topology version

### `ContinuationDomain`

- source boundary/patch provenance
- world-arclength `s`와 world-distance `t`로 고정 단위를 가진 sampled `(s, t)` grid(world/tangent_s/tangent_t/normal)
- first-order canonical position만 제공 — second derivative는 curvature-growth/uncertainty diagnostic 전용이며 position에 반영하지 않음
- validity mask(normal/direction/sample) + local_surface_scale/continuation_extent
- 상태: `valid`, `degenerate`, `rejected`(Phase E의 `candidate`와 혼동 방지를 위해 별도 명칭)

### `OccludedRegionCandidate`

- stable candidate ID
- supporting boundary IDs
- endpoint correspondence와 closed candidate topology
- overlap/proximity/visibility/empty-space soft evidence
- hard-gate rejection reason
- conflict edges

### `OccludedChartResult`

- constrained NURBS surface
- observed/shared boundary constraints
- fit diagnostics
- validity diagnostics
- uncertainty vector
- 상태: `candidate`, `validated`, `rejected`

### `UncertainGaussianProposal`

- position, UV, occluded chart ID
- color/covariance/opacity prior
- confidence/uncertainty
- 아직 `TorchGaussianModel`에 append되지 않은 격리된 산출물

## 3. Phase A — NURBS와 Boundary Interface

상태: **구현·검증 완료.** 근거: docs/worklogs/75_boundary_conditioned_phase_ab.md.

### 작업

1. `TorchNURBSSurface`에 read-only knot access를 추가한다.
2. rational `S_uu`, `S_uv`, `S_vv`를 제공하는 second-derivative API를 추가한다.
3. Phase 2 contour를 ordered, oriented boundary segment로 변환한다.
4. generic trimmed patch에서 support interior 방향을 결정한다.
5. boundary-adjacent inner isocurve를 생성한다.
6. Boundary-First state가 boundary result를 transient local variable로 버리지 않고 stable record로 보존하게 한다.
7. export에는 patch ID, boundary ID, UV/world samples, orientation, provenance만 추가한다. Production consumer 동작은 바꾸지 않는다.

### 검증

- Plane, sine, annulus에서 analytic/finite-difference derivative agreement
- Reversed loop ordering에 대한 outward/inward invariance
- Inner isocurve가 support 내부에 있고 boundary와 교차하지 않음
- 동일 입력에서 boundary ID와 payload hash 결정성
- 기존 benchmark 결과와 main training default 불변

### 승인 게이트 A

새 데이터 계약과 derivative/boundary test 결과를 보고하고 멈춘다.

## 4. Phase B — Artificial Patch-Boundary Reconciliation

상태: **구현·검증 완료, Gate B 승인.** 근거: docs/worklogs/75_boundary_conditioned_phase_ab.md. Production membership 및 global merge에는 연결하지 않았다.

### 작업

1. Spatial neighborhood로 adjacent boundary 후보를 만들되 voxel ID만으로 병합하지 않는다.
2. 양방향 endpoint correspondence와 parameter reversal을 평가한다.
3. 기존 Step 5-A의 shared-variable assembly를 generic constraint map으로 분리한다.
4. 전체 control column이 아닌 partial boundary segment, 서로 다른 orientation을 지원한다.
5. 우선 같은 degree/resolution의 canonical fixture만 지원하고 degree elevation/knot refinement는 후속으로 둔다.
6. Pre-fit evidence와 post-fit C0/Jacobian validity를 분리한다.
7. 통과한 경계만 `reconciled_internal`로 표시한다. Patch membership의 global union은 수행하지 않는다.

### Hard reject

- correspondence가 finite하지 않음
- overlap이 없음
- fit domain을 구성할 수 없음
- post-fit fold/singularity
- scale-normalized extension 한도 초과

Facing/normal/tangent은 hard reject로 쓰지 않는다.

### Benchmark

- 같은 평면을 둘로 자른 artificial seam
- 같은 곡면을 여러 chart로 자른 seam
- annulus cyclic wedge seam
- orthogonal/oblique visible patch pair
- true crease
- close parallel sheets
- disconnected-close negative control
- density/seed/rotation sweep

### 승인 게이트 B

Artificial seam은 reconciliation되고 true crease/parallel/disconnected는 extension 이전 상태로 남아야 한다. Scene-specific threshold나 GT runtime branch가 없어야 한다. 결과 보고 후 멈춘다.

## 5. Phase C — Observation Evidence와 Free-Space Query

상태: **구현·검증 완료, Gate C 승인.** 근거: docs/worklogs/77a_observation_evidence_phase_c.md, docs/worklogs/78_observation_evidence_phase_c_gate_c_followup.md, docs/worklogs/79_observation_evidence_phase_c_gate_c_round2.md.

### 작업

1. Constructor에 `TorchScene` 전체를 결합하지 않고 read-only `ObservationEvidence`를 전달한다.
2. CUDA inverse depth와 fallback depth를 하나의 명시적 world/view-depth convention으로 변환한다.
3. coverage/유효 depth mask를 제공한다.
4. 후보 world sample마다 per-view 상태와 aggregate 상태를 분리해 계산한다(최종 확정 enum, worklog 79).

```text
per-view (카메라 하나, sample 하나):
  known_free_space
  on_observed_surface
  behind_first_observed_surface
  unobserved
  outside_valid_view

aggregate (여러 카메라 종합, per-view와 다른 enum):
  known_free_space
  on_observed_surface
  occluded_candidate
  unobserved
  outside_valid_view
  conflicting_evidence   -- free/behind/on_surface 중 2개 이상이 공존할 때. on_surface_in이
                              비어있지 않은 sample은 known_free_space/occluded_candidate로
                              집계되지 않는다는 불변식을 유지한다.
```

5. Empty voxel AABB query를 추가하되 결과는 `no observed support`로만 기록한다.
6. Evidence cache는 Gaussian topology version과 camera set에 의해 무효화한다(`evidence_cache_key`는 렌더링 후에만 계산 가능한 post-build result fingerprint).

### 검증

- Camera 앞의 sample은 known free space
- 관측 surface 뒤 sample은 잠재 occlusion
- 모든 camera 밖 sample은 unobserved
- CUDA/fallback depth convention parity
- Empty voxel 단독으로 occlusion candidate가 되지 않음

### 승인 게이트 C

Free-space false acceptance가 없는지 보고하고 멈춘다.

## 6. Phase D — Parametric Continuation Domain

상태: **구현·검증 완료, Gate D 승인.** 근거: `docs/worklogs/81_phase_d_continuation_domain_implementation.md`. 아래는 완료된 canonical 계약과 검증 요약이다.

Phase D는 최종 parametric surface나 NURBS chart를 만들지 않는다. 역할은 `Visible NURBS boundary → boundary-local world-space sampled continuation strip → geometric validity 및 uncertainty metadata`로 제한된다.

### 작업

1. Boundary world tangent를 world-arclength 기준 유한차분으로 계산한다(§4.1) — `s`는 cumulative world arclength로 단위를 고정하고, sample index/normalized parameter와 혼용하지 않는다. **Closed boundary는 `boundary_length`(closing segment 포함 전체 perimeter) 필드를 별도로 두고, periodic 차분의 wrap-around denominator에 사용한다** — `s_world` 자체는 closing segment를 포함하지 않는다(revision 3).
2. Outward 방향은 UV-space perpendicular가 아니라 **world-space에서 직접** 정의한다: `N = normalize(S_u x S_v)`, `C = normalize(N x T)`, `inner_world` 방향과의 내적으로 부호 선택(§4.2). UV parameterization의 skew/scale에 영향받지 않는다.
3. First-order sampled grid를 canonical baseline이자 유일한 position 표현으로 구현한다(§4.4). Second-order position variant는 제공하지 않는다.
4. Second derivative는 UV로 투영한 뒤(§4.3) **`second_order_displacement_at_extent`/`second_order_growth_ratio`라는 명칭의 diagnostic에만 사용**하고 canonical position에는 반영하지 않는다 — intrinsic mean/Gaussian/normal/geodesic curvature 추정치가 아님을 docstring에 명시한다(revision 3, 이전 `curvature_growth_ratio` 명칭 폐기).
5. `inner_probe_distance`(부호 선택용)와 `local_surface_scale`(`L_boundary`/`L_inner`/`L_control` 중 finite·positive 값의 median, 최소 2개 이상 필요 — §4.5.1의 canonical 공식으로 고정)과 `continuation_extent`(`= extent_multiplier * local_surface_scale`)를 분리한다 — 하나의 probe distance를 extent로 직접 재사용하지 않는다.
6. Jacobian/orientation/degeneracy를 mask(`normal_valid_mask`/`direction_valid_mask`/`sample_valid_mask`)로 표현한다 — NaN을 쓰지 않는다(§5).
7. **(완료된 prerequisite)** `torch_annulus_chart.py`의 Jacobian singular-value 계산과 orientation-consistency 계산을 **두 개의 별도** surface-agnostic 헬퍼(`compute_parametric_jacobian_metrics`, `compute_orientation_consistency`)로 분리 추출하는 순수 리팩터를 Phase D 구현 착수 전에 완료해야 한다(§9). 회귀 검증은 byte-identical이 아니라 상태 분류/report field/수치 tolerance 동일 여부로 확인한다.
8. 입력 오류와 build 실패와 사후 품질 문제를 3단계로 분리한다(§2.2-2.4, revision 3): `reconciled_internal` boundary/patch_id 불일치/shape 불일치/최소 sample 수 미달/인접 duplicate·zero-length segment/명시적 scale·multiplier가 non-finite 또는 `<=0` → **`ValueError`** 즉시 발생. `local_surface_scale` 자동 도출 실패(유효 후보 2개 미만) 등 grid 자체를 만들 수 없는 경우 → **`ContinuationDomainBuildError`**(신규 예외 타입, `ContinuationDomain` 객체를 반환하지 않음). Grid 구성 후의 부분적 품질 문제(Jacobian collapse 등) → `state=degenerate`/`rejected`인 `ContinuationDomain` 반환.
9. 상태 이름은 `valid`/`degenerate`/`rejected`를 쓴다 — Phase E의 `candidate`와 혼동되는 이름은 쓰지 않는다(§5).

### 검증

- UV axis swap, **UV scale/skew**, loop reversal에 대한 world-space continuation invariance(핵심: world-space outward 공식이 UV 왜곡에 영향받지 않음을 증명)
- Plane, curved, rotated, **boundary resampling density 변경**, orthogonal/oblique(policy-regression: normal/facing hard gate 없음 확인), annular/radial fixture
- Degenerate Jacobian/normal/tangent — NaN 없이 mask로 표현, 예외로 죽지 않음
- **Closed boundary closing segment**가 `boundary_length`에 정확히 반영됨(negative control: adjacent duplicate/closing-segment zero-length는 `ValueError`)
- **최소 boundary sample 수 계약**(open 3개/closed 4개 unique) 충족 여부
- `reconciled_internal`이 `ValueError`로 거부됨(negative control)
- **`local_surface_scale` 자동 도출 실패가 `ContinuationDomainBuildError`로 구분됨**(`ValueError`/`state=rejected`와 혼동되지 않음, negative control)
- `local_surface_scale`/`continuation_extent`가 단일 probe(inner probe distance/boundary density/control grid resolution 각각)에 비정상적으로 종속되지 않음
- 모든 fixture에서 `state`가 `occluded_candidate`나 `validated chart`로 승격되지 않음
- Second-order diagnostic이 `second_order_*` 명칭으로 보고되고 intrinsic curvature로 오인되지 않음
- (§7에 fixture 전체와 실패 조건 명시)

### Phase D 범위 밖으로 명시적으로 미룬 것(§9)

- Boundary pairing, overlap 판정, 공동 empty-region 지지 여부 → Phase E.
- 전체 self-intersection 검사, 독립적 visible-surface-penetration 검사, control-net 기반 진짜 constrained NURBS chart → Phase F.
- Phase C `ObservationEvidence`와의 실제 결합 → interface만 정의(§3, `ContinuationDomain.world.reshape(-1,3)`가 `classify_world_samples` 입력과 이미 일치), 호출 없음.
- Multi-scale strip → Phase E recall 부족 시 후속 검토.

### 승인 게이트 D

Continuation strip의 방향·크기·결정성을 보고하고 멈춘다. **보고 완료(worklog 81).** 신규 테스트 22개 전부 통과, 전체 suite 회귀 없음(pytest 260 passed/1 skipped, unittest 261 tests OK). Phase E 착수는 별도 승인 전까지 시작하지 않는다.

## 7. Phase E — High-Recall Bounded Candidate Builder

아래가 Phase E의 canonical 계약 및 완료된 구현 요약이다.

### 초기 canonical scope

`ContinuationDomain A + ContinuationDomain B → pairwise OccludedRegionCandidate`만 구현 대상. 3개 이상 domain의 multi-sided aggregation은 pairwise 결과 위의 후속 확장(범위 밖). one-sided domain 하나만으로는 candidate를 만들지 않는다.

### Data flow와 모듈 경계

```
ContinuationDomain registry
→ geometric broad-phase pairing        (torch_aabb_broad_phase.py, 신규 공용 sweep-and-prune)
→ sampled-strip correspondence          (narrow phase, mutual-nearest + s-continuity component)
→ pairwise bounded-region candidate     (torch_occluded_region_candidate.py, evidence import 없음)
→ candidate-region ObservationEvidence validation  (torch_candidate_evidence.py)
→ candidate/conflict provenance
```

- geometric builder(`build_geometric_region_candidates`)는 `torch_observation_evidence`를 import하지 않는다. evidence는 `validate_candidate_observation_evidence`에서만 적용한다.
- broad phase는 deprecated Stage 2(`torch_surface_candidate_graph.py`)를 직접 dependency로 갖지 않는다. 신규 surface-agnostic 공용 모듈을 만들되 Stage 2는 회귀 방지를 위해 재배선하지 않는다(상세 §3.1).

### `ContinuationDomain.state` 입력 정책

`valid`=pairing 입력, `degenerate`=입력 가능하되 provenance에 degeneracy 기록, `rejected`=입력에서 제외.

### Narrow phase / topology (필수 교정 반영)

correspondence edge는 full `(s_a,t_a)`/`(s_b,t_b)` index를 보존한다. 동일 `(s_a,s_b)`는 min normalized distance 하나로 축약, 동일 s는 mutual-nearest→normalized distance→index ordering으로 canonical 1:1 매칭한다. `s_b` step 부호가 일정한 maximal run이 monotonic component이며 branch는 별도 component(→ 별도 candidate), closed는 modulo continuity로 duplicate seam edge 없이 처리한다. support chain은 full `(s,t)` sequence로 저장한다. bridge cell은 인접 correspondence pair `[A_i, A_{i+1}, B_{i+1}, B_i]`로 구성하고 zero-area cell은 unsupported/rejected로 표시(self-intersection·complete orientation은 Phase F).

### 최소 structural hard gate (facing/normal/tangent/curvature는 hard gate 아님)

- 서로 다른 source boundary/domain (같은 domain_id self-pair, 같은 source boundary duplicate pair 금지)
- 최소 2개의 연속 correspondence pair
- finite nonzero connector separation
- nonzero support interval
- finite bridge geometry(명백한 zero-area cell 거부)
- (evidence 단계) interior만으로 판정: 평가 가능한 nondegenerate bridge section이 ≥1이고 그 **모든** section이 interior hard contradiction(interior 유효 sample 전부 known_free_space, behind/on_surface/unobserved/conflicting 전무)일 때만 hard reject. support endpoint는 이 계산에 포함하지 않는다. interior 평가 sample이 없는 section은 `insufficient_evidence`(reject 아님), partial free section은 보존.

### Coplanar close pair는 negative control이 아니다 (필수 교정)

coplanarity/small gap/normal similarity/facing은 hard reject가 아니다. Phase B에서 reconcile되지 않고 도달한 서로 다른 coplanar boundary는 topology가 성립하면 candidate로 유지한다. 진짜 structural negative는 zero connector separation / zero support interval / duplicate·same-source domain / 잘못 유입된 reconciled_internal이다.

### Soft evidence / 보존 원칙

facing·normal·tangent·curvature·support symmetry·boundary distance·continuation overlap·behind-observation·empty/unobserved는 전부 raw로 병렬 보존한다. hidden weighted score나 조기 scalarization을 쓰지 않는다. `conflicting_evidence`와 empty-voxel `no_observed_support`는 승인/거부 근거로 자동 사용하지 않고 보존·metadata로만 남긴다.

### Conflict edge

`build_candidate_conflicts`가 (1) 동일 source domain/boundary 공유 + bridge AABB overlap, (2) 서로 다른 pair가 유사 bridge 공간 점유, (3) 동일 영역 evidence summary 양립 불가 시 conflict edge를 raw reason과 함께 생성·보존한다. ranking/selection/optimization/pruning은 하지 않는다.

### `OccludedRegionCandidate` 상태

`state ∈ {candidate, unsupported, rejected}` + `reason`. `ContinuationDomain.state`의 `{valid, degenerate, rejected}`와 의미를 혼용/승격하지 않는다.

### 승인 게이트 E

input domain count, broad-phase pair count, narrow-phase surviving pair count, candidate/unsupported/rejected count, conflict edge count, maximum candidate degree를 scene/seed sweep으로 보고한다. arbitrary candidate-count cap은 두지 않고 benchmark 후 필요성을 검토한다.

## 8.1 Phase F.1 — Sampled Chart Hardening

**보완 구현 진행 중, Gate F.1 승인 대기.** Phase F의 open pairwise constrained occluded NURBS chart 뒤에 sampled self-intersection, sampled visible-surface penetration, unresolved chart conflict, Phase G eligibility를 별도 safety result로 계산한다. 이는 complete analytic/CAD validity 보장이 아닌 proposal 전 deterministic sampled safety gate다. chart control grid/state는 변경하지 않으며, global ranking/selection·conflict resolution·Gaussian proposal·production integration은 미착수다. 상세 계약은 `OSN_GS_Phase_F1_Chart_Hardening_Design.md`, 구현 기록은 `docs/worklogs/84_phase_f1_chart_hardening_implementation.md`.

## 8. Phase F — Minimal Constrained Occluded NURBS Bridge

**구현·검증 완료, Gate F 승인(worklog 83 최종 구현 기록).** 상세 계약은 `OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md`. 실제 구현 요약: `torch_occluded_chart.py`(`fit_occluded_chart`, `OccludedChartResult`, `OccludedChartFitConfig`) + `torch_coons_patch.py` + `torch_constrained_chart_lsq.py`. 초기 scope는 **open pairwise quadrilateral ribbon만**(cyclic candidate는 `state=unsupported, reason="cyclic_topology_deferred"`). chart는 **두 continuation-supported matched chain 사이**를 잇고 visible `t=0`까지 되늘리지 않는다(Phase D/E bounded candidate 범위 유지). Solver는 **Coons transfinite seed + new single-chart constrained LSQ**(`_lsq_normal_system`/`_second_difference_penalty` 재사용, `fit_coupled_patch_graph_lsq` 미개조). weight: `support_weight >> connector_weight`, connector는 seed+저weight soft regularizer. 현재 support C0는 exact equality constraint가 아니라 high-weight constrained fitting + post-fit C0 validation이다. C0 post-fit gate, G1/G2 diagnostic만(occluded chart가 어느 visible normal도 강제로 따르지 않음). evidence는 fit 후 validation/metadata/rejection provenance로만(solver weight 아님). self-intersection/penetration은 `checked=False` + cheap proxy. state ∈ `{fitted, validated, unsupported, rejected}`. Phase F 테스트 21개 통과, 전체 suite `311 passed, 1 skipped`. Phase G Gaussian proposal, global selection, multi-sided, production integration 미착수.

### (원래 설계 노트 — 상세 Phase F 설계 문서로 대체됨)

초기 canonical topology는 두 visible boundary와 endpoint correspondence로 생성한 두 finite connector가 닫는 quadrilateral domain이다. 두 boundary만으로 surface가 유일하게 정해진다고 가정하지 않는다.

### 작업

1. Visible boundary geometry는 C0 exact/near-exact shared constraint로 둔다.
2. Connector는 candidate-domain bound이며 observed boundary로 취급하지 않는다.
3. Boundary와 interior control geometry를 하나의 constrained system에서 푼다.
4. Arbitrary boundary angle을 허용한다.
5. G1/G2는 hard constraint가 아니라 mismatch diagnostic/regularizer 후보로 둔다.
6. Post-hoc boundary overwrite를 사용하지 않는다.

### 필수 diagnostics

- C0 mean/max
- tangent/normal mismatch
- Jacobian flip
- minimum singular value
- condition p95/p99/max
- self-intersection
- visible-surface penetration
- extension length/area
- curvature magnitude/change
- free-space/visibility consistency
- candidate conflict

### 승인 게이트 F

Validated chart와 rejected chart를 동일 fixture에서 재현하고, rejection reason과 deterministic payload를 보고한다.

## 9. Phase G — Uncertainty와 Gaussian Proposal

### 작업

- Boundary distance
- Supporting boundary count
- Continuation overlap
- Boundary prediction disagreement
- Required curvature
- Extension size
- Chart condition
- Visibility support
- Candidate conflict

위 값을 raw vector로 보존한다. 초기에는 calibration되지 않은 단일 confidence로 압축하지 않는다.

Validated chart에서만 `UncertainGaussianProposal`을 생성한다. Color/covariance/opacity prior는 기존 visible neighborhood와 surface tangent frame에서 얻되 production model에는 append하지 않는다.

### 승인 게이트 G

Proposal 위치가 surface에 있고 metadata/source binding이 완전하며, certain Gaussian과 production checkpoint가 변하지 않음을 보고한다.

## 10. Phase H — Production Adoption Gate

다음 조건을 모두 만족하고 사용자가 별도로 승인할 때만 production integration을 계획한다.

- 기존 Boundary-First/legacy/main-training regression 통과
- Positive occlusion fixture의 geometry/visibility gate 통과
- artificial seam, crease, parallel, disconnected, one-sided negative control 통과
- scene/seed/rotation/density/camera sweep에서 결정성 확인
- Runtime에서 GT topology/component count/oracle 미사용
- Candidate 수와 solver memory/time의 상한 확인
- Certain Gaussian으로 reverse gradient가 흐르지 않음
- Uncertain-to-certain promotion 없음
- 실패 시 silent bridge가 아니라 explicit rejected/unsupported 상태
- 사용자에게 chart, boundary frame, observed/occluded separation, uncertainty, 정량 결과 보고

Production default 변경, uncertain Gaussian append, old plan retirement는 이 승인 뒤의 별도 작업이다.

## 11. 승인 상태와 다음 요청 범위

**Canonical 현재 상태(2026-07-23 기준)**:

| 범위 | 상태 | 근거 |
|---|---|---|
| Phase A | 완료, 승인 | `docs/worklogs/75_boundary_conditioned_phase_ab.md` |
| Phase B | 완료, Gate B 승인 | `docs/worklogs/75_boundary_conditioned_phase_ab.md` |
| Phase C | 완료, Gate C 최종 승인 | `docs/worklogs/77a_observation_evidence_phase_c.md` → `78_observation_evidence_phase_c_gate_c_followup.md` → `79_observation_evidence_phase_c_gate_c_round2.md`(최종 승인) |
| Phase D | 완료, Gate D 승인 | `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`(design revision 3), `docs/worklogs/80_phase_d_continuation_domain_design.md`(설계 이력), `docs/worklogs/81_phase_d_continuation_domain_implementation.md`(구현·Gate D 최종 기록) |
| Phase E | 구현·검증 완료, Gate E 승인 | `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`(상세 설계), `docs/worklogs/82_phase_e_bounded_candidate_implementation.md`(구현·Gate E 최종 기록) |
| Phase F | 구현·검증 완료, Gate F 승인 | `docs/Urgent_Work/OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md`(상세 설계), `docs/worklogs/83_phase_f_constrained_occluded_chart_implementation.md`(Phase F 최종 구현 기록) |
| Production pipeline/trainer integration | 미승인 | — |

Gate C의 non-blocking note 2건(후속 참고용, blocking 아님): (1) 완전히 동일한 duplicate camera가 리스트 순서만 바뀌면 fingerprint가 같을 수 있음 — 문제로 보지 않음. (2) synthetic per-view payload 기반 순수 aggregation truth-table 테스트를 향후 추가 권장 — Phase D 착수의 선행조건 아님.

사용자가 Phase D 구현 착수를 지시했다. `osn_gs/surface/torch_continuation_domain.py`(`ContinuationDomain`, `ContinuationDomainBuildError`, `build_continuation_domain`, `interpolate_boundary_arclength`)와 prerequisite 리팩터(`osn_gs/surface/torch_parametric_diagnostics.py`, `torch_annulus_chart.py`가 이를 호출하도록 수정)를 구현했다. `tests/test_continuation_domain.py` 22개 전부 통과, `tests/test_annulus_chart.py` 48개 회귀 없음, 전체 pytest `260 passed, 1 skipped, 8 subtests passed`, 전체 unittest `261 tests, OK`. 구현 중 설계 §2.3("모든 샘플이 degenerate하면 `ContinuationDomainBuildError`")을 최초 구현이 빠뜨렸던 간극을 테스트 작성 중 발견해 수정했다(설계 문서는 변경 없음, 구현만 설계를 따르도록 수정). Production(`torch_pipeline.py`/`torch_trainer.py`) 미변경, Phase C evidence 미호출. 사용자가 Gate D를 승인했다. **Phase C evidence 실제 결합, Phase E candidate 생성, Phase F NURBS fitting, production integration은 모두 별도 승인 전까지 시작하지 않는다.**

Phase E·F 구현·검증을 완료했고 Gate F를 승인했다(Phase F: `torch_occluded_chart.py`, `torch_coons_patch.py`, `torch_constrained_chart_lsq.py`, tests, `worklogs/83`). 현재 달성 범위는 **open pairwise quadrilateral occluded NURBS hypothesis construction**으로 한정한다. Phase G 전에 Occluded Chart Hardening and Selection Gap Audit을 수행하며, Phase G Gaussian proposal, global candidate selection/ranking, multi-sided topology, production integration은 별도 승인 전까지 시작하지 않는다.