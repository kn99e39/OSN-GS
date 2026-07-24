# OSN-GS Phase E — High-Recall Bounded Candidate Builder: Design

상태: **설계 확정, 구현 진행 승인됨(2026-07-24).** 상위 계획은 `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §7. 이 문서는 그 §7의 상세 계약이다. 사용자가 전체 방향과 이 문서의 필수 교정 6건(coplanar negative control 폐기, `(s,t)` 기반 correspondence canonicalization, bridge topology 계약, evidence interior/endpoint 분리, geometry source of truth, conflict edge 규칙)을 승인하고 구현 착수를 지시했다. 구현 결과는 `docs/worklogs/82_phase_e_bounded_candidate_implementation.md`(Gate E) 하나로 보고한다.

## 0. 이 문서가 근거하는 audit

실존하는 코드 계약만 근거로 삼는다.

- `ContinuationDomain`(`osn_gs/surface/torch_continuation_domain.py`): `world`(s×t×3), `sample_valid_mask`(s×t bool), `direction_valid_mask`(s), `outward_tangent_world`(s×3), `normal`(s×t×3), `tangent_s`/`tangent_t`(s×t×3), `s_world`, `boundary_length`, `aabb_min/max`(3), `local_surface_scale`(float), `continuation_extent`(float), `source_patch_id`(int), `source_boundary_id`(str), **`closed`(bool, 실존 확인됨 — Phase D가 이미 저장)**, `state ∈ {valid, degenerate, rejected}`. 원본 `PatchBoundarySegment`/`TorchNURBSSurface` 객체 참조는 없고 `source_*` id만 있다. closed domain의 `world`는 Phase D의 `_strip_closed_duplicate`로 이미 중복 종료점이 제거된 unique s 샘플이다(s 인접은 modulo s_count).
- `PatchBoundarySegment`(`torch_patch_boundary.py`): `boundary_id`, `patch_id`, `world`(closed면 `world[-1]==world[0]` 중복 종료점 저장), `inner_world`, `closed`, `state`, `confidence`.
- `ObservationEvidence`/`classify_world_samples`(`torch_observation_evidence.py`): `classify_world_samples(evidence, world_points(N,3)) -> list[SampleEvidence]`, `status ∈ {known_free_space, on_observed_surface, occluded_candidate, unobserved, outside_valid_view, conflicting_evidence}`.
- `query_empty_voxel_support(...) -> EmptyVoxelSupportResult`: `support`는 `"no_observed_support"` 하나뿐(결여 신호).
- Broad-phase 선례 `torch_surface_candidate_graph.py`(**Stage 2, diagnostics-only, deprecated, 결과 회귀-잠금**): sweep-and-prune 패턴만 참고하고 import/재배선하지 않는다.

## 1. 승인된 Phase E 책임

소유: `ContinuationDomain registry → geometric broad-phase pairing → sampled-strip correspondence → pairwise bounded-region candidate → candidate-region ObservationEvidence validation → candidate/conflict provenance`.

미소유: final NURBS chart fitting, global candidate selection/optimization, Gaussian proposal, production integration, control-net extrapolation, multi-scale continuation, multi-sided joint topology solver, self-intersection/visible-surface-penetration의 완전한 검사.

### 초기 canonical scope — pairwise two-sided만

`ContinuationDomain A + ContinuationDomain B → pairwise OccludedRegionCandidate`. 3개 이상 domain의 multi-sided aggregation은 pairwise 결과 위의 후속 확장(범위 밖). one-sided domain 하나만으로는 candidate를 만들지 않는다.

## 2. 입력 계약과 모듈 경계

### 2.1 Registry 입력

```python
domains: Sequence[ContinuationDomain]
boundaries_by_id: Mapping[str, PatchBoundarySegment]
surfaces_by_patch_id: Mapping[int, TorchNURBSSurface] | None
observation_evidence: ObservationEvidence | None
empty_voxel_query: Callable[[Any, Any], EmptyVoxelSupportResult] | None
```

### 2.2 Geometry source of truth (필수 교정 5)

Candidate geometry의 canonical source는 **`ContinuationDomain`뿐이다**: `world`, `sample_valid_mask`, `closed`, source IDs로 support chain과 bridge geometry를 구성한다. `boundaries_by_id`는 **provenance / confidence / source state 확인 / ID consistency validation 용도로만** 쓴다. `PatchBoundarySegment.world`로 `ContinuationDomain` geometry를 덮어쓰거나 재구성하지 않는다 — closed boundary의 duplicate-endpoint 표현 차이가 candidate geometry에 유입되지 않게 한다(Phase D가 이미 duplicate를 벗겨 저장했으므로 `ContinuationDomain.world`만 쓰면 안전하다).

### 2.3 모듈 분리 (evidence import 격리)

```
osn_gs/surface/torch_aabb_broad_phase.py            (surface-agnostic sweep-and-prune)
osn_gs/surface/torch_occluded_region_candidate.py   (geometric only — OccludedRegionCandidate,
        build_geometric_region_candidates, ConflictEdge, build_candidate_conflicts;
        torch_observation_evidence 를 import 하지 않는다)
osn_gs/surface/torch_candidate_evidence.py          (validate_candidate_observation_evidence;
        torch_observation_evidence 를 import 한다)
```

### 2.4 `ContinuationDomain.state` 입력 정책 (필수 교정 6)

```
valid      → pairing에 사용
degenerate → valid sample만 사용하고 degeneracy provenance 보존
rejected   → pairing 입력에서 제외
```

같은 `domain_id` self-pair, 같은 `source_boundary_id`를 공유하는 duplicate pair는 candidate를 만들지 않는다. 입력 domain은 `domain_id`로 먼저 dedup한다.

## 3. Broad phase (필수 교정 7)

신규 surface-agnostic `torch_aabb_broad_phase.py`. Deprecated Stage 2를 import/재배선하지 않는다(회귀 방지 — 상세는 impl plan §7). 제공:

- deterministic sweep-and-prune(entry 정렬 tie-break에 label 포함).
- canonical ordered pair `(label_a < label_b)`.
- raw AABB distance, scale-normalized AABB distance(`aabb_distance / max(scale_a, scale_b, eps)`).
- threshold/expand factor payload 보존.

입력: 각 domain의 `aabb_min/max`, `local_surface_scale`(정규화·확장용), `expand_factor`. 확장 AABB(`aabb ± expand_factor * scale`) 겹침으로 검사 대상을 추리고, raw `aabb_distance <= expand_factor * max(scale_a, scale_b) + tol`인 pair만 남긴다. self-pair 및 같은 `source_boundary_id` pair는 broad phase 결과에서 제외한다. Stage 2 기존 테스트/결과는 변경하지 않는다.

## 4. Narrow phase — `(s,t)` index 기반 correspondence (필수 교정 2)

`ContinuationDomain.world` 전체 valid sample(`sample_valid_mask`)을 대응에 사용하되, candidate topology 구성 **전에** edge를 canonical하게 축약한다. 각 edge는 full index를 보존한다.

```
sample_index_a = (s_a, t_a)
sample_index_b = (s_b, t_b)
world_distance
scale_normalized_distance = world_distance / max(scale_A, scale_B, eps)
mutual_nearest            bool
outward_dot / normal_dot / tangent_dot   (soft, hard gate 아님)
position_kind             endpoint | interior | closed_cyclic
```

### 4.1 Canonical edge 선택 순서

1. A의 각 valid sample에 대한 최근접 B valid sample, B의 각 valid sample에 대한 최근접 A valid sample을 계산(`torch.cdist`). 양방향이 서로를 가리키면 `mutual_nearest=True`.
2. `scale_normalized_distance <= correspondence_threshold`인 edge만 남긴다(raw와 threshold-used를 payload에 보존).
3. **동일 `(s_a, s_b)` 조합에 여러 edge가 있으면 scale-normalized distance 최소 edge 하나만 유지.**
4. **동일 `s_a` 또는 `s_b`가 여러 상대에 연결되면** canonical edge를 `mutual_nearest` 우선 → normalized distance → deterministic `(s_a, s_b)` index ordering으로 선택한다(greedy 1:1 매칭: `(not mutual, normalized_distance, s_a, s_b)`로 정렬해 s_a·s_b가 모두 미사용일 때만 채택). 결과는 각 s가 최대 한 번 매칭되는 1:1 대응.

### 4.2 Monotonic component 분리

선택된 correspondence를 `s_a` 오름차순으로 정렬하고 양 domain s-direction에서 monotonic한지 검사한다.

- 인접 edge의 `s_b` step 부호가 일정한 maximal run = **하나의 monotonic component**(부호는 +/− 모두 허용 — 두 boundary의 s 방향이 반대일 수 있음).
- 부호가 갈라지면(branch) 별도 component로 분리한다.
- component가 finite ordered support chain(연속 correspondence pair ≥ 2)을 만들 수 없으면 그 pair는 `unsupported`.
- 단일 pair가 여러 분리된 monotonic component를 만들면 각 component는 **별도 candidate**로 유지한다.
- **Closed domain**: s_a를 원형으로 보고 modulo continuity로 run을 잇되(seam을 한 번 넘는 run 허용), 각 s는 한 번만 등장하므로 **duplicate seam edge를 만들지 않는다**. 전체가 하나의 modulo-monotonic run이면 cyclic component(연결자 end 없음).

## 5. Pairwise bounded-region topology (필수 교정 3)

AABB overlap이나 strip intersection 자체를 bounded-region 조건으로 쓰지 않는다.

### 5.1 Support chain / bridge cell

```
support_chain_a: ordered [ (s_a, t_a), world_a, ... ]      # full (s,t) sequence, ContinuationDomain.world에서
support_chain_b: ordered [ (s_b, t_b), world_b, ... ]
connector_start: [ world_a[0], world_b[0] ]
connector_end:   [ world_a[-1], world_b[-1] ]   (cyclic이면 None)
bridge cell k:   [ A_i, A_{i+1}, B_{i+1}, B_i ]  (인접 correspondence pair)
```

closed cyclic component는 마지막→처음 wrap bridge cell도 만든다.

### 5.2 Cell별 최소 검사

각 bridge cell에 대해 검사: finite vertices, nonzero edge lengths, nonzero area proxy(예: 두 대각 삼각형 cross product norm 합), connector start/end finite, support interval nonzero(support_chain_a·b의 world arclength > eps). Self-intersection과 complete orientation consistency는 **Phase F로 미룬다**. 다만 **명백히 zero-area인 bridge cell**은 `unsupported` 또는 `rejected`로 표시한다.

### 5.3 관계 표현력

orthogonal / oblique / parallel 관계가 모두 표현 가능해야 한다. 이는 hard gate가 아니라 soft evidence(`outward_dot`/`normal_dot`/`tangent_dot`)와 bridge geometry로 표현되며 어떤 각도 관계도 gate로 거부하지 않는다.

## 6. `OccludedRegionCandidate` 최소 계약

```python
candidate_id
supporting_domain_ids        # 정확히 2개
supporting_boundary_ids
supporting_patch_ids
support_chain_a              # SupportChain(st_indices, world)
support_chain_b
correspondence_edges         # list[CorrespondenceEdge]
connector_start              # (2,3)
connector_end                # (2,3) | None
bridge_cells                 # (K,4,3)
aabb_min / aabb_max
raw_distance_statistics / normalized_distance_statistics
outward_soft_evidence / normal_soft_evidence / tangent_soft_evidence
free_space_contradiction / behind_observation_support / on_surface_evidence
unobserved_evidence / conflicting_evidence / empty_voxel_support   # evidence 단계에서 채움
state ∈ {candidate, unsupported, rejected}
reason
provenance
```

house 관례: plain dataclass + `payload()`, string state를 `__post_init__`에서 `set` 검증. geometric 단계는 evidence 필드 6개를 빈 dict로 두고 validation 단계에서만 채운다.

## 7. Proximity scale

```
distance_normalized = world_distance / max(local_surface_scale_A, local_surface_scale_B, eps)
```

threshold(`correspondence_threshold`)와 `broad_phase_expand_factor`는 configurable parameter로 두고 raw value와 threshold-used를 payload에 모두 보존한다. 이번 설계에서 최종 scientific constant로 확정하지 않는다(isolated fixture sweep으로 검증할 parameter). hidden weighted score / 조기 scalarization 금지 — 모든 통계는 raw 병렬 보존.

## 8. ObservationEvidence 적용 — interior와 endpoint 분리 (필수 교정 4)

geometric candidate 생성 후 **validation 단계에서만** Phase C evidence를 적용한다. `validate_candidate_observation_evidence`만 `torch_observation_evidence`를 import한다.

각 bridge cross-section(= 하나의 correspondence pair, A_k↔B_k)을 다음처럼 샘플링한다.

```
support endpoint A
one or more interior bridge samples   (A_k↔B_k 선분 내부 linspace, 양 끝 제외)
support endpoint B
```

Evidence summary를 **별도 필드로** 나눈다: `support_endpoint_evidence`, `bridge_interior_evidence`. **support endpoint는 known-free-space hard reject 계산에 포함하지 않는다.** support endpoint의 `on_observed_surface`는 정상적인 source-boundary evidence이며 bridge interior contradiction을 상쇄하는 positive occlusion evidence로 쓰지 않는다.

## 9. Known-free-space 정책 (필수 교정 4)

비율 threshold를 hard gate로 쓰지 않는다.

**Section hard contradiction** (interior만 평가):
```
- 최소 한 개의 평가 가능한 interior sample 존재 (status != outside_valid_view)
- 모든 유효 interior sample이 known_free_space
- interior에 behind / on_surface / unobserved / conflicting evidence가 하나도 없음
```
평가 가능한 interior sample이 없거나 전부 outside_valid_view인 section은 hard contradiction이 아니라 **`insufficient_evidence`**로 기록한다.

**Candidate hard reject** → `state=rejected`:
```
- 평가 가능한 nondegenerate bridge section이 최소 하나 존재
- 그 모든 section이 hard contradiction
```
partial known-free-space section은 자동 reject하지 않고 raw 통계를 보존한다. 다음 raw 통계를 모두 보존: known-free sample count, known-free section count, behind support count, on-surface count, unobserved count, conflicting count, insufficient-section count, per-camera provenance.

## 10. Empty voxel / conflicting evidence 정책 (필수 교정 9)

- `no_observed_support`: metadata only, positive occlusion evidence 아님, candidate 승격/거부에 사용하지 않음.
- `conflicting_evidence`: raw provenance로 보존, 승인/거부에 자동 사용하지 않음.

## 11. Candidate dedup과 conflict edge (필수 교정 10)

deterministic candidate_id = canonical domain-pair key(정렬된 `domain_id` 쌍) + correspondence-component identity(component 멤버 `(s_a,s_b)` 쌍 정렬 튜플 해시). 같은 pair의 서로 다른 component는 별도 candidate.

**Conflict edge 최소 생성 규칙**(보존만, 해결 안 함): `build_candidate_conflicts(candidates)`가 다음에 conflict edge를 생성하고 raw reason을 기록한다.
1. 동일 source domain 또는 boundary를 공유하는 두 candidate의 bridge 영역이 공간적으로 중첩(AABB overlap).
2. 서로 다른 supporting pair가 유사한 bridge 공간을 점유(AABB overlap + bridge centroid 근접).
3. Candidate evidence summary가 동일 영역에서 명백히 양립 불가(예: 한쪽 full free-space contradiction, 다른 쪽 behind support가 같은 겹치는 영역).

conflict 생성 함수는 candidate의 evidence dict를 읽되 `torch_observation_evidence`를 import하지 않는다(dict 필드만 참조). Ranking/selection/optimization/pruning은 하지 않는다.

Gate E 규모 지표: input domain count, broad-phase pair count, narrow-phase surviving pair count, candidate count, unsupported count, rejected count, conflict edge count, maximum candidate degree. arbitrary candidate-count cap 없음.

## 12. Fixture 및 테스트 계획 (필수 교정 1: Fixture 7 교체)

**Coplanar close pair를 negative control로 쓰지 않는다.** coplanarity/small gap/normal similarity/facing은 hard reject가 아니다 — Phase E는 "artificially separated visible surface"와 "narrow occluded region between coplanar boundaries"를 geometric pairwise rule만으로 구분할 수 없다. Phase B에서 reconcile되지 않고 Phase E까지 도달한 서로 다른 coplanar boundary는 proximity·correspondence topology가 성립하면 candidate로 유지한다. 진짜 structural negative는 zero/near-zero connector separation, zero support interval, duplicate/same-source domain, 잘못 유입된 `reconciled_internal` provenance다.

threshold 경계 fixture는 구현 시 **numeric probe를 먼저 수행**해 precondition 성립을 확인한다(Phase D 경계 fixture 실수 전례 반영). 대부분 `tests/test_continuation_domain.py`의 헬퍼로 두 `ContinuationDomain`을 배치해 구성한다.

반드시 검증(26항목):

1. Planar two-sided gap → candidate
2. Coplanar narrow pair가 topology 성립 시 임의로 reject되지 않음 → candidate
3. Parallel bounded ribbon → candidate
4. Orthogonal corner → candidate(facing gate 없음)
5. Oblique corner → candidate
6. Curved two-sided gap → candidate
7. Closed annular/radial candidate → candidate(cyclic band 비퇴화)
8. Zero connector 또는 zero support interval structural negative → unsupported/rejected
9. Disconnected-close negative(valid sample 대응 없음) → candidate 없음
10. One-sided negative(domain 하나) → candidate 없음
11. AABB overlap/no narrow correspondence → candidate 없음
12. 동일 `(s_a,s_b)` edge canonicalization(min normalized distance 유지)
13. Non-monotonic correspondence component 분리
14. Multiple components remain distinct(별도 candidate_id)
15. Domain order reversal determinism((A,B)와 (B,A) 동일 candidate_id/payload)
16. Degenerate domain provenance 보존
17. Full bridge-interior known-free contradiction → rejected
18. Partial free contradiction preserved → candidate + 통계 보존
19. Support endpoint on-surface가 interior reject를 막지 않음
20. No interior evidence는 insufficient이지 reject 아님
21. Conflicting evidence preserved
22. Empty voxel non-promotion(state 불변, metadata만)
23. Duplicate candidate deduplication
24. Conflict edge generation
25. Deprecated Stage 2 regression 없음
26. 전체 suite 회귀 없음

## 13. Phase F로 미룰 것 (명시적 제외)

actual constrained NURBS chart fitting, candidate global ranking/selection, multi-sided joint topology, full polygon/voxel solid reconstruction, self-intersection, visible-surface penetration, multi-scale continuation, scalar confidence calibration, candidate conflict resolution, Gaussian proposal, production integration.

## 14. 구현 순서

1. `torch_aabb_broad_phase.py` + 단위 테스트. Stage 2 미변경 회귀 확인.
2. `torch_occluded_region_candidate.py`: `OccludedRegionCandidate`/`SupportChain`/`CorrespondenceEdge`/`ConflictEdge`, `build_geometric_region_candidates`(broad→narrow→canonicalize→component→topology), `build_candidate_conflicts`. evidence import 없음.
3. `torch_candidate_evidence.py`: `validate_candidate_observation_evidence`(§8–10, interior/endpoint 분리).
4. `tests/test_occluded_region_candidate.py`, `tests/test_candidate_evidence.py`(26항목 분배).
5. `osn_gs/surface/__init__.py` export(entry point + result/candidate/conflict 타입만).
6. 전체 suite 회귀 확인 → worklog 82(Gate E) → 멈춤.

## 15. 승인 게이트 E 보고 형식 (worklog 82에서 채움)

fixture별 결과 표 + 규모 지표(§11) + threshold/config 값과 근거 + 전체 suite 결과 + Phase F/production 미착수 확인.
