# Worklog 82: Phase E — High-Recall Bounded Candidate Builder 구현 및 Gate E 보고

날짜: 2026-07-24

상태: **Phase E 구현·검증 완료. 승인 게이트 E 보고. Phase F NURBS bridge, global candidate selection/ranking, multi-sided aggregation, Gaussian proposal, production integration 모두 미착수.**

## 1. 배경

사용자가 Phase E 전체 방향과 상세 설계(`docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`)를 승인하고, 필수 교정 6건을 반영한 뒤 재승인 요청 없이 구현까지 진행하도록 지시했다. 이 문서는 그 구현 결과와 마스터 플랜(`OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §7)의 승인 게이트 E를 보고한다. 설계 단계 worklog는 지시대로 만들지 않았고, 기존 설계 문서 두 개(상세 설계, impl plan §7)에 교정을 반영했다.

## 2. 변경 파일

**신규 구현:**
- `osn_gs/surface/torch_aabb_broad_phase.py` — surface-agnostic sweep-and-prune broad phase.
- `osn_gs/surface/torch_occluded_region_candidate.py` — geometric candidate builder(evidence import 없음).
- `osn_gs/surface/torch_candidate_evidence.py` — ObservationEvidence validation.
- `tests/test_occluded_region_candidate.py`(22개), `tests/test_candidate_evidence.py`(8개).

**수정:**
- `osn_gs/surface/__init__.py` — entry point + result/candidate/conflict 타입 export.
- `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` — 필수 교정 6건 반영.
- `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` — §7 구체화, 상태선/§11 표/다음 승인 요청 갱신.

**미변경(회귀 방지):** deprecated Stage 2 `torch_surface_candidate_graph.py`는 import·재배선하지 않았다. Production(`torch_pipeline.py`/`torch_trainer.py`) 미변경.

## 3. 최종 module/data contract

```
ContinuationDomain registry
→ torch_aabb_broad_phase.sweep_and_prune_pairs   (surface-agnostic, canonical ordered pair + raw/normalized AABB distance)
→ torch_occluded_region_candidate.build_geometric_region_candidates
      (broad → (s,t) correspondence → canonicalize → monotonic component → bridge topology; evidence import 없음)
→ torch_candidate_evidence.validate_candidate_observation_evidence
      (bridge cross-section 샘플링 → classify_world_samples → interior-only hard reject)
→ torch_occluded_region_candidate.build_candidate_conflicts   (conflict edge 생성·보존, 해결 없음)
```

- `OccludedRegionCandidate.state ∈ {candidate, unsupported, rejected}` — `ContinuationDomain.state`의 `{valid, degenerate, rejected}`와 혼용/승격하지 않는다(provenance에 domain state 별도 기록).
- geometric builder는 `torch_observation_evidence`를 import하지 않음(grep 확인: docstring 언급 2건뿐, import문 0). evidence는 `torch_candidate_evidence`만 import.

## 4. Broad phase (설계 §3, 필수 교정 7)

`sweep_and_prune_pairs(labels, aabb_min, aabb_max, scales, *, expand_factor, tol, excluded_pairs)`:
- deterministic sweep-and-prune(entry 정렬 tie-break에 label 포함), canonical ordered pair(`label_a < label_b`), sort by `(label_a, label_b)`.
- 확장 AABB(`aabb ± expand_factor*scale`) 겹침으로 broad-phase, raw `aabb_distance <= expand_factor*max(scale_a,scale_b)+tol`로 정밀 필터.
- raw AABB distance, scale-normalized distance, expand_factor, threshold를 `BroadPhasePair.payload()`에 보존.
- same-source-boundary pair는 `excluded_pairs`로 사전 제외. Stage 2를 import/재배선하지 않아 Stage 2 결과 회귀 위험 0(테스트 11/11 불변).

## 5. Narrow phase — correspondence canonicalization과 component topology (설계 §4, 필수 교정 2)

- valid sample(`sample_valid_mask`)만 사용, `torch.cdist` 양방향 최근접으로 `mutual_nearest` 판정.
- `scale_normalized_distance <= correspondence_threshold` 필터.
- **동일 `(s_a, s_b)`는 min normalized distance edge 하나로 축약.**
- **동일 s는 greedy 1:1 매칭**(`(not mutual, normalized_distance, s_a, s_b)` 정렬 후 s_a·s_b 미사용 시 채택).
- `s_a` 오름차순 정렬 후 `s_b` step 부호 일정 구간 = monotonic component, 부호 반전 시 분리. Closed는 modulo 부호(`_signed_step`)로 full-ring 여부 검사, seam 한 번 넘는 run 허용, duplicate seam edge 없음(각 s 1회 등장).
- 단일 pair의 여러 component는 별도 candidate(deterministic candidate_id = 정렬 domain-pair key + component `(s_a,s_b)` 해시).

## 6. Candidate geometry (설계 §5, 필수 교정 3, 5)

- geometry canonical source는 `ContinuationDomain.world`뿐. `boundaries_by_id`는 provenance/state/ID 검증에만 사용, `PatchBoundarySegment.world`로 덮어쓰지 않음(closed duplicate-endpoint 차이 유입 없음 — Phase D가 이미 strip한 unique s 샘플 사용).
- `support_chain_a/b` = full `(s,t)` sequence + world. bridge cell = 인접 correspondence pair `[A_i, A_{i+1}, B_{i+1}, B_i]`, cyclic이면 wrap cell 추가.
- structural hard gate: 서로 다른 source, correspondence pair ≥2, finite nonzero connector separation, nonzero support interval, finite bridge geometry, 명백한 zero-area cell 거부. self-intersection/complete orientation은 Phase F.
- state: `candidate`(ribbon 구성), `unsupported`(pair<2 또는 zero support interval), `rejected`(non-finite/zero-connector/zero-area).

## 7. Evidence validation과 interior-only hard reject (설계 §8–9, 필수 교정 4)

- 각 correspondence pair를 cross-section으로: `[endpoint A, interior samples(exclusive linspace), endpoint B]`.
- `support_endpoint_evidence`와 `bridge_interior_evidence` 분리. **support endpoint는 hard reject 계산에서 제외.**
- section hard contradiction: 평가 가능한 interior sample ≥1 && 모두 known_free_space && behind/on/unobserved/conflicting 전무. 평가 가능 interior 없음 → `insufficient_evidence`.
- candidate hard reject: 평가 가능 nondegenerate section ≥1 && 그 전부가 hard contradiction → `state=rejected`("full_bridge_interior_known_free_space"). partial free는 보존.
- support endpoint의 `on_observed_surface`는 `on_surface_evidence.endpoint_on_surface_count`로만 기록, interior contradiction을 상쇄하지 않음(테스트 19로 잠금).
- raw 통계 전부 보존: known-free sample/section count, insufficient section count, behind/on-surface/unobserved/conflicting count, per-camera provenance.

## 8. Empty voxel / conflict 정책 (설계 §10–11, 필수 교정 9–10)

- `no_observed_support`: `empty_voxel_support` metadata(`used_as_evidence=False`), 승격/거부에 미사용.
- `conflicting_evidence`: `interior_conflicting_count` 보존, `used_as_evidence=False`.
- `build_candidate_conflicts`: (1) 공유 source + bridge AABB overlap, (2) 서로 다른 pair 유사 bridge 공간, (3) 동일 영역 evidence 양립 불가(free-contradiction vs behind support) 시 conflict edge를 raw reason과 함께 생성. ranking/selection/pruning 없음. `torch_observation_evidence` import 없이 candidate의 evidence dict만 참조.

## 9. Fixture별 결과

`tests/test_occluded_region_candidate.py`(22) + `tests/test_candidate_evidence.py`(8), 설계 §12의 26항목을 모두 커버:

| # | 항목 | 결과 |
|---|---|---|
| 1 | Planar two-sided gap → candidate | PASS |
| 2 | **Coplanar narrow pair가 임의 reject되지 않음** | PASS(candidate) |
| 3 | Parallel bounded ribbon | PASS |
| 4 | Orthogonal corner(facing gate 없음, outward_dot≈0) | PASS |
| 5 | Oblique corner | PASS |
| 6 | Curved two-sided gap | PASS |
| 7 | Closed annular/radial candidate(cyclic, connector_end=None) | PASS |
| 8 | Zero connector separation structural negative → rejected | PASS |
| 9 | Disconnected-close(threshold 초과) → candidate 없음 | PASS |
| 10 | One-sided → candidate 없음 | PASS |
| 11 | AABB overlap/no narrow correspondence → candidate 없음 | PASS |
| 12 | 동일 `(s_a,s_b)` canonicalization(tip 선택) | PASS |
| 13 | Non-monotonic component 분리 | PASS(≥2 component) |
| 14 | Multiple components 별도 candidate_id | PASS |
| 15 | Domain order reversal determinism | PASS(id·bridge 동일) |
| 16 | Degenerate provenance 보존 / rejected domain 제외 | PASS |
| 17 | Full bridge-interior known-free → rejected | PASS |
| 18 | Partial free contradiction 보존 → candidate | PASS |
| 19 | Support endpoint on-surface가 interior reject 막지 않음 | PASS |
| 20 | No interior evidence → insufficient(reject 아님) | PASS |
| 21 | Conflicting evidence 보존(used_as_evidence=False) | PASS |
| 22 | Empty voxel non-promotion | PASS |
| 23 | Duplicate candidate/domain deduplication | PASS |
| 24 | Conflict edge generation | PASS |
| 25 | Deprecated Stage 2 regression 없음 | PASS(11/11 불변, 코드 미변경) |
| 26 | 전체 suite 회귀 없음 | PASS |

threshold 경계 fixture(2·8·9·11)는 numeric probe로 normalized distance를 먼저 확인해 precondition을 검증했다.

## 10. Threshold/config 값과 근거

- `broad_phase_expand_factor` 기본 1.5, `correspondence_threshold` 기본 1.0(= scale-normalized distance 1배). **확정 scientific constant가 아니라 잠정 시작값**이며, raw value와 threshold-used를 payload에 병렬 보존한다. 실제 scene/seed sweep으로 재검증할 parameter다(설계 §7). hidden weighted score / 조기 scalarization 없음.
- `interior_samples` 기본 3(cross-section 내부 exclusive linspace).

## 11. 규모 지표

대표 입력(facing pair 2쌍 + 이격 배치 4 domain) 실측:
```
input_domain_count      4
broad_phase_pair_count  2   (이격된 cross-pair는 broad-phase에서 정확히 배제)
candidate_count         2
states                  {candidate: 2}
conflict_edge_count     0   (분리된 비중첩 영역)
max_candidate_degree    0
```
API가 모든 지표(input/broad/narrow-surviving/candidate/unsupported/rejected count, conflict edge count, max degree)를 노출한다. arbitrary candidate-count cap 없음 — benchmark 후 필요성 검토.

## 12. 전체 suite 결과

- 신규만: `tests/test_occluded_region_candidate.py` 22 passed, `tests/test_candidate_evidence.py` 8 passed.
- deprecated Stage 2 `tests/test_surface_candidate_graph.py`: 11 passed(회귀 없음).
- 전체 pytest: `290 passed, 1 skipped, 8 subtests passed`(Gate D 시점 260에서 +30).
- 구조 불변식 grep 확인: geometric 모듈에 `torch_observation_evidence` import 0, broad-phase에 `torch_surface_candidate_graph` import 0.

## 13. Phase F/production 미착수 확인

Phase F NURBS bridge fitting, candidate global ranking/selection, multi-sided joint topology, full polygon/voxel solid reconstruction, self-intersection·visible-surface penetration 완전 검사, multi-scale continuation, scalar confidence calibration, candidate conflict resolution, Gaussian proposal, production integration 모두 미착수. `osn_gs/core/` 어디에도 이번 신규 모듈을 import하지 않는다.

## 14. 중단 및 다음 승인

계획대로 Phase E 구현과 Gate E 보고까지 완료하고 멈춘다. 다음 승인 요청은 **Phase F — Minimal Constrained Occluded NURBS Bridge**로 한정하며, 별도 사용자 승인 없이는 Phase F, global candidate selection, production integration을 시작하지 않는다.
