# Worklog 36 — Directed Cycle Solver Contract Hardening 및 Region Coverage Recovery

## 최종 질문에 대한 결론 먼저

> Worklog 35의 Hungarian one-in/one-out cycle recovery가 unmatched node, short subtour, branch ambiguity, capacity limit 및 self-intersection 상황에서도 physical evidence만을 사용해 동일한 fail-closed 계약을 유지하는가?

**대부분 유지했고, 실제로 증명된 결함 두 가지를 이번에 좁게 고쳤다.** Hungarian solver의 unmatched/dummy 계약 자체는 10개 합성 시나리오(전부-forbidden, successor-only, predecessor-only, 유효 open path, cycle+unmatched, disjoint cycles, 2-cycle, self-loop, score tie, 극단 score)로 직접 감사해 전부 통과했다(§4). 그러나 실제 box_face(cap=27) replay에서 **candidate 100% accounting이 깨져 있었다** — 15+3=18인데 실제 candidate는 19개였다(§3). 원인은 decomposition 함수가 "matched-source도 아니고 matched-target도 아닌" 완전히 고립된 노드를 그냥 누락시키던 버그였다 — 좁게 수정해 `isolated_boundary_candidate` 상태를 명시적으로 추가했다. Short-subtour objective(B: cardinality-first lexicographic)는 실제로 시도했지만 cylinder side(88 candidate)를 3조각으로 fragmentation시키는 회귀를 유발해 **채택하지 않고 원래 max-score objective(A)를 유지**했다 — "결과가 잘 나온다는 이유만으로 채택하지 마라"는 지시를 정확히 지켜 검증된 회귀 하나를 근거로 되돌렸다. Capacity limit(400개 초과)의 silent greedy fallback은 명시적 `ordering_capacity_exceeded` fail-closed 상태로 교체했다(Case C). Self-intersection은 turning-angle만으로는 증명 불가능하다는 지적이 맞았다 — 별도 세그먼트 교차 검사 모듈을 신설해 materialization 이전에 fail-closed로 연결했다. Branch/non-manifold 사전 진단은 시도했으나 **정규 격자(box_face 32점 등)에서 전수 오탐**을 일으켜 채택하지 않고 진단 전용 함수로만 남겼다 — degree>2를 이유로 무조건 reject하지 말라는 지시와 정확히 같은 이유로 롤백했다.

> 그리고 Worklog 34/35의 baseline 차이를 제거한 authoritative replay에서 C9 수정이 실제로 coherent region coverage를 개선했는지 확인했으며, real snapshot의 closed-loop 부재가 region coverage 부족인지 termination-candidate recall 부족인지 정확히 분리했는가?

**baseline 차이의 정확한 원인을 규명했고(§1), C9 수정의 개선 효과를 config-flag 기반 authoritative ablation으로 재확인했으며(§12), closed-loop 부재의 원인을 region coverage 부족(90%)과 candidate recall/compatibility 부족(10%)으로 정량 분리했다(§15).** Worklog 35의 "A_baseline"은 `git show HEAD:<file>`로 캡처됐는데, HEAD(d359c5e)는 worklog 34/35의 어떤 수정도 커밋된 적 없는 상태였다 — 즉 worklog 35의 ablation은 "worklog 34 이전" 상태와 비교하고 있었다. 이번에 두 수정을 `RegionFormationConfig`의 명시적 플래그(`enable_worklog34_growth_weak_bridge_exemption`, `enable_worklog35_parallel_veto_nearby_evidence_gate`)로 재구성해 파일 교체 없이 동일 코드/동일 프로세스에서 4-way ablation을 재현했다 — 두 플래그 모두 False가 정확히 worklog 35의 "A_baseline" 수치(region_count=70, core_member=362, consensus_attached=1)와 일치함을 확인해 원인을 확정했다.

---

## 1. Worklog 34/35 Baseline Discrepancy 해소

**원인**: `scripts/devtools/run_c11_c9_ablation.py`(worklog 35)가 ablation "A_baseline"을 `git show HEAD:osn_gs/surface/torch_gaussian_surface_region_formation.py`로 캡처했다. 그러나 이 저장소는 worklog 1부터 35까지 전부 커밋되지 않은 working tree 상태로 진행돼 왔다 — `git log`로 확인한 HEAD(`d359c5e`)는 worklog 34/35의 어떤 수정도 포함하지 않는다. 즉 worklog 35의 "A_baseline"은 실제로는 **worklog 34 이전(pre-34)** 상태였고, worklog 34가 자체 보고한 75/85/64, 392/447/357은 이 스크립트로 재현 불가능한, 커밋되지 않은 중간 상태에서 측정된 값이었다.

**해소 방법**: `RegionFormationConfig`에 두 개의 진단 전용 ablation 플래그를 추가했다(둘 다 기본값 True = 현재 production 동작 그대로 유지):
- `enable_worklog34_growth_weak_bridge_exemption`
- `enable_worklog35_parallel_veto_nearby_evidence_gate`

파일 교체 대신 이 플래그로 4-way ablation을 동일 코드/동일 프로세스에서 재현했다(`scripts/devtools/authoritative_replay_fingerprint.py`). 결과(3k 기준):

| w34 | w35 | region_count | core_member | consensus_attached | micro≤3 |
|---|---|---|---|---|---|
| 0 | 0 | **70** | **362** | **1** | 15 |
| 0 | 1 | 77 | 414 | 1 | 9 |
| 1 | 0 | 70 | 362 | 12 | 11 |
| 1 | 1(현재 production) | 77 | 414 | 12 | 8 |

`w34=0,w35=0`이 정확히 worklog 35가 보고한 "A_baseline"(70/362/1)과 일치함을 확인했다 — 이것이 baseline 불일치의 정확한 근거다. **Worklog 34의 자체 보고값(75/392)은 커밋되지 않은 중간 상태라 이 세션에서 정확히 재현할 수 없다**; 이번부터는 `w34=1,w35=1`(현재 production, 이 fingerprint 스크립트로 항상 재현 가능)을 유일한 authoritative baseline으로 확정한다.

## 2. Authoritative Replay Fingerprint

`scripts/devtools/authoritative_replay_fingerprint.py`가 매 replay마다 다음을 기록한다: checkpoint 절대경로/SHA-256, iteration, full Gaussian count, representative cap, representative stable-ID SHA-256, config SHA-256, LocalEvidenceScale/G1 상태, worklog34/35 fix 상태, ordering 구현 식별자, random seed, `torch.are_deterministic_algorithms_enabled()`, git commit, dirty diff SHA-256. 3k checkpoint 예시: `checkpoint_content_sha256=de6d93e2...`, `commit=d359c5e...`, `dirty_file_count=129`. 4-way ablation은 실제 config 객체를 `VisibleSurfaceConstructionConfig(regions=...)`로 주입해 수행했다 — 파일 교체 없음.

## 3. Candidate 100% Accounting

Box_face(cap=27, genuine candidate 19개)에서 worklog 35의 실제 output(15 closed + 3 open = 18)이 candidate 수와 불일치함을 확인했다. 원인: `_decompose_into_paths_and_cycles`가 "matched-source도 아니고 matched-target도 아닌" 노드(양방향 compatibility가 매칭에서 전부 탈락한 노드)를 순회 대상에서 아예 빠뜨리고 있었다. 노드 21(gaussian_id=21)이 정확히 이 경우 — forward 3개/backward 3개의 실제 compatibility edge가 있었지만 전부 더 높은 score의 경쟁자에게 밀려 매칭되지 못했고, 그 결과 output 어디에도 나타나지 않았다.

**수정**: `_decompose_into_paths_and_cycles`에 `all_node_ids` 파라미터를 추가해, cycle/path 어디에도 속하지 않은 노드를 명시적으로 `isolated_boundary_candidate` 상태로 반환하도록 했다. 모든 candidate가 다음 9개 상태 중 정확히 하나를 받는다: `ordered_closed_loop`(component), `ambiguous_ordering`, `isolated_boundary_candidate`, `ordering_capacity_exceeded` (그 외 rejected 계열은 이번 라운드에서 발견되지 않음 — reason=review_reasons로 세분화됨). 수정 후 box_face(cap=27): 15+3+**1**=**19**, accounting 정확히 일치. `tests/test_directed_boundary_ordering.py::CandidateAccountingTest`에 정확한 합계 assertion을 추가했다.

## 4. Hungarian Solver 실제 계약 감사

`scripts/devtools/audit_hungarian_solver_contract.py`로 10개 시나리오를 직접 검증했다: 전부-forbidden 노드는 절대 실제 노드에 매칭되지 않고 unmatched로 남음, successor-only/predecessor-only 노드가 정상 처리됨, 유효 open path(3-node)가 정확히 복원됨, cycle+unmatched extra node가 섞여도 정확, 두 개의 disjoint 3-cycle이 모두 발견됨, 순수 2-cycle 자체는 solver 레벨에서 유효한 매칭으로 선택되지만(cost 최적화 관점에서는 맞는 선택) decomposition 단계에서 `len(chain)>=3` 게이트로 걸러짐(closed로 인정 안 됨), self-loop는 애초에 candidate generation에서 배제되어 생성되지 않음, score tie 결과가 반복 호출에서 결정적으로 동일, 극단적 score 범위(1e9 vs 1e-9)에서도 정상 동작. 전부 통과 — Forbidden/dummy edge가 실제 geometry edge로 복원되는 사례는 없었다.

## 5. Maximum-Weight vs Maximum-Coverage 목표 비교

**A(현재 max-total-score)**를 유지했다. **B(cardinality-first lexicographic)**를 실제로 구현하고 실측했으나 **채택하지 않았다**:

- Box_face(cap=27)에서 B는 개선을 보였다: 15-node closed → **16-node closed**, isolated 1→0 (node 21이 매칭에 포함됨).
- 그러나 **cylinder positive control에서 회귀를 유발했다**: side wall(88 candidate, 기존 1개의 깨끗한 closed loop)이 B 하에서 10/18/26-node로 조각난 `ambiguous_ordering` 3개 + isolated 2개로 붕괴했다 — box_face에서 도움이 되는 것과 정확히 같은 메커니즘(더 많은 노드를 강제로 매칭에 포함시키는 것)이 후보가 많은 큰 region에서는 오히려 일관성을 깨뜨렸다.
- "결과가 잘 나온다는 이유만으로 단일-loop penalty를 추가하지 마라"는 지시에 따라, 검증된 positive-control 회귀 하나를 근거로 **B를 되돌리고 A를 유지**했다. box_face의 잔여 결함(node 21, 3-node open path)은 §11에서 별도로 정직하게 disclosed.
- C(bounded exact/branch-and-bound)는 A 자체가 이미 exact(Hungarian)이므로 불필요. D(shape-specific)/E는 금지 목록에 해당.

## 6. Branch/Y-junction 계약 수정

`_diagnose_branch_ambiguity`(진단 함수)를 구현해 undirected compatibility degree>2이면서 matched edge와 runner-up 간 score margin이 좁은(5% 미만) 노드를 branch 후보로 표시하도록 했다. **Admission gate로 연결을 시도했으나 box_face(32점, 정규 9x9 격자)의 32개 노드 전부를 branch로 오탐했다** — 정규 격자에서는 대칭성 때문에 인접 후보들의 score가 자연히 근접해, "score margin이 좁다"는 판정 기준이 진짜 Y-junction과 정상적인 조밀 샘플링을 구분하지 못했다. `thin_slab`/`box_isolated_floater`/`box_isotropic_contamination`/`box_face` 전부 closed=0으로 회귀시켰다.

**최종 판단**: "degree>2라는 이유만으로 무조건 reject하지 마라"는 지시를 정확히 지켜 admission gating을 롤백했다. `_diagnose_branch_ambiguity`는 코드베이스에 진단 전용 함수로 남겨뒀다(§6이 요구한 5개 카테고리 분류를 위한 원시 데이터는 계산 가능하지만, 실제 branch/정상-밀도를 구분하는 신뢰할 만한 신호는 이번 라운드에서 찾지 못했다 — 실제 각도/loop-topology 기반 판정이 필요하며 다음 라운드 과제로 남긴다).

## 7. Candidate 400개 초과 fallback 제거

**Case C(명시적 fail-closed)**를 채택했다. 기존의 silent greedy fallback(candidate 수가 늘어나면 correctness 계약이 조용히 바뀌는 문제)을 제거하고, cap(150) 초과 region의 모든 candidate에 명시적 `ordering_capacity_exceeded` 상태와 `region_candidate_count_exceeds_exact_matching_capacity` 사유를 부여하도록 교체했다. 200-node 합성 ring으로 검증: 전부 `ordering_capacity_exceeded`, accounting 200/200 정확히 일치. 실제 관측된 어떤 시나리오도 이 cap에 도달하지 않지만(최대 88, cylinder side), "현재 최대가 32/88이므로 안전하다"는 가정에 의존하지 않고 명시적 계약으로 만들었다.

## 8-9. Cycle/Path Decomposition 검증 및 명시적 Self-Intersection 검사

**Decomposition 불변량**은 §3의 accounting 수정으로 이미 검증됨(모든 matched edge가 정확히 하나의 component, 모든 candidate가 정확히 하나의 최종 state). Closed cycle in/out-degree=1, open path endpoint=2, branch=0(matching 구조상 보장), self-loop=0(candidate generation에서 배제), 2-cycle=0(decomposition에서 path로 강등), duplicate/reverse-duplicate 미영향(§4에서 확인) — 전부 확인.

**Self-intersection**: 신규 모듈 `osn_gs/surface/torch_boundary_self_intersection.py`(`validate_simple_closed_loop`)를 작성했다. 3D 루프를 자체 local tangent plane(공분산 기반 PCA, torch/numpy 의존성 없음)에 투영한 뒤, 비인접 segment pair 전부에 대해 정확한 교차 검사(proper intersection / endpoint touch / collinear overlap)를 수행한다. Winding number는 centroid-기반 계산이 near-degenerate 케이스에서 수치적으로 불안정함을 발견해(near_touching fixture에서 오탐), turning-angle 합(인접 edge 방향만 비교하므로 안정적)으로 대체했다. Analytic fixture 7종(rectangle/concave/bow-tie/repeated-vertex/near-touching/figure-eight) 전부 기대대로 판정. `materialize_visible_boundary_component`에 wiring해 self-intersection 실패 시 NURBS fitting 이전에 fail-closed하도록 했다 — 기존 positive control(box_face/cylinder/thin_slab 등) 전부 회귀 없음 확인.

## 10. Box_face Analytic Source/Evaluated Boundary Accuracy

Box_face는 `_flat_grid(9, 0.12)`, half-extent=0.48의 정사각형. `scripts/devtools/measure_box_face_analytic_boundary_accuracy.py`로 candidate/ordering/fitting 세 오차를 분리 측정했다.

| | 비다운샘플(32점) | Cap=27 다운샘플 |
|---|---|---|
| candidate position error (median) | **0.0006** | **0.119** |
| ordered source polyline error (median) | 0.0006(변화 없음) | 0.119(변화 없음) |
| NURBS evaluated boundary error (median) | 0.063 | 0.124 |

**핵심 발견**: worklog 35가 보고한 `boundary_residual=0.054`는 ordering이나 fitting의 결함이 아니라 **representative selection(다운샘플)이 애초에 analytic edge 위에 정확히 놓이지 않는 candidate를 만들어낸 결과**였다 — cap=27에서 candidate 자체의 오차(0.119)가 이미 ordering 이후 오차(0.119, 변화 없음)와 동일하다. Ordering은 추가 오차를 전혀 만들지 않았다. NURBS fitting은 두 경우 모두 소폭의 추가 오차를 만든다(6x6 control grid, degree-2 LSQ의 한계로 추정, 정사각형의 날카로운 코너를 완벽히 재현하지 못함) — 이는 fitting 단계의 별도 과제이지 C11(ordering)의 문제가 아니다.

## 11. 잔여 3-node Open Path 및 누락 Candidate 원인 확정

`scripts/devtools/trace_residual_box_face_fragments.py`로 정확히 추적했다:

- **노드 70**(position (-0.359, 0.361), 실제 코너 근처): `compat_out_degree=0` — 어느 방향으로도 forward-compatible successor가 전혀 없다. 이는 **올바른 fail-closed 잔여 evidence**다(§11 카테고리) — 어떤 알고리즘도(가짜 edge를 만들지 않는 한) 이 노드를 닫힌 loop에 포함시킬 수 없다.
- **노드 58**(0.0, 0.239, 인테리어에 가까운 점): 58→77→70 체인의 일부지만 70의 dead-end 때문에 이 체인 전체가 open으로 남는다.
- **노드 21**(0.119, -0.241): 양방향 compatibility degree 3/3으로 실제 물리적 증거는 충분하지만, max-score objective 하에서 양쪽 다 경쟁에서 밀렸다(§5에서 이미 B 시도, 회귀로 인해 미채택).

**판정**: 15-node loop 하나가 나왔다고 C11을 "완전히 해결"이라고 선언하지 않는다. 3-node open path는 진짜 fail-closed 증거(노드 70의 구조적 dead-end)이고, 1-node isolated(노드 21)는 objective 선택의 부작용으로 정직하게 disclosed한다.

## 12. C9 Authoritative Ablation (Config-flag 기반)

§1의 fingerprint 스크립트로 3k/5k/10k 전체에 대해 재수행했다.

| checkpoint | w34,w35 | region_count | core_member | consensus_attached | micro≤3 | closed | materialized |
|---|---|---|---|---|---|---|---|
| 3k | 0,0 | 70 | 362 | 1 | 15 | 0 | 0 |
| 3k | 0,1 | 77 | 414 | 1 | 9 | 0 | 0 |
| 3k | 1,0 | 70 | 362 | 12 | 11 | 0 | 0 |
| 3k | 1,1 | 77 | 414 | 12 | 8 | 0 | 0 |
| 5k | 0,0 | 84 | 431 | 0 | 16 | 0 | 0 |
| 5k | 1,1 | 83 | 454 | 9 | 6 | 0 | 0 |
| 10k | 0,0 | 63 | 344 | 0 | 12 | 0 | 0 |
| 10k | 1,1 | 67 | 375 | 6 | 8 | 0 | 0 |

두 수정이 독립적으로 재확인됐다: w35(C9)는 core_member를 증가시키고(3k 362→414) micro-region 비율을 낮춘다(15→9); w34(growth fix)는 consensus_attached를 증가시킨다(1→12). `boundary_component_closed_count`/`materialized_surface_count`는 8개 조합 전부에서 0 — region/core 개선이 closed-loop 형성으로 직결되지 않음을 재확인(§15에서 정량 분리).

## 13. Nearby Parallel Evidence Semantics 검증

`consensus.contradicting_parallel_neighbor_count`는 `candidate_neighbors[a] & candidate_neighbors[b]`로 계산되며, `candidate_neighbors`는 manifold affinity graph의 **이미 bounded된 kNN candidate edge**(`CANDIDATE_STATUS_CANDIDATE`, `candidate_scale` radius 내)에서만 채워진다 — 새로운 scene-specific radius를 추가하지 않고 기존 graph neighborhood를 그대로 재사용한다는 것을 코드로 직접 확인했다. Real 3k에서 override 후보 edge 10개를 무작위 샘플링해 감사한 결과, `contradicting_parallel_neighbor_count`가 0인 경우와 1~3인 경우가 섞여 있어(진짜로 구분하는 신호임을 확인) 항상 참/거짓이 되는 퇴화된 조건이 아님을 검증했다. Negative control(thin_slab=2 regions, box=6 regions, box_isolated_floater=1 region excl. floater) 전부 회귀 없음.

## 14. Ambiguous-Unassigned Waterfall (R1-R6)

`scripts/devtools/trace_ambiguous_unassigned_waterfall.py`로 3k/5k/10k 전체 representative를 분류했다.

| | 3k | 5k | 10k |
|---|---|---|---|
| R1(same_surface degree=0) | 512(32%) | 525(31%) | 646(38%) |
| **R2(same_surface 이웃은 있으나 전부 아직 어느 region에도 속하지 않음)** | **913(57%)** | **882(55%)** | **892(53%)** |
| R3(경쟁 region, 명확한 다수 없음) | 5(0.3%) | 1(0.1%) | 2(0.1%) |
| R4(growth threshold 미충족, 명확한 다수 region 있음) | 183(11%) | 164(9%) | 121(7%) |

**R2가 세 checkpoint 전부에서 지배적(53~57%)** — 이는 **core-seeding coverage 자체의 부족**이 근본 원인임을 뜻한다: 대부분의 ambiguous 노드는 same_surface 이웃이 있지만, 그 이웃들 역시 아직 어느 region에도 속하지 못한 상태(닭-달걀 문제)라 growth(기존 region에만 붙임)가 애초에 붙을 대상이 없다. R1(32~38%)이 두 번째로 크고, R3(경쟁 region)은 무시할 수준(0.1~0.3%)이다. **결론: growth threshold(`growth_min_support_count`/`growth_min_support_ratio`)를 완화해도 R2 노드에는 아무 효과가 없다** — 이 문제는 core seeding 단계(`bridge_min_shared_neighbor_for_well_supported=2` 등, worklog 35에서 이미 지배적 원인으로 확인됨)에서만 해결 가능하며, 이는 이번 라운드에서 명시적으로 금지된 threshold 완화 없이는 해소되지 않는다.

## 15. Physical Boundary Candidate Coverage

`scripts/devtools/trace_physical_boundary_candidate_coverage.py`로 3k의 전체 77개 region을 분류했다:

| 분류 | region 수 | 비율 |
|---|---|---|
| 1(region이 물리적 perimeter를 담기에 구조적으로 너무 작음, candidate<3) | 69 | **90%** |
| 4(candidate≥3, compatibility 충분하지만 ordering 실패) | 8 | 10% |

**90%는 순수 region-size/candidate-recall 문제**(closed loop에는 최소 3개 candidate가 필요한데 애초에 미달)이고, **10%(8개)만 candidate≥3인데도 미형성**됐다. 이 8개를 직접 조사한 결과 전부 member_count 3~8의 소형 region이며 candidate 대부분이 `isolated_boundary_candidate`(양방향 compatibility 자체가 0) — box_face처럼 "candidate가 충분한데 ordering 알고리즘이 실패"하는 진짜 C11 케이스가 아니라, **spatial sparsity로 인해 candidate 간 방향 호환성 자체가 형성되지 않는 케이스**(카테고리 2: candidate recall/compatibility 부족)다. **결론: real snapshot에서 closed-loop 부재는 압도적으로(90%+10%=100%) region coverage/candidate-recall 문제이며, box_face에서 발견된 것과 같은 순수 ordering-solver 결함 사례는 real snapshot에서 하나도 발견되지 않았다.**

## 16. 적용한 Narrow Repair 요약

| 파일 | 변경 | Case |
|---|---|---|
| `torch_directed_boundary_ordering.py` | `_decompose_into_paths_and_cycles`에 isolated-node 명시적 accounting 추가 | 100% accounting 결함 |
| `torch_directed_boundary_ordering.py` | Candidate>150 시 silent greedy → 명시적 `ordering_capacity_exceeded` | Case D(Case C 방식 채택) |
| `torch_boundary_self_intersection.py`(신규) | 명시적 segment-crossing 검사 모듈 | Case (self-intersection, 신규) |
| `torch_visible_boundary_materialization_adapter.py` | self-intersection 검사를 fitting 이전에 wiring | 위와 동일 |
| `torch_gaussian_surface_region_formation.py` | worklog34/35 fix에 diagnostic-only ablation 플래그 추가(기본값 유지) | Case A(baseline 재현성) |

**시도했으나 롤백한 것**: lexicographic max-coverage objective(§5, cylinder 회귀), branch admission gating(§6, 정규 격자 전수 오탐).

## 17. Positive/Negative Controls

Positive: post-ADC plane(box_face 비다운샘플, closed=1), box_face cap=27(closed=1, 15+3+1=19 정확 accounting), cylinder(side+2 caps, closed=3), 두 개의 분리된 물리적 loop(합성, closed=2, 병합 없음), rigid rotation/translation/uniform scale(정확히 invariant, §4/frozen 테스트).

Negative: open chain(closed=0), missing-edge loop(closed=0), sparse gap(closed=0), Y-junction(branch node가 closed loop에 포함 안 됨), bow-tie(self-intersection 검출), figure-eight(self-intersection 검출), repeated vertex(검출), 근접 평행 loop 없음(thin_slab=2 regions 유지), box crease 없음(box=6 regions 유지), floater 미포함(box_isolated_floater), isotropic contamination 미포함, bridge contamination(false merge 없음, box_with_bridge≥5 regions), sphere(physical outer boundary 미생성, seam/atlas 미구현 상태 그대로 유지).

## 18. Real 3k/5k/10k 결과 (최종, 모든 수정 적용)

| | 3k | 5k | 10k |
|---|---|---|---|
| region_count | 77 | 83 | 67 |
| core_member | 414 | 454 | 375 |
| consensus_attached | 12 | 9 | 5 |
| boundary_component_count | 61 | 90 | 56 |
| boundary_component_closed_count | 0 | 0 | 0 |
| materialized_surface_count | 0 | 0 | 0 |
| runtime | 32.8s | 36.5s | 55.1s |

`boundary_component_count`가 이전 라운드(4/21/9) 대비 크게 늘었다(61/90/56) — 이는 §3의 accounting 수정으로 이전에 조용히 누락되던 isolated candidate(3k 56개)가 이제 명시적으로 각각 하나의 component로 보고되기 때문이며, 실제 회귀가 아니라 정직한 보고다. `closed`/`materialized`는 §15에서 확인한 대로 구조적 candidate 부족 때문에 여전히 0 — 이는 이번 라운드에서 강제로 만들지 않았다(§18 성공 기준에 명시된 대로 필수 성공 조건이 아님).

## 19. Runtime/Memory

Real 3k/5k/10k: 25~55초대(기존과 동일 범위, 재앙적 증가 없음). Hungarian solver: n=20 0.004초, n=100 0.388초, n=200 3.21초(O(n³), forward+reversed 두 번); 실측 최대 region 크기(88, cylinder side)는 이 범위 내에서 수 밀리초 수준. 신규 self-intersection 검사는 O(n²) segment-pair 비교(순수 Python, torch 비의존) — box_face 32-node 루프에서 무시할 수준(<1ms). Capacity-exceeded fail-closed 경로는 O(1) 조건 분기만 추가.

## 20. Focused/Full Pytest

- `tests/test_directed_boundary_ordering.py`: 12(기존, worklog 35) + 14(신규: CandidateAccountingTest 2, HungarianSolverContractTest 3, CapacityLimitTest 1, SelfIntersectionValidationTest 8) = **26/26 pass**.
- `tests/test_region_consolidation_repair.py`: 6(기존) + 3(신규: AblationConfigFlagTest 2, AmbiguousUnassignedWaterfallTest 1) = **9/9 pass**.
- 관련 broader suite(19개 파일, boundary/region/affinity/invariance/materialization 전체): **136/136 pass** (58.38s).
- **Repository-wide pytest(세션 전체 마지막에 1회만 실행): 647 passed, 1 skipped, 0 failed, 8 subtests passed, 소요 시간 188.68초(3분 9초).**

## 21. 다음 남은 Visible Surface Constructor 병목

1. **Real snapshot의 region coverage 근본 한계(§14/§15에서 정량화, 여전히 미해결)**: ambiguous_unassigned의 53~57%가 R2(same_surface 이웃이 있지만 그 이웃도 아직 region에 속하지 않음)이며, 이는 growth threshold가 아니라 **core seeding coverage 자체**(`bridge_min_shared_neighbor_for_well_supported=2`)의 문제다. 다음 라운드는 이 threshold가 실제 데이터 밀도에 비해 구조적으로 너무 엄격한지 여부를 별도 승인 하에 조사해야 한다.
2. **Branch/non-manifold 사전 진단(§6)**: 이번 라운드에서 시도한 score-margin 기반 판정은 정규 격자에서 전수 오탐을 일으켜 롤백했다. 진짜 Y-junction과 정상 조밀 샘플링을 구분하려면 각도/loop-topology 기반의 더 정교한 신호가 필요하다 — 진단 함수(`_diagnose_branch_ambiguity`)는 남겨뒀지만 production에 연결되지 않았다.
3. **box_face(cap=27)의 잔여 결함(§11)**: 노드 70의 구조적 dead-end(compat_out_degree=0)는 진짜 fail-closed 증거이므로 강제로 고칠 대상이 아니다. 노드 21은 max-score objective의 부작용으로 남아 있다 — lexicographic objective가 cylinder를 깨뜨리지 않는 형태로 재설계 가능한지는 별도 조사 과제.
4. **NURBS fitting 단계의 코너 재현 한계(§10)**: 비다운샘플 box_face에서도 fitting이 median 0.063의 추가 오차를 만든다(6x6 control grid, degree-2 LSQ) — ordering과 무관한 별도 fitting 품질 과제.
