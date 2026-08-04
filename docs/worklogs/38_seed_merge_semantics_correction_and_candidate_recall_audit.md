# Worklog 38 — Seed Existence / Weak-Bridge Merge Semantics 교정 및 Termination Candidate Recall 감사

## 최종 질문에 대한 결론 먼저

> Worklog 37의 intra-raw-component bridge-veto exemption은 seed existence와 component merge를 올바르게 분리한 것인가, 아니면 raw same-surface component 내부에서 weak-bridge protection을 사실상 무력화한 것인가?

**후자다. 그것도 부분적 무력화가 아니라 완전한 무력화이며, 수학적으로 증명 가능한 tautology였다.**

Worklog 37의 raw component는 `core_eligible` edge 집합의 connected component로 계산된다. Connected component의 정의상 그 edge 집합에 속한 **모든** edge의 양 endpoint는 같은 component에 속한다. 따라서 "두 endpoint가 같은 raw component면 bridge veto를 건너뛴다"는 조건은 core-eligible edge 100%에 항상 참이다. 실측(3k checkpoint): core-eligible edge 2092개 중 intra-component 2092개, inter-component **0개**. Bridge veto가 평가한 edge 수는 flag ON에서 **0개**, OFF에서 1244개(그중 862개가 `weak_bridge_candidate`)였다. 3k/5k/10k 모두 동일 패턴(ON=0 / OFF=1244·1180·1072).

결과적으로 **articulation bridge(제거 시 그래프가 끊어지는 단일 fragile edge, 양쪽 모두 3노드 이상) 47개가 그대로 union**됐다 — 7노드 클러스터와 23노드 클러스터를 독립 cross-support 없이 단 하나의 edge로 잇는 사례 포함. 이는 bridge veto가 막으라고 존재하는 바로 그 실패 모드다. §2의 disqualifying condition 중 최소 3개(raw component 내부 weak single bridge union / one-edge connection으로 large component 병합 / bridge veto가 평가 대상 edge 대부분에서 우회)에 해당하므로 production canonical semantics로 유지할 수 없다.

> 이를 검증한 뒤 valid local seed component는 독립적으로 보존하되 weak bridge를 통한 surface merge는 허용하지 않는 구조로 교정했는가?

**교정했다.** `exempt_intra_raw_component_unions_from_bridge_veto`를 기본값 False(diagnostic-only)로 원복하고, §5가 권장한 **명시적 2-phase DSU**를 `separate_seed_and_merge_phases=True`(canonical)로 구현했다. Phase 1은 `seed_strong_edge`(=`CONSENSUS_WELL_SUPPORTED`, 즉 실제 shared same_surface neighbor 지지가 확인된 edge)만으로 union해 각 component가 자기 evidence로 독립적인 seed가 되게 하고, Phase 2는 남은 weak cross-edge를 **component pair 단위**로 모아 distinct-endpoint aggregate support(`merge_min_distinct_endpoint_support=2`)를 요구한다. 단일 fragile edge는 endpoint support 1/1이므로 구조적으로 merge 불가이고, merge가 거부돼도 **양쪽 seed는 그대로 유지**된다. 검증: articulation bridge union 수가 worklog 37의 47 → **0**. `bridge_min_shared_neighbor_for_well_supported`는 2로 그대로 유지했다.

> 그리고 termination-candidate rejection waterfall과 analytic precision/recall을 통해 real closed-loop 부재가 region extent 부족인지 candidate extraction 부족인지 정확히 분리했으며, Worklog 37에서 발생한 cylinder cap 회귀를 false boundary 생성 없이 해소했는가?

**분리는 완료했다. Cylinder cap 회귀는 근본 원인까지 정확히 규명했으나 해소하지 못했고, 정직하게 미해결로 보고한다.**

Candidate waterfall(§10)과 analytic precision/recall(§12)을 구현했다. **box_face candidate precision=1.000, recall=1.000**(ground-truth 경계 노드 32개 전부 생성, false positive 0) — candidate generator 자체는 정확하다. Sphere는 genuine candidate 22개에도 closed=0(임의 seam 미생성, 올바름). Box는 face당 candidate 16~20개가 있으나 compatible directed edge가 10~17개뿐 — N-노드 ring에는 N개 edge가 필요하므로 **구조적으로 edge-starved(C4)**이며 ordering solver 결함이 아니다.

Cylinder cap 회귀의 root cause는 정확히 특정했다: sector histogram의 ±0.15 bin-margin smearing이 실제 geometric gap을 무력화한다. 노드 257/266은 gap 1.568/1.590 rad(threshold 1.178 rad를 명백히 통과)인데도 smearing 때문에 8개 sector 전부 occupied로 표시돼 `runs=()`가 되고 탈락한다. Core seeding이 개선돼 accepted neighbor가 하나 늘면(5→6) 이 상태로 전환되는 것이 worklog 37에서 "cap 회귀"로 관측된 실체다. 그러나 시도한 모든 교정(gap-dominates-histogram 독립 임계값, measured gap span에서 run 유도)이 결국 **이 fixture의 1.568 rad 바로 위에 오는 상수를 고르는 일**로 귀결됐고, 이는 §13이 명시적으로 금지한 scene-tuning이다. 원복하고 미해결로 보고한다.

---

## 1. Worklog 37 수정 재평가

§1이 요구한 판정 명제 — "raw component를 구성한 모든 edge의 endpoint는 정의상 같은 raw component에 속하므로 exemption은 bridge veto를 사실상 항상 우회하는가?" — 는 **참**이다. 코드에서 raw component는 `core_eligible`(veto loop가 순회하는 바로 그 집합)로부터 BFS로 계산된다. 수학적 tautology이며 데이터 의존성이 없다.

## 2. Raw-Component Exemption 적용 범위 (§1 요구 수치)

3k checkpoint 기준:

| 항목 | 값 |
|---|---|
| Core-eligible edge 전체 | 2092 |
| Raw component 내부 edge | **2092 (100%)** |
| Raw component 간 edge | **0** |
| Bridge veto 평가 대상 (exemption ON) | **0** |
| Bridge veto 평가 대상 (exemption OFF) | 1244 |
| OFF 시 `well_supported_connection` | 382 |
| OFF 시 `weak_bridge_candidate` | **862** |
| Exemption 때문에 union된 articulation bridge (양쪽 ≥3노드) | **47** |
| conflict edge 수 ON / OFF | 452 / 1342 |

5k: ON=0 / OFF=1180(weak 773). 10k: ON=0 / OFF=1072(weak 717).

## 3. Bridge-Veto 실효성 판정

Exemption ON 상태에서 bridge veto는 **단 하나의 edge도 평가하지 않는다**. §1의 판정 기준("평가 대상 edge의 대부분 또는 전부가 exemption 대상이라면 seed/merge 분리가 아니라 bridge veto의 실질적 비활성화")을 100%로 충족하므로, 이 수정은 **분리가 아니라 비활성화**로 확정한다.

## 4. Seed/Merge Semantic 분리 (교정 구조)

§3이 정의한 세 연산을 코드 구조로 분리했다.

- **A. Local seed existence** → Phase 1: `seed_strong_edge`만으로 DSU union. 각 결과 component는 자기 local evidence로 독립 seed다.
- **B. Core component merge** → Phase 2: 남은 weak cross-edge를 component pair로 그룹핑, distinct-endpoint aggregate support 요구. 거부돼도 양쪽 seed 유지.
- **C. Single-node growth** → worklog 34의 typed conflict semantics 그대로(변경 없음).

핵심 계약: weak bridge edge는 union에 사용하지 않으며(§4 요구), merge 거부가 seed existence를 제거하지 않는다.

## 5. Edge Category 정의 (§4)

`_classify_core_edge()`로 core-eligible edge를 typed category로 분류한다.

| category | 판정 근거 |
|---|---|
| `consensus_contradicted_edge` | `consensus_state == CONSENSUS_CONTRADICTED` |
| `phase_alias_edge` | `path_status == PATH_PHASE_ALIAS` |
| `oversized_footprint_edge` | normal-separation-over-thickness 및 mutual-tangent-residual 초과 + nearby parallel evidence(worklog 35 gate) |
| `seed_strong_edge` | 위 veto 전부 통과 **AND** `consensus_state == CONSENSUS_WELL_SUPPORTED` |
| `weak_bridge_edge` | veto는 통과했으나 consensus가 well-supported가 아님 |
| `merge_supported_edge` | Phase 2에서 component-pair aggregate support를 만족해 실제 merge에 사용된 edge |

§4의 요구("단순히 raw same_surface라는 이유만으로 seed_strong_edge로 승격하지 마라")를 지켰다 — `CONSENSUS_WELL_SUPPORTED`는 이미 "실제 shared same_surface neighbor 지지 + contradiction ratio 임계값 이하"를 의미하므로 relation class 단독 승격이 아니다.

## 6. Seed Admission 후보 비교 (§6)

S0(worklog 36 baseline) / S1(worklog 37 exemption) / S2(strong-edge local component seed, 채택)를 실측 비교했다. S3(cycle/2-core 필수)는 §6이 명시적으로 경고한 대로 채택하지 않았다 — 희소 sampling된 평면의 valid surface는 tree 형태일 수 있어 cycle 존재를 절대 조건으로 삼으면 안 된다. S2는 topology를 가정하지 않고 **edge evidence 품질**로만 판정하므로 이 함정을 피한다.

## 7. Major/Micro Region 품질 재평가 (§7)

3k 기준 (frozen replay, 동일 input):

| config | region | core_member | ambiguous | max | micro(≤3) | major(>10) | major_coverage |
|---|---|---|---|---|---|---|---|
| A (worklog 36) | 77 | 414 | 1613 | 17 | 8 (10%) | 4 | 57 |
| B (worklog 37) | 182 | 908 | 1137 | 48 | 85 (47%) | 15 | 259 |
| **C (two-phase, 채택)** | **157** | **799** | **1245** | **21** | **62 (39%)** | **11** | **163** |

기하 품질(≥4멤버 region):

| config | region 수 | median diameter | median normal dispersion | **max normal dispersion** |
|---|---|---|---|---|
| A | 69 | 1.493 | 0.0209 | 0.0538 |
| B | 97 | 2.143 | 0.0273 | **0.1333** |
| C | 95 | 1.938 | 0.0240 | **0.0995** |

**§7 질문 답변**: core_member 증가는 major region 확대(major_coverage 57→163, 2.9배)와 작은 component 대량 seed(micro 8→62)가 **둘 다** 기여한다 — 한쪽만이 아니다. Region 77→157 증가의 상당 부분은 이전에 seed 자체를 못 만들던 작은 raw component가 이제 독립 seed를 얻은 결과이며, 이는 §3의 "valid local seed component 보존" 계약상 **의도된 동작**이다. 다만 max normal dispersion이 A의 0.054에서 C의 0.0995로 상승한 것은 일부 region이 더 굽은 표면을 포괄하기 시작했다는 뜻으로, worklog 37의 "core-region coverage 복구" 결론은 **B에 대해서는 철회**하고(bridge veto 무력화로 얻은 수치였음) C에 대해서만 축소된 형태로 유지한다.

## 8. Adversarial Weak-Bridge Controls (§8)

`tests/test_seed_merge_separation.py`에 구현했다. Threshold는 fixture에 맞춰 조정하지 않았다.

- **Two coherent sheets**: gap=1.5로 분리(gap=0.9에서는 affinity graph가 실제로 19개의 genuine same_surface cross edge를 내보내며, coplanar patch가 그만큼 가까우면 하나의 surface인 것이 맞으므로 fixture 자체를 수정했다). 두 sheet 모두 독립 core_member를 얻고 어느 region도 두 sheet에 걸치지 않음을 검증.
- **Articulation bridge rejection**: `box_with_bridge`에서 two-phase는 articulation bridge union **0개**, worklog 37 exemption은 그 이상.
- **Thin slab**: front/back 각각 z부호 검증까지 포함해 완전 분리.
- **Box crease**: 6 face 각각 한 축이 near-flat임을 bounding box로 검증(수직 두 face에 걸친 region 없음).
- **Floater / isotropic contamination**: 오염 노드가 core_member가 되지 않음.

## 9. Shared-Neighbor Semantics (§9)

`bridge_min_shared_neighbor_for_well_supported=2`를 그대로 유지했다. §9가 지적한 혼용 — edge 하나의 shared-neighbor count와 component pair 전체의 merge support가 섞이는 문제 — 를 구조적으로 분리했다:

- **Seed edge admission**: local edge coherence(`CONSENSUS_WELL_SUPPORTED`, per-edge)
- **Component merge**: component-pair aggregate(`merge_min_distinct_endpoint_support`, 양쪽 distinct endpoint 수)

Per-edge bridge veto는 Phase 2에서도 그대로 실행되며, component-pair 요건은 **추가** 조건이므로 legacy 단일 edge 경로보다 결코 느슨해지지 않는다. 고정 threshold를 낮추지 않고 valid seed coverage를 복구할 수 있는지에 대한 §9의 질문에 대한 답은 **가능하다**(core_member 414→799, threshold 변경 0).

## 10. Candidate Rejection Waterfall (§10)

`scripts/devtools/trace_candidate_rejection_waterfall.py`로 `extract_support_termination_candidates`의 전 gate를 순서대로 재현했다. Cylinder 실측:

| first failure | count |
|---|---|
| `generated_genuine_termination` | 78 |
| `insufficient_angular_gap` | 188 |
| `sampling_gap` | 4 |

Region별: region 0(side) 48 생성 / 168 gap 미달, region 1(cap) 16 생성 / 9 미달 / 2 sampling_gap, region 2(cap) 14 생성 / 11 미달 / 2 sampling_gap.

각 행에 measured value / threshold / signed margin(예: `largest_gap_radians`, `gap_threshold_radians`, `gap_margin_radians`, `local_neighbor_count`, `occupied_sector_count`)을 기록한다.

## 11. Region Extent vs Candidate Recall (§11)

- **box_face**: C6/정상 — region이 perimeter에 완전히 도달하고 candidate recall 1.000, closed=1.
- **box (face별)**: **C4** — candidate 16~20개로 충분하고 perimeter에도 도달했으나 compatible directed edge가 10~17개뿐(ring 폐합에 필요한 N개 미만). Isolated candidate 0~3, zero-out-degree 3~6. Candidate generator가 아니라 compatibility 부족.
- **cylinder side**: 정상 — 48 candidate로 24-node ring 2개 폐합.
- **cylinder cap**: **C2** — perimeter에 도달했고 대부분의 candidate가 생성되지만 §13의 gate 결함으로 1~2개가 누락돼 50°대 hole이 남는다.
- **sphere**: **C6** — 물리적 outer boundary가 실제로 없음(closed=0이 정답).

## 12. Analytic Candidate Precision/Recall (§12)

box_face(half-extent 0.48, ground-truth 경계 노드 32개):

| metric | 값 |
|---|---|
| generated candidate | 32 |
| true positive | 32 |
| false positive | **0** |
| false negative | **0** |
| **precision** | **1.000** |
| **recall** | **1.000** |
| largest angular gap | 14.2° |

Sphere: genuine candidate 22개가 생성되지만 closed component 0 — 임의 seam을 만들지 않는다(§12 요구 충족).

## 13. Cylinder Cap Regression (§13) — 근본 원인 규명, 미해결

**정확한 원인**: `extract_support_termination_candidates`의 sector histogram은 bin 경계 jitter 방어를 위해 각 neighbor의 인접 sector까지 occupied로 표시한다(±0.15 bin). Neighbor가 충분히 많으면 **실제로 큰 각도 gap이 있어도 8개 sector 전부가 occupied**가 되어 `_missing_sector_runs`가 빈 튜플을 반환하고, `not runs` 절이 노드를 탈락시킨다 — `gap`이 threshold를 명백히 통과했는데도.

실측(cylinder cap): 노드 257/266은 gap 1.568 / 1.590 rad, threshold 1.178 rad, 그런데 occupied = 8개 전부, runs = (). Accepted neighbor가 5→6으로 하나 늘면 이 상태로 전환된다. Core seeding 개선(worklog 37 B 또는 worklog 38 C 모두)이 cap당 accepted edge를 68→72로 늘리면서 이 전환이 발생했고, candidate가 16→14/12로 줄어 candidate ring에 50~64° hole이 생겨 폐합이 불가능해졌다.

Ablation으로 확인: A(worklog 36) closed=3 / candidate 56·16·16, B(worklog 37) closed=2 / 48·14·12, C(two-phase) closed=2 / 48·14·12. **회귀는 worklog 37 고유가 아니라 "core seeding이 개선되면 발현되는" 잠재 결함**이다.

**미해결 사유**: 시도한 교정 두 가지 — (a) `gap >= width*2.0`를 histogram 독립 충분조건으로 추가, (b) measured gap span에서 missing run을 유도 — 모두 결국 이 fixture의 1.568 rad 바로 위/아래에 오도록 상수를 고르는 문제로 귀결됐다(1.568/width = 1.996으로 `floor`가 1이 되는 등). §13이 "threshold 완화나 강제 loop closure로 복구하지 마라", §19가 "scene-specific tuning 금지"를 명시했으므로 **원복하고 미해결로 보고한다.** 코드에는 진단 내용을 주석으로 남겼다. 올바른 해법은 smeared histogram과 geometric measurement를 원리적으로 화해시키는 것이며 별도 라운드가 필요하다.

## 14. Box Ordering Diagnostic (§14, 진단만)

Directed ordering solver는 변경하지 않았다. Face별 실측:

| face | candidate | compat edges | isolated | zero_out | accepted_core_pairs |
|---|---|---|---|---|---|
| 0 | 19 | 16 | 1 | 3 | 32/342 |
| 1 | 18 | 14 | 1 | 5 | 28/306 |
| 2 | 16 | 10 | 3 | 6 | 22/240 |
| 3 | 19 | 16 | 0 | 3 | 32/342 |
| 4 | 18 | 12 | 3 | 6 | 28/306 |
| 5 | 20 | 17 | 0 | 3 | 36/380 |

Component states: `ambiguous_ordering` 18, `isolated_boundary_candidate` 8. **N-노드 ring에는 N개의 directed edge가 필요한데 모든 face에서 compat edges < candidate**이므로 solver가 무엇을 하든 폐합 불가다. 병목은 `accepted_core_pairs`(가능한 pair의 9~11%만 accepted topology에 존재)이며 candidate generator도 ordering solver도 아닌 **compatibility 단계**다.

## 15. 적용한 Narrow Repair / 원복

| 항목 | 조치 |
|---|---|
| `exempt_intra_raw_component_unions_from_bridge_veto` | **기본값 True → False 원복** (Case A). diagnostic opt-in으로만 유지 |
| `separate_seed_and_merge_phases` (신규, 기본 True) | 2-phase DSU 도입 (Case B/C) |
| `merge_min_distinct_endpoint_support` (신규, 2) | component-pair aggregate merge 요건 |
| `_classify_core_edge()` (신규) | typed edge category (Case C) |
| `_seed_core_components_two_phase()` (신규) | Phase 1 seed / Phase 2 merge |
| Sector-histogram gate | **변경 없음** — 진단 주석만 추가 (§13) |
| `bridge_min_shared_neighbor_for_well_supported` | **변경 없음 (2 유지)** |
| Directed ordering solver | **변경 없음** |
| NURBS fitting | **변경 없음** |

## 16. Independent Ablation (§16, config injection, 파일 스왑 없음)

동일 frozen state에서 config만 주입해 비교했다.

| checkpoint | config | region | core | ambiguous | max | micro | major | runtime |
|---|---|---|---|---|---|---|---|---|
| 3k | A worklog36 | 77 | 414 | 1613 | 17 | 8 | 4 | 0.25s |
| 3k | B worklog37 | 182 | 908 | 1137 | 48 | 85 | 15 | 0.22s |
| 3k | **C two-phase** | 157 | 799 | 1245 | 21 | 62 | 11 | 0.28s |
| 5k | A | 83 | 454 | 1572 | 17 | 6 | 5 | 0.26s |
| 5k | B | 186 | 882 | 1158 | 23 | 90 | 18 | 0.26s |
| 5k | **C** | 148 | 770 | 1270 | 23 | 56 | 9 | 0.27s |
| 10k | A | 67 | 375 | 1661 | 21 | 8 | 6 | 0.24s |
| 10k | B | 175 | 782 | 1260 | 28 | 75 | 9 | 0.21s |
| 10k | **C** | 141 | 689 | 1354 | 28 | 51 | 7 | 0.30s |

D(candidate recall repair only) / E(combined)는 §13의 candidate 결함을 scene-tuning 없이 고칠 방법을 찾지 못해 **적용 대상 수정이 없으므로 실행하지 않았다**(정직한 미실행, 빈 결과를 지어내지 않음).

## 17. Real 3k/5k/10k 결과

§16 표의 C행이 production 상태다. `boundary_component_closed_count` / `materialized_surface_count`는 세 snapshot 모두 여전히 **0** — §18이 명시한 대로 real snapshot의 materialization은 필수 성공 조건이 아니며, §11/§14의 분석대로 병목은 region extent가 아니라 candidate compatibility 단계에 있다.

Positive/negative control 최종 상태: box_face closed=1, cylinder closed=2(cap 1개 미해결), thin_slab closed=2, box_isolated_floater closed=1, box_isotropic_contamination closed=1, box closed=0(C4), sphere closed=0(정답), box_with_bridge closed=0.

## 18. Runtime/Memory

Region formation replay는 config 무관하게 0.21~0.30초(3k/5k/10k). 2-phase 구조는 edge를 두 번 순회할 뿐 새로운 O(N²) 연산이 없다. Frozen state 구축(선택+evidence+affinity)은 3k 기준 29초로 이전과 동일. Full-cloud all-pairs, 신규 unbounded all-pairs, 반복 tensor transfer 없음.

## 19. Focused / Full Pytest

- `tests/test_seed_merge_separation.py`(신규 14 tests): exemption tautology 회귀 증명, typed edge category, 독립 seed 보존, articulation bridge rejection, thin slab / box crease / bridge contamination / floater negative control, box_face·cylinder positive control.
- `tests/test_core_seeding_coverage_repair.py`: worklog 37 계약을 주장하던 테스트를 교정된 계약으로 갱신(exemption 기본값 False 및 two-phase 기본값 True 검증 포함).
- 관련 broader suite(16 파일): 121 passed + 갱신분.
- **Repository-wide pytest(세션 마지막 1회): 672 passed, 1 skipped, 0 failed, 8 subtests passed, 218.67초(3분 38초).**

## 20. 다음 Visible Surface Constructor 병목

1. **Sector histogram과 geometric gap의 원리적 화해(§13, 최우선)**: 정확히 특정된 실제 결함이며 cylinder cap 폐합을 막고 있다. Scene-tuned 상수 없이 두 신호를 일관되게 만드는 설계가 필요하다 — 예컨대 smearing을 occupancy가 아니라 gap 측정 자체에 반영하거나, histogram을 완전히 제거하고 gap + neighbor count만으로 판정하는 방향. 별도 라운드로 승인 필요.
2. **Box의 accepted_core_pairs 희소성(§14)**: face당 가능한 candidate pair의 9~11%만 accepted topology에 존재해 ring 폐합에 필요한 edge 수를 채우지 못한다. Candidate generator도 ordering solver도 아닌 **compatibility/accepted-topology 단계**의 문제로 새로 특정됐다.
3. **Two-phase seeding의 micro-region 비율(§7)**: 10%→39%로 상승했다. 작은 raw component가 독립 seed를 얻는 것은 의도된 동작이지만, 그중 실제 surface fragment와 노이즈를 구분하는 seed admission criterion(§6의 S4/S5 계열)은 이번에 채택하지 않았다.
4. Real snapshot materialization은 여전히 0이며, 위 2번이 해소되기 전까지는 real 데이터에서도 같은 compatibility 병목이 지배적일 것으로 추정된다.
