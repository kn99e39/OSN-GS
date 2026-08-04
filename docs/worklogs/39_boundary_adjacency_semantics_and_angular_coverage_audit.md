# Worklog 39 — Local Seed Admission Hardening 및 Boundary Evidence Compatibility 복구

## 최종 질문에 대한 결론 먼저

> Worklog 38의 two-phase seed/merge 구조는 valid local surface fragment를 보존하면서 micro-region과 false bridge merge를 동시에 통제할 수 있는가?

**False bridge merge는 통제하지만 micro-region은 통제하지 못한다.** Articulation bridge union은 여전히 0이고 thin_slab/box crease/floater false merge도 없다. 그러나 micro-region 비율은 10%→36~39%로 그대로이며, 이번 라운드에서 이를 억제하는 seed admission criterion을 채택하지 못했다. §16의 canonical 채택 조건 중 "micro-region 폭증이 억제됨"을 충족하지 못하므로 **two-phase는 여전히 provisional이다**(§2의 선택지 2에 해당: 정의 수정 없이는 canonical 확정 불가).

> Sector histogram smearing을 scene-specific 상수 없이 rotation-invariant한 continuous angular coverage로 교체해 cylinder cap regression과 sphere false termination candidate를 함께 해결했는가?

**아니다 — 그리고 그 전제 자체가 틀렸음을 측정으로 확인했다.** Worklog 38은 histogram smearing을 "유효한 geometric gap을 무력화하는 결함"으로 규정했으나, 두 gate를 노드 단위로 전수 비교한 결과 **histogram은 실제로 하중을 지탱하는 gate**였다. 닫힌 구(sphere, 물리적 경계가 존재하지 않음)에서 geometric gap만 쓰면 154개 노드가 통과하지만 histogram이 108개를 veto해 22개로 줄인다. 즉 §8이 제안한 A→B/E 교체(histogram을 diagnostic으로 강등하고 geometric coverage를 canonical로)는 sphere의 false physical candidate를 22개에서 154개로 **7배 악화**시킨다. Histogram은 교체 대상이 아니라 유지 대상이며, 이 판정을 근거와 함께 기록한다.

Cylinder cap regression은 대신 §11의 compatibility 수정으로 해결됐다(closed 2→3, threshold 변경 없음). Sphere의 22개 false candidate는 **미해결**이며, 그 근본 원인은 histogram이 아니라 region fragmentation이었다(§10).

> 그리고 region-seeding용 accepted core topology와 boundary-ordering용 adjacency evidence를 분리해 box/cylinder의 physical boundary compatibility를 false cross-surface connection 없이 복구했는가?

**그렇다. 이번 라운드의 핵심 성과다.** `accepted_core_pair`가 boundary adjacency로 오용되고 있음을 waterfall로 특정하고, §11의 P1(same region + bounded region-graph path)을 채택했다. 결과: **cylinder closed 2→3**(§16 필수 조건 충족), **box closed 0→5**(6면 중 5면), box_face/thin_slab/floater/contamination 전부 유지, **sphere는 여전히 0**(false boundary 미생성). 복구된 11개 closed loop 전부가 self-intersection 검사를 통과했고, bounding box로 검증했을 때 모두 단일 평면 face 위에 머문다(min axis extent ≤0.006). Cross-surface connection은 발생하지 않았다.

---

## 1. Worklog 38 승인/보류 항목

승인 유지: raw-component exemption 원복(tautology 증명), two-phase 구조 자체, `bridge_min_shared_neighbor_for_well_supported=2`, Hungarian 계약, candidate 100% accounting, capacity fail-closed, self-intersection + planarity precondition.

**보류/수정**: worklog 38이 "sector histogram smearing이 유효한 gap을 veto한다 → 원리적 교정 필요"로 남긴 최우선 병목은 이번 측정으로 **오진**이었음이 밝혀졌다(§8). Worklog 38이 2순위로 지목한 "box accepted_core_pairs 희소성"이 실제 병목이었고, 이번에 수정했다(§11).

## 2. Two-Phase Production 상태 — provisional 유지

§2가 요구한 4지선다 중 **2번(정의 수정 후 canonical 유지)의 전제 조건 미충족**으로, 현 상태는 provisional이다. Worklog 36 baseline과 항상 비교 가능하도록 `separate_seed_and_merge_phases` flag는 그대로 유지했다.

| checkpoint | config | region | core | micro(≤3) | major(>10) | major_cov | runtime |
|---|---|---|---|---|---|---|---|
| 3k | A worklog36 | 77 | 414 | 8 (10%) | 4 | 57 | 0.21s |
| 3k | C two-phase | 157 | 799 | 62 (39%) | 11 | 163 | 0.23s |
| 5k | A | 83 | 454 | 6 (7%) | 5 | 68 | 0.21s |
| 5k | C | 148 | 770 | 56 (38%) | 9 | 139 | 0.27s |
| 10k | A | 67 | 375 | 8 (12%) | 6 | 85 | 0.19s |
| 10k | C | 141 | 689 | 51 (36%) | 7 | 108 | 0.20s |

Major-region coverage는 2.9배(57→163) 증가했으나 micro-region 비율도 4배 가까이 증가했다. §16 기준상 "micro-region 폭증 억제" 미충족.

## 3-4. Local Seed Admission 분석 — 미완

시간 배분상 §3의 전수 component 감사(cycle rank, 2-core, articulation point, edge-disjoint path 등)와 §4의 S2~S7 후보 비교를 수행하지 못했다. Micro-region 원인 분해(coherent_sparse_surface_fragment vs weak_chain_fragment 등)도 미완이다. **정직하게 미수행으로 보고하며**, §16 판정은 이 미완을 근거로 provisional을 유지했다(측정 없이 canonical 채택하지 않음).

## 5. Merge Aggregate Support — 변경 없음

`merge_min_distinct_endpoint_support=2`를 유지했다. 이번 라운드에서 M0~M5 비교를 수행하지 않았으므로 값 변경도 하지 않았다.

## 6. Adversarial Weak-Bridge Fixture — worklog 38 상태 유지

Worklog 38에서 추가한 fixture(two separated sheets, articulation bridge, thin slab, box crease, floater, contamination)는 전부 통과 상태를 유지한다. §6이 요구한 gap/local_graph_scale 비율 sweep은 이번에 추가하지 못했다.

## 7. Real 3k/5k/10k Candidate Rejection Waterfall — 부분 수행

Worklog 38에서 구현한 `trace_candidate_rejection_waterfall.py`가 존재하며 analytic scene에 대해 동작한다. Real 3k/5k/10k 전체에 대한 전수 실행(§7의 전체 필드 기록 포함)은 이번에 완료하지 못했다. 대신 real snapshot의 candidate 수는 authoritative replay로 확인했다(3k 153, 5k 181, 10k 121).

## 8. Sector Histogram vs Continuous Angular Coverage — **교체하지 않음, 근거 기록**

`scripts/devtools/audit_angular_coverage_semantics.py`로 모든 analytic fixture에서 두 gate를 노드 단위 비교했다.

| scene | 평가 노드 | both_accept | both_reject | **geometric accepts / histogram vetoes** |
|---|---|---|---|---|
| box_face | 81 | 32 | 49 | **0** |
| box | 256 | 124 | 63 | **69** |
| cylinder | 270 | 78 | 186 | **6** |
| **sphere** | 154 | 46 | 0 | **108** |

**결정적 관측**: sphere는 닫힌 곡면으로 물리적 경계가 어디에도 없다. 그런데 geometric gap 단독으로는 154개 노드 전부가 통과한다(46 both_accept + 108 histogram-vetoed). Histogram이 108개를 막아 22개로 줄이는 것이다. 따라서 §8이 제안한 "histogram을 diagnostic으로 강등하고 continuous geometric coverage를 canonical로"(후보 E) 또는 "exact sorted circular gap"(후보 B)를 채택하면 sphere의 false physical candidate가 22 → 154로 악화된다.

Histogram은 worklog 38이 규정한 "measurement를 무력화하는 결함"이 아니라, sampling 밀도가 높아 인접 이웃이 각도상 뭉칠 때 발생하는 **가짜 gap을 걸러내는 실제 방어 장치**다. Bin count와 smearing multiplier를 다른 상수로 바꾸지 않았고, 교체도 하지 않았다. 후보 C/D/F(neighbor별 angular uncertainty interval union)는 이번에 구현·비교하지 못했다 — histogram이 유지 대상으로 판정된 이상 우선순위가 낮아졌다.

## 9. Continuous Angular Coverage Controls — 미수행

§8에서 교체를 채택하지 않았으므로 §9의 continuous formulation 전용 control은 실행 대상이 없어졌다. 기존 gate에 대한 positive/negative control은 §17에 기록했다.

## 10. Sphere Genuine Candidate 22개 감사 — **false positive 확정, 미해결**

Sphere의 22개 `observed_support_termination` candidate를 감사한 결과:

- Sphere는 **2개 region(99/90 members)으로 fragmentation**된다.
- 22개 candidate가 region별 11/11로 정확히 양분되며, 위치는 z∈[0.084,0.293]과 z∈[-0.292,-0.084] — **두 region의 seam에 정확히 집중**된다(구 전체 z 범위는 [-0.299,0.298]).
- 각 candidate는 자기 support radius 안에 **같은 region 이웃 ~25개, 다른 region 이웃 ~26개**를 가진다. 즉 "support가 없다"고 판정된 방향은 실제 관측 Gaussian으로 가득 차 있으며, 그 Gaussian들이 단지 다른 region id를 가질 뿐이다.

**판정: `nonphysical_region_frontier`가 `observed_support_termination`으로 승격되고 있다.** 이는 §15가 금지하는 "region frontier를 physical boundary로 사용"에 해당하는 현행 결함이다.

**시도한 수정과 원복**: 선택된 outward 방향 arc 안에 out-of-region 관측 support가 있으면 `reliability_frontier`로 강등하는 guard를 구현하고 측정했다. Sphere는 22→0으로 정확히 고쳐졌으나, **box 110→0, cylinder 74→0(closed 2→0), thin_slab 48→3**으로 genuine candidate를 함께 파괴했다. 다면체 solid에서는 실제 physical patch boundary가 진짜 crease를 사이에 두고 다른 region과 정당하게 접하기 때문이다. Out-of-region support만으로는 "이 표면이 계속된다"와 "다른 표면이 여기서 만난다"를 구분할 수 없다 — 구분하려면 affinity graph가 이미 계산하는 crease/parallel relation evidence를 이 단계까지 전달해야 하며, 이는 guard patch가 아니라 구조 변경이다. 원복하고 코드에 진단을 주석으로 남겼다.

## 11. Accepted-Core-Pair Semantic 감사 — **결함 확정 및 수정(Case D)**

§11의 질문에 대한 답:

- **A.** `accepted_core_pair`(=`internal_accepted_edge_ids`)는 **region seeding/merge를 위한 topology evidence**다. `by_pair`(bounded-kNN affinity candidate graph)에서 생성되므로 `candidate_neighbor_count` 희소성을 그대로 상속한다.
- **B.** 두 boundary candidate가 perimeter상 이웃이라는 증거가 **아니다**. 서로 다른 질문에 답하는 값이다.
- **C.** 금지할 이유가 없다 — 이것이 결함이었다.
- **D.** 그렇다. 측정: box의 perimeter-adjacent pair 중 direct accepted edge가 없는 45개가 **전부(45/45) 2-hop region-graph path로 도달 가능**하다(3-hop 필요 0개, path 없음 0개).
- **E.** Direct affinity edge가 아니라 **bounded local path support**가 필요하다.

**채택: P1(same region + bounded region-graph path)**, 단 §12의 측정에 따라 경로 중간 노드를 **non-candidate interior node로 제한**했다.

## 12. Boundary Compatibility Waterfall

`scripts/devtools/trace_boundary_compatibility_waterfall.py`로 analytic-adjacent pair(서로의 최근접 2개 후보 안에 드는 쌍 = ring이 실제로 필요로 하는 연속 쌍)의 first-failure를 분해했다.

| scene / region | adjacent pairs | accepted | non_forward | **no_direct_accepted_core_pair** |
|---|---|---|---|---|
| box_face r0 (closed=1) | 64 | 32 | 32 | **0** |
| box r0 | 38 | 16 | 16 | **6** |
| box r1 | 36 | 14 | 13 | **9** |
| box r2 | 32 | 10 | 11 | **11** |
| box r3 | 38 | 16 | 16 | **6** |
| box r4 | 36 | 12 | 14 | **8** (+2 tangent) |
| box r5 | 40 | 17 | 18 | **5** |

box_face(폐합 성공)는 `no_direct_accepted_core_pair` 손실이 **0**이고 accepted 32 / non_forward 32의 정확한 50:50 분할을 보인다(무방향 인접 1개당 정방향 1 + 역방향 1). Box의 각 면은 5~11개를 이 gate에서 잃어 N-node ring에 필요한 N개 edge를 확보하지 못한다.

## 13-14. 적용한 Repair 및 Analytic 결과

**수정 내용**(`torch_directed_boundary_ordering.py`):
- `_build_accepted_adjacency()`: region-internal accepted edge의 무방향 인접 구조.
- `_has_region_topology_support()`: direct accepted edge **또는** 공유 이웃을 통한 2-hop path. 단 공유 이웃이 **boundary candidate가 아닌 interior node**여야 한다.
- 기하 gate(forward/distance/lateral/tangent/normal)는 전부 그대로. Threshold 변경 0. Cross-region 연결 없음. Directed Hungarian objective와 cycle decomposition 미변경. NURBS fitting 미변경.

**non-candidate 제한이 필수인 이유**(측정으로 확인): 공유 이웃을 제한하지 않으면 Y-junction 음성 통제가 깨진다. 반지름 0.6의 **내부** stub(ring은 반지름 1.0)이 ring node 0과의 단일 accepted edge만으로 node 1과 "인접"해져 13-node cycle에 삽입됐다 — perimeter가 아닌 노드를 지나는 가짜 physical boundary. Perimeter상 연속인 두 candidate 사이에는 내부 표면이 있으므로 bridging evidence가 interior node인 반면, branch stub의 유일한 경로는 다른 candidate를 지난다. 이 비대칭이 판별식이다.

**Analytic 결과**:

| scene | closed (before → after) | materialized |
|---|---|---|
| box_face | 1 → 1 | 1 |
| **box** | **0 → 5** | 5 |
| **cylinder** | **2 → 3** | 3 |
| sphere | 0 → 0 | 0 |
| thin_slab | 2 → 2 | 2 |
| box_with_bridge | 0 → 5 | 5 |
| box_isolated_floater | 1 → 1 | 1 |
| box_isotropic_contamination | 1 → 1 | 1 |

복구된 11개 closed loop 전부 `validate_simple_closed_loop` 통과(self-intersection 0). Bounding box 검증에서 전부 단일 평면 face에 머묾(min axis extent 0.0023~0.0059). thin_slab의 두 loop는 z부호로 완전 분리 확인.

## 15. Real Region 병목 재분류 — 부분 수행

§14가 요구한 R1~R6 전수 분류는 완료하지 못했다. 확인된 것: real 3k/5k/10k는 candidate 153/181/121개를 생성하고 region 157/148/141개를 형성하지만 closed=0이다. Compatibility 수정 적용 중간 단계에서 3k가 일시적으로 closed=1(3-node triangle, boundary_residual 0.033)을 만들었으나, Y-junction 안전 수정 이후 다시 0이 됐다 — 그 loop가 candidate-to-candidate 2-hop 경로로 성립했기 때문이며, 안전 제약이 우선이므로 올바른 결과다.

## 16. 적용한 narrow repair 요약

| 항목 | 조치 |
|---|---|
| Boundary adjacency gate | direct accepted edge → direct **또는** non-candidate interior node 경유 2-hop (Case D) |
| Sector histogram | **변경 없음** — 교체 제안이 오진이었음을 측정으로 확인, 유지 판정 |
| Region-frontier guard | 구현·측정 후 **원복**(genuine candidate 파괴) |
| Threshold 전체 | **변경 없음** |
| Directed solver / NURBS fitting | **변경 없음** |

## 17. Positive/Negative Controls

Positive: box_face(closed=1), box 6면 중 5면(closed=5), cylinder side+2caps(closed=3), thin_slab(closed=2), floater/contamination(closed=1) — 전부 통과. 복구 loop 전부 simple polygon.

Negative: Y-junction stub 미삽입(전용 테스트), sphere false physical boundary 0, thin_slab 양면 미병합, box crease 횡단 없음, box_with_bridge false merge 없음, articulation bridge union 0(worklog 38 계약 유지), floater/contamination 미편입.

## 18. Runtime/Memory

Region formation replay 0.19~0.27초로 변화 없음. 2-hop 지원 판정은 region-local accepted adjacency의 집합 교집합 1회로 O(degree)이며, bounded-k 구조를 그대로 재사용한다. Full-cloud all-pairs, 신규 unbounded all-pairs, 반복 tensor transfer 없음. Real snapshot 전체 runtime 29.7/32.1/49.4초로 기존 범위 유지.

## 19. Focused/Full Pytest

- `tests/test_boundary_adjacency_semantics.py`(신규 14 tests): region topology support 계약(direct / 2-hop via interior / 2-hop via candidate 거부 / no path / 3-hop 거부), Y-junction stub 미삽입, cylinder side+2caps, box 5면 복구, box_face 유지, sphere false boundary 0, box/box_with_bridge/thin_slab의 단일 face 제약, 전 loop simple polygon 검증.
- `tests/test_directed_boundary_ordering.py`: 30/30 통과(Y-junction 음성 통제 포함).
- **Repository-wide pytest(세션 마지막 1회): 686 passed, 1 skipped, 0 failed, 8 subtests passed, 172.52초(2분 53초).**

## 20. 다음 Visible Surface Constructor 병목

1. **Sphere region fragmentation (§10, 최우선)**: 닫힌 구가 2개 region으로 쪼개지고 그 seam이 22개 false physical candidate를 만든다. 근본 해법은 candidate 단계 guard가 아니라 (a) sphere가 하나의 region으로 형성되게 하거나, (b) affinity graph의 crease/parallel relation evidence를 termination 판정까지 전달해 "다른 표면이 만나는 곳"과 "같은 표면이 계속되는 곳"을 구분하는 것이다.
2. **Micro-region 억제 criterion (§3/§4 미완)**: two-phase는 micro-region 비율을 10%→36~39%로 올린 채로 남아 있고, 이를 억제하는 evidence-normalized seed admission을 아직 채택하지 못했다. 이것이 해결되기 전까지 two-phase는 provisional이다.
3. **Real snapshot closed loop**: analytic box/cylinder는 이번에 구조적으로 폐합 가능해졌으나 real 3k/5k/10k는 여전히 0이다. §7의 real waterfall 전수 실행이 다음 라운드의 선결 과제다.
4. **Box 6번째 face**: 5면은 폐합하나 1면(region 4, tangent_misaligned 2건 포함)은 여전히 미폐합이다.
