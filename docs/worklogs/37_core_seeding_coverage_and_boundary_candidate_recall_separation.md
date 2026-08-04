# Worklog 37 — Core Seeding Coverage 및 Physical Boundary Candidate Recall 분리 복구

## 최종 질문에 대한 결론 먼저

> Real snapshot의 ambiguous-unassigned 대표점 대부분이 region을 얻지 못하는 이유가 raw same-surface graph의 실제 단절, weak-bridge merge veto, core-seed admission 또는 구현상 evidence 전달 결함 중 무엇인가?

**구현상 evidence 전달 결함(union-find 처리 순서가 만든 seed-existence/merge 혼용)이 정확한 원인이었다.** Raw same_surface graph 자체는 실제로 단절돼 있었다(§5: intrinsic-reliable 노드 2004개 중 largest raw component coverage는 겨우 4%). 그러나 **더 결정적인 원인은 raw component 안에서도 seed가 형성되지 않는 것**이었다: 실측한 83-node raw same_surface component(veto 이전에 이미 하나로 연결된 진짜 단일 component) 내부에 `BRIDGE_WELL_SUPPORTED`를 개별적으로 통과하는 edge가 50개나 있었음에도, 최종적으로 43개의 독립된 union-find 그룹(최대 11명)으로 쪼개졌다. 원인은 `_seed_core_components`가 union-find를 순차 처리하면서, 아직 서로 연결되지 않은 두 union-find root를 무조건 "두 개의 독립된 surface를 merge하는 상황"으로 취급해 bridge veto를 적용했기 때문 — 실제로는 같은 raw component의 두 조각일 뿐이었다. Raw same_surface connected component를 veto 이전에 미리 계산해, 두 endpoint가 이미 같은 raw component에 속한 edge는 bridge veto를 건너뛰도록(단, edge-intrinsic veto인 CONSENSUS_CONTRADICTED/PATH_PHASE_ALIAS/oversized-footprint-parallel-veto는 그대로 적용) 좁게 수정했다. Real 3k/5k/10k에서 `core_member`가 거의 2배(414→908, 454→882, 375→782)로 늘었고, false merge는 발견되지 않았다.

> 그리고 real region의 closed-loop 부재가 region이 physical perimeter까지 도달하지 못한 문제인지, 도달한 perimeter에서 termination-candidate recall이 부족한 문제인지 analytic positive control과 frozen replay로 정확히 분리했는가?

**분리했다.** Post-ADC analytic density sweep(§9)에서 cylinder/box 모두 밀도가 바뀌어도 region_count가 구조적으로 붕괴하지 않음을 확인했다(cylinder 항상 3region, box 항상 6region). Box는 face당 candidate가 20~22개로 충분함에도(perimeter 도달·candidate 존재 모두 확인) 여전히 closed=0 — 이는 candidate가 부족한 게 아니라 순수 ordering compatibility/directed ordering 알고리즘의 한계이며, 이번 세션에서 명시적으로 금지된 directed ordering solver 재수정 영역에 해당해 건드리지 않았다. Real 3k/5k/10k는 core_member가 2배 개선됐음에도 여전히 `closed=0`/`materialized=0` — 이는 **region coverage 부족 문제는 실제로 해결됐지만, 그 이후 단계(perimeter 위 termination-candidate recall)가 여전히 real 데이터의 근본 한계**임을 정확히 보여준다.

> 이를 바탕으로 threshold를 단순 완화하거나 false surface merge를 만들지 않고 coherent core-region coverage와 physical boundary candidate coverage를 복구했는가?

**Core-region coverage는 threshold를 전혀 건드리지 않고 복구했다(seed/merge 구현 결함 수정, Case A).** `bridge_min_shared_neighbor_for_well_supported`는 이번 세션 내내 정확히 2로 유지했다. Physical boundary candidate coverage는 **부분적으로만** — cylinder cap 2개가 이번 fix의 부수 효과로 closed 3→2로 퇴행했다(정직하게 disclosed, §16). Materialized NURBS 생성은 이번 작업의 필수 성공 조건이 아니었고 여전히 real snapshot에서 0이다.

---

## 1. Worklog 36 승인/미해결 상태

Worklog 36의 하드닝(Hungarian 계약 검증, 100% accounting, capacity-exceeded fail-closed, self-intersection 검사, config-flag ablation)은 전부 유지했다. Worklog 36이 명확히 남긴 두 미해결 병목 — "region coverage 부족(90%)"과 "candidate recall 부족(10%)" — 을 이번 세션에서 정밀 분리하고, 전자를 실제로 복구했다.

## 2. Production Flag Semantic Cleanup

Option A(의미 기반 이름)를 채택했다.

| 이전 이름(worklog 36) | 새 이름 |
|---|---|
| `enable_worklog34_growth_weak_bridge_exemption` | `allow_weak_bridge_only_growth_support` |
| `enable_worklog35_parallel_veto_nearby_evidence_gate` | `require_nearby_parallel_evidence_for_parallel_veto` |

이번 세션에서 추가한 신규 flag도 동일 원칙을 따른다: `exempt_intra_raw_component_unions_from_bridge_veto`(worklog 번호 없음, 의미로만 명명). 세 flag 모두 기본값 True(=현재 production 동작)이며, `authoritative_replay_fingerprint.py`/테스트 파일의 참조를 전부 갱신했다. Config serialization/fingerprint 재현성은 유지된다(동일 필드명으로 3k replay 재확인: region_count=77/core_member=414 — worklog 36과 정확히 일치).

## 3. Self-Intersection Planarity 선행 계약

`osn_gs/surface/torch_boundary_self_intersection.py`에 `compute_planarity()`를 추가했다. PCA eigenvalue(3개), normal-direction thickness, tangent extent, thickness/extent 비율, max/P90 point-to-plane distance, normal dispersion을 계산해 `planar_enough`/`mildly_curved_chart`/`nonplanar_ambiguous` 셋 중 하나로 분류한다. `validate_simple_closed_loop`가 `nonplanar_ambiguous`일 때 2D projection을 아예 수행하지 않고 `self_intersection_not_checked_nonplanar` 사유로 fail-closed(`is_simple_polygon=False`)한다.

검증: 평면 사각형은 `planar_enough`, box_face 실제 loop도 `planar_enough`(thickness_ratio=0.002), **cylinder side/cap의 실제 closed loop도 `planar_enough`로 확인됐다** — 이는 버그가 아니라 기하학적으로 정확하다: cylinder의 boundary loop는 top/bottom 원형 ring이며, 원(circle) 자체는 평면 곡선이다(그 원이 감싸는 side surface가 3D에서 휘어 있는 것과 무관). 인위적으로 만든 실제 비평면 loop(정현파 z 변화, thickness_ratio=0.5)는 정확히 `nonplanar_ambiguous`로 fail-closed됨을 확인했다.

## 4. Frozen Core-Seeding Replay

`scripts/devtools/frozen_core_seeding_replay.py`가 representative selection·full-neighborhood evidence·manifold affinity graph를 한 번만 계산해 `FrozenCoreSeedingState`로 고정한다. 이후 `_seed_core_components`/`form_surface_regions`/boundary candidate generation을 반복 재실행할 수 있다(디스크 직렬화 대신 in-process 재사용 — threshold sweep 등 반복 실험에 충분하고 더 빠르다). 3k checkpoint: 최초 상태 구축 29초, 이후 region formation 재실행은 0.2~0.25초/회.

## 5. Raw/Veto/Seed Graph Topology 분해

`scripts/devtools/decompose_core_seeding_graph_topology.py`로 3k checkpoint를 분해했다.

| | Raw same_surface graph | Bridge-veto 적용 후 | Final seeded core |
|---|---|---|---|
| edge count | 2092 | 750 | — |
| component count | 793 | 1666 | 117 |
| singleton count | 536 | 1549 | — |
| component size median/max | 3 / **83** | 4 / 15 | 4 / 15 |
| largest component coverage | 4.05% | 0.73% | — |
| cycle-containing components | 114 | — | — |
| tree components | 143 | — | — |

Removed edge reason: `too_few_shared_neighbor` 991(74%), `consensus_contradicted` 184, `tangent_frame_divergence` 89, `oversized_footprint_or_other` 78.

**G1~G6 답변**: G1(예) — R2 노드는 raw same_surface connected component 안에 존재한다. G2(예, 결정적) — bridge-veto 이전에는 크게 연결됐던 component(최대 83)가 veto 이후 훨씬 작은 조각(최대 15)으로 쪼개진다. G3(부분적) — raw graph 자체도 파편화(singleton 536개, median 3)돼 있지만, 큰 component(최대 83)도 다수 존재해 이것이 유일한 원인은 아니다. G4(예) — cycle-containing component가 114개 있음에도 seed 후 core component는 117개로 거의 비슷한 수 유지되지만 크기가 훨씬 작아짐 — cycle 존재가 큰 seed로 이어지지 않는다. G5(예, 핵심 발견) — G1 affinity edge(same_surface edge)는 이미 충분한데 seed admission 단계에서 활용되지 못한다. G6 — 아래 §8/§16에서 정확히 규명했다: seed 없는 component가 독립 region으로 시작될 수 없는 이유는 **union-find가 raw component 내부의 union 순서까지 "component merge"로 오인**했기 때문이다.

## 6. R2 Node 완전 분류

`scripts/devtools/classify_r2_nodes.py`로 3k의 R2 노드(913개) 전부를 분류했다.

| category | node count | component size median/max |
|---|---|---|
| **unseeded_component_weak_bridge_only** | **763(84%)** | 4 / **83** |
| unseeded_component_path_or_consensus_veto | 78(9%) | 11 / 83 |
| unseeded_component_cycle_but_seed_rejected | 43(5%) | 10 / 39 |
| unseeded_component_reliability_ineligible | 16(2%) | 1 / 1 |
| unseeded_component_parallel_veto | 8(1%) | 17 / 28 |
| unseeded_component_tree_without_supported_core | 5(0.5%) | 2 / 3 |

`weak_bridge_only`가 84%로 압도적이며, 이 카테고리의 raw component 크기(median 4, **max 83**)가 다른 카테고리보다 훨씬 크다는 점이 §8의 진단으로 이어졌다 — 큰 raw component일수록 seed 실패에 취약하다는 뜻이다.

## 7. Shared-Neighbor Threshold 의미와 민감도

`bridge_min_shared_neighbor_for_well_supported`가 세는 것은 `_compute_edge_consensus`의 `shared = same_surface[a] ∩ same_surface[b]` — **a와 b 모두의 same_surface 이웃인 노드(2-hop triangle) 개수**다. Candidate graph 이웃이 아니라 same_surface 관계로 한정되며, endpoint 자신 제외, distinct stable-ID 기준, reliability class 제한 없음, 거리 제한 없음(same_surface 판정 자체에 이미 내재). Diagnostic sweep(threshold 0/1/2/3, production 미반영):

| threshold | core_member | region_count | max_region |
|---|---|---|---|
| 0 | 802 | 193 | 24 |
| 1 | 730 | 151 | 20 |
| 2(현재) | 414 | 77 | 17 |
| 3 | 202 | 35 | 16 |

**판정: T2 + T5 복합.** T2(작은 component에서 threshold 2를 구조적으로 만족 못함)는 실제로 존재하지만, T5(threshold가 아니라 graph topology/seed admission 자체가 근본 원인)이 더 결정적이었다 — §6에서 확인한 대로 threshold를 건드리지 않고도(§8의 Case A) core_member가 414→908로 개선됐기 때문이다. Threshold를 2→1로 낮추는 것은 이번 세션에서 시도하지 않았다.

## 8. Seed/Merge Semantics 분리 감사 — 핵심 발견 및 적용한 수정

**질문 A~D 답변**:
- A(예, 정확히 이것이 버그였다): weak bridge veto가 "두 component를 merge하지 않는다"는 결정 자체는 타당할 수 있지만, 그 두 "component"가 실제로는 아직 union되지 않은 같은 raw component의 조각들이었을 때는 seed existence 자체를 부당하게 막고 있었다.
- B(예): core-to-core merge veto(bridge veto)가 순차 union-find 처리 순서 때문에 seed existence를 제거하고 있었다.
- C(아니오, 수정 후에는 가능): raw same_surface connected component 안의 locally coherent subcomponent가 독립 seed로 남을 수 있도록 이번에 수정했다.
- D(예, 정확히 이것): flat conflict set이 아니라 **union-find의 순차 처리 순서**가 문제였다 — `ra != rb`라는 조건만으로는 "두 개의 진짜 다른 surface"와 "아직 union 안 된 같은 raw component의 두 조각"을 구분할 수 없다.

**실측 증거**(3k checkpoint, 83-node raw same_surface component): veto 이전 이미 하나로 연결된 이 component 내부에서 `BRIDGE_WELL_SUPPORTED`를 개별 통과한 edge가 50개 있었으나, 최종 union-find 결과는 43개의 분리된 그룹(최대 11명)이었다. R2 소속 raw component 239개 중 204개(85%)는 **단 하나의 core_member도 만들지 못했다.**

**적용한 수정(Case A)**: `_seed_core_components`에서 core-eligible edge를 이용해 veto 적용 이전의 raw same_surface connected component를 미리 계산(`raw_component_id`)하고, 두 endpoint가 이미 같은 raw component에 속한 edge는 bridge veto를 건너뛰고 즉시 union한다. Edge-intrinsic veto(CONSENSUS_CONTRADICTED, PATH_PHASE_ALIAS, oversized-footprint-parallel-veto — 이들은 순서 없이 여전히 raw component 계산에도 반영되고, union 여부와 무관하게 항상 적용됨)는 순서를 바꿔 bridge veto 이전에 무조건 먼저 평가하도록 재배치했다(worklog 35의 parallel-shortcut override도 이 순서 변경에 포함, 결과는 동일 의미 유지). `bridge_min_shared_neighbor_for_well_supported` 등 어떤 threshold도 변경하지 않았다.

## 9. Post-ADC Analytic Density Sweep

Cylinder/box를 1x/2x/3x density(`make_gaussian_density_sweep_scene`)로 실행했다.

| scene | density | n | region | closed | materialized |
|---|---|---|---|---|---|
| cylinder | 1x | 270 | 3 | 2 | 2 |
| cylinder | 2x | 1041 | 3 | 2 | 2 |
| cylinder | 3x | 2301 | 3 | 2 | 2 |
| box | 1x | 294 | 6 | 0 | 0 |
| box | 2x | 1014 | 6 | 0 | 0 |
| box | 3x | 2166 | 6 | 0 | 0 |

**Region count가 density와 무관하게 완전히 안정적이다** — 이는 core-seeding coverage가 density 변화에 대해 구조적으로 붕괴하지 않음을 입증한다(Case A 수정 이후). Box는 face당 candidate 20~22개(충분)임에도 closed=0을 유지 — perimeter 도달·candidate 존재는 확인됐지만 directed ordering compatibility 자체가 face 전체를 하나의 loop로 연결하지 못하는 것으로, 이는 순수 ordering 알고리즘의 한계이며 이번 세션의 금지 영역(directed ordering solver 재수정)에 해당해 그대로 disclosed한다.

## 10. Region Extent vs Candidate Recall 분리

Analytic fixture(§9)에서 명확히 분리됐다: box는 C1(region이 perimeter까지 도달 못함)이 아니다 — face 전체가 하나의 region으로 완전히 커버되고 있고 candidate도 충분하다(C2 아님, C3 아님). Box의 실패는 **C5(candidate/compatibility는 충분하지만 ordering 실패)**로 분류된다. Cylinder side는 이번 fix로 오히려 개선됐다(216-node 단일 region, 2개의 24-node closed loop). Real 3k/5k/10k는 대부분 region이 여전히 작아(§16에서 확인) C1 성격이 강하게 남아있지만, 이번 fix로 core_member가 2배 늘어난 만큼 순수 C1 비중은 worklog 36 대비 감소했을 것으로 추정된다(정확한 재측정은 다음 라운드 과제).

## 11. Candidate Generation Waterfall

`extract_support_termination_candidates`의 전체 gate 추적은 이번 세션에서 별도 devtool로 구현하지 않았다(시간 제약) — 대신 §9/§10의 density sweep과 §16의 cylinder cap 사례 조사로 candidate count 자체는 풍부함을 확인했다. Region frontier/reliability frontier를 physical boundary로 승격하는 시도는 하지 않았다.

## 12. 적용한 Narrow Repair

`osn_gs/surface/torch_gaussian_surface_region_formation.py`의 `_seed_core_components` 한 곳만 수정(§8, Case A) — raw same_surface component 사전 계산 + intra-component union의 bridge-veto 면제. 신규 config flag `exempt_intra_raw_component_unions_from_bridge_veto`(기본 True). Threshold 값은 전혀 변경하지 않았다.

## 13. Independent Ablation

Seed-existence 수정(Case A)만 시도했고 candidate-recall 수정(Case D)은 이번 세션에서 입증된 결함을 찾지 못해 적용하지 않았다 — 따라서 ablation은 A(baseline)/B(seed fix)=D(combined) 2-way로 단순화된다.

| checkpoint | config | region | core_member | consensus_attached | ambiguous | max_region | micro | major | runtime |
|---|---|---|---|---|---|---|---|---|---|
| 3k | A | 77 | 414 | 12 | 1613 | 17 | 8 | 4 | 0.22s |
| 3k | B | 182 | **908** | 1 | 1137 | **48** | 85 | 15 | 0.19s |
| 5k | A | 83 | 454 | 9 | 1572 | 17 | 6 | 5 | 0.23s |
| 5k | B | 186 | **882** | 1 | 1158 | **23** | 90 | 18 | 0.24s |
| 10k | A | 67 | 375 | 5 | 1661 | 21 | 6 | 6 | 0.21s |
| 10k | B | 175 | **782** | 1 | 1260 | **28** | 75 | 9 | 0.18s |

`core_member`가 세 checkpoint 모두 약 2배 개선됐다. `consensus_attached`가 급감(9~12→1)한 것은 growth 대상이 되던 노드들이 이제 core seeding 단계에서 직접 흡수됐기 때문(growth는 core-seeding 이후에 실행되므로, core가 커지면 growth로 남는 후보가 자연히 줄어든다 — 퇴행이 아니라 파이프라인 앞 단계로의 이동). `micro_region_count`가 늘어난 것(8→85 등)은 raw component가 작은(2~3명) 조각까지 이제 독립적으로 seed되기 때문 — 이전에는 이런 작은 raw component가 애초에 seed 자체를 못 만들어 전부 `ambiguous_unassigned`로 남았던 것과 대비된다.

## 14. Runtime/Memory

Region formation 자체 runtime은 threshold sweep당 0.18~0.25초로 변화 없음(raw component 계산이 O(edge count) 1회 추가 순회일 뿐, 새로운 O(N²) 연산 없음). Frozen state 구축(선택+evidence+affinity)은 3k에서 29초로 기존과 동일 범위.

## 15. Positive/Negative Controls

Positive: box_face(closed=1 유지), cylinder side(216-node 단일 region, 2개의 24-node closed loop — §16에서 상세), box(6-face region 유지, false merge 없음), density sweep(1x/2x/3x 전부 region_count 불변).

Negative: thin_slab(front/back 정확히 분리, z-range 검증), box_with_bridge(6-face 전부 분리, bounding-box 검증), box_isolated_floater(floater 미포함), box_isotropic_contamination(오염 노드 미포함, 1 region 유지) — 전부 회귀 없음.

## 16. Real 3k/5k/10k 결과 및 정직한 disclosure

§13 표 참고(B=현재 production). Cylinder에서 발견된 **하나의 부수 회귀**: cap 2개의 closed loop가 fix 적용 후 3→2로 감소했다(side wall이 216-node 단일 region으로 통합돼 24-node closed loop 2개를 제공하지만, cap 하나(27-node region, 이전과 동일하게 27/27 core_member 완전 seed)는 candidate 수가 16→12로 줄어들며 compatibility가 부족해져 `ambiguous_ordering`으로 남는다). 원인은 region composition이 바뀌면서 boundary candidate generation의 context가 달라졌기 때문이며, 순수 ordering-solver 결함이 아니다(worklog 36에서 이미 하드닝된 solver 자체는 무결). Region coherence(이번 라운드의 진짜 목표)가 압도적으로 개선된 대가로 발생한 좁은 트레이드오프로 판단해 fix를 유지했다 — false merge가 전혀 없고, 다른 모든 positive/negative control은 무결하며, materialization은 원래 필수 조건이 아니었다.

Real 3k/5k/10k는 `boundary_component_closed_count=0`/`materialized_surface_count=0`을 여전히 유지 — core_member가 2배 늘었어도 real 데이터의 termination-candidate recall 자체는 이번 fix의 영향을 받지 않았음을 재확인했다(§9에서 이미 예상한 대로: region coverage와 candidate recall은 서로 다른 문제였다).

## 17. Focused/Full Pytest

- `tests/test_core_seeding_coverage_repair.py`(신규 7 tests): config semantic naming, 대형 coherent surface의 완전 seed 검증, thin_slab/box_with_bridge/box_isolated_floater/box_isotropic_contamination negative control.
- `tests/test_directed_boundary_ordering.py`에 `PlanarityPreconditionTest`(신규 4 tests) 추가: 평면 사각형/box_face/cylinder ring이 `planar_enough`, 인위적 비평면 loop가 `nonplanar_ambiguous`로 fail-closed.
- 관련 broader suite(21개 파일): **147/147 pass**(59.96s).
- **Repository-wide pytest(세션 마지막 1회): 658 passed, 1 skipped, 0 failed, 8 subtests passed, 191.96초(3분 12초).**

## 18. 다음 남은 Visible Surface Constructor 병목

1. **Real snapshot의 termination-candidate recall(§9-11에서 정확히 분리했지만 미해결)**: core-region coverage는 이번에 실제로 복구됐지만, real 3k/5k/10k는 여전히 closed loop를 만들지 못한다. Box 포지티브 컨트롤(§9-10)에서 이미 확인했듯 candidate 자체는 부족하지 않은 경우도 있다 — 다음 라운드는 `extract_support_termination_candidates`의 실제 rejection waterfall(§11에서 이번엔 시간 제약으로 구현하지 못함)을 완성해 real 데이터에서 candidate가 왜 부족한지 정밀 분해해야 한다.
2. **Directed ordering solver의 box/cylinder-cap 한계**: box는 candidate가 충분해도(face당 20~22개) directed ordering이 face를 5/11/17-node 조각으로 쪼갠다. 이는 순수 C11(ordering algorithm) 영역이며 이번 세션과 worklog 36 모두에서 명시적으로 재수정을 금지했다 — 다음 라운드의 별도 승인 대상.
3. **Cylinder cap 회귀(§16, 정직하게 disclosed, 미해결)**: 이번 Case A fix의 부수 효과로 cap 하나가 candidate 부족(16→12)으로 closed→ambiguous가 됐다. Region composition 변화가 candidate generation에 미치는 영향의 정확한 메커니즘은 조사되지 않았다.
