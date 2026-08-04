# Worklog 35 — Core Region Consolidation 및 Directed Boundary Cycle Recovery

## 최종 질문에 대한 결론 먼저

> Candidate가 충분한 physical boundary에서 current mutual-agreement + greedy ordering이 simple closed cycle을 복원하지 못하는 정확한 이유는 무엇이며, 이를 open chain이나 false boundary를 강제로 닫지 않는 deterministic cycle recovery로 복구했는가?

**정확한 원인을 밝히고 복구했다(C11).** box_face(27-member region, genuine candidate 19개)를 frozen replay로 정확히 재현한 결과, 19개 노드 중 13개가 forward-compatible successor를 2개 이상 가지고 있었다(§5). 각 노드가 독립적으로 자신의 최고 점수 successor를 고르는 greedy 방식은 지역적으로만 최적이라 인접한 두 노드가 서로 다른 세 번째 노드를 선호하는 상황이 발생했고, 그 결과 forward/backward mutual agreement가 19개 중 11개 edge에서만 성립했다. 나머지는 score 순 greedy augmentation으로 채워졌는데, 이 과정에 "하나의 ring을 유지"하는 메커니즘이 전혀 없어 6개의 조각(node count [1,2,3,5,7,1])으로 쪼개졌다. 이를 **동일한 compatibility edge 집합 위에서 수행하는 정확한 최대-가중치 one-in/one-out 매칭(Hungarian algorithm)**으로 교체해, in-degree≤1/out-degree≤1을 구조적으로 보장하고 결과를 disjoint simple cycle/path로 분해했다. box_face는 15-node closed loop + 3-node 잔여 open path로 복구됐고(6조각 → 1 closed + 1 open), `materialized_surface_count`가 0→1이 됐다. Open chain/Y-junction/두 개의 분리된 loop/중복 edge/sparse gap 등 모든 negative control에서 강제 폐쇄가 발생하지 않음을 직접 검증했다(§6).

> 또한 real snapshot의 same-surface region이 작은 core cluster로 남는 원인이 weak-bridge/merge evidence semantics에 있는지 입증하고, false surface merge 없이 coherent major region과 physical boundary candidate coverage를 복구했는가?

**입증했고, 하나의 실제 결함을 좁게 수정했다(C9).** `_seed_core_components`의 weak-bridge veto 자체(§6의 두 threshold: `bridge_min_shared_neighbor_for_well_supported`, `bridge_tangent_divergence_threshold`)는 대체로 정당했다 — real 3k snapshot에서 cross-component pair의 83%(925/1109)는 실제로 단 하나의 fragile bridge edge뿐이었다(진짜 weak bridge, Case D, 건드리지 않음). 그러나 **2개 이상의 cross edge를 가진 184개 pair 중 56개(30%)는 개별 edge 중 최소 하나가 `_evaluate_bridge_veto`를 통과(`well_supported_connection`)했음에도 최종적으로 병합되지 않았다.** 원인을 정확히 추적한 결과: `_seed_core_components`의 별도 parallel-shortcut override(§9-10)가 `normal_direction_separation_over_thickness`(개별 Gaussian의 `normal_thickness`로 정규화된 raw metric — worklog 30-33이 이미 real 데이터에서 사용 불가로 판정한 것과 동일 계열의 양)를 고정 threshold "4.0"으로 재검사해, bridge veto를 이미 통과한 edge를 다시 vetoing하고 있었다. 실측: 이 metric은 same_surface 분류 edge(median 108)와 parallel_separate 분류 edge(median 645) 사이에 극심한 overlap이 있어 어떤 고정 threshold로도 두 클래스를 구분하지 못한다(§10). Override의 주석 자체가 "nearby parallel-separated evidence"를 요구한다고 명시했지만 코드는 그 nearby 조건을 전혀 검사하지 않았다 — 이미 계산되어 있던 `consensus.contradicting_parallel_neighbor_count > 0` 조건을 추가해 주석이 원래 의도한 대로 동작하게 좁게 수정했다. Real 3k에서 `consensus_attached` 1→12, 5k 0→9, 10k 0→5, 최대 core component 크기(3k 15→17, 5k 15→18, 10k 16→22)가 개선됐다. Threshold를 완화하거나 core-to-core 병합 정책 자체를 바꾸지 않았고, negative control(thin_slab/box_with_bridge/box_isotropic_contamination) 전부 false merge 없음을 확인했다.

---

## 1. C9/C11 기존 상태

Worklog 34가 남긴 두 원인이 그대로 확인됐다: real 3k/5k/10k는 region당 genuine candidate가 1~2개뿐인 C9(구조적 candidate 부족), post-ADC box_face(27-member, candidate 19개, same-region partner median 18)는 candidate가 충분한데도 directed ordering이 6조각으로 쪼개는 C11이다. Worklog 34의 growth-loop 수정(weak-bridge veto를 growth에 재사용하지 않도록 한 것)은 이번 세션에서도 그대로 유지했다 — `torch_gaussian_surface_region_formation.py`의 해당 코드는 이번 diff에 포함되지 않았다.

## 2. Frozen Boundary Candidate Replay

`scripts/devtools/trace_c11_box_face_ordering.py`가 `construct_visible_nurbs_from_gaussians`의 `boundary_halfedge_candidates`/`accepted_local_topology`를 그대로 사용해 `_recover_directed_boundary_components`의 forward/backward/mutual 단계를 독립적으로 재현한다. Candidate generation·region formation·reliability는 재실행하지 않는다.

## 3. Current Ordering Failure Decomposition

Box_face(cap=27 강제 다운샘플, 프롬프트가 지정한 worklog 33/34와 동일 재현 조건)에서 실측:

| 단계 | 값 |
|---|---|
| physical boundary candidate | 19 |
| same-region pair | 342 |
| accepted core edge pair | 118 |
| distance-compatible | 61 |
| lateral-compatible | 61 |
| tangent-compatible | 46 |
| normal-compatible(=candidate) | 46 |
| forward-compatible directed edge | 18 |
| backward-compatible directed edge | 19 |
| mutual-agreement edge | 11 |
| nodes with >1 forward candidate | 13 |
| nodes with >1 backward candidate | 13 |
| forward/backward disagree(둘 다 존재) | 7 |

기존 output: node_count [1,2,3,5,7,1] (6조각), `boundary_component_closed_count=0`, `materialized_surface_count=0`.

**판정: O4(경쟁 successor 다수) + O1(forward/backward local optimum 불일치)의 복합.** 19개 중 13개 노드가 2개 이상의 지리적으로 타당한 successor를 가지며, greedy 방식이 지역 최적만 보장해 mutual agreement가 11/19에서만 성립했다. O2(mutual-only 조건 과도한 제거)는 아니다 — mutual 자체는 상당수(11/19) 성립했고, 문제는 greedy augmentation이 남은 8개를 하나의 ring으로 묶지 못한 것이다. O3(greedy 처리 순서)도 부분 원인이지만 근본 원인은 애초에 알고리즘에 "하나의 전역 cycle"이라는 제약이 전혀 없었다는 것(O8 복합으로 최종 판정).

## 4. Ordering 후보 비교

A(기존 mutual+greedy) vs B(deterministic cycle extraction) vs C(one-in/one-out matching) vs D(bounded exact search) 중, **C: one-in/one-out constrained matching(Hungarian algorithm)**을 채택했다. 근거:
- B(단순 deterministic cycle extraction)만으로는 "어떤 edge를 선택할지"의 근본 문제가 남는다 — matching이 이 선택 자체를 전역 최적화한다.
- D(bounded exact/branch-and-bound)는 C가 이미 전역 최적(같은 compatibility edge 집합에서)이므로 별도로 필요 없다 — one-in/one-out 매칭 자체가 정확히 이 문제의 최적해다.
- Region-centroid polar-angle 정렬, convex hull, nearest-neighbor 강제 연결, 마지막-첫 endpoint 강제 연결, shape별 분기는 모두 금지 목록에 해당해 배제했다.

## 5. 선택한 Deterministic Cycle Recovery

`osn_gs/surface/torch_directed_boundary_ordering.py`를 재작성했다. Region별로:
1. 기존과 동일한 compatibility gate(same-region, accepted core edge, forward>0, distance≤1.6×local_spacing, lateral≤0.9×local_spacing, tangent≥-0.15, normal≥0.45)로 directed edge 전체 집합을 구성(단일 selection이 아니라 전체 compatibility graph).
2. 이 edge들 위에서 O(n³) Hungarian algorithm으로 최대 총점 one-in/one-out matching을 계산 — "edge 없음"은 무한대 비용(선택 불가)으로 취급, 강제 매칭 없음.
3. 결과를 disjoint simple cycle/path로 분해 — self-loop/1·2-node 순환은 closed로 인정하지 않음(open path로 남김).
4. Region당 candidate 400개 초과 시 안전장치로 deterministic greedy fallback(실측 O(n³) 비용 n=200에서 3.2초, n=400에서 27초 — 실제 관측된 어떤 region도 32개를 넘지 않으므로 이 경로는 현재 도달되지 않는다).

`recover_directed_boundary_components`의 기존 순방향/역방향 tangent 비교(unoriented surface 대응)는 그대로 유지했다.

## 6. C11 Positive/Negative Control

Positive:
- box_face(cap=27): 6조각 → **1 closed 15-node loop + 1 open 3-node** 잔여, `materialized_surface_count` 0→1.
- box_face(비다운샘플, 81점): 이미 성공하던 케이스(1 closed 32-node loop) 유지.
- cylinder(비다운샘플): side+2 caps = 3 closed loop 유지(worklog 33 positive control과 동일 결과).
- Concave/두 개의 분리된 loop: 합성 halfedge fixture로 직접 테스트(§ 아래), 병합 없이 2개의 8-node closed loop로 정확히 분리됨.

Negative(`tests/test_directed_boundary_ordering.py`에 영구 테스트로 반영):
- Open chain(edge 1개 제거): closed=0 유지, 강제 폐쇄 없음.
- Sparse gap(연속 2개 노드 제거): closed=0 유지, bridging 없음.
- Y-junction(가지 노드 추가): 가지 노드가 어떤 closed loop에도 포함되지 않음(matching 구조상 degree>1인 branch는 애초에 cycle에 들어갈 수 없음).
- 중복/역방향 중복 accepted pair: `accepted_pairs`가 frozenset 집합이라 자연히 중복 제거, 결과 영향 없음.
- thin_slab/box_isolated_floater/box_isotropic_contamination/box_with_bridge/box/sphere: 전부 기존(A) 대비 회귀 없음(§12 ablation 표 참고), false merge/false closed loop 없음.

## 7. Source/NURBS Boundary Accuracy

Box_face(cap=27) 복구된 15-node closed loop의 world-space 좌표에서 total turning angle을 측정 — 정확히 ±2π(6.283 rad)로, self-intersection 없는 단순 폐곡선임을 직접 확인했다(`scripts/devtools/verify_c11_loop_geometry.py`). z-표준편차 0.001로 평면 face에 정확히 위치. NURBS materialization 후 `boundary_residual=0.054`(face half-extent ~0.48 대비 낮은 잔차, 6×6 control grid LSQ fitting 해상도에 부합) — **source loop 자체가 정확했고, fitting 단계도 정상 동작**함을 확인했다. Ordering 수정과 fitting 수정을 섞지 않았다(fitting 코드 미변경).

## 8. Core Component Merge Trace

`scripts/devtools/trace_c9_core_component_merge.py`/`trace_c9_all_cross_edges_per_pair.py`로 real 3k snapshot의 `_seed_core_components` 전체를 추적했다.

| 항목 | 값(수정 전) |
|---|---|
| core-eligible edge | 2092 |
| bridge_state: well_supported_connection | 542 |
| bridge_state: weak_bridge_candidate | 1093 |
| weak-bridge reason: too_few_shared_neighbor | 995 (91%) |
| weak-bridge reason: tangent_frame_divergence | 270 |
| weak-bridge reason: local_cut_splits | 66 |
| cross-component pair, exactly 1 edge | 925 (83%) |
| cross-component pair, ≥2 edge | 184 |
| **≥2-edge pair에서 개별 edge가 bridge veto를 통과했지만 병합 안 됨** | **56/184 (30%)** |

## 9. Weak-Bridge Rejection 분포 — R1~R7 판정

- **R1(불필요한 independent-support 요구)**: 아니다 — 91%는 진짜 shared-neighbor 부족(threshold 2 미달)이며 이는 `bridge_min_shared_neighbor_for_well_supported`를 감으로 낮추지 말라는 지시에 해당해 건드리지 않았다.
- **R2(진짜 fragile single bridge)**: 맞다 — 83%(925/1109)는 실제로 단 하나의 edge뿐이다. 이는 진짜 fragile bridge이므로 그대로 veto 유지(Case D).
- **R3(고정 threshold가 구조적으로 달성 불가능)**: 부분적으로 관련되나, 이번 라운드에서 threshold 자체는 변경하지 않았다.
- **R4(G1 evidence가 merge evidence에 제대로 전달되지 않음)**: **맞다, 이것이 실제 근본 원인이었다.** 56/184 pair는 G1 기반으로 계산된 개별 edge가 `_evaluate_bridge_veto`를 정상 통과했음에도, 별도의 parallel-shortcut override(§10)가 다시 vetoing해 union이 일어나지 않았다.
- **R5(path/consensus/oversized-footprint veto가 실제 지배적)**: 아니다 — bridge_state 분포상 `too_few_shared_neighbor`가 압도적이며, 이번에 수정한 override는 소수(56/184)에만 영향을 미쳤다.
- **R6(growth 후에도 남는 mergeable component)**: 이번 라운드의 fix가 R4를 해소해 일부 해당 사례를 이미 흡수했다.
- **R7(candidate generator recall 부족이 별도로 존재)**: 이번 세션에서 candidate generator(affinity graph candidate 단계) 자체는 건드리지 않았으며, `core_eligible_edge_count=2092`가 이미 상당한 양이므로 candidate 부족이 R4의 원인은 아니다.

## 10. C9 Root Cause — 정확한 코드 위치

`_seed_core_components`(torch_gaussian_surface_region_formation.py) 내 parallel-shortcut override:

```python
if (
    direct_metrics.normal_direction_separation_over_thickness
    > config.bridge_normal_separation_with_parallel_veto   # 고정 4.0
    and direct_metrics.mutual_tangent_residual
    > config.bridge_borderline_tangent_residual_veto
):
    boundary_conflict_edges.add((a, b))
    continue
```

`normal_direction_separation_over_thickness`는 `frame.normal_thickness`(개별 Gaussian footprint 두께)로 정규화된 raw metric — worklog 30-33이 이미 real 장기학습 데이터에서 사용 불가로 판정한 것과 동일 계열이다. 실측(3k checkpoint): same_surface로 이미 분류된 edge의 이 값은 median 108(범위 0.09~4453), parallel_separate로 분류된 edge는 median 645 — **두 분포가 심하게 겹쳐 어떤 고정 threshold로도 구분되지 않는다.** 주석은 "A direct pair that has a large normal-direction separation AND **nearby parallel-separated evidence**"라고 명시했지만, 코드는 "nearby parallel-separated evidence"를 전혀 검사하지 않고 direct pair 자신의 raw metric만 봤다.

## 11. 적용한 Region Repair

```python
and consensus.contradicting_parallel_neighbor_count > 0
```

한 줄을 override 조건에 추가했다. `consensus.contradicting_parallel_neighbor_count`는 이미 `_compute_edge_consensus`가 계산해 놓은, a/b의 공통 candidate 이웃 중 실제로 parallel_separate 관계를 가진 개수 — override 주석이 원래 요구했던 "nearby parallel-separated evidence"를 정확히 구현한다. Case A(merge-conflict/growth-conflict/boundary-conflict 역할 분리)는 worklog 34가 이미 완료했으므로 이번엔 해당 없음; Case B(threshold를 상대값으로 바꾸는 것)는 시도하지 않음(threshold 자체는 안 건드림); **Case C(evidence가 제대로 전달되지 않는 결함)**에 해당하며, 이미 계산된 값을 재사용했을 뿐 새 연산을 추가하지 않았다.

## 12. C11-only / C9-only Ablation

`scripts/devtools/run_c11_c9_ablation.py`로 4개 조합(A baseline / B C11-only / C C9-only / D combined)을 real 3k/5k/10k + box_face(cap=27)에 대해 각각 독립 실행(파일 스왑 방식, 작업 트리는 최종적으로 D로 복원 확인).

| config | 3k region/core/consensus_attached/micro≤3 | 5k 동일 | 10k 동일 | box_face closed/materialized |
|---|---|---|---|---|
| A baseline | 70/362/1/15 | 84/431/0/16 | 63/344/0/12 | 0/0 |
| B C11-only | 70/362/1/15(불변) | 84/431/0/16(불변) | 63/344/0/12(불변) | **1/1** |
| C C9-only | 77/414/**12**/**8** | 83/454/**9**/**6** | 67/375/**5**/**8** | 0/0(불변) |
| D combined | 77/414/12/8 | 83/454/9/6 | 67/375/5/8 | **1/1** |

**두 수정이 완전히 독립적으로 기여함을 확인했다** — B는 real snapshot의 region formation에 전혀 영향을 주지 않고(A와 완전 동일), C는 box_face ordering에 전혀 영향을 주지 않으며(A와 완전 동일), D는 두 이득을 동시에 정확히 재현한다(C의 region 결과 + B의 box_face 결과, 상호작용 비용 없음). Runtime도 세 구성 모두 유사한 범위(3k 25~26초, 5k 29~30초, 10k 43~46초)로 재앙적 증가 없음.

## 13. Real 3k/5k/10k Combined Replay 결과

| iteration | region_count | core_member | consensus_attached(전→후) | region_member_max(전→후) | micro≤3(전→후) | boundary_component_count(전→후) | closed | materialized | runtime |
|---|---|---|---|---|---|---|---|---|---|
| 3000 | 77 | 414 | 1→12 | 16→17 | 15→8 | 3→4 | 0 | 0 | 25.9s |
| 5000 | 83 | 454 | 0→9 | 17→18 | 16→6 | 19→21 | 0 | 0 | 29.6s |
| 10000 | 67 | 375 | 0→5 | 21→22 | 12→8 | 7→9 | 0 | 0 | 45.2s |

`boundary_component_closed_count`는 여전히 0 — real snapshot은 C9의 근본 한계(region당 genuine candidate 1~2개, closed loop에는 최소 3개 필요)가 여전히 지배적이며, 이는 worklog 34가 정확히 진단한 대로 `bridge_min_shared_neighbor_for_well_supported` 등 threshold를 감으로 완화하지 않는 한 해소되지 않는, 이번 라운드에서 명시적으로 건드리지 않은 영역이다.

## 14. Region Quality

3k: `consensus_attached` 12배 증가(1→12), 최대 core component 크기 16→17, micro-region 비율 감소(15→8/75, 20%→10%대). 5k/10k도 동일 방향 개선. `region_count` 자체는 3k만 소폭(75→77, worklog34 대비) 변화, 5k/10k는 거의 불변(84→83/64→67) — region 개수보다 **기존 region이 더 크고 조밀해지는** 효과가 지배적이다(=목표한 core consolidation).

## 15. Boundary Component 결과

Real snapshot에서 boundary_component_count는 소폭 증가했으나(region이 커진 만큼 candidate가 늘어난 자연스러운 결과), closed=0/materialized=0은 유지 — C9 fix가 region을 실제로 closed loop를 만들 만큼(≥3 genuine candidate) 키우지는 못했다(이는 §13에서 설명한 근본 한계). box_face는 C11 fix로 closed=1/materialized=1 달성.

## 16. Materialization 결과

Real 3k/5k/10k: `materialized_surface_count=0`(불변, 정직하게 미해결). box_face(cap=27): 0→1. 그 외 post-ADC positive control(비다운샘플 box_face/box/cylinder) 모두 이번 수정으로 회귀 없음 — cylinder는 여전히 3개 closed loop(side+2 caps), box_face 비다운샘플은 여전히 1개 closed loop(32-node).

## 17. Runtime/Memory

- C11 신규 Hungarian matching: n=20에서 0.004초, n=100에서 0.388초, n=200에서 3.21초(O(n³), forward+reversed 두 번 실행). 실측 최대 region 크기(32, 비다운샘플 box_face)는 이 범위 내에서 수 밀리초 수준. Region당 400 candidate 초과 시 O(n²) greedy fallback으로 전환하는 안전장치를 추가했다(현재 어떤 시나리오도 도달하지 않음).
- C9 fix: 기존에 이미 계산되어 있던 `consensus.contradicting_parallel_neighbor_count`를 재사용하는 조건 하나 추가 — 신규 O(N²) 등 연산 없음.
- Real 3k/5k/10k 전체 runtime: 수정 전후 25~46초대 유지(worklog 33/34와 동일 범위), 재앙적 증가 없음.

## 18. Focused / Full Pytest

- `tests/test_directed_boundary_ordering.py`(신규 12 tests): simple cycle recovery, box_face fragmentation 재현·복구, 다중 disjoint loop 분리, open chain/sparse gap/Y-junction/중복 edge negative control, rigid-transform exact invariance, NURBS boundary residual·winding 검증, thin_slab/floater negative control.
- `tests/test_region_consolidation_repair.py`(신규 6 tests): scale-mismatch 전제 조건 실증, cylinder side consolidation positive control, thin_slab/box/box_with_bridge negative control, override 수정 조건의 직접 재현 검증.
- `test_density_preserving_representative_selection.py`/`test_full_cloud_continuation_shell.py`의 기존 invariance test에 region-count 5배 조건 외 member-coverage/major-region 존재/candidate-coverage 지표를 추가(§14 task 요구사항) — 기존 assertion은 삭제하지 않고 추가만 했다.
- 관련 broader suite 재확인(17개 파일, boundary/region/affinity/invariance/materialization 전체): **101/101 pass** (55.80s).
- **Repository-wide pytest(세션 전체 마지막에 1회만 실행): 630 passed, 1 skipped, 0 failed, 8 subtests passed, 소요 시간 187.60초(3분 7초).**

## 19. 다음 남은 Visible Surface Constructor 병목

1. **C9의 근본 한계(미해결)**: real 3k/5k/10k는 region당 genuine termination candidate가 여전히 1~2개 수준으로, closed loop 최소 요구치(3개)에 못 미친다. 이번 라운드는 `_seed_core_components`의 명백한 결함(§10) 하나만 좁게 고쳤을 뿐 — `bridge_min_shared_neighbor_for_well_supported=2` 자체가 real 데이터 밀도에 비해 구조적으로 너무 엄격한지 여부는 여전히 미해결이며, 이는 threshold를 완화하지 말라는 이번 세션의 명시적 금지에 해당해 다음 라운드의 별도 승인이 필요한 과제다.
2. **C11의 잔여 조각**: box_face(cap=27)는 15-node closed loop 1개 + 3-node open path 1개로 개선됐지만 완전한 19-node 단일 loop는 아니다 — 남은 3개 후보가 매칭에서 탈락한 정확한 이유(동일 compatibility gate 하의 순수 매칭 결과)는 이번 라운드에서 더 파고들지 않았다.
3. Worklog 33/34가 이미 disclosed한 sphere의 8-region 과분할, LocalEvidenceScale의 3k 소폭 악화는 여전히 미해결로 남아 있다.
