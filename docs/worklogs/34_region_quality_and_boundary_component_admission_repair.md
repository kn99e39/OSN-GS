# Worklog 34 — Region Quality 검증 및 Boundary Component Admission / Closed-Loop 복구

## 최종 질문에 대한 결론 먼저

> Worklog 33에서 생성된 64~85개 region은 실제 coherent visible-surface regions인가, 아니면 fragmented micro-regions인가?

**대부분 작지만(median 4~5명), fragmented micro-region이라고 단정할 수는 없다.** Representative 2048개 중 core_member는 19~22%(392/447/357)뿐이고, 나머지 80% 가까이(1580~1674)는 여전히 `ambiguous_unassigned` — 즉 region 자체는 진짜 same_surface 증거(worklog 33의 G1)로 형성됐지만 GROWTH(주변 ambiguous 노드를 region에 편입시키는 단계)가 거의 작동하지 않아 작은 core cluster 상태로 멈춰 있었다.

> C_component_admission_failed의 정확한 최초 원인은 region fragmentation, candidate 누락, endpoint matching, directed orientation, ordering, closure scale, ownership 또는 false-boundary provenance 중 무엇인가?

**두 가지 서로 다른 원인이 snapshot 종류에 따라 다르게 지배적이다.**

1. **실제 3k/5k/10k 장기 학습 snapshot**: **C9(region이 구조적으로 너무 작음) + 실제로 입증된 growth-loop 버그(아래) 복합.** 각 region의 genuine termination candidate가 median 1~2개뿐이라(closed loop는 최소 3개 필요) 애초에 ordering 알고리즘이 무엇을 하든 닫힌 loop를 만들 후보 자체가 없다.
2. **Post-ADC synthetic 단일 큰 region(box_face, 27 representative 1개 region, genuine candidate 19개)**: 후보는 충분한데도(15/19이 개별적으로는 유효한 forward successor를 찾음) 최종적으로 1~7개짜리 조각난 open chain으로 쪼개진다 — 이건 **C11(directed ordering의 strict mutual-agreement + greedy augmentation 구현 한계)**이며, region이 작아서가 아니다.

> 이를 입증한 뒤 physical observed-support boundary만을 사용해 simple closed component와 정확한 NURBS boundary를 복구했는가?

**부분적으로만.** 입증된 좁은 결함(growth-loop이 core-merge 전용 veto를 growth에도 잘못 재사용) 하나를 production에서 복구해 real snapshot의 `consensus_attached`가 0~1 → 7~10으로 늘었지만, closed loop/materialization은 여전히 0이다. Region 크기 자체(C9)와 directed ordering 알고리즘(C11)은 이번 세션에서 감으로 재설계하지 않았다 — 강제로 loop를 닫는 방식은 명시적으로 금지됐고, 두 원인 모두 이번 한 번의 좁은 수정으로 해결될 성질이 아니라고 판단했다.

---

## 1. Worklog 33 graph repair 상태 — 고정 유지

G1(representative kNN spacing, candidate_scale/residual_scale 양쪽)을 그대로 유지했다. 이번 세션에서 `torch_gaussian_manifold_affinity.py`는 전혀 수정하지 않았다(git diff 없음).

## 2. Region membership breakdown (3k/5k/10k, 수정 전)

| iteration | core_member | consensus_attached | ambiguous_unassigned | conflict_boundary | rejected | representatives_in_any_region |
|---|---|---|---|---|---|---|
| 3000 | 392 (19.1%) | 1 | 1653 | 2 | 0 | 395 (19.3%) |
| 5000 | 447 (21.8%) | ~1 | 1580 | 13 | 0 | ~448 |
| 10000 | 357 (17.4%) | ~1 | 1674 | 9 | 1 | ~358 |

## 3. Region 크기 및 micro-region 분포 (3k, 수정 전)

- region_count=75, member count median=4, p90=10, max=16.
- singleton region 1개, 2-member 0개, 3-member 16개. micro(≤3)=17개(22.7%), major(>10)=3개.
- 즉 region은 대부분 작지만 완전히 무의미한 singleton은 아니다 — "fragmented micro-region"이라는 가설(질문 B)은 부분적으로만 맞다: 진짜 문제는 growth가 거의 작동하지 않아 CORE cluster 크기 자체가 작게 멈춘 것이지, 큰 region이 쪼개진 게 아니다.

## 4. 기존 invariance test 계약 — 근거 보강

Worklog 33이 도입한 5배 tolerance를 "삭제/약화가 아니다"라고만 서술하지 않고 실측 근거를 테스트 docstring에 직접 추가했다: 이 두 fixture(seed=2, seed=3)에서 실측된 최악 비율은 각각 5:3(1.67배), 4:1(4배) — 4배가 이 코드베이스에서 실측된 두 rigid-transform fixture 중 최악값이다. 5배는 이 실측 최악값에 여유를 두면서도 진짜 회귀(예: 한쪽이 50개의 가짜 region을 만드는 경우)는 걸러낼 수 있는 값으로 정당화했다. Test A(frozen exact invariance, worklog 33에서 이미 구축)와 Test B(선택 재실행 후 topology stability)의 계층 분리는 그대로 유지했다.

## 5. Boundary pipeline 실제 호출 경로 (코드 직접 확인)

```text
construct_visible_nurbs_from_gaussians
  -> extract_support_termination_candidates(...)          [candidate 생성, boundary_reason 분류]
  -> normalize_continuation_candidates(...)
  -> build_boundary_compatibility(halfedges)               [진단용, ordering에 미사용]
  -> recover_directed_boundary_components(halfedges, accepted)
       -> _recover_directed_boundary_components(직접/역방향 두 번 시도, 더 나은 쪽 채택)
            1. candidates = boundary_reason=="observed_support_termination"만 필터
            2. local_spacing = 같은 region 내 candidate 간 최근접 거리의 median
            3. max_distance=1.6*local_spacing, max_lateral=0.9*local_spacing
            4. forward-successor 탐색: accepted_pairs(코어 same_surface 합집합)에 있는 쌍만, forward>0, distance<=max_distance, lateral<=max_lateral, tangent>=-0.15, normal>=0.45
            5. backward-predecessor 독립 탐색(동일 게이트)
            6. mutual = forward와 backward가 서로 합의하는 edge만
            7. 남은 미사용 slot을 score 순 greedy augmentation으로 채움
            8. adjacency 기반 chain 추적, 시작점으로 돌아오면 closed(>=3 node)
  -> materialize_visible_boundary_component(component, ...)   [closed만 admission 시도]
```

`recover_directed_boundary_components`의 `candidates` 필터(1단계)가 `observed_support_termination`만 통과시키므로, reliability frontier나 sampling gap이 물리적 경계로 승격되는 경로는 구조적으로 없다 — 코드로 직접 재확인했다.

## 6. Boundary-local scale 감사

`local_spacing`(§5-2)은 REGION별 candidate 간 실측 거리의 median이다 — Gaussian footprint도, LocalEvidenceScale도, RepresentativeGraphScale도 아니고, **boundary candidate 자기 자신들의 실제 spacing**을 이미 쓰고 있다. Worklog 30~33이 발견한 "개별 Gaussian footprint를 엉뚱한 local scale로 오용"하는 패턴이 여기서는 **발견되지 않았다** — boundary linking 단계는 이미 올바르게 설계돼 있다. 따라서 boundary-local scale 교체는 이번 세션에서 하지 않았다(불필요).

## 7. Component ambiguity 원인 분해 — `ambiguous`를 세분화

신규 스크립트 `scripts/devtools/trace_directed_ordering_failure.py`(offline 전용, production 미변경)로 forward-successor 탐색 단계에서 각 candidate가 실패하는 정확한 이유를 재현했다.

### 3000 (수정 전, genuine candidate 67개)

| per-node 실패 원인 | 개수 |
|---|---|
| succeeded(성공) | 3 |
| non_forward_direction | 22 |
| **no_same_region_partner_at_all** | **27** |
| not_accepted_core_edge | 10 |
| distance_exceeds_max | 5 |

Same-region partner 수의 median=1. 즉 대부분의 candidate는 자기 region 안에 이어붙일 후보가 **1개 이하**밖에 없다 — closed loop는 최소 3개 필요하므로 알고리즘이 무엇을 하든 구조적으로 닫을 수 없다.

### 5000 / 10000 (수정 전)

| | 5000 | 10000 |
|---|---|---|
| genuine candidate | 122 | 59 |
| no_same_region_partner_at_all | 24 | 25 |
| not_accepted_core_edge | 35 | 16 |
| succeeded | 23 | 5 |
| same-region partner median | 2 | 1 |

세 snapshot 모두 동일 패턴: **candidate 누락(C10)이 아니라 candidate가 애초에 region당 1~2개뿐인 region 크기 문제(C9)**가 지배적이다.

### box_face(post-ADC synthetic, 단일 27-member region, genuine candidate 19개) — 다른 패턴

| | 값 |
|---|---|
| same-region partner median | **18** (region 전체가 서로 후보) |
| per-node succeeded | 15/19 |
| 실제 output component | node_count [1,2,3,5,7,1] — 6개 조각 |

Candidate는 풍부하고 개별적으로는 15/19가 성공적인 forward successor를 찾지만, **mutual 합의(forward와 backward가 서로 일치해야 함) + greedy augmentation**을 거치면서 하나의 19-node closed loop 대신 6개의 조각난 open chain으로 쪼개진다. 이건 candidate 부족이 아니라 **C11: directed ordering 구현의 mutual-matching 휴리스틱 한계**다.

## 8. Stage C 최초 실패 지점 확정

```text
실제 장기학습 3k/5k/10k: C9 (region이 구조적으로 너무 작아 candidate 자체가 부족) — 지배적
                        + 실제 입증된 growth-loop 버그(§9) — 부차적 기여
post-ADC 합성 box_face: C11 (directed ordering의 mutual-matching이 충분한 candidate를 조각냄)
```

C1(연결 안 됨)/C2(endpoint matching 실패)/C7(ownership 충돌)/C8(false boundary 혼입)은 확인되지 않았다. C4(같은 loop가 여러 component로 분리)는 box_face에서 관찰된 현상과 표면적으로 비슷해 보이지만 실제 원인은 candidate 부족이 아니라 mutual-matching 알고리즘이므로 C11로 분류했다.

## 9. 실제로 입증하고 수정한 결함 — Growth-loop의 boundary_conflict_edges 오용

`_seed_core_components`가 만드는 `boundary_conflict_edges`는 4가지 서로 다른 이유로 채워진다: (1) `CONSENSUS_CONTRADICTED`, (2) `PATH_PHASE_ALIAS`, (3) oversized-footprint-parallel-veto, (4) **weak bridge**(두 개의 **이미 형성된 core 컴포넌트를 병합**할 때 독립 cross-support가 부족하다는 veto). 3000 snapshot 실측: `boundary_conflict_edges` 총 1429개 중 1039개(73%)가 (4) weak bridge 사유였다.

`form_surface_regions`의 GROWTH 단계(§8, 단일 미배정 노드를 이미 만들어진 region에 편입시키는 과정)가 이 **같은 flat set**을 재사용해 지원 edge를 걸러내고 있었다 — 그런데 growth는 "두 컴포넌트를 병합"하는 게 아니라 "노드 하나를 기존 region에 붙이는" 전혀 다른 연산이다. 직접 검증: 3000 snapshot에서 자기 region의 core member에 대해 same_surface degree>=2인 ambiguous 노드 93개 전부가, 지원 edge 중 최소 하나가 `boundary_conflict_edges`에 있다는 이유로 growth가 막혀 있었다(93/93).

**수정**: `torch_gaussian_surface_region_formation.py`의 growth 루프에서, edge가 `boundary_conflict_edges`에 있어도 그 이유가 (1)CONTRADICTED/(2)PHASE_ALIAS/(3)oversized-footprint-veto 중 하나일 때만 계속 제외하고, (4) weak-bridge-only인 경우는 더 이상 growth를 막지 않도록 했다. `_seed_core_components`가 이미 반환하는 `consensus_by_pair`/`path_by_pair`를 재사용했을 뿐 새 계산을 추가하지 않았다. Core-to-core 병합(bridge veto 자체)과 region merge threshold는 전혀 건드리지 않았다 — 금지 목록을 그대로 지켰다.

## 10. Post-ADC Positive Control (수정 후)

| scene | region | boundary_comp | closed | materialized | stage |
|---|---|---|---|---|---|
| box_face | 1 | 6 | 0 | 0 | C |
| box | 6 | 28 | 0 | 0 | C |
| cylinder | 3 | 5 | 0 | 0 | C |
| sphere | 8 | 2 | 0 | 0 | C |

Plane/box_face에서도 closed loop·materialization은 이번 수정으로 복구되지 않았다 — §9의 좁은 수정은 real-checkpoint의 C9(작은 region)에는 일부 도움이 됐지만 box_face가 겪는 C11(ordering 알고리즘 fragmentation)에는 애초에 해당되지 않는 별개의 결함이기 때문이다. **작업 완료 기준(§15 원문)이 "plane/box_face에서도 실패하면 미완료"라고 명시했으므로, 이 부분은 솔직히 미완료로 보고한다.**

## 11. Negative Control (수정 후)

| scene | region_count | 수정 전 대비 |
|---|---|---|
| thin_slab | 2 | 동일 |
| box_isolated_floater | 1 | 동일(floater 미포함 재확인) |
| box_isotropic_contamination | 1 | 동일 |
| box_with_bridge | 6 | 동일 |
| box | 6 | 동일 |

False merge, false split 전부 없음 — growth-loop 수정이 negative control 결과를 전혀 바꾸지 않았다(즉 새로운 false attachment를 만들지 않았다).

## 12. Inaccurate Boundary 원인 — 해당 사례 없음

Closed loop가 0개이므로 materialize된 patch가 아예 없다. Source boundary/fitted NURBS boundary/visualization 세 카테고리 중 무엇이 부정확한지 구분할 대상 자체가 없다 — §5에서 확인한 대로 candidate 생성 단계의 provenance 필터는 정상 작동 중이므로, 이번 세션에서 재현한 "-surf 부정확한 boundary" 문제는 여전히 별도 조사가 필요하다(이번 라운드 범위 밖).

## 13. 적용한 narrow repair — 요약

`osn_gs/surface/torch_gaussian_surface_region_formation.py`의 growth 루프 1곳만 수정(§9). `torch_gaussian_manifold_affinity.py`, `torch_directed_boundary_ordering.py`, `torch_boundary_support_termination.py`는 전혀 건드리지 않았다.

## 14. Repair 전후 real 3k/5k/10k 결과

| iteration | consensus_attached(전→후) | conflict_boundary(전→후) | region_count | boundary_component_count | materialized |
|---|---|---|---|---|---|
| 3000 | 1→10 | 2→10 | 75(불변) | 3(불변) | 0(불변) |
| 5000 | ~1→8 | 13→13 | 85(불변) | 19(불변) | 0(불변) |
| 10000 | ~1→7 | 9→9 | 64(불변) | 5(불변) | 0(불변) |

Growth 자체는 개선됐지만(대표점이 region에 새로 편입) region 개수·경계 컴포넌트·materialization은 이번 수정만으로는 바뀌지 않았다 — §9에서 분석한 대로 이 수정은 C9(candidate 부족)의 원인(작은 core cluster)을 완화하는 방향이지만, 이미 형성된 75/85/64개 region 중 어느 것도 이번 라운드에서 3개 이상의 genuine termination candidate를 새로 얻지는 못했다(candidate 생성은 growth 이후 단계라 이번 수정과 독립적으로 재계산되지만, growth로 늘어난 core 규모가 아직 후속 candidate 생성 임계치를 넘길 만큼은 아니었다).

## 15. Materialization 결과

3k/5k/10k, box_face/box/cylinder/sphere 전부 `materialized_surface_count=0`. Closed loop 자체가 없으므로 강제로 닫지 않는 한 필연적인 결과다.

## 16. Runtime/memory

3k snapshot 기준 `_construct_canonical_with_full_evidence` 전체 runtime 34~35초대로 worklog 33과 동일한 범위 유지 — growth 루프 수정은 기존 O(count × neighbor) 연산에 조건 분기 하나를 추가한 것뿐이라 새로운 O(N²) 등 비용이 없다.

## 17. Focused / full pytest

- 관련 suite 재확인: `test_gaussian_surface_region_formation`, `test_surface_region_adversarial_validation/invariance/phase_alias/validation`, `test_density_preserving_representative_selection`, `test_full_cloud_continuation_shell`, `test_representative_graph_scale`, `test_local_evidence_scale`, `test_visible_surface_construction`, `test_adc_synchronized_visible_nurbs` = **51/51 pass**.
- **Repository-wide pytest(이번 세션 전체 마지막에 1회만 실행): 612 passed, 1 skipped, 0 failed, 8 subtests passed, 소요 시간 152.18초(2분 32초).**

## 18. 다음 남은 Visible Surface Constructor 병목

1. **C9 (region 크기 제한)**: growth-loop 수정은 부분적으로만 도움이 됐다 — region이 여전히 대부분 4~5명 수준으로 작다. 근본적으로는 CORE 단계의 bridge-veto(두 core 컴포넌트 병합 조건)가 보수적으로 설계돼 있어서인데, 이건 명시적으로 이번 세션의 금지 목록(region merge threshold)에 해당해 손대지 않았다. 다음 라운드에서 "왜 core cluster 자체가 커지지 않는가"(bridge veto의 independent-cross-edge 요구치가 실제 데이터에 비해 너무 엄격한지)를 별도로 조사할 필요가 있다.
2. **C11 (directed ordering의 mutual-matching 한계)**: box_face처럼 candidate가 충분한 경우에도 mutual-agreement + greedy augmentation이 하나의 closed loop 대신 여러 조각으로 쪼갠다. `torch_directed_boundary_ordering.py`의 알고리즘 자체를 다음 라운드에서 별도로, 신중하게 검토할 필요가 있다 — 이번 세션에서는 "G1 graph repair를 다시 설계하지 마라"는 지시와 유사한 이유로, 이 핵심 ordering 알고리즘도 감으로 재설계하지 않았다.
3. `-surf`로 관찰됐던 "부정확한 boundary" 문제는 애초에 closed loop가 하나도 안 만들어지는 현재 상태에서는 재현/분류할 대상이 없다 — 위 두 병목이 먼저 해소돼야 검증 가능하다.
