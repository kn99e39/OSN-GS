# Worklog 40 — Cross-Region Continuation-Aware Termination 및 Boundary Support Certificate Hardening

## 최종 질문에 대한 결론 먼저

> Sphere의 region seam에서 발생한 `observed_support_termination` candidate를 cross-region smooth-continuation evidence로 정확히 재분류하면서도 box/cylinder의 crease boundary와 thin-slab의 parallel-separated boundary를 보존했는가?

**그렇다.** Sphere의 22개 physical candidate가 0이 됐고(22개 전부 `reliability_frontier`로 재분류, 삭제 아님), 동시에 box 110개/cylinder 74개/thin_slab 48개 genuine candidate와 box closed=5, cylinder closed=3, thin_slab closed=2가 **전부 그대로 유지**됐다.

핵심은 판정을 per-Gaussian-pair가 아니라 **REGION PAIR 단위**로 내린 것이다. Sphere의 22개 candidate를 개별 추적하면 "cross-region support인데 relation evidence 없음"으로 보이지만(bounded-kNN이 candidate↔support Gaussian 직접 edge를 자주 누락), region pair (0,1) 전체로는 `same_surface=12, crease=0`으로 affinity graph가 두 반구를 하나의 매끄러운 표면으로 이미 판정하고 있었다. 반면 box의 모든 face pair는 `crease=32~33`, cylinder side/cap은 `crease=88~90`, thin_slab은 `parallel_but_separate=57`이다. **Sphere만이 crease-free이면서 same_surface를 가진 유일한 fixture**이며, 이것이 정확한 판별식이다.

> Worklog 39의 non-candidate interior 2-hop support가 concavity, hole, narrow neck, near-touching loop 및 Y-junction에서도 false shortcut 없이 physical perimeter adjacency만 복구한다는 것을 입증했는가?

**입증했다.** 신규 adversarial topology fixture 전부에서 shortcut이 발생하지 않았다.
- **U-shape**: 56-node closed loop 1개, notch 내부 노드 0개, loop가 notch 벽면을 따라 내려감(x=±0.36 구간 추적, |x|<0.34 진입 0)
- **Hole**: outer 48 + inner 4의 **2개 분리 loop** — hole 가로지르기 없음
- **Narrow neck**: 27+27 **2개 분리 loop** — neck 반대편 연결 없음
- **Near-touching loops** (gap 0.30/0.45/0.60): 항상 24+24 분리, 양쪽 patch를 걸치는 loop 0개
- **Y-junction / interior stub**: worklog 39 음성 통제 그대로 통과

복구된 모든 loop가 `validate_simple_closed_loop`를 통과했다.

> 그리고 real 3k/5k/10k rejection waterfall을 통해 현재 closed-loop 부재가 candidate extraction, cross-region misclassification, compatibility 또는 ordering 중 어디에서 발생하는지 확정했는가?

**확정했다: candidate extraction이다.**

| snapshot | R1/R2 (candidate 부족) | R3 (compatibility 부족) | **R4 (ordering 실패)** |
|---|---|---|---|
| 3k | 138/157 (**88%**) | 19 (12%) | **0** |
| 5k | 124/148 (**84%**) | 24 (16%) | **0** |
| 10k | 132/141 (**94%**) | 9 (6%) | **0** |

세 snapshot 모두 **ordering 실패 region이 0개**다. 가장 큰 region(21~28 members, 공간 지름 6~9)조차 genuine candidate를 0~4개만 만든다. Cross-region misclassification도 원인이 아니다(3k에서 `reliability_frontier` 3개, 5k 1개, 10k 0개로 미미).

---

## 1. Worklog 39 승인/보류 항목

승인 유지: accepted_core_pair와 boundary adjacency의 semantic 분리, direct-or-2-hop support certificate, geometry gate/Hungarian solver 불변, histogram이 geometric gap 단독보다 방어적이라는 측정.

보류였다가 이번에 해소: sphere 22개 false candidate(§2~§6), 2-hop certificate의 topology safety 미검증(§8).

여전히 보류: two-phase seeding의 micro-region 문제(이번 작업에서 의도적으로 미변경, §1 지시), box 6번째 face(§10에서 원인은 확정, 강제 복구 안 함), histogram이 최종 canonical model이라는 증명(§11).

## 2. Sphere Candidate Authoritative Trace

`scripts/devtools/trace_cross_region_support_relations.py`로 모든 fixture의 genuine candidate에 대해 outward arc 안의 same-region/out-of-region support와 그 relation evidence를 기록했다.

| scene | genuine | 분류 결과 |
|---|---|---|
| sphere | 22 | `cross_region_support_without_relation_evidence` 22 |
| box | 110 | `crease_adjacent` 108, relation evidence 없음 2 |
| cylinder | 74 | `crease_adjacent` 58, evidence 없음 15, cross-region support 없음 1 |
| thin_slab | 48 | `parallel_separate_neighbor` 23, evidence 없음 14, support 없음 11 |
| box_face | 32 | `no_cross_region_support` 32 |

Sphere 22개는 전부 두 region seam(z∈[0.084,0.293] / [-0.292,-0.084], 전체 z 범위 [-0.299,0.298])에 집중되며 각각 같은 region 이웃 ~25개, 다른 region 이웃 ~26개를 support radius 안에 가진다.

**중요한 방법론적 발견**: per-Gaussian-pair 조회로는 sphere 22개 전부가 "relation evidence 없음"으로 나온다. 그러나 이는 bounded-kNN이 candidate와 support Gaussian 사이 직접 edge를 만들지 않았기 때문이며, region pair 전체로 집계하면 evidence가 명확히 존재한다(아래 §3).

## 3. Cross-Region Relation Taxonomy (REGION PAIR 단위)

| scene | region pair | crease | parallel | same_surface | verdict |
|---|---|---|---|---|---|
| **sphere** | (0,1) | **0** | 2 | **12** | **smooth_continuation** |
| box | (0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5) | 32~33 | 0 | 0 | crease_adjacent |
| cylinder | (0,1),(0,2) | 88~90 | 0 | 0 | crease_adjacent |
| thin_slab | (0,1) | 0 | **57** | 0 | parallel_separate |

판정 우선순위는 보수적으로 설계했다: crease evidence가 하나라도 있으면 crease_adjacent; parallel이 same_surface 이상이면 parallel_separate; crease-free이고 same_surface가 있을 때만 smooth_continuation; 그 외 ambiguous.

## 4-5. Continuation Certificate 및 Suppression 원칙

`classify_cross_region_pairs()`(신규, `torch_boundary_support_termination.py`)는 affinity graph가 **이미 계산한** relation만 region pair 단위로 집계한다. 새 geometry, 새 threshold, scene-specific 상수 없음. Bounded: 기존 candidate edge 집합만 순회한다.

Candidate 분류 시점에 outward arc를 차지하는 out-of-region support의 region pair verdict가 `smooth_continuation`일 때만 `reliability_frontier`로 재분류한다. crease_adjacent / parallel_separate / ambiguous는 **전부 physical candidate로 유지**한다(§5의 suppress 금지 조건 준수). 재분류된 candidate는 provenance를 유지한 채 emit되며, `ordering_state="ambiguous_ordering"`이므로 directed ordering에 도달하지 않는다.

Worklog 39가 시도한 조잡한 버전(arc에 out-of-region support가 있으면 무조건 suppress)은 box 110→0, cylinder 74→0(closed 2→0), thin_slab 48→3으로 genuine candidate를 파괴했다. Relation class를 참조하는 것이 이 둘을 가르는 지점이다.

## 6. Sphere 결과

| 항목 | before | after |
|---|---|---|
| genuine `observed_support_termination` | 22 | **0** |
| `reliability_frontier` | 0 | **22** |
| closed component | 0 | 0 |
| materialized | 0 | 0 |
| arbitrary seam | 0 | 0 |

§6의 expected 계약을 전부 충족한다. Candidate generation 자체를 무력화하지 않았다(다른 fixture의 candidate 수 불변).

## 7. Crease/Parallel Negative Controls

| scene | genuine | closed | 판정 |
|---|---|---|---|
| box | 110 (불변) | 5 (불변) | crease boundary 보존 |
| cylinder | 74 (불변) | 3 (불변) | side/cap crease 보존 |
| thin_slab | 48 (불변) | 2 (불변) | parallel outer edge 보존 |
| box_face | 32 (불변) | 1 (불변) | 단일 region, 영향 없음 |
| box_with_bridge | 110 (불변) | 5 (불변) | 영향 없음 |
| box_isolated_floater | 32 (불변) | 1 (불변) | 영향 없음 |
| box_isotropic_contamination | 32 (불변) | 1 (불변) | 영향 없음 |

**Folded sheet crease angle sweep**(신규 fixture):

| fold 각도 | regions | region-pair verdict | genuine | 재분류 |
|---|---|---|---|---|
| 180° (평면) | 1 | (해당 없음) | 44 | 0 |
| 150° | 3 | (1,2) smooth, (0,1)/(0,2) parallel | 40 | 6 |
| 120° | 2 | ambiguous | 48 | **0** |
| 90° | 2 | **crease_adjacent** | 48 | **0** |

실제 fold(90°/120°)에서는 재분류가 전혀 일어나지 않고, 완만한 150°에서만 일부가 continuation으로 판정된다. §7이 요구한 "smooth 영역은 continuation, 실제 fold는 crease"를 만족한다.

## 8. 2-Hop Certificate Topology Safety

신규 adversarial fixture(`tests/test_boundary_topology_safety.py`):

| fixture | n | regions | closed loops | 결과 |
|---|---|---|---|---|
| U-shape (concavity) | 134 | 1 | **1** (56-node) | notch 내부 노드 0, 벽면 추적 확인 |
| Sheet with hole | 160 | 1 | **2** (48 + 4) | outer/inner 분리, hole 가로지르기 없음 |
| Narrow neck | 109 | 1 | **2** (27 + 27) | neck 반대편 연결 없음 |
| Near-touching (gap 0.30) | 98 | 2 | 2 (24+24) | 양 patch 걸침 0 |
| Near-touching (gap 0.45) | 98 | 2 | 2 (24+24) | 양 patch 걸침 0 |
| Near-touching (gap 0.60) | 98 | 2 | 2 (24+24) | 양 patch 걸침 0 |

U-shape loop 경로 실측: (0.72,-0.0)→(0.72,0.72)→(0.36,0.72)→(0.36,0.24)로 notch 벽을 따라 내려가며, 해당 구간 노드의 distinct x는 {±0.36, ±0.48, ±0.60, ±0.72}뿐 — |x|<0.34 진입 0개. Shortcut이면 이 벽면 노드가 나타나지 않는다.

모든 복구 loop가 self-intersection 검사 통과. Y-junction/interior stub은 worklog 39 음성 통제 그대로 유지.

## 9. Box 6번째 Face 원인 확정 — 강제 복구 안 함

Certificate 적용 후 box compatibility:

| face | candidates | compat edges | zero_out | 결과 |
|---|---|---|---|---|
| 0 | 19 | 25 | 0 | CLOSED |
| 1 | 18 | 27 | 0 | CLOSED |
| 2 | 16 | 19 | 0 | CLOSED |
| 3 | 19 | 26 | 0 | CLOSED |
| **4** | 18 | 23 | **2** | **FAILED** |
| 5 | 20 | 27 | 0 | CLOSED |

Face 4의 dead-end 2개(gid 216, 242)를 gate 단위로 추적한 결과 **compatibility나 topology support 문제가 아니다**:
- gid=216: 최근접 이웃이 뒤쪽(fwd=-0.111), 전방 이웃은 tangent 반전(tan=-0.384)
- gid=242: 동일 패턴(fwd=+0.000/tan=-0.699, fwd=-0.171)

더 근본적으로 face 4의 candidate 구성 자체가 불규칙하다: **corner candidate가 4개 중 2개뿐**이고(gid 196, 238), edge 12개 외에 **interior 4개**가 candidate로 잡혀 있다. 정사각형 perimeter를 이루기에 evidence가 실제로 부족한 상태다. §10의 "Evidence가 실제로 부족하다면 강제하지 말고 부족 원인을 수치로 보고한다"에 따라 강제 복구하지 않았다. Directed solver와 objective는 변경하지 않았다.

## 10. Histogram 역할 감사 (§11, diagnostic only)

Histogram은 provisional baseline으로 유지했고 production을 변경하지 않았다.

**Rotation sensitivity**(회전각 0.0 / 0.37 / 0.91 / 1.57 rad, 임의 축):

| scene | genuine candidate | closed |
|---|---|---|
| box_face | 32 / 32 / 32 / 32 | 1 / 1 / 1 / 1 |
| cylinder | 74 / 74 / 74 / 74 | 3 / 3 / 3 / 3 |
| sphere | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

모든 각도에서 완전히 동일 — bin origin 회전 민감성은 관측되지 않았다.

**역할 분리 확인**: histogram은 "local angular exposure가 있는가"에, cross-region certificate는 "그 노출 방향에 실제 surface continuation이 있는가"에 답한다. 두 gate는 코드상 별도 지점에서 평가되며 서로의 판정을 덮어쓰지 않는다. Continuous uncertainty-interval 후보는 이번에 구현하지 않았다(§11이 production 반영을 금지했고, histogram 유지 판정 후 우선순위가 낮음).

## 11. Real 3k/5k/10k Waterfall

`scripts/devtools/trace_real_snapshot_boundary_waterfall.py`(신규).

| snapshot | regions | genuine | nonphysical | closed | R1/R2 | R3 | **R4** |
|---|---|---|---|---|---|---|---|
| 3k | 157 | 153 | reliability_frontier 3, ambiguous_continuation 16, sampling_gap 1 | 0 | **138** | 19 | **0** |
| 5k | 148 | 181 | ambiguous_continuation 18, parallel_sheet_conflict 1, reliability_frontier 1, sampling_gap 1 | 0 | **124** | 24 | **0** |
| 10k | 141 | 121 | ambiguous_continuation 28, sampling_gap 1 | 0 | **132** | 9 | **0** |

최대 region 표본(3k): r4(21 members, 지름 6.29, candidate 3), r9(20 members, 지름 6.51, **candidate 0**), r61(19 members, candidate 3). 10k: r41(28 members, 지름 8.17, candidate 3), r6(21 members, 지름 9.31, candidate 4).

**Cross-region misclassification은 real snapshot의 병목이 아니다** — `reliability_frontier` 재분류는 3k 3개, 5k 1개, 10k 0개에 불과하다.

## 12. R1~R6 분류

R4(ordering 실패)는 세 snapshot 모두 **0개**. R5/R6에 해당하는 region은 이번 분류 기준에서 나타나지 않았다. 지배적인 것은 R1/R2(84~94%)이며, R3는 6~16%다. 즉 region이 공간적으로 크더라도(지름 6~9) perimeter를 따라 termination candidate가 거의 생성되지 않는 것이 실제 병목이다.

## 13. 적용한 Narrow Repair

| 항목 | 조치 |
|---|---|
| `classify_cross_region_pairs()` (신규) | region-pair 단위 crease/parallel/same_surface 집계, 기존 evidence만 재사용 (Case A) |
| Candidate 분류 | `smooth_continuation` region pair의 frontier candidate만 `reliability_frontier`로 재분류 (Case A) |
| `extract_support_termination_candidates` | `affinity_graph` optional 파라미터 추가(미전달 시 기존 동작 완전 보존) |
| Histogram | **변경 없음** |
| 2-hop certificate | **변경 없음**(검증만) |
| Box face 4 | **변경 없음**(원인 보고만) |
| Threshold 전체 / Hungarian objective / cycle decomposition / NURBS fitting | **변경 없음** |

## 14. Independent Ablation

실제 수정이 하나(cross-region continuation certificate)뿐이므로 A/B만 유효하다. C(2-hop hardening)와 D(box tangent repair)는 **수정을 적용하지 않았으므로 실행하지 않았다**(§14 지시에 따라 명시).

| config | sphere genuine | box closed | cylinder closed | thin_slab closed | box_face closed |
|---|---|---|---|---|---|
| A (worklog 39) | 22 | 5 | 3 | 2 | 1 |
| **B (+certificate)** | **0** | 5 | 3 | 2 | 1 |

Seed admission 결과는 변경하지 않았으므로 region/core 수치는 A와 B가 동일하다(3k region=157/core=799, 5k 148/770, 10k 141/689).

## 15. Analytic 결과 요약

box_face closed=1, box closed=5, cylinder closed=3, thin_slab closed=2, floater closed=1, contamination closed=1, box_with_bridge closed=5, **sphere closed=0 & genuine=0**.

## 16. Positive/Negative Controls

Positive: box_face, box 6면, cylinder side+2caps, U-shape concave sheet, sheet with hole, narrow neck, two disjoint loops, folded sheet 180°, rigid rotation 4각도.

Negative: closed sphere(genuine 0), thin_slab, close/near-touching patches(3 gap), curved near-contact, box crease, Y-junction, interior stub, hole shortcut, narrow-neck shortcut, floater, isotropic contamination, folded sheet 90°/120°.

## 17. Runtime/Memory

`classify_cross_region_pairs`는 기존 candidate edge 집합 1회 순회(O(E))이며 region pair 수만큼의 dict 항목만 유지한다. Candidate 분류 시 추가되는 루프는 이미 존재하던 support radius 순회와 동일 범위다. Real snapshot runtime은 3k 28~30초, 5k 32초, 10k 47~49초로 기존 범위 유지. Full-cloud all-pairs, 신규 unbounded all-pairs, 반복 tensor transfer, hop count 확대 없음.

## 18. Focused/Full Pytest

- `tests/test_cross_region_continuation.py`(신규 15 tests): region-pair taxonomy 4종(sphere smooth / box crease / cylinder crease / thin_slab parallel), sphere false boundary 제거 3종, crease/parallel 보존 4종, folded sheet sweep 2종, ambiguous 미suppress 1종, rotation invariance 1종.
- `tests/test_boundary_topology_safety.py`(신규 6 tests): U-shape concavity 2종, hole shortcut 2종, narrow neck 1종, near-touching loops 1종.
- 기존 boundary suite(`test_boundary_adjacency_semantics` / `test_directed_boundary_ordering` / `test_seed_merge_separation`): 58/58 통과.
- **Repository-wide pytest(세션 마지막 1회): 707 passed, 1 skipped, 0 failed, 8 subtests passed, 184.67초(3분 4초).**

## 19. 다음 Visible Surface Constructor 병목

1. **Real snapshot의 termination candidate 생성 (최우선, §11/§12에서 확정)**: R1/R2가 84~94%로 압도적이다. 21~28 member/지름 6~9의 큰 region조차 candidate를 0~4개만 만든다. Ordering(R4=0)도 compatibility(R3 6~16%)도 아닌 **candidate extraction 단계**가 유일한 실질 병목임이 세 snapshot에서 일관되게 확정됐다. 다음 라운드는 이 단계의 gate(angular exposure, neighbor count, continuation query)를 real 데이터 밀도 기준으로 분해해야 한다.
2. **Box 6번째 face의 candidate 불규칙성 (§9)**: corner candidate가 4개 중 2개만 생성되고 interior 4개가 잘못 candidate가 된다. 이는 1번과 같은 candidate-extraction 문제의 analytic 축소판이므로 함께 다루는 것이 효율적이다.
3. **Two-phase seeding micro-region (§1에 의해 이번 미변경)**: 36~39% 비율이 그대로이며 여전히 provisional이다.
4. **Histogram의 canonical 여부 (§11)**: rotation invariance는 확인됐으나 density/neighbor-count sensitivity와 continuous uncertainty-interval 대안 비교는 미수행이다.
