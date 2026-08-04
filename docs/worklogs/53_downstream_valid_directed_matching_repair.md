# Worklog 53: Downstream-valid Directed Matching 복구

## 배경

Worklog 52는 region 52의 진단은 맞았지만(cycle 자체가 위상적으로 불가능), region 56은 재검토가 필요했다. 이번 작업은 Hungarian이 decomposition에서 무효 처리되는 길이-2 cycle에 점수를 주고 capacity를 낭비해 실제 feasible한 길이-3 cycle을 밀어내는지 확인하고, 확인되면 solver를 수정하는 것이었다.

## 확인: 2-cycle이 진짜 capacity를 낭비한다

Region 56 (candidates: 1039800, 1110285, 278207, 819956)의 순수 max-score 결과는 `1039800→819956`(3.18)과 `819956→1039800`(2.71)를 **둘 다** 선택한다 — 상호 2-cycle이다. `_decompose_into_paths_and_cycles`는 길이 3 미만인 닫힌 구조를 처음부터 유효한 loop로 인정하지 않고 `ambiguous_ordering` path로 강등한다(이 규칙 자체는 worklog 35부터 있던 기존 계약, 이번에 바꾸지 않았다). 그 결과 이 2-cycle은 두 후보 모두의 in/out capacity를 통째로 써버리고 아무 것도 만들지 못하며, 남은 `278207→1110285`(2.05)만 별도의 열린 조각으로 남는다 — 실제 원래 보고: `paths: [['1039800','819956'], ['278207','1110285']]`.

완전 탐색(전수 조합)으로 2-cycle을 배제한 **진짜 최선의 대안**을 계산했다: 4-node open path `1039800→819956→278207→1110285`, 총점 **7.224** — 이는 2-cycle을 포함한 원래 assignment(7.9408, 그중 5.89가 무효 2-cycle에 낭비됨)보다 낮지만, feasible한 3-cycle(`819956→278207→1110285→819956`, 6.4619)보다는 **높다**. 즉 2-cycle을 금지하고 공정하게(closure bonus 없이) 재최적화하면, 진짜 최댓값은 **닫힌 3-cycle이 아니라 4-node open path**다 — closure를 강제하면 그 자체가 금지된 "closure bonus"가 된다.

## 적용한 수정 (1차)

`osn_gs/surface/torch_directed_boundary_ordering.py`:

- `_solve_one_in_one_out_assignment(node_ids, edges, forbidden)`: 기존 solve 로직을 그대로 유지하되 특정 (source, target) pair를 feasible set에서 제외할 수 있게 분리했다.
- `_find_two_cycle`: matching 결과에서 상호 pair(`a→b`, `b→a` 둘 다 선택됨)를 결정론적으로(사전순 최솟값) 찾는다.
- `_matching_forbidding_two_cycles`: 2-cycle이 발견되면 두 방향 중 하나씩 금지한 두 branch를 재귀적으로 풀고, 점수가 더 높은 branch를 선택한다(동점이면 닫힌 cycle 수가 더 많은 branch, 그것도 동점이면 결정론적 고정 선택). 새 edge 추가나 compatibility 우회는 없다 — 순수하게 기존 compatible edge 중 무엇을 배제할지의 문제다. 병리적 확산을 막기 위해 `_MAX_TWO_CYCLE_BRANCH_EXPANSIONS=16`으로 재귀를 bound했다(기존 `_EXACT_MATCHING_MAX_CANDIDATES_PER_REGION`과 같은 "명시적 fail-closed" 철학).

## 발견한 2차 결함: 안전하지 않은 branch가 기존에 안전했던 결과를 대체

1차 수정만 적용하고 thin_slab negative-control(cap 64)을 재확인하니 `closed=3→2, materialized=3→0`으로 회귀했다. 원인을 추적한 결과:

- 2-cycle을 제거하면 Hungarian이 확보된 여유 capacity를 다른 node를 큰 loop에 흡수하는 데 쓸 수 있다 — thin_slab region 0에서 이게 실제로 일어나 이전에는 안전했던 두 개의 작은 loop(5-node+9-node)를, 자기교차(self-intersecting)하는 16-node "closed loop" 하나로 바꿔버렸다.
- `recover_directed_boundary_components`는 direct/reverse tangent 두 orientation을 계산해 **전체 scene 단위로** 더 나은 쪽을 고르는데(worklog 130 이전부터 있던 기존 로직), `quality()`가 `ordered_closed_loop` 개수만 세고 자기교차 여부를 확인하지 않았다. 이번 수정이 REVERSE orientation의 **다른** region(0이 아님)에서 진짜 2-cycle 낭비를 하나 고쳐 REVERSE의 closed-loop 개수가 2→3으로 올라갔고, 그 결과 전체 quality 비교에서 REVERSE가 DIRECT를 이기게 됐다 — 그런데 REVERSE에는 원래부터(이번 수정과 무관하게, 2-cycle도 없이) 자기교차하는 16-node loop가 region 0에 있었고, 그게 함께 "당첨"돼버렸다.

## 적용한 수정 (2차, 안전 조건 보존)

`recover_directed_boundary_components`의 `quality()`가 `ordered_closed_loop`로 보고된 component라도 `validate_simple_closed_loop`(materialization이 이미 쓰는 것과 동일한 self-intersection 검사)를 통과한 것만 "닫힌 loop"로 인정하도록 수정했다. 자기교차하는 "닫힌" 구조는 어차피 materialization에서 거부되므로, 이걸 quality 비교에서 진짜 승리로 세는 것 자체가 결함이었다 — 이번 2-cycle 수정이 그 결함을 드러낸 것이지 새로 만든 것은 아니다.

## Before/After (Region 56, 3k, cap 2048)

| | before | after |
|---|---|---|
| matched assignment | `1039800↔819956`(2-cycle, 무효), `278207→1110285` | `1039800→819956→278207→1110285`(4-node open path) |
| matched total score | 7.9408 (5.89가 낭비됨) | 7.2239 (전부 유효) |
| cycles | 0 | 0 |
| paths | 2개 fragment | 1개 4-node path |

Region 52는 완전히 불변이다(`['1085315','1020950']`, `['1086120','666904']`) — feasible cycle이 애초에 없으므로 강제 폐쇄하지 않았다.

## Real 3k/5k/10k

| checkpoint | physical(raw/normalized) before -> after | closed regions before -> after |
|---|---:|---|
| 3k | 154/148 -> 154/148 (불변) | [] -> [] |
| 5k | 185/182 -> 185/182 (불변) | [130,141] -> [130,141] (불변) |
| 10k | 125/122 -> 125/122 (불변) | [] -> [] |

3k의 실제 region 56도 합성 테스트와 동일하게 fragment 2개 -> 4-node open path 1개로 병합됐다(직접 확인). candidate 수와 closed-loop 수 자체는 불변이다 — region 56/10k region 52 둘 다 진짜 최선의 결과가 여전히 열린 상태이기 때문이며, 이는 강제로 닫지 않는다는 원칙과 일치한다.

## Negative-control (cap 64)

| fixture | physical | closed | materialized |
|---|---:|---:|---:|
| Box | 51 | 6 | 6 |
| Cylinder | 16 | 2 | 2 |
| Sphere | 14 | 0 | 0 |
| Thin slab | 37 | 3 | 3 |

worklog 52 baseline과 완전히 동일 — 2차 수정(quality의 self-intersection 인식) 이후 전부 복구됐다. Box density-sweep(multiplier 1/2/4/8, cap 128/256)도 전부 `region_count=6` 유지 확인.

## 검증

```text
python -m pytest -q tests/test_directed_boundary_ordering.py tests/test_boundary_topology_safety.py \
    tests/test_visible_surface_construction.py tests/test_full_cloud_continuation_shell.py \
    tests/test_gaussian_surface_region_formation.py tests/test_surface_region_invariance.py \
    tests/test_boundary_adjacency_semantics.py tests/test_cross_region_continuation.py
95 passed in 44.33s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed in 266.78s
```

## 완료 조건 처리

- **Region 56/10k region 52 assignment before/after**: 위에 제시. Region 56은 fragment 2개→4-node open path 1개로 개선(닫히진 않음, 진짜 최선이 open이므로). 10k region 52는 애초에 2-cycle이 없어 완전히 불변(worklog 52의 결론 그대로 — feasible cycle이 matching 선택보다 15% 낮은 진짜 evidence 열세 케이스).
- **valid 3-cycle 선택 시 compatibility/topology safety 통과**: 이번 라운드에서 실제로 3-cycle이 선택된 사례는 없다(region 56은 4-path가 진짜 최댓값이었다) — 하지만 solver 자체에 self-intersection 안전장치를 추가했으므로, 앞으로 3-cycle이 선택되는 경우가 생기면 자동으로 이 조건이 적용된다.
- **Region 52 강제 폐쇄 금지**: 완전히 불변으로 유지됨, 확인.
- **Box 6 / Cylinder 3 / Sphere 0 / Thin slab 분리**: 전부 worklog 52 baseline과 동일하게 복구·유지.
- **real 3k/5k/10k replay 및 pytest**: 위에 제시, 전부 통과.

## 남은 단일 병목

worklog 47~52와 동일하다. 이번 수정은 진짜 matching 낭비(2-cycle)와 그로 인해 드러난 quality 비교의 안전 결함을 고쳤지만, closed-loop 총 개수 자체는 바뀌지 않았다 — 여전히 candidate evidence의 밀도/위상 한계다.
