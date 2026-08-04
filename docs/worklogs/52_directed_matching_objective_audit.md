# Worklog 52: Directed Matching Competition 감사

## 배경

Worklog 51은 region 52의 `666904 ↔ 1086120`, region 56의 `1110285 ↔ 278207`이 raw·representative 양쪽에 physical evidence가 있고 compatibility도 통과하지만 "Hungarian matching 경쟁에서 선택되지 않아 chain이 열린다"고 보고했다. 이번 작업은 그 정확한 objective/assignment 원인을 확정하고, 진짜 matching 결함이면 production solver를 수정하는 것이었다.

## 감사 방법

`scripts/devtools/trace_directed_matching_objective_audit.py`(신규)로 production `_compatible_directed_edges`/`_max_weight_one_in_one_out_matching`/`_decompose_into_paths_and_cycles`를 그대로 재사용해:

- 두 region의 **전체** compatible directed edge와 각 edge의 score 성분(forward, lateral, normalized distance, tangent/normal alignment)을 전부 기록하고,
- 실제 선택된 matching과 그 총점을 기록하고,
- **완전 탐색**(region이 4~6개 candidate뿐이라 permutation 전수 탐색이 정확하고 저렴함)으로 길이 3 이상인 feasible cycle 중 최고 점수 조합을 찾아 matching의 선택과 비교했다.

## 결과: worklog 51의 "탈락" 서술을 먼저 정정

`666904↔1086120`과 `1110285↔278207` edge는 실제로는 **matching에서 탈락하지 않았다** — 둘 다 최종 assignment에 그대로 포함돼 있다(region 52: `1086120 -> 666904` score 3.1172, region 52 내 최고점; region 56: `278207 -> 1110285` score 2.0504). worklog 51의 결론은 각 endpoint의 "가장 가까운 후보 1개"만 보는 진단 스크립트의 `first_gate`에 근거했는데, 이는 전체 그래프의 one-in/one-out feasibility를 보지 못한다. 실제 fragmentation의 원인은 이 두 edge가 아니라 **같은 region의 다른 candidate들**에 있었다.

### Region 52 (candidates: 1020950, 1085315, 1086120, 666904)

전체 compatible edge는 4개뿐이다: `1085315→1020950`, `1085315→1086120`, `1085315→666904`, `1086120→666904`. `1020950`과 `666904`는 **outgoing edge가 단 하나도 없다** — 즉 이 두 노드는 어떤 조합으로도 cycle의 일부가 될 수 없다(cycle은 모든 멤버가 next-hop을 가져야 함). `1085315`는 incoming edge가 하나도 없다. 완전 탐색 결과 **길이 3 이상 feasible cycle이 단 하나도 존재하지 않는다**(`best_feasible_cycle_score = None`). Matching은 가능한 최선(두 개의 2-노드 경로, 총점 5.2767)을 정확히 찾았다 — 이건 matching 결함이 아니라 candidate 자체의 위상적 한계다.

### Region 56 (candidates: 1039800, 1110285, 278207, 819956)

6개 compatible edge가 있고, 실제로 길이-3 feasible cycle이 **존재한다**: `819956→278207→1110285→819956`, 점수 6.4619. 하지만 matching이 실제로 선택한 assignment(`1039800→819956`, `819956→1039800`(2-cycle, 길이<3이라 path로 강등), `278207→1110285`)의 총점은 **7.9408**로, feasible cycle보다 **1.4789점(23%) 더 높다**. 이 차이는 tie가 아니다 — `1039800→819956`(score 3.1818)과 `819956→1039800`(score 2.7086)이라는 두 개의 강한 edge가, cycle을 만들려면 반드시 포기해야 하는 `819956→278207`(score 1.9918)보다 명백히 우월한 evidence이기 때문이다.

## 판정: Hungarian solver 결함 없음

두 region 모두 **objective/assignment 결함이 아니다**:

- Region 52는 feasible cycle이 아예 존재하지 않는다(구조적 불가능).
- Region 56은 feasible cycle이 존재하지만 명백히 더 약한 evidence(1.48점, 23% 차이)를 강제로 골라야만 닫힌다 — 이건 이번 task가 명시적으로 금지한 "primary evidence가 열등한 edge를 closure만을 위해 선택"하는 경우에 정확히 해당한다.

score 공식(`forward/distance + tan_align + normal_align + outward_align - lateral/max_lateral`) 자체도 감사했다: 모든 항이 `[0,1]` 범위의 무차원 비율이고 부호(페널티 항만 감산)도 일관돼 있어 단위/부호 결함은 없었다. Tie-break 여부도 확인했으나 두 region 다 진짜 tie가 아니므로(1.48/1.27점 차이) lexicographic tie-break를 적용해도 결과가 달라지지 않는다 — 적용할 tie가 없다.

## 추가 확인 (10k)

같은 감사를 10k의 fragmented region 4곳(104, 6, 11, 52)에도 적용했다. 3곳은 feasible cycle이 전혀 없고, 1곳(region 52)은 feasible cycle(점수 7.4278)이 matching의 실제 선택(8.6985)보다 1.2706점(15%) 낮다 — 3k와 동일한 패턴이다. 이는 우연이 아니라 이 checkpoint들의 fragmentation이 일관되게 candidate 위상 부족 또는 evidence 우열 문제이지 solver 결함이 아님을 보여준다.

## 결론

Region 52, 56 둘 다 실제 objective상 열등하거나(56) 아예 topology적으로 불가능(52)하다는 근거를 확정했다. Production solver는 변경하지 않았다 — 변경할 결함이 없었다.

## 완료 조건 처리

- **region 52/56의 탈락 원인과 before/after assignment**: 위에 정확한 score와 완전 탐색 결과로 제시했다. production 변경이 없으므로 before/after는 동일하다.
- **결함 확인 시 production 수정**: 결함이 확인되지 않아 수정하지 않았다.
- **real 3k/5k/10k closed/materialized 전후 비교**: 코드 변경이 없으므로 worklog 51의 최종 상태(physical 154/185/125, closed/materialized 0/0, 2/2, 0/0)가 그대로 유지된다.
- **negative control / pytest**: 코드 변경이 없어 자동으로 유지되지만, 완료 조건에 따라 재실행해 확인했다.

```text
python -m pytest -q tests/test_directed_boundary_ordering.py tests/test_boundary_topology_safety.py tests/test_visible_surface_construction.py
43 passed in 11.27s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed
```

## 남은 단일 병목

worklog 47~51과 동일하다. 이번 감사는 그 결론을 ordering/matching 단계에서도 재확인했을 뿐이다 — 남은 fragmentation은 candidate evidence 자체(밀도·위상)의 한계이지, no_gap 분류·representative selection·raw-to-representative 전달·directed matching 중 어느 단계의 결함도 아니다. 억지로 닫지 않았다.
