# Boundary-proximate `no_gap` local evidence 복구

## 수행 내용

- frozen real checkpoint `3000`, `5000`, `10000`을 representative cap `2048`로 replay했다.
- `scripts/devtools/trace_real_physical_candidate_chains.py`를 확장했다. 주요 region의 `no_gap`, `parallel_sheet_conflict`, `crease_discontinuity`, `ambiguous_continuation` node마다 support radius, same-mode/competing mass, supporting stable ID fingerprint, world normal, outward arc, bounded affinity relation count와 residual range를 JSON으로 기록한다.
- 5k의 closed region `133`/`143`을 대조군으로 사용했다. 각각 observed termination 3/4개가 실제 loop를 이루며, 같은 region 안의 parallel/ambiguous diagnostic node는 loop에 포함되지 않았다.

## 확인한 production 결함

기존 `build_continuation_shells()`는 `same_mode`, `parallel_conflict`, `crease`, `ambiguous`의 footprint를 하나의 `occupied` angular bin에 모두 기록했다. 따라서 local parallel/crease/competing support가 same-mode gap을 전부 덮으면 `best_length == 0`이 되어 `no_gap`으로 반환됐다. 이는 "smooth continuation"이 아니라 "어떤 mode든 존재"를 의미하므로 cross-surface leakage였다.

## 적용한 수정

`osn_gs/surface/torch_full_cloud_continuation_shell.py`에서 다음을 적용했다.

- same-mode footprint만 기록하는 `same_mode_occupied`를 별도로 만들었다.
- `no_gap`은 오직 same-mode coverage의 largest gap이 termination minimum보다 작을 때만 반환한다.
- 이전에는 `no_gap`이던 node에서 same-mode gap을 local parallel/crease/competing evidence가 덮는 경우, physical termination으로 승격하지 않고 각각 `parallel_sheet_conflict`, `crease_discontinuity`, `ambiguous_continuation`으로 fail-closed 처리한다.
- 기존 total gap이 실제로 존재하던 termination의 outward arc와 state path는 그대로 유지했다.
- policy version을 `full_cloud_continuation_shell_worklog130_v2`로 올렸다.

Ordering, topology threshold, histogram threshold, seed admission, gap interpolation, Hungarian solver, NURBS fitting은 변경하지 않았다.

## Real evidence 분포

아래는 수정 후 continuation query state의 분류다. `no_gap`은 true smooth continuation, parallel+crease는 false continuation caused by nonlocal/cross-surface support, 나머지 typed uncertainty는 insufficient/ambiguous evidence로 해석한다.

| checkpoint | true smooth (`no_gap`) | false continuation (parallel+crease) | insufficient/ambiguous | physical candidate | closed/materialized |
|---|---:|---:|---:|---:|---:|
| 3k | 252 | 368 | 21 | 153 | 0 / 0 |
| 5k | 181 | 384 | 22 | 181 | 2 / 2 |
| 10k | 217 | 317 | 32 | 121 | 0 / 0 |

대표 local evidence:

- 3k true smooth: stable ID `31845`, same-mode support 385, bounded graph relation `same_surface=7`, `parallel_but_separate=2`.
- 3k false continuation: stable ID `809871`, state `parallel_sheet_conflict`, same-mode support 142, competing mass 86.34, bounded graph relation `parallel_but_separate=1`.
- 10k true smooth: stable ID `1183889`, same-mode support 748, bounded graph relation에는 same-surface와 crease/parallel/ambiguous가 함께 있으나 same-mode coverage 자체는 circularly complete했다.
- 10k false continuation: stable ID `1100200`, state `parallel_sheet_conflict`, same-mode support 153, competing mass 491.63, bounded graph relation `parallel_but_separate=4`, crease=1.
- 5k closed region `133`/`143`은 observed termination source가 각각 3/4개로 유지됐다. 해당 region의 competing diagnostic source는 physical loop를 늘리거나 줄이지 않았다.

## 전후 replay

| checkpoint | physical before -> after | closed/materialized before -> after | nonphysical parallel/crease after |
|---|---:|---:|---:|
| 3k | 153 -> 153 | 0 / 0 -> 0 / 0 | 335 / 1 |
| 5k | 181 -> 181 | 2 / 2 -> 2 / 2 | 356 / 2 |
| 10k | 121 -> 121 | 0 / 0 -> 0 / 0 | 302 / 2 |

수정은 loop 수를 늘리기 위한 것이 아니다. 이전에 smooth `no_gap`으로 사라지던 cross-surface evidence를 nonphysical typed diagnostic으로 보존했고, physical ordering input은 그대로 유지했다.

## Fixture와 검증

`tests/test_full_cloud_continuation_shell.py`에 다음 local fixture를 추가했다.

- complete smooth ring -> `no_gap`
- close-parallel ring -> `parallel_sheet_conflict`
- mixed smooth/crease ring -> `crease_discontinuity`
- smooth/gap -> `observed_support_termination`
- smooth ring의 단일 touching parallel point -> `no_gap`

Synthetic contract:

| fixture | physical | closed | materialized |
|---|---:|---:|---:|
| Box | 110 | 6 | 6 |
| Cylinder | 74 | 3 | 3 |
| Sphere | 0 | 0 | 0 |
| Thin slab | 48 | 2 | 2 |

Focused suite:

```text
84 passed in 33.97s
```

Repository-wide pytest:

```text
720 passed, 1 skipped, 1 warning, 8 subtests passed in 199.85s
```

## 남은 단일 병목

3k/10k는 false continuation이 typed nonphysical state로 정확히 분리된 뒤에도 physical candidate와 closed loop 수가 증가하지 않는다. 남은 병목은 true smooth `no_gap` source가 큰 region perimeter에서 우세하고, 별도의 observed termination source가 chain을 이룰 만큼 충분하지 않은 real evidence coverage다.
