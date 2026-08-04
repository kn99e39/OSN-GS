# Worklog 48: Candidate-local Smooth Continuation 복구

## 배경

Worklog 47은 `build_continuation_shells()`의 same-mode footprint가 parallel/crease/ambiguous와 섞여 기록되던 cross-surface leakage를 고쳤지만, 3k/10k의 physical candidate/closed loop 수는 그대로였다. 남은 결론은 "큰 region perimeter에서 true smooth `no_gap`이 우세해 observed termination chain이 형성되지 않는다"였다.

이번 작업은 그 `no_gap`을 만드는 same-mode support 자체가 candidate 주변에서 실제로 국소 연속된 표면인지 감사했다.

## 감사 방법

`scripts/devtools/trace_no_gap_local_connectivity.py`를 새로 작성했다. `build_continuation_shells`가 계산하는 것과 동일한 pointwise same-mode 마스크(normal alignment / tangent residual / footprint ratio)를 재현하되, 각 same-mode full-cloud member에 대해 추가로:

- `nearest_representative_index`로 그 member의 소유 representative를 찾고,
- query representative에서 그 representative까지 **region 자신의 `internal_accepted_edge_ids` 그래프**로 BFS(무제한 hop)를 돌려 도달 가능한지, 몇 hop인지 측정하고,
- 도달 가능하면 `hop_count * 이 노드의 representative 간격` 대 실제 직선거리(straight-line distance) 비율을 계산했다.

첫 시도는 hop을 2로 제한한 BFS였는데, region 43(3k, 멤버 14개)에서 실제로는 hop 3~5에 있는 정상 이웃까지 "unreachable"로 잘못 분류하는 결함이 있어 무제한 BFS + "같은 accepted-edge connected component인가"로 교정했다. 이 교정 후에는 3k/5k/10k 전부 no_gap same-mode support가 100% 같은 connected component 안에 있었다 — union-find가 accepted edge 없이 두 조각을 억지로 합친 흔적(worklog 35의 parallel-shortcut 같은)은 이 real 데이터에서 확인되지 않았다.

그 다음 "직선거리 대비 과도하게 먼 경로" 신호(경로 길이가 직선거리의 3배 이상, hop 2 이상)를 추가하자 국소적이지만 겉보기 same-mode인 support가 드러났다: 3k/5k에서 각각 region 8/61/76/54와 66/55/83의 no_gap node 몇 개가 same-mode support의 일부를 hop 2~6, fold_ratio 3~7배 경로로만 갖고 있었다 — 즉 "직선상으로는 근처지만, region의 accepted topology는 그 근처로 가려면 멀리 돌아가야 한다"는 fold/gap-crossing 서명이다.

## 분류 결과 (3k/10k vs 5k 성공 region 133/143)

major region의 no_gap node 89(3k)/53(5k)/62(10k)개 중:

| checkpoint | local_connected_smooth | mixed_local_and_nonlocal | 순수 nonlocal |
|---|---:|---:|---:|
| 3k | 81 | 8 | 0 |
| 5k | 50 | 3 | 0 |
| 10k | 61 | 1 | 0 |

5k의 성공한 closed region(133, 143)에는 no_gap node가 아예 없었다 — 두 region 다 candidate(4개)가 곧 눈에 보이는 gap node로만 구성되고, no_gap으로 판정되는 대형 perimeter 자체가 없었다. 즉 성공 사례는 "no_gap 오염이 없어서" 성공한 게 아니라 애초에 region이 작고 gap이 대부분 관측 가능해서 성공했다.

순수 nonlocal(same-mode support 전체가 국소 연결 안 됨)은 0건이었다 — 발견된 결함은 "완전 오염"이 아니라 "일부 지분 오염"이다.

## 확인한 production 결함

`build_continuation_shells()`의 same-mode 판정은 순수 pointwise 기하 조건(법선 정렬, tangent-offset residual, footprint ratio)뿐이었고, 그 member가 query와 candidate-local하게 연결되어 있는지는 전혀 확인하지 않았다. Radius 포함 + pointwise 정렬만으로 same-mode를 인정했기 때문에, 직선거리로는 반경 안에 들어오지만 실제로는 region의 accepted topology를 몇 hop 돌아가야 닿는 support가 same-mode gap을 메워 `no_gap`을 만들 수 있었다.

## 적용한 수정

`osn_gs/surface/torch_full_cloud_continuation_shell.py`:

- `ContinuationShellConfig`에 `fold_signature_min_hops=2`, `fold_signature_path_ratio_min=3.0`을 추가했다. hop 1(accepted 그래프가 직접 인정한 이웃)은 정의상 fold 대상에서 제외한다.
- 새 헬퍼 `_accepted_hop_distances`: 한 region의 accepted-edge adjacency 위에서 무제한 BFS. 긴 hop 자체는 문제가 아니다 — 진짜 연속된 얇은 region은 accepted-edge 지름이 커도 정상이다. 문제는 hop을 실제 거리로 환산한 경로 길이가 직선거리보다 압도적으로 길 때(같은 component이지만 돌아가야만 닿는 경우)다.
- `build_continuation_shells`/`build_continuation_shells_from_input`에 `region_internal_accepted_edges`를 새로 plumbing했다. `_from_input`은 `region_result.regions`에서 이미 존재하는 `internal_accepted_edge_ids`를 그대로 전달하므로 프로덕션 호출부(`torch_visible_surface_construction.py`)는 변경이 필요 없었다.
- same-mode pointwise 마스크(`same_mode`)는 그대로 두고, 그 위에서 `same_mode_local`(gap을 닫을 자격) / `nonlocal_same_mode`(자격 없음, 새 카테고리 `nonlocal_same_mode`)로만 나눴다. `parallel_conflict`/`crease`/`ambiguous`는 원래의 `same_mode` 기준 그대로 정의해 영향받지 않는다.
- `same_mode_occupied`(= `no_gap` 판정에 쓰이는 각도 커버리지)는 `same_mode_local`만으로 계산한다. `nonlocal_same_mode`는 general `occupied`(전체 점유)에는 포함시켜 "거기 뭔가 있다"는 사실은 보존하되, same-mode gap을 닫지는 못한다.
- gap이 확인된 뒤 state 분류에서 `nonlocal_same_mode`가 gap 경계에 있으면 `STATE_AMBIGUOUS`로 fail-closed 처리한다(기존 `"ambiguous" in border_categories` 분기에 병합). 즉 국소 경로가 없거나 모순되면 physical candidate(`observed_support_termination`)로 강제 승격하지 않는다.
- ordering, histogram threshold, topology threshold, seed admission, gap 보간, Hungarian solver, NURBS fitting은 변경하지 않았다.

## Before/After 실 checkpoint replay (cap 2048)

| checkpoint | no_gap | parallel_sheet_conflict | physical(raw/normalized) | closed/materialized |
|---|---:|---:|---:|---:|
| 3k | 252 -> 251 | 367 -> 368 | 153 -> 153 (불변) | 0/0 -> 0/0 |
| 5k | 181 -> 180 | 382 -> 383 | 181 -> 181 (불변) | 2/2 -> 2/2 |
| 10k | 217 -> 217 (불변) | 315 -> 315 (불변) | 121 -> 121 (불변) | 0/0 -> 0/0 |

3k/5k에서 각 1개 node가 `no_gap`에서 `parallel_sheet_conflict`로 재분류됐다 — fold-signature support를 제외하자 실제로 same-mode gap이 열렸고, 그 경계에 이미 있던 parallel evidence가 정직하게 타입화됐을 뿐, 어느 것도 `observed_support_termination`으로 강제 승격되지 않았다. 10k에서 유일하게 걸렸던 node(region 41, stable id 3834009, same_mode 255개 중 nonlocal 44개)는 nonlocal support를 제외해도 남은 local support(211개)가 여전히 원을 다 덮어 `no_gap`으로 정직하게 유지됐다 — 강제로 바꾸지 않는다는 원칙대로다.

physical candidate 수와 closed/materialized 수는 세 checkpoint 전부 불변이다 — 이번 수정은 결함을 정확히 타입화했을 뿐, 남은 병목(큰 region perimeter의 evidence 밀도 부족)을 해소하지 않는다.

## Negative-control 검증

`scripts/devtools/compare_fold_signature_toggle.py`로 fold-signature gate를 비활성화한 config(`fold_signature_path_ratio_min=1e18`)와 기본 config을 같은 cap(64)에서 A/B 비교했다.

| fixture | physical | closed | materialized | 기본 config와 차이 |
|---|---:|---:|---:|---|
| Box | 51 | 6 | 6 | 없음 |
| Cylinder | 16 | 2 | 2 | 없음 |
| Sphere | 14 | 0 | 0 | 없음 |
| Thin slab | 37 | 3 | 3 | 없음 |

네 fixture 전부 fold-signature gate를 켜고 끄고에 차이가 없다 — 합성 negative control에는 애초에 fold/gap-crossing 서명이 없으므로 이 수정이 건드리지 않는다. (Sphere의 physical=14는 이 A/B 비교 스크립트가 쓴 cap=64 자체의 특성이며 fold-signature gate와 무관 — 두 config에서 동일하다.)

## 검증

```text
python -m pytest -q tests/test_full_cloud_continuation_shell.py
15 passed in 11.23s

python -m pytest -q tests/test_visible_surface_construction.py tests/test_full_cloud_continuation_shell.py \
    tests/test_density_preserving_representative_selection.py tests/test_surface_ownership.py \
    tests/test_adc_synchronized_visible_nurbs.py tests/test_uncertain_gaussian_append_adapter.py \
    tests/test_training_regressions.py
159 passed in 53.78s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed in 207.58s
```

Full pytest 통과 수(720 passed, 1 skipped)는 worklog 47과 동일 — 회귀 없음.

## 남은 단일 병목

이번 라운드는 `no_gap`의 진짜 오염(fold/gap-crossing support가 same-mode gap을 메우는 것)을 정확히 찾아 고쳤지만, 그 오염은 87~98%의 no_gap node에는 없었고 순수 오염은 0건이었다. 3k/10k의 physical candidate/closed loop 수는 이번 수정으로 변하지 않았다 — 남은 병목은 여전히 worklog 47이 보고한 대로, 큰 region perimeter에서 real observed termination evidence 자체의 밀도/분포가 chain을 이루기에 부족한 것이다. 이건 threshold나 알고리즘 결함이 아니라 real checkpoint의 관측 밀도 문제로, 억지로 닫지 않는 원칙을 유지한다.
