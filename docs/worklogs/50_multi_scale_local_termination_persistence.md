# Worklog 50: Multi-scale Local Termination 복구

## 배경

Worklog 49의 boundary-evidence swap-in은 그대로 두고, 이번 작업은 `build_continuation_shells()`가 same-mode support를 모으는 반경(`radius = max(6x tangent_major_scale, 4x representative_mean_spacing)`, 사실상 단일 `4x candidate_scale` neighborhood)이 가까운 곳의 진짜 gap을 더 먼 same-surface support로 덮어버려 `no_gap`을 만드는지 확인하는 것이었다.

## 감사: multi-scale sweep

`scripts/devtools/trace_multi_scale_termination_persistence.py`(신규)로 production `build_continuation_shells_from_input`을 `radius_spacing_multiplier ∈ {1, 2, 3, 4}`(1x~4x, 4x가 production 기본값)에서 그대로 재실행해 major region node와 5k의 실제 closed region(130, 141 — worklog 49의 swap-in으로 region id가 133/143에서 이동)의 state 변화를 추적했다.

Scene 전체 집계(모든 representative-eligible node, 3k 기준):

| scale | observed_support_termination | no_gap |
|---|---:|---:|
| 1x | 308 | 153 |
| 2x | 208 | 200 |
| 3x | 162 | 234 |
| 4x (production) | 154 | 243 |

10k도 동일한 단조 패턴(termination 229→125, no_gap 151→201)이었다. 반경이 커질수록 termination이 줄고 no_gap이 느는 것 자체는 반경이 커지면 더 많은 support를 보게 되는 당연한 결과일 수 있어, 이 집계만으로는 결함 여부를 판단할 수 없었다.

**결정적 대조군**은 5k의 실제 closed region(130, 141) 멤버였다: 이 region의 모든 node는 1x~4x 전 scale에서 `no_gap`을 단 한 번도 보이지 않았다(`observed_support_termination`이거나, 모든 scale에서 일관되게 `parallel_sheet_conflict`이거나, 4x에서 termination으로 복귀). 반면 3k/10k의 major region에서는 "1x~3x에서 well-supported(23~267개 same-mode Gaussian)된 26~74도 gap이 존재하다가 4x에서만 먼 support가 그 gap을 닫아 no_gap이 되는" node가 3k 8개/10k 6개 확인됐다. 5k 대조군에는 이 패턴이 전혀 없었다.

**결론: 단일 4x 반경이 가까운 진짜 gap을 먼 support로 덮어 `no_gap`을 만드는 결함이 확인됐다.**

## 적용한 수정

`osn_gs/surface/torch_full_cloud_continuation_shell.py`:

- `ContinuationShellConfig`에 `persistence_check_radius_ratio=0.5`를 추가했다.
- production radius의 절반(같은 `same_mode_local` 멤버 집합의 부분집합 — 새 쿼리 없음)만으로 same-mode angular occupancy를 다시 계산해 `persistence_length`(작은 반경에서의 최대 gap)를 구한다.
- 기존 코드가 `same_mode_length < gap_threshold_bins` (전체 반경 기준 "닫힘")로 `no_gap`을 승인하려는 시점에, `persistence_length >= gap_threshold_bins`(작은 반경에서는 아직 안 닫힘) **그리고** `persistence_same_mode_count >= min_same_mode_support_for_termination`(작은 반경의 gap이 sparsity 착시가 아니라 실제 근거가 있음)이면 `no_gap`을 승인하지 않고 `STATE_AMBIGUOUS`로 fail-closed 처리한다. 두 조건 다 만족하지 않으면(즉 작은 반경도 이미 닫혀 있거나, 작은 반경의 gap이 근거 부족이면) 기존 동작 그대로 `no_gap`을 승인한다.
- `observed_support_termination` 승인 로직 자체는 변경하지 않았다 — 이번 결함은 명확히 "no_gap 오판"이었고, termination 승인에 대한 추가 방향-일치 요구는 5k의 기존 정상 동작(oscillate 하다가 4x에서 termination으로 돌아오는 node 포함)을 건드릴 위험이 있어 이번 범위에 넣지 않았다.
- policy version을 `full_cloud_continuation_shell_worklog130_v4`로 올렸다.

Threshold 완화, scene별 예외, selection/ordering/compatibility/gap 보간/NURBS fitting은 변경하지 않았다.

## Real checkpoint before/after (cap 2048)

| checkpoint | no_gap before -> after | ambiguous_continuation before -> after | observed_support_termination | physical(raw/normalized) | closed/materialized |
|---|---:|---:|---:|---:|---:|
| 3k | 243 -> 167 | 18 -> 94 | 154 -> 154 (불변) | 154/148 -> 154/148 (불변) | 0/0 -> 0/0 |
| 5k | 174 -> 138 | 19 -> 55 | 185 -> 185 (불변) | 185/182 -> 185/182 (불변) | 2/2 -> 2/2 (region 130/141 유지) |
| 10k | 201 -> 162 | 28 -> 67 | 125 -> 125 (불변) | 125/122 -> 125/122 (불변) | 0/0 -> 0/0 |

`no_gap → ambiguous_continuation`으로 정확히 재분류된 수만큼(3k 76개, 5k 36개, 10k 39개) 이동했고, `observed_support_termination`과 physical candidate 수는 세 checkpoint 전부 **정확히 불변**이다 — 이 수정은 잘못된 `no_gap` 판정을 타입화된 불확실 상태로 고쳤을 뿐, 어떤 node도 강제로 physical termination으로 승격시키지 않았다(ambiguous는 physical candidate를 만들지 않는다). closed/materialized도 세 checkpoint 전부 불변이다.

## Negative-control 검증

`cap=64`에서 표준 fixture 4종(worklog 47/48/49와 동일 비교 기준):

| fixture | physical | closed | materialized |
|---|---:|---:|---:|
| Box | 51 | 6 | 6 |
| Cylinder | 16 | 2 | 2 |
| Sphere | 14 | 0 | 0 |
| Thin slab | 37 | 3 | 3 |

worklog 49 baseline과 완전히 동일 — 합성 fixture는 균일 밀도라 persistence 반경에서도 이미 충분히 조밀해 이 수정에 전혀 영향받지 않는다.

## 검증

```text
python -m pytest -q tests/test_full_cloud_continuation_shell.py
15 passed in 11.90s

python -m pytest -q tests/test_visible_surface_construction.py tests/test_full_cloud_continuation_shell.py \
    tests/test_density_preserving_representative_selection.py tests/test_surface_ownership.py \
    tests/test_adc_synchronized_visible_nurbs.py tests/test_uncertain_gaussian_append_adapter.py \
    tests/test_training_regressions.py tests/test_gaussian_surface_region_formation.py \
    tests/test_surface_region_invariance.py tests/test_representative_graph_scale.py \
    tests/test_boundary_topology_safety.py tests/test_directed_boundary_ordering.py
211 passed in 64.61s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed
```

## 5k와 10k의 boundary evidence 차이

5k가 2개 loop를 닫는 이유는 "single-radius suppression이 없어서"가 아니다 — closed region(130/141)의 termination node들은 애초에 no_gap을 전혀 보이지 않았고, 이번에 고친 결함의 영향을 받지도 않았다. 5k와 3k/10k의 진짜 차이는 region 크기와 perimeter당 관측 밀도다: 5k의 성공 region은 candidate가 4개뿐인 작은 region이라 gap이 항상 관측 가능한 범위 안에 있었던 반면, 3k/10k의 큰 region은 perimeter 자체가 넓어 real observed termination evidence 밀도가 chain을 이루기에 부족하다(worklog 47/48/49와 동일 결론).

## 남은 단일 병목

이번 수정은 진짜 결함(단일 큰 반경이 근거리 gap을 원거리 support로 덮는 것)을 확인하고 정확히 고쳤지만, 그 결함은 `no_gap` 오분류였지 physical candidate 부족의 원인이 아니었다. worklog 47/48/49와 동일하게, 남은 병목은 큰 region perimeter의 real observed termination evidence 밀도 그 자체다. 억지로 닫지 않았다.
