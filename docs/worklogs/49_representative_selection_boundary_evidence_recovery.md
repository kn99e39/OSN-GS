# Worklog 49: Full-cloud Boundary Evidence → Representative Candidate Coverage 복구

## 배경

Worklog 48은 candidate-local fold/gap-crossing 차단을 완료했고, 남은 병목은 "real observed termination evidence의 밀도/분포가 chain을 이루기에 부족하다"였다. 이번 작업은 그 evidence가 원본 full cloud에는 존재하는데 representative selection 또는 region assignment/candidate admission 단계에서 유실되는지를 stable ID lineage로 확인했다.

## 감사: representative selection 단계

`select_density_preserving_representatives`는 이미 voxel cell마다 `_split_cell_into_modes`로 normal/offset 기준 로컬 mode를 분리한다 — 한 cell에 mode가 2개 이상이면 그 자체로 "이 cell은 서로 다른 두 표면 orientation을 품고 있다"는 증거다. 문제는 그 다음 단계인 weighted-farthest-point 예산 선택이 전체 scene의 모든 mode-candidate와 전역 경쟁을 붙인다는 것 — cap을 넘는 순간 일부 mode가 예산 경쟁에서 밀려 완전히 탈락한다.

`scripts/devtools/trace_representative_selection_boundary_loss.py`(신규)로 production `_voxel_cells`/`_split_cell_into_modes`를 그대로 재사용해 전체 candidate 목록(선택된 것 + 탈락한 것)을 복원하고, "같은 cell 안에서 형제 mode 하나는 선택됐는데 이 mode는 탈락"한 경우를 추출했다.

- 3k: total candidate 5524개 중 2048 선택, 1452개가 "형제-선택-drop" 패턴. 그러나 이 중 대부분(median 형제-정렬 0.94)은 노이즈성 근중복 분리였다.
- 형제와의 normal alignment가 0.3 미만(모드 분리 게이트 0.6보다 훨씬 낮은, 진짜 orientation 차이)이고 source_count(실제 뒷받침하는 원본 Gaussian 수) >= 3인 것만 추리면: 3k 26개, 10k 100개. alignment 최저값은 3k 0.0066, 10k 0.0016 — 거의 직교하는 진짜 crease 신호이고, source_count는 최대 1456(3k)/3006(10k)까지 — 명백히 노이즈가 아닌 실제 evidence.

**결론: representative selection이 실제로 well-supported, 거의 직교하는 boundary evidence를 예산 경쟁으로 탈락시키고 있었다.** 원본 full cloud에는 존재하되 representative 단계에서 유실된 사례로 확정.

## 적용한 수정

`osn_gs/surface/torch_density_preserving_representative_selection.py`:

- `RepresentativeSelectionConfig`에 `boundary_evidence_alignment_max=0.3`, `boundary_evidence_min_source_count=3`를 추가했다.
- 새 함수 `_boundary_evidence_swap_in()`: FPS 예산 경쟁으로 탈락한, 같은 cell 안의 형제-대비 orientation이 진짜로 갈라지는(≤0.3) mode를 결정론적으로 다시 채운다. cap 2048은 그대로 유지 — swap-in 하나당 기존 선택 representative 하나를 정확히 교체한다.
- `select_density_preserving_representatives`가 이 결과를 사용하고, `SelectionDiagnostics.boundary_evidence_swap_in_count`로 몇 개가 교체됐는지 노출한다.

### 안전장치 (여러 차례 반복 끝에 확정)

첫 구현(단순 전역 최근접-중복 대체)은 real 3k에서는 문제없었지만 **box 합성 fixture에서 region_count 6→8**로 회귀시켰다(작은 2개 region이 새로 쪼개짐). 원인을 좁혀가며 다음을 확인·적용했다:

1. **Eviction은 swap-in이 아니라 "형제(cell에서 이긴 mode)"의 orientation 주변에서만** 찾는다 — swap-in 위치 주변에서 찾으면 정작 swap-in을 올바른 face에 연결해줄 대표점을 없애버릴 수 있다.
2. **Redundancy(최근접 거리)만으로는 안전하지 않다** — 진짜 연결 다리(articulation point)를 제거할 수 있다. Eviction 후보는 형제-orientation pool 안에서 **명시적 connected-component 계산**(고정 이웃 그래프, degree>=4 요구)으로 제거해도 component 수가 늘지 않는 것만 허용한다.
3. **Pool이 15개 미만이면 swap을 시도하지 않는다** — 안전성 proxy graph를 신뢰하기엔 국소 밀도가 너무 낮다.
4. **Swap-in 후보 자체도 서로 최소 `median_spacing * 3`만큼 떨어지도록 greedy dedup** — 하나의 real edge가 여러 voxel cell에 걸쳐 있으면 각 cell이 독립적으로 swap을 제안하는데, 이걸 전부 받아들이면 그 edge 한 구간에 eviction이 몰려 face 자체가 kNN 그래프에서 끊긴다. 이 dedup을 넣기 전까지는 위 1-3 안전장치를 다 걸어도 box_4/cap256에서 여전히 region 6→7이 재현됐다 — 실제 원인은 eviction 대상이 아니라 **swap-in 후보 밀도 자체**였다.

이 네 가지를 전부 적용한 뒤 box(multiplier 1/2/4/8, cap 128/256 전 조합)에서 swap-in이 발생하거나(9~57개) 발생하지 않거나 상관없이 **region_count가 항상 원래 값(6)으로 유지**됐다.

## Negative-control 검증

`cap=64`에서 표준 fixture(worklog 47/48과 동일 비교 기준):

| fixture | physical | closed | materialized | region_count | swap_in |
|---|---:|---:|---:|---:|---:|
| Box | 51 | 6 | 6 | 6 | 0 |
| Cylinder | 16 | 2 | 2 | 3 | 0 |
| Sphere | 14 | 0 | 0 | 2 | 0 |
| Thin slab | 37 | 3 | 3 | 4 | 0 |

worklog 48의 baseline과 완전히 동일 — 이 cap에서는 swap-in이 발동하지 않아(pool 크기/spacing 조건) fixture에 영향이 없다.

Box density-sweep(multiplier 1/2/4/8, cap 128/256)에서는 swap-in이 실제로 발동하지만(9~57개) region_count는 항상 6으로 보존됐다.

## Real checkpoint before/after (cap 2048)

swap-in 발동 개수: 3k=22, 5k=51, 10k=76 (전부 형제 alignment<=0.3, source_count>=3, connectivity-safe 조건을 모두 통과한 진짜 evidence).

| checkpoint | physical(raw/normalized) before -> after | closed/materialized before -> after |
|---|---:|---:|
| 3k | 153/153 -> 154/148 | 0/0 -> 0/0 |
| 5k | 184/181 -> 185/182 | 2/2 -> 2/2 |
| 10k | 124/121 -> 125/122 | 0/0 -> 0/0 |

physical candidate 수는 거의 변화가 없고(±1~5, 방향도 checkpoint마다 다름 — representative set이 바뀌며 region 재구성이 재조정된 결과이지 일관된 개선이 아니다), **closed/materialized는 세 checkpoint 전부 완전히 불변**이다.

## 평가

Representative selection이 진짜 boundary evidence를 예산 경쟁으로 잃고 있다는 가설은 **확인됐고**, production 경로를 안전하게 수정해 그 evidence를 복구했다(cap 증가 없이, deterministic swap-in, negative-control 100% 보존). 그러나 이 evidence 손실은 real 3k/5k/10k의 closed-loop 부재를 설명하는 지배적 원인이 **아니었다** — 복구 후에도 closed/materialized는 조금도 움직이지 않았다.

이는 worklog 47/48의 결론과 일치한다: 남은 병목은 representative-selection이나 candidate-admission의 전달 결함이 아니라, 큰 region perimeter에서 true smooth `no_gap`이 우세하고 독립적인 observed termination evidence 밀도 자체가 chain을 이루기에 부족한 것이다 — representative 몇 개를 더 정확한 위치에 복구해도 그 밀도 자체는 바뀌지 않는다. 억지로 닫지 않는다.

## 금지 항목 준수

termination/compatibility threshold 변경 없음, gap 보간이나 강제 폐쇄 없음, shape별 예외 없음, ordering solver/NURBS fitting 변경 없음, representative cap(2048) 증가 없음.

## 검증

```text
python -m pytest -q tests/test_density_preserving_representative_selection.py tests/test_full_cloud_continuation_shell.py tests/test_visible_surface_construction.py
29 passed in 36.55s

python -m pytest -q tests/test_visible_surface_construction.py tests/test_full_cloud_continuation_shell.py \
    tests/test_density_preserving_representative_selection.py tests/test_surface_ownership.py \
    tests/test_adc_synchronized_visible_nurbs.py tests/test_uncertain_gaussian_append_adapter.py \
    tests/test_training_regressions.py tests/test_gaussian_surface_region_formation.py \
    tests/test_surface_region_invariance.py tests/test_representative_graph_scale.py \
    tests/test_boundary_topology_safety.py tests/test_directed_boundary_ordering.py
211 passed in 60.77s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed in 268.94s
```

Full pytest 통과 수(720 passed, 1 skipped)는 worklog 47/48과 동일 — 회귀 없음.

## 남은 단일 병목

Worklog 47/48과 동일 — 3k/10k의 real evidence 밀도 자체가 perimeter를 chain으로 닫기에 부족하다. representative selection과 candidate admission 양쪽 다 이번에 감사·확인했고 (representative selection은 실제 결함을 찾아 안전하게 수정, candidate admission/region assignment 쪽은 이번 라운드에서 별도 결함을 발견하지 못함) closed-loop 수를 움직이지 못했다는 사실 자체가 병목이 알고리즘/전달 결함이 아니라 real checkpoint의 관측 밀도 문제임을 다시 확인시켰다.
