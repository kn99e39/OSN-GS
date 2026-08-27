# Worklog 119-3 — Performance Track Phase 1~4

## 상태

**구현·검증 완료, 후보 미채택.** 기존 serial chart path와 `torch.cdist` exact KNN reference path를 그대로 유지한다. 이 작업은 Main Architecture Track과 독립된 Performance Track이며, Worklog 119의 scientific result를 변경하거나 재해석하지 않는다.

## 승인 계약

- chart/component/topology/ID/order exact identity
- chart별 mathematical dependency, solve/projection 횟수, membership, metric semantics 불변
- continuous tensor 초기 기준 `atol=1e-6`, `rtol=1e-5`
- tolerance 외에 per-chart delta, worst case, NaN/Inf parity, pathological chart, research conclusion invariance 검증
- official plan/backend 고정 및 기록; runtime OOM splitting·silent fallback 금지
- CUDA Graph, multi-stream, custom chart CUDA kernel 금지
- exact KNN은 full-scene neighbor→graph→partition exact identity 후에만 채택

## Phase 1 — 실제 chart corpus와 deterministic plan

동일 WL119 checkpoint, `images_8`, 8 views, 앞 512 charts를 별도 immutable corpus로 추출했다.

- corpus: `output/confirmed/119_performance_track_20260827/wl119_chart_corpus_8v_512.pt`
- chart 512개, pixel 206,889개
- min/median/p95/max: 32 / 60 / 486.2 / 114,571 pixels
- bucket count: `<=64` 277, `65~128` 132, `129~256` 57, `257~512` 21, `513~1024` 14, `1025~2048` 6, `2049~4096` 2, `>4096` 3
- top-10 `(chart_id, pixels)`: `(333,114571)`, `(17,16540)`, `(258,9394)`, `(298,3674)`, `(299,2051)`, `(155,1741)`, `(279,1687)`, `(232,1585)`, `(149,1533)`, `(218,1456)`
- plan digest: `815c1fc7a2034817373e8719306c5ad131a3bec42b401dbfb7ad3b9ae098edeb`
- batch plan은 ordered chart lengths와 고정 bucket/point limits만으로 생성되며 report에 전체 batch 목록을 기록한다. 실행 중 재분할은 없다.
- 최소/경계/상위 10개/oversize chart를 pathological validation set에 포함했다.

추가한 opt-in corpus export는 `--performance-corpus-out`을 지정할 때만 chart fitting 전에 종료한다. 옵션이 없는 기존 WL119 serial report 경로는 그대로다.

## Phase 2 — batched LSQ 후보

고정 8×4 degree-2 topology를 이용해 `[B,32,32]` system과 `[B,32,3]` RHS를 묶는 후보를 구현했다. 그러나 synthetic regular/near-line/duplicate/large-offset corpus에서 batched solve/reduction 순서가 control grid 일부를 초기 tolerance 밖으로 이동시켰다.

- regular CPU worst control delta: `1.28e-5`
- regular CUDA에서도 초기 기준을 넘는 원소가 발생
- near-line chart에서는 후속 projection까지 차이가 증폭

허용오차로 덮지 않고 batched normal-system/solve 후보를 제거했다. 채택 경로는 seed, 모든 normal-system/solve, ARM A의 다음 solve에 들어가는 중간 projection을 immutable reference 호출 순서로 유지한다.

## Phase 3 — dependency-terminal projection/evaluation batching

두 번째 solve 이후의 ARM A final projection과 ARM B evaluation projection은 더 이상 solve에 피드백되지 않으므로 두 terminal projector만 묶는 후보를 검증했다.

### 후보 1: grid initialization부터 terminal projection 전체 batch

- serial reference: 19.153초
- candidate: 7.840초
- 잠재 speedup: 2.443배
- NaN/Inf parity: 통과
- continuous equivalence / pathological / conclusion invariance: 실패

Batched surface-grid/`cdist`가 nearest initialization을 바꾸면서 일부 point가 다른 Gauss–Newton basin으로 진입했다. 이 후보는 즉시 미채택했다.

### 후보 2: grid evaluation/nearest는 chart별 exact, GN update만 batch

nearest basin을 reference와 같게 고정해 재검증했다.

- serial reference: 19.015초
- candidate: 9.245초
- 잠재 speedup: 2.057배
- control grid A/B: 512 charts 전부 exact
- NaN/Inf parity: 통과
- 초기 continuous 기준 실패: 509/512 charts
- field별 실패 chart: `uv_footpoint` 506, `uv_geo_b` 506, fitted point 390, normal 509, G-A 331, G-B 345, C-A 246, C-B 280
- research winner/tie 관계 변경 chart: 8개 (`130,168,186,204,458,470,474,481`)

통과한 3개는 plan상 애초 `>4096` oversize라 serial-reference로 실행된 chart였다. 따라서 실패 ID blacklist를 만들면 사실상 모든 batched chart를 serial로 돌리는 과적합이 된다. Explicit blacklist 방안도 채택하지 않았다.

최종 Phase 2~3 판정은 **POTENTIAL_SPEEDUP_CONFIRMED_BUT_EQUIVALENCE_REJECTED**다. Candidate/harness는 Performance Track evidence로만 남고 production/default 경로에 연결하지 않는다.

## Phase 4 — exact KNN backend adoption gate

설치된 후보 중 `scipy.spatial.cKDTree(eps=0)` exact Euclidean query를 독립 모듈로 구현했다. row-index self exclusion을 명시하고, 선택된 좌표의 최종 거리는 reference와 같은 torch 좌표 차이 norm으로 재계산했다. 작은 distinct-random corpus에서는 index/distance exact identity를 통과했고 coincident point도 자기 row만 제외했다.

1,190,469 surfel full scene 결과:

| 항목 | `torch.cdist` reference | cKDTree candidate |
|---|---:|---:|
| KNN+graph 시간 | 70.552초 | 0.647초 |
| 잠재 speedup | 1.0배 | 108.98배 |
| candidate edges | 6,016,599 | 6,016,779 |
| spatial edges | 5,132,180 | 5,132,203 |
| accepted edges | 3,986,975 | 3,987,027 |

Equivalence gate:

- neighbor element mismatch 303,889개, row mismatch 136,277개
- local spacing mismatch 81개, max abs delta `0.00156833`
- candidate/accepted edge shape 불일치
- partition root mismatch 1,251 surfels

후보가 수학적 Euclidean exact search이더라도 현재 reference의 float32 `torch.cdist` ranking/tie semantics와 full-scene identity가 다르다. 정확한 원인을 임의 교정하지 않고 **reject-contract-mismatch**로 판정했다. Production backend 인자나 fallback은 추가하지 않았다.

## 구현 파일

- `osn_gs/surface/torch_nurbs_performance_batch.py`: deterministic plan, pathology preflight, immutable serial executor, 미채택 terminal-batch candidate
- `scripts/devtools/wl119_performance_track.py`: corpus 전용 benchmark/equivalence harness
- `osn_gs/surface/torch_exact_knn_performance.py`: 미채택 cKDTree exact candidate와 reference graph post-processing
- `scripts/devtools/wl119_exact_knn_backend_validation.py`: full-scene adoption gate
- `tests/test_nurbs_performance_batch.py`
- `tests/test_exact_knn_performance.py`

공식 evidence:

- `output/confirmed/119_performance_track_20260827/validation_8v_512.json`
- `output/confirmed/119_performance_track_20260827/validation_exact_nearest_full_fail_8v_512.json`
- `output/confirmed/119_performance_track_20260827/exact_knn_full_scene_validation.json`

## 검증

WL119 관련 기존 tests와 신규 Performance tests를 합친 focused suite: **89 passed**.

Repository-wide suite: **1,422 passed, 22 skipped, 18 subtests passed, 1 existing warning** (`249.21s`).

Scientific invariants:

- 기존 serial chart path/default CLI 동작 유지
- 기존 `_knn`/`torch.cdist` 본문 유지
- chart corpus의 chart/component/ID/pixel ordering exact 보존
- candidate plan의 output chart order exact
- NaN/Inf parity 검증
- Worklog 119 scientific report 미수정

## 최종 판정과 남은 위험

승인된 계약 아래 채택 가능한 추가 wall-clock 개선은 이번 Phase 1~4에서 발견하지 못했다. 측정상 chart batching은 약 2배, cKDTree는 약 109배의 잠재 성능이 있지만 각각 continuous/conclusion invariance와 full-scene topology identity를 위반한다.

다음 선택지는 별도 승인 대상이다.

1. 현재 `torch.cdist` ranking/tie semantics 자체를 새 contract로 재정의
2. reference와 동일한 ranking을 재현하는 전용 exact KNN kernel
3. serial projector arithmetic을 재현하는 custom chart CUDA kernel
4. CUDA Graph 또는 multi-stream

이번 승인에서 금지된 2~4의 구현은 수행하지 않았고, 1은 scientific/topology contract issue이므로 임의 변경하지 않았다.
