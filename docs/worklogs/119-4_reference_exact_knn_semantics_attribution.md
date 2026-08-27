# Worklog 119-4 — Reference-Exact KNN Semantics Attribution and Conditional Accelerated Backend

## Agent Interpretation of Intent

### DIRECTION

Worklog 119의 현재 exact-KNN 구현과 결과를 immutable scientific reference로 유지한다. Worklog 119-3 cKDTree는 108.98배의 잠재 속도를 보였지만 full-scene neighbor/edge/partition identity를 실패한 역사적 후보로만 보존한다.

### PURPOSE

수학적으로 더 정확하거나 빠른 새 이웃 정의를 고르는 것이 아니라, **현재 reference가 실제로 생성하는 ordered neighbor relation을 더 빠르게 동일 재현할 수 있는지** 판정한다.

### CENTRAL INTENT

최적화 전에 `torch.cdist`의 실제 CUDA 수치 경로, ranking, tie, shape/chunk 의존 계약을 규명한다. reference가 안정적이고 mismatch 원인을 설명할 수 있으며 topology를 재정의하지 않는 신뢰할 만한 재현 경로가 있을 때만 가속 후보 하나를 구현한다. 그렇지 않으면 negative attribution verdict로 종료한다.

### PRESERVE

- Worklog 119 serial scientific path와 scientific conclusion
- Worklog 119-2 accepted exact-semantics 최적화
- Worklog 119-3 `_knn`/`torch.cdist` reference와 historical cKDTree evidence
- checkpoint, visible coordinates, dtype, device, `K`, chunking, self exclusion
- local spacing, candidate/spatial/normal/accepted edge, component root semantics
- official reference run의 PyTorch/CUDA/backend 설정

### CHANGE ONLY

- production에서 분리된 attribution module, devtool, focused tests, Worklog
- diagnostic-only `torch.cdist` compute-mode 비교
- complete mismatch classification, K-boundary margin, arithmetic recomputation, repeatability evidence
- Attribution Gate 통과 시에만 정확히 하나의 opt-in accelerated candidate

### DO NOT

- cKDTree 채택, KNN contract 재정의, float64 canonical KNN
- tolerance, row blacklist, mismatch 기반 fallback, 튜닝된 extra-neighbor 수
- `K`, threshold, dtype/precision, reference compute mode 변경
- chart batching, NURBS, custom chart CUDA, CUDA Graph, multi-stream 재방문

### PROMPT-REQUIRED DECISION

1. reference repeatability와 numerical path를 먼저 규명한다.
2. full-scene 136,277 mismatch 행 전부를 A~H로 분류한다.
3. reference arithmetic/ranking을 명시적으로 재현할 수 있을 때만 후보 하나를 구현한다.
4. 구현 시 ordered IDs부터 roots까지 zero mismatch일 때만 채택한다.
5. gate 실패는 유효한 완료이며 속도만으로 성공을 선언하지 않는다.

### AGENT-INTRODUCED OPERATIONAL CHOICE

아래는 topology 선택에 사용하지 않는 진단 규칙이며 full-scene 결과 해석 전에 고정했다.

- 동일 default call을 두 번 수행해 IDs/order, recomputed distance/local spacing, graph tensors, roots를 exact 비교한다.
- production chunk 전체에서 default와 explicit MM을 비교하고, direct mode는 고정 첫 production chunk 및 고정 S1~S8 fixture에서만 비교한다.
- K-boundary는 self exclusion 후 raw reference matrix의 sorted `topk(K+1)`로 관찰한다. 상대 margin epsilon `1e-30`은 보고 전용이다.
- `B`는 membership 변경 여부의 독립 축이다. primary 우선순위는 `G → A → C → D → E → F → H`이다.
- E는 결과에 맞춘 threshold가 아니라 float32 `u=2^-24`, `gamma9=9u/(1-9u)`에서 유도한 3D MM squared-distance forward-error bound를 사용한다.
- boundary-only는 앞 `K-1` membership이 같고 symmetric difference가 정확히 2인 경우다.
- S1~S8 seed/좌표는 코드 상수로 사전 고정했다. cKDTree `K+1` query는 관찰용이며 candidate safety bound가 아니다.
- 최초 skeleton의 “모든 primary class arithmetic 표본” 규칙은 구현 검토 후 수정했다. A는 동일 set의 order-only라 entering/leaving pair가 존재하지 않으므로 pair 필드는 무효다. artifact v2에 `pair_fields_valid`를 추가하고 D/E 각 첫 32행만 arithmetic audit에 사용했다. 이 정정은 후보 튜닝이 아니며 집계 오류를 방지한다.

## Reference KNN Contract

Reference는 `osn_gs/surface/torch_coverage_first_subset_partition.py::_knn`(200–222행)이다.

| 항목 | 동결된 실제 계약 |
|---|---|
| 입력 | checkpoint iteration 30,000의 visible surfel positions, shape `(1,190,469, 3)`, `torch.float32`, `cuda:0` |
| K / chunk | `K=8`, `_auto_chunk_size`가 3 GiB raw matrix 목표로 `676` 선택; 마지막 chunk도 33행 |
| 거리 ranking | `torch.cdist(positions[start:end], positions)`; `compute_mode` 생략, non-squared Euclidean 출력 |
| self exclusion | full distance matrix 계산 후 row-global diagonal 하나를 `+inf`로 설정 |
| selection | `torch.topk(distance, 8, dim=1, largest=False)`; API default `sorted=True` |
| neighbor order | `topk`가 반환한 ordered IDs 자체가 immutable reference |
| 반환 거리 | ranking raw `cdist` 값을 버리고 선택 ID 좌표차의 `.norm(dim=-1)`로 재계산 |
| post-filter | KNN 단계에는 추가 distance filter 없음 |
| local spacing | 재계산된 K개 거리의 row median (`median(dim=1).values`) |
| graph | row-neighbor를 `(min(id), max(id))` canonical pair로 만든 뒤 deduplicate; 직접 좌표 norm과 local spacing으로 spatial gate, normal gate 후 accepted edges 생성 |
| components | accepted edges의 deterministic connected-component roots |

현재 downstream graph는 neighbor rank를 직접 사용하지 않는다. median은 set에 대해 불변이고 edge는 flatten 후 canonicalize/deduplicate되므로 A order-only 행 자체는 graph를 바꾸지 않는다. 그러나 21,465개의 set membership mismatch가 존재하므로 weaker set contract도 cKDTree가 통과하지 못한다. 또한 본 batch의 기본 adoption contract는 ordered identity 그대로 유지했다.

## Reference Numerical-Path Audit

- fresh authoritative reference: `70.5122 s`.
- full-scene explicit `compute_mode="use_mm_for_euclid_dist"`: `70.4815 s`; ordered IDs와 recomputed distances 모두 mismatch 0.
- 첫 production chunk `(676, 3) × (1,190,469, 3)`의 default raw matrix는 explicit MM과 bitwise exact였고 `topk(K+1)` IDs/raw도 exact였다.
- 같은 chunk의 direct mode는 default 대비 676행 중 61행, top-9 ID 원소 131개가 달랐고 raw top-9 위치값 6,084개가 달랐다. 최대 positional raw delta는 `0.00182125`였다.
- production의 모든 query chunk는 행 수가 33 이상이고 상대 집합도 25개보다 훨씬 크므로 PyTorch default conditional은 MM 경로를 사용한다.
- 단, 동일 raw matrix에 `topk(K)` 대신 diagnostic `topk(K+1)`을 호출하면 첫 K에서 91,410 ID 원소가 달랐다. 따라서 tie 선택은 output-size까지 포함한 `topk` 구현에 결합돼 있다. K+1 margin은 raw numerical semantics의 attribution 수치이지 production K 호출의 대체 결과가 아니다.
- pair shape `(1,1,3)`에서 강제 MM을 호출해도 production GEMM 순위를 일반적으로 재현하지 못했다. D 표본 32개에서 pair-MM은 reference 우선 2, candidate 우선 21, tie 9였고, E 표본 32개에서는 각각 15/8/9였다. explicit/direct norm과 float64는 64개 모두 cKDTree가 포함한 candidate를 더 가깝게 판정했다.

결론적으로 reference ranking은 추상적 Euclidean norm이 아니라 production-sized float32 MM의 `||x||² + ||y||² - 2x·y` 상쇄/반올림과 `topk(K)` tie 선택의 합성 결과다.

## Environment / Reproducibility State

| 항목 | 값 |
|---|---|
| PyTorch | `2.12.1+cu130` |
| CUDA runtime / toolkit | runtime `13.0`, `nvcc 13.3 (V13.3.33)` |
| GPU / capability | NVIDIA GeForce RTX 5080, compute capability 12.0, 16,303 MiB |
| driver | `596.49` |
| float32 matmul precision | `highest` |
| matmul TF32 | `False` |
| FP16/BF16 reduced-precision reduction | 각각 `True` |
| cuDNN TF32 / deterministic / benchmark | `True / False / False` |
| deterministic algorithms / debug mode | `False / 0` |

동일 환경의 두 번째 full reference는 `70.4833 s`였다. neighbor IDs/order, recomputed distance, local spacing, candidate edges, spatial mask, normal mask/alignment, accepted edges, partition roots가 모두 exact identity(`mismatch=0`)였다. 따라서 **현재 환경 안의 reference는 결정론적**이다. 이는 다른 PyTorch/CUDA/GPU에서의 cross-environment bitwise portability를 증명하지 않는다.

## cKDTree Mismatch Attribution

전체 mismatch를 artifact로 저장하고 모든 row에 primary class를 부여했다.

| 축/클래스 | 행 수 | 해석 |
|---|---:|---|
| 전체 ordered-row mismatch | 136,277 | ordered element mismatch 303,889 |
| A order-only | 114,812 | K-set 동일, order만 다름 |
| B membership mismatch | 21,465 | 독립 축; K-set 자체가 다름 |
| C exact coordinate/distance tie | 0 | production full scene 해당 없음 |
| D float32-reference raw tie | 10,466 | membership mismatch인데 두 경쟁 raw MM 거리가 bit-identical |
| E near tie within derived MM error bound | 10,999 | float64 squared gap이 보수적 float32 MM forward-error bound 이내 |
| F material disagreement | 0 | bound 밖 membership reversal 없음 |
| G duplicate/self effect | 0 | production mismatch 해당 없음 |
| H unattributed | 0 | 미분류 없음 |

Membership 21,465행의 decisive pair는 reference-only 중 가장 먼 항목과 cKDTree candidate-only 중 가장 가까운 항목이다. 이 비교는 다중 교환 행에서도 가장 큰 수학적 순위 역전을 검사한다. float64 squared-distance gap은 min `7.48e-11`, q25 `1.56e-6`, median `4.95e-6`, p95 `2.63e-5`, p99 `4.73e-5`, max `7.02e-4`였다. D 10,466행은 production raw gap이 정확히 0이고, E 10,999행은 모두 사전 유도한 MM 오차 상계 안이다. `F=0`, `H=0`이므로 136,277행은 완전 분류됐다.

왜 cKDTree가 다른가: cKDTree는 저장된 float32 좌표의 수학적 Euclidean 관계에 따라 정렬하지만, reference는 큰 절대 좌표 항의 float32 MM 조합에서 상쇄된 raw 거리를 정렬한다. 이 과정이 실제로 더 가까운 후보들을 raw tie 또는 역전으로 만들며, `topk`와 cKDTree의 tie/order 규칙도 다르다. 이후 direct norm 재계산은 이미 선택된 IDs의 거리만 바꾸므로 잘못된 membership을 되돌리지 않는다.

## K-Boundary Margin Analysis

`dK`, `dK+1`은 reference raw MM matrix에서 self exclusion 후 diagnostic sorted `topk(K+1)`로 측정했다. epsilon `1e-30`은 relative report 분모에만 사용했다.

| 집단 / margin | min | q25 | median | p95 | max | raw zero |
|---|---:|---:|---:|---:|---:|---:|
| matched abs (1,054,192행) | 0 | 6.3695e-4 | 1.6477e-3 | 1.0648e-2 | 3.0619e-1 | 7,150 |
| all mismatched abs (136,277행) | 0 | 3.0839e-4 | 1.1204e-3 | 9.2462e-3 | 2.1269e-1 | 13,556 |
| membership-mismatched abs (21,465행) | 0 | 0 | 0 | 4.4233e-4 | 3.9063e-3 | 11,053 |
| matched rel | 0 | 1.3675e-2 | 3.3380e-2 | 1.7419e-1 | 6.8528 | 7,150 |
| all mismatched rel | 0 | 9.6549e-3 | 3.5258e-2 | 2.6025e-1 | 3.9063e27 | 13,556 |
| membership-mismatched rel | 0 | 0 | 0 | 2.8992e-2 | 3.9063e27 | 11,053 |

Membership mismatch 중 19,073행(88.85%)은 K-boundary-only이며 나머지 2,392행은 둘 이상의 rank/set 차이를 포함한다. Membership 행의 11,053개(51.49%)가 raw zero margin이다. 상대 max의 비정상적으로 큰 값은 `dK=0`에서 보고용 epsilon으로 나눈 결과이며 선택이나 policy에 사용하지 않는다.

## Synthetic Adversarial Contracts

Fixtures는 `torch_knn_reference_attribution.adversarial_knn_fixtures`에 seed 1194와 고정 좌표로 정의했다.

| fixture | 핵심 계약 결과 |
|---|---|
| S1 well-separated random | default/direct/MM/cKDTree ordered IDs 모두 exact |
| S2 duplicate coordinates | arithmetic mode exact; cKDTree order 10원소 및 set 2행 mismatch |
| S3 exact equal-distance | arithmetic mode exact; cKDTree order 22원소 및 set 8행 mismatch |
| S4 float32 ULP near-equal | default=direct; forced MM ID 2원소 mismatch; cKDTree set 1행 mismatch |
| S5 K-boundary tie | arithmetic mode exact; cKDTree order 23원소 및 set 9행 mismatch |
| S6 large offset/small spacing | default=direct; forced MM ID 20원소 mismatch; cKDTree set 5행 mismatch |
| S7 anisotropic magnitudes | default=MM; direct ID 32원소 mismatch; cKDTree set은 동일하나 order 44원소 mismatch |
| S8 production-like clusters | default=MM; direct ID 11원소 mismatch; cKDTree order 11원소/set 1행 mismatch |

모든 fixture에서 default와 explicit conditional mode는 exact였다. S4/S6처럼 작은 shape에서는 default가 direct와 같고, S7/S8처럼 크기가 25를 넘으면 default가 MM과 같아 shape-dependent dispatch를 확인했다. Duplicate/equal-distance/K-boundary fixture는 backend별 tie semantics가 ordered/set 결과를 바꿀 수 있음을 확인했다.

## Attribution Gate Verdict

### 판정

**C. REFERENCE SEMANTICS TOO IMPLEMENTATION-COUPLED FOR A CLEAN EXACT REPLACEMENT**

- 결정론 gate: 통과 — 동일 환경 반복 full scene은 전 항목 exact.
- attribution gate: 통과 — `H=0`, `F=0`; mismatch 원인이 전부 설명됨.
- explicit arithmetic gate: 부분 통과 — frozen production shape에서는 explicit MM이 default와 exact.
- clean accelerated reproduction gate: 실패.

실패 이유는 explicit MM 거리식만 맞추면 충분하지 않기 때문이다. Candidate-set reduction은 GEMM shape와 reduction/tile 반올림을 바꾸며 pair-MM 표본도 production 순위를 재현하지 못했다. full candidate matrix를 유지하면 현재 70.5초 병목을 그대로 수행한다. 또한 `topk` output size만 바꿔도 first-K가 달라져 ordered tie semantics가 호출 세부사항에 결합돼 있다. 이 조건에서 spatial pruning이나 축소 후보 수가 모든 reference top-K를 포함하고 production MM ranking까지 exact 재현한다는 correctness bound는 제시할 수 없다.

따라서 승인 조건에 따라 speculative accelerated backend를 구현하지 않았고 production/default path를 변경하지 않았다. 이는 “가속이 영구 불가능”하다는 증명이 아니라, 현재 contract 아래에서 깨끗하고 검증 가능한 exact replacement 경로가 이번 batch에서 성립하지 않았다는 판정이다. 다음 선택지는 별도 승인을 받아 KNN scientific contract/tie semantics를 명시적으로 재정의하거나, 동일 PyTorch/CUDA GEMM 및 `topk(K)` 실행을 유지한 저수준 최적화 가능성을 별도 연구하는 것이다.

## Implementation Fidelity Statement

- immutable reference `torch_coverage_first_subset_partition.py::_knn` 및 production/default backend는 수정하지 않았다.
- historical `scipy_ckdtree_exact_knn` 구현과 WL119-3 결과를 수정·재해석하지 않았다.
- diagnostic 전용 코드만 추가했다.
  - `osn_gs/surface/torch_knn_reference_attribution.py`
  - `scripts/devtools/wl119_reference_exact_knn_attribution.py`
  - `tests/test_knn_reference_attribution.py`
- full-scene evidence:
  - `output/confirmed/119_performance_track_20260827/reference_exact_knn_attribution_full_scene.json`
  - `output/confirmed/119_performance_track_20260827/reference_exact_knn_mismatch_rows.pt` (`v2`, 136,277행, pair fields는 membership 21,465행에서만 valid)
- focused tests: attribution/order/membership/tie/near-tie/self/duplicate/K-boundary/compute-mode 및 기존 exact KNN/partition tests를 포함해 `28 passed`.
- attribution-only batch라는 명시 조건에 따라 full repository regression은 실행하지 않았다.
- accelerated candidate, production integration, fallback, blacklist, margin policy는 모두 0건이다.

### 최종 답

1. cKDTree가 136,277행에서 다른 이유는 production float32 MM `cdist`의 상쇄/반올림 및 `topk(K)` tie/order semantics가 수학적 Euclidean 순위와 다르기 때문이다. 114,812행은 order-only, 10,466행은 raw tie membership 교환, 10,999행은 derived MM error bound 내 순위 역전이다.
2. reference는 **현재 환경에서는 결정론적**이고 frozen production call을 explicit MM으로 재현할 수 있다. 그러나 더 빠른 축소 backend가 요구하는 독립적·명시적 계약으로는 충분하지 않다. shape/tile 및 `topk(K)` 구현 결합을 제거하면 바로 ordered/set identity가 달라진다.
3. 그러므로 이번 batch에서는 exact accelerated backend를 채택하지 않는다.