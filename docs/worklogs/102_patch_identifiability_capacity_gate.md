# Worklog 102 — pre-fit patch identifiability / adaptive capacity gate

## 상태

**완료 — Decision C: TENSOR_PRODUCT_PATCH_STILL_FAILS.** Worklog 101은 고정 6×6 grid 자체의 제어점 수(36)를 밑도는 chart를 구조적으로 부족하다며 즉시 기각했다 — 그러나 이 count 기반 컷오프는 기존 정규화 NURBS solver(Tikhonov anchor + second-difference penalty)의 수학적 요구사항이 아니다: 정규화된 시스템은 raw data design matrix가 underdetermined여도 항상 풀린다. 이번 배치는 그 count 기반 컷오프를 fit 이전에 확인 가능한 명시적 **algebraic identifiability contract**로 대체하고, A(고정 6×6/degree2, 무변경)/B(adaptive quadratic, degree 2 고정 3×3~6×6 중 최대 식별가능 grid)/C(support-adaptive, degree 1(2×2 최소)~degree 2(3×3 최소), 최대 6×6, identifiability만으로 order·capacity 결정)를 fallback 없이 비교했다. 실측 결과: **C는 identifiable chart 비율을 A의 23.5%(27/115)에서 68.7%(79/115)로 크게 늘렸지만, VALID_SUPPORTED는 세 candidate 모두 정확히 2/115(1.7%)로 동일하게 유지됐다** — C가 새로 식별 가능하게 만든 52개 chart(79−27) 중 어느 하나도 안전한 fit으로 이어지지 못했다(identifiable 79개 중 unsafe 48개(60.8%)·extrapolative 29개(36.7%), 합쳐 97.5%). **Intrinsic parameterization·patch capacity 부족이 아니라, 현재의 local tensor-product NURBS fitting/representation 자체가 한계라는 근거가 확보됐다.** Worklog 101의 local intrinsic chart atlas, Worklog 100의 synchronized field·global differential integration·수정된 source-graph validator, held-out evaluation, 기존 정규화 tensor-product NURBS solver는 모두 미변경이다. Chart 생성·capacity 선택 어디에도 fit 오차·held-out 오차·extrapolative/unsafe 분류·rendering metric을 사용하지 않는다(AST로 검증).

## Worklog 101 해석 정정 (이번 라운드에서 반영)

Worklog 101이 보고한 "chart 단위 domain validity 100%(115/115)"는 이미 materialize된 chart만을 분모로 삼은 항등적 수치이며, intrinsic parameterization이 evidence 전체에 대해 보편적으로 해결됐다는 뜻으로 인용해서는 안 된다 — 의미 있는 지표는 source-evidence 커버리지(combined 1241개 node 중 1089개, 87.8%)이고, 나머지 12.2%는 명시적으로 unchartable로 남는다. Worklog 101, Master 문서, README, architecture.md, 메모리 파일에 정정 문구를 추가했다.

## 구현

### 1. Pre-fit algebraic identifiability

신규 `osn_gs/surface/torch_patch_identifiability.py`: chart의 고정 intrinsic `(u,v)` sample에 대해, **실제 solver가 쓰는 것과 완전히 동일한** `TorchNURBSSurface._basis_tables`(재구현 아님)로 tensor-product B-spline design matrix `rows[q, i*n_v+j] = N_i(u_q) N_j(v_q)`를 만들고, SVD로 singular value spectrum·condition number·수치적 rank(`torch.linalg.matrix_rank`와 동일한 관례 — `max(shape) * eps * largest_singular_value`, replay로 튜닝하지 않음)를 구한다. **Identifiable 판정**: design matrix가 자기 shape이 허용하는 최대 rank(`min(sample_count, control_variable_count)`)에 도달하는가 — sample이 충분하면 고전적 full column rank, sample이 부족(underdetermined)하면 모든 observation이 서로 독립인지(full row rank)를 요구한다. Sample 수가 36 미만이라고 자동 기각하지 않으며, 대신 u/v extent가 퇴화됐거나(단일 u/v 값으로 붕괴) 실제로 rank가 achievable보다 부족한 경우만 무효로 판정한다.

### 2. Candidate A — FIXED_6x6_DEGREE2

Worklog 101의 downstream probe를 그대로 유지한다(degree=2, 6×6, 기존 정규화 solver). Identifiability를 fitting outcome과 분리해 별도로 report한다.

### 3. Candidate B — ADAPTIVE_QUADRATIC_NURBS

신규 `osn_gs/surface/torch_adaptive_patch_capacity.py::select_adaptive_quadratic_capacity`: degree=2 고정, 3×3부터 6×6까지(직사각형 포함) 모든 grid를 pre-fit identifiability로 평가해 **capacity(=control-variable count)가 가장 큰 identifiable grid**를 선택한다. Capacity가 동률이면 chart의 실제 intrinsic u/v extent 비율과 grid aspect ratio가 log-space에서 가장 가까운 쪽을 결정론적으로 선택한다(fit 오차로 고르지 않음).

### 4. Candidate C — SUPPORT_ADAPTIVE_LOCAL_NURBS

`select_support_adaptive_capacity`: degree 2(최소 3×3)를 먼저 시도하고, identifiable한 grid가 하나도 없으면 degree 1(최소 2×2)로 낮춘다 — order/capacity 승격을 fit·held-out 성능으로 하지 않고 오직 pre-fit identifiability로만 결정한다("가장 높은 order, 그다음 가장 큰 capacity"). 여전히 tensor-product NURBS이며 새 surface family를 도입하지 않는다.

### 5~6. Paired 비교 + 5개 범주 분리

신규 `scripts/devtools/patch_identifiability_capacity_gate_replay.py`가 Worklog 101의 동일 chart membership·intrinsic UV·source evidence·overlap·synchronized frame·held-out evidence에 대해 A/B/C를 fallback 없이 독립 실행한다. PATCH_NOT_IDENTIFIABLE을 PARAMETER_DOMAIN_INVALID와 절대 혼동하지 않는다 — Worklog 101의 chart domain contract는 이미 통과한 상태다.

### 7. Overlap-consistency 평가(reconciliation 없음)

신규 `osn_gs/surface/torch_chart_overlap_consistency.py`: 두 chart가 공유하는 source node마다 각자 독립적으로 fit된 patch를 그 node의 (chart별) UV에서 평가해 위치 불일치·normal 불일치(각도)를 report한다. Merge/재적합/capacity 조정은 하지 않는다 — 순수 평가 지표다.

## 검증

신규 focused 테스트 20개: `test_patch_identifiability.py` 8개(36 미만 sample이 자동 기각되지 않음, B-spline basis rank 계산, 기하학적으로 붕괴된 UV의 rank deficiency, 유효한 3×3/degree2·2×2/degree1 식별, sample 0개 처리, fit/held-out/렌더링 오차 비의존(AST), 실제 basis-table 재사용 확인), `test_adaptive_patch_capacity.py` 6개(잘 샘플링된 경우 최대 identifiable grid 선택, fit/held-out 오차 비의존, 둘 다 식별 가능하면 높은 order 우선, degree 2가 불가능할 때 fail-closed 확인, 결정론적 tie-break, 새 surface family 미도입), `test_chart_overlap_consistency.py` 3개(양쪽 다 fit된 pair만 실제 수치 report, chart를 수정하지 않음, capacity 튜닝에 쓰이지 않음), `test_patch_identifiability_capacity_gate_replay.py` 3개(A/B/C 동일 chart evidence, 고정 6×6 baseline 무변경, atlas/field 재구성 없이 재사용). 전체 회귀 실행함(아래).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(held-out evidence 10117, 실행 541초), final(held-out evidence 11457, 실행 800초). 산출물: `output/extent_ab/val102/patch_identifiability_capacity_gate_replay_2900.json`, `patch_identifiability_capacity_gate_replay_final.json`.

### Identifiability + capacity/degree 분포(combined 115 chart)

| candidate | identifiable | 비율 | capacity 분포 | degree 분포 |
|---|---:|---:|---|---|
| A FIXED_6x6_DEGREE2 | 27/115 | 23.5% | 6×6: 27 | 2: 27 |
| B ADAPTIVE_QUADRATIC | 36/115 | 31.3% | 3×3: 9, 6×6: 27 | 2: 36 |
| C SUPPORT_ADAPTIVE_LOCAL | **79/115** | **68.7%** | 2×2: 43, 3×3: 9, 6×6: 27 | 1: 43, 2: 36 |

### Fitting outcome(combined, identifiable chart 대상)

| candidate | PATCH_NOT_IDENTIFIABLE | FIT_FAILED | EXTRAPOLATIVE | UNSAFE | **VALID_SUPPORTED** |
|---|---:|---:|---:|---:|---:|
| A | 88 | 0 | 0 | 25 | **2** |
| B | 79 | 0 | 8 | 26 | **2** |
| C | 36 | 0 | 29 | 48 | **2** |

C가 identifiable chart를 A 대비 3배 가까이 늘렸지만(23.5%→68.7%), **VALID_SUPPORTED는 세 candidate 모두 정확히 2개로 완전히 동일하다.** C가 새로 식별 가능하게 만든 52개 chart(주로 degree-1/2×2, 매우 작은 chart) 중 어느 하나도 안전한 fit에 도달하지 못했고, C의 identifiable 79개 중 97.5%(unsafe 48 + extrapolative 29)가 여전히 불안전하다.

### Overlap consistency(pre-reconciliation, 평가만)

| candidate | pair 수 | 양쪽 다 fit됨 | position 불일치 p95(평균) | normal 불일치(도, p95 평균) |
|---|---:|---:|---:|---:|
| A | 133 | 6 | 0.44 | 55.0 |
| B | 133 | 14 | 0.60 | 55.3 |
| C | 133 | 74 | 2.19 | 50.0 |

C는 더 많은 chart가 fit에 도달해 겹치는 pair 수도 늘었지만, position/normal 불일치는 A/B보다 오히려 크다 — capacity를 늘려도 겹치는 patch 간 일관성이 개선되지 않는다(이번 배치에서는 reconciliation을 시도하지 않았으므로 조정하지 않는다).

### Source-evidence 커버리지(Worklog 101과 동일, 미변경)

Combined 1241개 source node 중 1089개(87.8%)가 유효 chart로 커버됐고, 152개(12.2%)는 명시적으로 unchartable이다.

## 결정

**C. TENSOR_PRODUCT_PATCH_STILL_FAILS.** Support-adaptive local NURBS(candidate C)는 identifiable chart 비율을 23.5%→68.7%로 대폭 늘렸다 — "대부분의 valid chart가 algebraically identifiable해졌다"는 조건을 만족한다. 그러나 fitted patch는 여전히 압도적으로 unsafe/extrapolative(identifiable 79개 중 97.5%)이고, VALID_SUPPORTED는 A/B/C 모두 동일하게 2/115(1.7%)에 머물렀다 — capacity를 늘리거나 degree를 낮춰도 안전한 patch 수가 전혀 늘지 않았다. **이는 intrinsic parameterization이나 과도한 고정 capacity의 문제가 아니라, 현재 local tensor-product NURBS fitting/representation 자체의 진짜 한계를 가리킨다.** 지시대로 upstream(intrinsic parameterization, chart 구성)으로 되돌아가지 않는다.
