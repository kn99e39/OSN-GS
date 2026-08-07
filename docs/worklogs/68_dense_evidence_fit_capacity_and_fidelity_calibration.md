# Worklog 68: Dense-Evidence Fit Capacity and Fidelity Calibration

## 목적

worklog 67에서 region-owned full evidence 도입 후 21/22 patch가 `extrapolative`로 재분류된 원인이(worklog 69에서 총계 오류 정정: 실제 총 22개, 기존 "21개"는 오기) (a) NURBS 표현력(fitting capacity) 부족, (b) dense nearest-neighbor 정규화 척도의 밀도 의존성, (c) 비균일 evidence weighting 중 무엇인지 구분한다. Region formation, representative topology, chart boundary, ownership gating은 전혀 변경하지 않는다 — worklog 67이 이미 복원해 둔 boundary+region-owned evidence 위에서만 동작한다.

## 방법

신규 `osn_gs/surface/torch_local_orientation_folding.py`: 기존 `compute_orientation_consistency`(단일 전역 참조 방향과의 일치도)와 명확히 분리된, **인접 UV grid 샘플 간의 국소 normal 부호 일치**만 검사하는 함수. Global reversal(patch 전체가 임의의 참조 방향과 반대로 향함 — 정의상 결함 아님)과 local folding(이웃 grid cell끼리 normal이 어긋남 — 실제 기하 결함 신호)을 구조적으로 분리한다.

각 patch(worklog 67에서 `full_evidence_state == "materialized"`인 22개 전부, worklog 69에서 총계 정정)에 대해 `scripts/devtools/dense_evidence_fit_capacity_validation.py`가:

- **deterministic spatial holdout**: 기존 `pca_parameterize_points`(미변경)로 evidence를 `[0,1]^2`에 결정론적으로 배치하고, 4×4 checkerboard 패리티로 train/holdout을 분리(공간적으로 섞여 있어 보간 성능을 검증하지, 외삽 성능을 검증하지 않음).
- **4개 정규화 척도**를 함께 기록: dense full-evidence NN spacing(worklog 66/67과 동일 관례), representative-only spacing(worklog 66의 원래 관례), robust normal-noise scale(고정 참조 fit — base 6×6 uniform — 대비 normal-direction residual의 MAD), patch diameter(boundary 최대 pairwise distance).
- **normal/tangent 잔차 분해**: 각 evidence 점의 offset을 최근접 sample의 local normal 방향 성분과 접선 방향 잔차로 분해.
- **boundary는 항상 hard constraint로 유지**하고 point_weight=1 고정(density compensation은 evidence에만 적용) — boundary constraint의 의미/provenance를 그대로 보존.
- 6×6/8×8/10×10 grid × {uniform, density-compensated} weighting 조합마다 train/held-out raw+정규화 오차, Jacobian condition/degenerate cell 수, local fold fraction, global orientation flip count, patch area를 기록.

판정은 과제가 준 4개 규칙을 그대로 코드화했다(우선순위: geometry 악화 > train만 개선 > train·held-out 동시 개선(capacity_insufficient) > raw 안정+dense-NN만 흔들림(metric_density_dependent) > weighting 단독 개선(weighting_problem) > inconclusive). Threshold는 사전에 고정한 두 상수(`MEANINGFUL_DROP_RATIO=0.10`, `FOLD_INCREASE_ABS_THRESHOLD=0.01`, 전체 adjacent pair의 1%)만 사용했고 결과를 보고 조정하지 않았다 — 첫 실행에서 절대 epsilon(1e-6) 비교가 사실상 모든 patch를 `overfitting`으로 과다판정하는 결함을 발견해 1%-절대-바닥 기준으로 고쳤고(패턴 자체는 재실행 후에도 20/22에서 유지됨), 그 이상은 손대지 않았다.

## 결과

**[worklog 69에서 정정] 총 22개 patch(기존 "21개"는 오기) 중 20개(90.9%)가 `overfitting`, 1개가 `capacity_insufficient`(region 10, baseline_compatible@3100), 1개가 `inconclusive`(region 11, 같은 checkpoint). `metric_density_dependent`/`weighting_problem`은 0건이다.**

`overfitting`으로 분류된 20개는 예외 없이 grid 해상도가 오를수록 **local orientation folding fraction이 절대 1%p 이상 증가**하거나(예: region 6가 0.09%→3.17%), train error만 의미 있게(≥10%) 줄고 held-out error는 그만큼 줄지 않는 패턴을 보였다. Jacobian degenerate cell은 모든 조합에서 0건이었다 — folding은 늘지만 완전히 특이(degenerate)한 수준까지는 가지 않는, 저수준의 국소 과적합 신호다.

`capacity_insufficient`로 분류된 유일한 사례(region 10)는 train(0.355→0.246, -31%)과 held-out(0.309→0.242, -21%) error가 함께, 의미 있게 줄었고 fold fraction은 0.36%→1.00%로 절대 기준 바로 아래 머물렀다(geometry safe) — 판정 로직이 실제로 차별화된 신호에 반응하고 있음을 보여주는 대조군이다.

`inconclusive` 사례(region 11)는 train error가 사실상 그대로(drop 0%)인 채 held-out만 소폭(7%, 유의 기준 10% 미달) 개선됐고, dense-NN 정규화 척도도 크게 흔들리지 않았다(spread 0.01) — 어느 규칙에도 깨끗이 들어맞지 않는 경계 사례로 정직하게 보고한다.

**density-compensated weighting은 base(6×6) 해상도에서 단독으로 유의미한(≥10%) train error 개선을 만든 patch가 0건**이었다 — 비균일 evidence weighting이 worklog 67의 재분류를 설명하는 요인이 아니었다.

**dense-NN 정규화가 raw error와 별개로 결과를 흔든 patch(`metric_density_dependent`)도 0건**이었다 — worklog 67에서 제기됐던 "정규화 척도의 밀도 의존성" 가설은 이번 라운드에서 지지되지 않았다.

## 완료 기준 대조

- raw point-to-surface/surface-to-evidence, 4종 정규화(dense-NN/representative/normal-noise/patch-diameter), normal/tangent 잔차, deterministic holdout error: **22개 patch 전부 기록 완료**(`output/extent_ab/val68/dense_evidence_fit_capacity_report.json`, worklog 69에서 총계 정정).
- 6×6/8×8/10×10 grid 비교(train+held-out error, Jacobian condition/degenerate, local fold, patch area): **완료.**
- uniform vs density-compensated weighting 비교: **완료, 0/21에서 weighting 단독 개선.**
- global reversal vs local folding 구분: **`torch_local_orientation_folding.py` 신규 모듈로 구조적으로 분리, 전용 테스트 5개로 두 개념이 실제로 다른 값을 낸다는 것까지 검증.**
- `validate_simple_closed_loop`는 이번에도 boundary-loop 검사로만 사용/보고(surface self-intersection이라는 표현 사용하지 않음).

## 테스트

신규 `tests/test_local_orientation_folding.py`(5개, global reversal과 local folding이 실제로 서로 다른 값을 내는지 직접 검증하는 케이스 포함) 전부 통과. 이번 라운드는 `torch_local_orientation_folding.py` 하나만 신규 production 코드이고 기존 파일은 변경하지 않았다. 지시대로 focused pytest만 실행했고 full pytest는 수행하지 않았다.
