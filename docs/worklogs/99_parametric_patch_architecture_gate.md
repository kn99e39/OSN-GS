# Worklog 99 — bounded parametric patch architecture gate

## 상태

**완료 — Decision D: PARAMETER_DOMAIN_LIMIT.** Worklog 98은 tangent frame **orientation** 일관성(인접 sample 간 방향 반전 없음, cycle holonomy 위반 없음)을 증명했지만, 이것이 tree-integrated `(u,v)` map 자체가 well-conditioned parameter domain이라는 것을 보장하지는 않는다. 이번 배치의 pre-fit 검증이 이를 직접 실측으로 확인했다: Worklog 98의 coherent(orientation-consistent) component 중 **combined 80.4%(37/46)가 local UV fold/orientation reversal로 즉시 무효화**됐고, 이는 A/B/C 세 candidate가 비교되기도 전에 일어난다. Domain-valid로 남은 소수 component에서도 valid_supported는 사실상 0%(2900: A/B만 0.59%, C는 0%; final: 세 candidate 모두 0%)였다. **A/B/C를 서로 비교해 patch representation의 우열을 가릴 근거 자체가 대부분의 evidence에서 성립하지 않는다.** NURBS capacity를 탓하지 않는다 — Worklog 98은 frame orientation은 풀었지만 전역적으로 사용 가능한 parameter-domain 구성은 풀지 못했다. 다음 architecture 결정은 intrinsic parameterization 자체를 다뤄야 하며, curve seeding으로 되돌아가지 않는다. Visible Gaussian training, ADC, region ownership, Worklog 95 latent-surface estimator, continuous support contract, Worklog 98 synchronized tangent-frame field·coherent component, 각 paired component의 input 3D evidence, held-out evaluation, 기존 extrapolative/unsafe/valid_supported 정의는 모두 미변경이다. Held-out 결과로 어떤 candidate도 튜닝하지 않았다.

## 구현

### 1. Pre-fit parametric domain validity(candidate 비교 이전)

신규 `osn_gs/surface/torch_parametric_domain_validity.py`: Worklog 98의 coherent component마다 이미 계산된 `(u,v)`에 대해 fitting 이전에 다음을 검사한다 — nonzero u/v extent, UV cell당 3D geometry 불일치를 확인하는 duplicate/incompatible UV assignment, **local Jacobian 기반 orientation reversal/foldover 검출**(각 점의 UV-공간 kNN 이웃에 `delta_position ≈ J @ delta_uv` least-squares affine fit을 적용해 `J`의 두 열 외적 방향을 그 점의 실제 normal과 비교 — 부호가 반대면 fold), local Jacobian singular value로 singularity와 condition number(=extreme stretching/compression, `median_spacing` 대비)를 함께 얻는다. 별도로 `cycle_position_drift_p95`가 Worklog 98의 기존 holonomy edge를 재사용해 orientation이 아닌 **position** 기준 drift(대안 경로가 있는 cycle에서 tree-integrated 좌표가 얼마나 벗어나는지)를 진단으로 report한다. PCA는 어디에도 없고, 이 module은 NURBS를 import하지 않는다(테스트로 확인). Fold가 하나라도 있으면 domain 전체를 invalid로 fail-closed한다 — 어떤 candidate로도 복구하지 않는다.

### 2. Candidate A — FIXED_6x6_LSQ

Worklog 98의 기존 경로(`fit_curve_lattice_native`)를 그대로 baseline A로 유지한다.

### 3. Candidate B — ADAPTIVE_REGULARIZED_NURBS

신규 `osn_gs/surface/torch_adaptive_nurbs_capacity.py::select_adaptive_control_grid_capacity`: capacity를 fit 이전에 순수 구조 정보(Worklog 98 자신의 U/V integral-curve 개수, component sample 수, `(u,v)` aspect ratio)로만 결정한다 — `sqrt(sample_count)` 기반 총 budget을 aspect ratio로 U/V에 분배하고, 관측된 curve family 개수 이상으로 유지하며, `[4,10]`으로 clamp한다. Fit이나 held-out 오차는 이 함수의 입력에 전혀 없다(AST 검사로 확인). 실제 fit은 Worklog 97의 기존 `fit_torch_visible_surface_from_uv`를 그대로 재사용한다 — 이미 second-difference/Tikhonov 정규화가 내장돼 있어 별도 regularizer를 새로 만들 필요가 없었다.

### 4. Candidate C — GORDON_CURVE_NETWORK

신규 `osn_gs/surface/torch_gordon_curve_network_surface.py`: 고전 Gordon transfinite interpolation 원칙(`S = S1 + S2 - S12`, U-curve family loft + V-curve family loft − intersection grid의 bilinear 보정)과 OSN-GS의 잡음 있는 산란 데이터용 근사를 모듈 docstring에서 명시적으로 분리했다. 근사: component의 `(u,v)`를 Worklog 98의 U/V curve 개수만큼 discrete level로 결정론적으로 나누고(fit 오차로 정하지 않음), 각 level에 실제 quadratic least-squares curve를 적합하며(`_fit_1d_curve`, raw scatter가 아닌 진짜 curve fit), 두 family가 독립적으로 적합한 intersection 값의 불일치를 `intersection_grid_residual`로 report한다. 최종 control grid는 고전 공식을 정규 parameter grid에서 평가한 값을 그대로 쓴다. 이 population이 `MIN_LEVEL_POPULATION=3` 미만인 level이 2개 미만 남으면 `insufficient_populated_curve_levels`로 fail-closed하고(가짜 boundary curve 없음, closed boundary 요구 없음) PCA로 복구하지 않는다.

### 5~7. Paired 비교 + real replay

신규 `scripts/devtools/parametric_patch_architecture_gate_replay.py`가 Worklog 98의 동일 coherent component evidence에 대해 pre-fit domain validity를 한 번 계산한 뒤, valid한 component에만 A/B/C를 fallback 없이 각각 독립 실행한다(한 component에 대해 세 candidate 모두 동일 3D evidence·동일 held-out 대상을 쓴다). 5개 범주(PARAMETER_DOMAIN_INVALID/FIT_FAILED/FIT_SUCCEEDED_BUT_EXTRAPOLATIVE/FIT_SUCCEEDED_BUT_UNSAFE/VALID_SUPPORTED)를 하나의 실패율로 합치지 않고 분리 보고한다.

## 검증

신규 focused 테스트 18개: `test_parametric_domain_validity.py` 7개(깨끗한 평면 UV의 유효성, 인위적으로 접힌 UV의 fold 검출, degenerate extent 검출, local spacing 대비 extreme stretch report, 평평한 field에서 cycle drift가 0에 가까움, PCA 미사용, NURBS 비의존), `test_adaptive_nurbs_capacity.py` 5개(결정론성, capacity bound 유지, 관측 curve 개수 이하로 축소 안 함, aspect ratio 반영, fit/held-out 오차 비의존을 시그니처·AST로 확인), `test_gordon_curve_network_surface.py` 6개(일관된 합성 U/V network에서 성공, crossing curve를 tolerance 내로 재현, 모순된 correspondence에서 fail-closed, PCA 미사용, level 배정이 입력 순서/ID에 의존하지 않음, intersection residual report). Worklog 79~98 관련 focused 186개 통과. 전체 회귀 **1054 passed, 1 skipped**(435.2초).

구현 중 real-data replay에서 Gordon 모듈의 device-placement 버그(CPU에서 생성한 anchor/grid tensor를 CUDA 위 데이터와 연산)를 발견해 수정했다 — CPU 기반 synthetic 테스트에서는 드러나지 않았던 결함이다.

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(held-out evidence 1915), final(held-out evidence 3173). 산출물: `output/extent_ab/val99/parametric_patch_architecture_gate_replay.json`, `..._final.json`.

| checkpoint | coherent component | **domain-valid** | domain-invalid | invalid 원인 |
|---|---:|---:|---:|---|
| 2900 | 29 | **7 (24.1%)** | 22 (75.9%) | `uv_orientation_reversal_or_foldover` 22/22(100%) |
| final | 17 | **2 (11.8%)** | 15 (88.2%) | `uv_orientation_reversal_or_foldover` 15/15(100%) |
| combined | 46 | **9 (19.6%)** | 37 (80.4%) | `uv_orientation_reversal_or_foldover` 37/37(100%) |

두 checkpoint 모두 domain-invalid의 유일한 원인은 UV foldover/orientation reversal이다 — duplicate-UV나 singular Jacobian은 하나도 없었다. Worklog 98의 holonomy 검사(방향 자체의 일관성)를 통과한 component조차 그 위에 통합된 `(u,v)` map은 대부분 국소적으로 접힌다는 뜻이다.

Evidence-weighted(PARAMETER_DOMAIN_INVALID이 세 candidate 모두 공유):

| checkpoint | candidate | PARAMETER_DOMAIN_INVALID | FIT_FAILED | EXTRAPOLATIVE | UNSAFE | VALID_SUPPORTED | held-out p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2900 | A FIXED_6x6_LSQ | 78.77% | 0.00% | 12.51% | 8.12% | 0.59% | 10.00 |
| 2900 | B ADAPTIVE_NURBS | 78.77% | 0.00% | 20.63% | 0.00% | 0.59% | 10.24 |
| 2900 | C GORDON | 78.77% | 6.12% | 15.10% | 0.00% | 0.00% | 20.93 |
| final | A FIXED_6x6_LSQ | 91.72% | 0.00% | 1.28% | 7.00% | 0.00% | 6.51 |
| final | B ADAPTIVE_NURBS | 91.72% | 0.00% | 8.28% | 0.00% | 0.00% | 6.87 |
| final | C GORDON | 91.72% | 0.00% | 8.28% | 0.00% | 0.00% | 9.05 |

Domain-valid component만 봐도(2900: size 8~20점, final: size 18~40점, 모두 소규모) B의 adaptive capacity는 매번 최소값 4×4로 수렴했다 — 관측 sample 수 자체가 워낙 적어 `select_adaptive_control_grid_capacity`의 budget이 하한에 걸린다. 2900의 region 3(11점)만 A와 B 모두 VALID_SUPPORTED에 도달한 유일한 사례이고, 나머지는 전부 extrapolative/unsafe/fit_failed다. C(Gordon)는 여러 domain-valid component에서 `insufficient_populated_curve_levels`로 fit 자체가 실패했다(component가 너무 작아 level당 최소 3점을 못 채움).

## 결정

**D. PARAMETER_DOMAIN_LIMIT.** Worklog 98의 synchronized coherent component 중 combined 80.4%(37/46)가 A/B/C 어느 것으로도 비교되기 전에 이미 local UV fold/orientation reversal로 무효다. 이는 NURBS capacity(candidate B)나 curve-network-native construction(candidate C) 문제가 아니다 — domain이 유효한 소수 component에서도 세 candidate 모두 valid_supported가 사실상 0%(2900에서 A/B만 0.59%)였으므로, 남은 근거로도 어느 patch-fitting 방식이 우월하다고 판정할 수 없다. **Worklog 98은 tangent frame orientation coherence는 풀었지만, 전역적으로 사용 가능한 parameter-domain 구성은 풀지 못했다.** 다음 architecture 결정은 NURBS capacity나 fitting formulation이 아니라 **intrinsic parameterization 자체**(현재의 tree-integrated arc-length 방식이 아닌 다른 parameter-domain 구성)를 다뤄야 한다. 지시대로 curve seeding으로 되돌아가지 않으며, 이 배치는 architecture 결정으로 끝난다.
