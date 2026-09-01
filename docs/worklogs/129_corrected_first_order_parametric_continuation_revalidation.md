# Worklog 129 — 올바른 1차 Parametric Continuation 계약 보정 및 Worklog 128 재검증

## 범위와 의도

Worklog 128과 commit `2d87366b910873562b9dfc223408d85257c5af9f`를 역사적
baseline으로 동결하고, 기존 모듈/출력은 변경하지 않았다. 별도
`devtools/demo/corrected_first_order_parametric_continuation.py`가 저장된
WL128 fit/control grid와 동일한 ROI, `holdout_u_cut=0.58`, 평가 행, `h`,
`mu`를 읽어 Arm A/B를 재생했다. fit을 다시 수행하지 않았으므로 이번 batch에서
바뀐 것은 continuation construction뿐이다.

## 구현 fidelity

- Arm A: WL128의 `build_continuation_control_grid`를 그대로 재생했다.
- Arm B: 기존 `TorchNURBSSurface.evaluate_with_derivatives`와 analytic
  `S_uv`만 사용해
  `S_pred(t,v) = S(1,v) + r*t*S_u(1,v)`, `r=(1-c)/c`를 평가했다.
- 정상 벡터는 `dS/dt=r*S_u`, `dS/dv=S_v+r*t*S_uv`로 계산했다. `S_uu`,
  세 control column, quadratic/curvature continuation은 사용하지 않았다.
- withheld XYZ는 Arm B의 `B(v)`, `T(v)`, `r`, prediction, derivative에
  들어가지 않는다. withheld reference는 평가/시각화에만 사용했다.
- ROI, affine axes/bounds, holdout mask, fitter configuration, evaluation
  population, `h=0.012105485424399376`, `mu=0.03631645627319813`은 모두
  그대로 유지했다.
- canonical OSN-GS production code, WL127 cache, Candidate B, Worklog 128
  source/output은 수정하지 않았다.

## Worklog 128 역사적 결함

기존 구현은 `boundary=P[-1]`, `tangent_step=P[-1]-P[-2]`와
`Q_i=boundary+linspace(0,1,8)[i]*r*tangent_step`를 사용했다. 따라서
`Q_1-Q_0 = r/7*(P[-1]-P[-2])`가 되어 analytic interface derivative
`r*S_u(1,v)`보다 약 `1/7`로 축소되었다. 새 report에서 Arm A의
`historical_control_grid_replay_equal=true` 및 geometry hash 일치를 확인했다.

## Figure 1 provenance/metric 보정

중간 패널은 TSDF voxel center가 아니라 WL127의 실제
`RENDERER_MEDIAN_SURFACE_POINTS` binary PLY marker tail
`21,896`개를 읽어 사용했다. 라벨은 `Renderer median surface event samples`
이며 deterministic stride `2003/view` provenance를 report에 남겼다.

Figure 1의 canonical annotation은 WL127과 같은 all-event 모집단으로
재계산했다.

| 항목 | 값 |
|---|---:|
| 전체 renderer median event | 43,817,760 |
| finite distance event | 43,746,284 |
| non-finite/no-local-surface event | 71,476 |
| canonical all-event coverage `<=h` | 89.8346789% |
| canonical all-event coverage `<=2h` | 98.4548001% |
| finite-only coverage `<=h` | 89.9814576% |
| ray-hit coverage | 99.8804069% |

따라서 WL128의 Figure 1 표기 `89.981%`는 finite-only 분모를 사용한 값이고,
WL127 canonical `89.835%`는 non-finite row를 miss로 유지한 all-event 값이다.
새 Figure 1에는 후자를 큰 annotation으로 사용하고 전자를 report에 별도
기록했다.

## Arm A — historical WL128

Arm A metric은 Worklog 128의 frozen `case_report.json` 값을 그대로 보존했다.
새로 계산한 항목은 actual generated extent와 analytic derivative ratio이다.

| ROI | predicted points | actual local-u range / span | target withheld extent | median/p95 `/h` | coverage `<=h / <=2h` | normal median/p95 |
|---|---:|---|---:|---:|---:|---:|
| `curved_table_rim` | 3,072 | `[-5.49515,-5.38109]` / 0.11406 | 0.79750 | 22.6283 / 55.6515 | 2.708% / 7.925% | 24.2586° / 79.8411° |
| `thin_table_leg_brace` | 3,072 | `[0.73405,0.81713]` / 0.08308 | 0.25096 | 10.1897 / 20.3371 | 0.000% / 0.671% | 59.1462° / 87.0900° |

Interface derivative magnitude ratio
`||dS_pred/dt|| / ||r*dS_observed/du||`는 두 ROI 모두 약 `0.142857`
였다. 이는 WL128의 underscaling을 직접 확인한다.

## Arm B — corrected first-order Taylor continuation

| ROI | predicted points | actual local-u range / span | target withheld extent | median/p95 `/h` | coverage `<=h / <=2h` | normal median/p95 |
|---|---:|---|---:|---:|---:|---:|
| `curved_table_rim` | 3,072 | `[-5.49515,-4.26007]` / 1.23508 | 0.79750 | **3.1787 / 10.9182** | **9.225% / 32.267%** | 24.5606° / 79.8883° |
| `thin_table_leg_brace` | 3,072 | `[0.58642,0.98247]` / 0.39605 | 0.25096 | 9.0197 / 19.8284 | 0.377% / 2.431% | 59.1723° / 86.7295° |

두 ROI의 interface position gap은 0에 가까웠고, normal-angle gap p95는
`0.0235°` 이하였다. Arm B derivative ratio는 두 case 모두 median/p95
`1.0 / 1.0`이었다. 다만 tangent가 수동 local-u 축과 정렬되지 않아 실제
생성 geometry의 local-u range가 target extent와 다를 수 있음을 숨기지 않고
보고했다. Arm B는 target endpoint에 맞추지 않았다.

## A/B 정성 비교와 판정

Primary curved rim에서 Arm B는 역사적 Arm A보다 median error를 `22.63h`에서
`3.18h`로 줄이고 coverage `<=h`를 `2.71%`에서 `9.23%`로 개선했다. 이는
WL128 결과가 continuation-scale 구현 오류의 영향을 크게 받았다는 증거다.
그러나 corrected result 자체도 p95 `10.92h`, coverage `<=h` `9.23%`,
normal p95 `79.89°`로 non-planar withheld geometry를 quantitatively
non-catastrophic하게 회복했다고 보기 어렵다. secondary thin structure도
여전히 median `9.02h`, coverage `<=h` `0.38%`로 약하다.

따라서 이번 batch의 정확한 verdict는:

> **B. WL128 implementation was wrong, but corrected first-order continuation
> still fails on the real non-planar holdout.**

corrected primary가 positive feasibility signal을 통과하지 못했으므로
true-occluded prototype과 canonical Occluded Surface는 실행하지 않았다.

## 산출물

- `devtools/demo/corrected_first_order_parametric_continuation.py`
- `tests/test_corrected_first_order_parametric_continuation.py`
- `output/129_demo_corrected_first_order_parametric_continuation/README.md`
- `output/129_demo_corrected_first_order_parametric_continuation/figure_1_visible_surface_is_real_corrected_provenance.png`
- `output/129_demo_corrected_first_order_parametric_continuation/figure_2_corrected_first_order_ab.png`
- `output/129_demo_corrected_first_order_parametric_continuation/corrected_first_order_parametric_continuation_report.json`

## 검증

`tests/test_corrected_first_order_parametric_continuation.py` focused suite는
`5 passed, 1 warning`이다. 검증 내용은 parameter-scale boundary derivative,
historical Arm A 재현성, observed-only fitter payload의 실제 row identity,
all-event/finite-only coverage 명칭, renderer median event provenance이다.
별도 corrected run은 두 ROI 모두 `cases=2`로 완료되었고 report verdict는
`B_CORRECTED_FIRST_ORDER_STILL_FAILS`이다.

## 남은 위험

이번 결과는 corrected first-order contract의 재검증이지 NURBS completion
prior의 최종 판단이나 canonical Occluded Surface 결과가 아니다. 특히 실제
generated local-u가 manual local-u target과 정렬되지 않는 점, 얇은 구조의
normal 오차가 큰 점, WL127 Visible Surface의 표현/ROI 한계를 후속 연구에서
분리해 다뤄야 한다. 이 batch에서는 이를 개선하기 위한 extent tuning,
second-order continuation, ROI 변경을 하지 않았다.
