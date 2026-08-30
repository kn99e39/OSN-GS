# Worklog 131 — First-order Continuation을 위한 명시적 Geometric Termination Mapping

## INTENT ALIGNMENT / 의도 정렬

Worklog 128(`2d87366`), Worklog 129(`1ca0da5`), Worklog 130(`8b2b4e7`)의
fit, ROI, holdout, historical metric을 동결하고 continuation 시작 경계만
바꾸는 별도 attribution 실험을 수행했다. 기존 `u=1`은 더 이상 Visible
Termination으로 부르지 않고, fitted observed-side NURBS와 고정 physical
holdout plane의 교차곡선을 `GEOMETRIC_TERMINATION_CURVE`로 정의했다.

second-order/curvature completion, true-occluded prototype, canonical
production 변경은 수행하지 않았다.

## IMPLEMENTATION FIDELITY / 구현 충실도

- 신규 모듈: `devtools/demo/explicit_geometric_termination_continuation.py`
- frozen WL128 control grid와 frozen WL129 corrected Arm A prediction을
  read-only로 사용했다. NURBS를 재설계하거나 target endpoint에 맞추지
  않았다.
- fixed ROI origin/axis/bounds와 `holdout_u_cut=0.58`만으로 termination
  plane을 만들었다. withheld XYZ는 plane, root, direction, horizon에
  사용하지 않았다.
- plane root는 `u∈[0,1]`에서 deterministic bisection으로 모두 찾았다.
  여러 root가 있으면 모두 report하고, prediction branch가 필요할 때는
  target error와 무관하게 largest-u root를 선택했다.
- physical direction은 fixed ROI `+axis_u`를 NURBS tangent plane에
  projection한 뒤 Jacobian `[S_u,S_v]` 계수로 표현하고,
  `d local_u_world / dl=+1`이 되도록 정규화했다.
- continuation은 `S(Gamma)+l J_S d_uv`의 1차식만 사용했다. `S_uu`,
  curvature scaling, support threshold sweep은 없다.
- mesh connectivity fragmentation은 fragmentation으로만 보고했으며,
  disconnected component를 distinct physical sheet라고 해석하지 않았다.

## CURVED-RIM PARAMETERIZATION REINTERPRETATION

Worklog 130의 값을 그대로 보존하되, 이전 `terminal initial u>=.90` 및
`final u>=.95` gate를 identity preservation 판정으로 사용하지 않았다.

| ROI | Pearson / Spearman | inversion fraction | shift median / p95 | 해석 |
|---|---:|---:|---:|---|
| curved rim | 0.99997 / 0.99994 | 0.411% | 0.00097 / 0.00446 | identity와 materially 다르지 않음 |
| thin leg/brace | 0.92186 / 0.92940 | 11.412% | 0.05648 / 0.25566 | materially reparameterized |

따라서 primary curved-rim은 parameterization 자체가 크게 변했다기보다,
rectangular `u=1` edge가 실제 termination 전체를 대표하는지가 핵심이다.

## EXPLICIT GEOMETRIC TERMINATION CONSTRUCTION

plane은 withheld geometry로 이동/회전하지 않았다.

| ROI | fixed plane |
|---|---|
| curved rim | `dot(x, axis_u) = -5.498` |
| thin leg/brace | `dot(x, axis_u) = 0.828` |

curved rim에서는 모든 고정 v sample에서 root를 찾았다.

- root coverage: `32/32 = 100%`
- multiple-root v samples: `0`
- selected `u_gamma` range: `0.9910–0.9976`
- maximum physical local-u root residual: `8.09e-7`
- generated physical local-u span: `0.798002`, fixed target extent `0.798`

이는 primary에서 새 시작점이 실제 fixed plane을 통과하도록 구성됐음을
보여주지만, root가 관측 termination 전체와 동일하다는 뜻은 아니다.

thin leg/brace에서는 plane intersection root가 `0/32`였다. 따라서 이
ROI에서는 explicit `Gamma`와 Arm B prediction surface가 정의되지 않았다.

## TERMINATION SUPPORT / 종료선 지지

고정 support rule은 `nearest observed fitting point <= 2h`이며 threshold를
조정하지 않았다. observed mesh support는 별도로 함께 기록했다.

| ROI | fit support `<=h / <=2h` | mesh support `<=h / <=2h` | attribution subset |
|---|---:|---:|---:|
| curved rim | 59.4% / 71.9% | 68.8% / 71.9% | Gamma samples의 71.9% |
| thin leg/brace | root 없음 | root 없음 | 0% |

curved rim의 supported termination만으로 별도 평가한 attribution population은
original 12,000 reference rows 중 11,977 rows(99.81%)였다. 이 결과는 full
population을 대체하지 않는다.

## PHYSICAL DIRECTION CONTRACT

curved rim의 `d local_u_world/dl`은 median `1.0`, p05 `1.0`, p95 `1.0`로
정규화 계약을 만족했다. `l`은 target error로 맞추지 않고 고정된
`(u1-u0)*(1-0.58)` 물리 extent를 사용했다.

## ARM A — HISTORICAL `u=1` FIRST ORDER

Arm A는 Worklog 129 corrected first-order prediction을 그대로 사용했다.
여기서 `u=1`은 rectangular NURBS parameter-domain edge로만 표기한다.

| ROI | points | actual local-u span | interface median / p95 (`/h`) | support `<=h / <=2h` | full median / p95 (`/h`) | coverage `<=h / <=2h` | normal median / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| curved rim | 3,072 | Worklog 129 frozen | 0.602 / 30.127 | 68.8% / 71.9% | 3.179 / 10.918 | 9.23% / 32.27% | 24.56° / 79.89° |
| thin leg/brace | 3,072 | Worklog 129 frozen | 5.322 / 16.507 | 0.0% / 3.1% | 9.020 / 19.828 | 0.38% / 2.43% | 59.17° / 86.73° |

Worklog 129의 original full-population metric은 재계산에서도 unchanged였다.

## ARM B — EXPLICIT-TERMINATION FIRST ORDER

| ROI | points | actual local-u span | interface median / p95 (`/h`) | support `<=h / <=2h` | full median / p95 (`/h`) | coverage `<=h / <=2h` | normal median / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| curved rim | 3,072 | `0.798002` | 0.375 / 30.298 | 68.8% / 71.9% | 3.275 / 10.807 | 8.83% / 30.31% | 24.51° / 80.02° |
| thin leg/brace | 0 | 없음 | root 없음 | root 없음 | 평가 불가 | 평가 불가 | 평가 불가 |

curved rim에서 interface median은 `0.602h→0.375h`로 가까워졌지만 p95는
개선되지 않았고, geometry full-population metric은 오히려 median/coverage
모두 악화됐다.

## FULL-POPULATION A/B

동일한 frozen Worklog 129 withheld evaluation population을 사용했다.

| 항목 | Arm A | Arm B | 변화 |
|---|---:|---:|---:|
| curved-rim median error / h | 3.179 | 3.275 | 악화 |
| curved-rim p95 error / h | 10.918 | 10.807 | 미세 개선 |
| curved-rim coverage `<=h` | 9.23% | 8.83% | 악화 |
| curved-rim coverage `<=2h` | 32.27% | 30.31% | 악화 |

따라서 termination 시작점을 physical plane으로 옮긴 것만으로는 primary
full holdout geometry를 materially recover하지 못했다.

## SUPPORTED-TERMINATION ATTRIBUTION

curved rim에서 observed-supported Gamma subset(23/32 samples)에 대응하는
fixed-v target population은 11,977 rows였다.

- median / p95: `3.308h / 10.814h`
- coverage `<=h / <=2h`: `8.83% / 29.96%`
- normal median / p95: `24.44° / 79.91°`

지원되는 종료선 구간만 보아도 full Arm A보다 materially 좋아지지 않았다.
이는 primary failure가 단순히 unsupported rectangular edge 하나 때문이라고
보기 어렵다는 attribution evidence다. 단, thin structure는 root 자체가
없으므로 secondary에 대해서는 종료선 계약이 여전히 실패했다.

## DISTANCE-TO-TERMINATION A/B

reference point에서 fixed face-derived interface까지의 Euclidean distance를
`/h`로 나눈 동일한 bin을 사용했다. geodesic distance가 아니며,
continuation horizon을 줄이지 않았다.

| ROI | bin | count | Arm A median / p95 | Arm B median / p95 | Arm A coverage `<=h` | Arm B coverage `<=h` |
|---|---|---:|---:|---:|---:|---:|
| curved rim | 0–1h | 158 | 1.509 / 3.414 | 1.518 / 3.400 | 27.85% | 29.75% |
| curved rim | 1–2h | 168 | 1.506 / 3.360 | 1.476 / 3.350 | 29.76% | 26.19% |
| curved rim | 2–4h | 391 | 1.681 / 3.294 | 1.655 / 3.327 | 23.79% | 23.53% |
| curved rim | 4–8h | 777 | 1.683 / 4.104 | 1.662 / 4.066 | 21.24% | 21.88% |
| curved rim | 8–16h | 1,824 | 2.025 / 4.359 | 2.045 / 4.384 | 13.76% | 14.14% |
| curved rim | >16h | 8,682 | 4.263 / 11.737 | 4.418 / 11.613 | 5.81% | 5.16% |
| thin leg/brace | 0–1h | 129 | 2.630 / 6.767 | root 없음 | 4.65% | — |
| thin leg/brace | 1–2h | 120 | 3.359 / 6.702 | root 없음 | 2.50% | — |
| thin leg/brace | 2–4h | 326 | 5.127 / 6.799 | root 없음 | 0.00% | — |
| thin leg/brace | 4–8h | 611 | 7.350 / 9.525 | root 없음 | 0.00% | — |
| thin leg/brace | 8–16h | 921 | 11.620 / 14.893 | root 없음 | 0.00% | — |
| thin leg/brace | >16h | 279 | 19.145 / 23.162 | root 없음 | 0.00% | — |

## QUALITATIVE RESULT / 정성 결과

curved-rim figure에서 physical plane과 `Gamma`는 역사적 `u=1` edge보다
고정 interface에 가까운 구간을 보여준다. 그러나 Arm B prediction은
withheld red surface와의 전체 겹침을 개선하지 않았고, 긴 거리에서의
divergence도 제거하지 않았다. `u=1` edge를 Gamma로 교체하는 조치만으로
artificial fan/divergence가 해결됐다고 말할 수 없다.

## TRUE-OCCLUDED PROTOTYPE

실행하지 않았다. controlled primary 결과가 positive feasibility signal을
주지 않았으므로 조건을 충족하지 않는다. canonical Occluded Surface도
추가하지 않았다.

## 산출물과 검증

- 모듈: `devtools/demo/explicit_geometric_termination_continuation.py`
- 테스트: `tests/test_explicit_geometric_termination_continuation.py`
- report: `output/demo_explicit_geometric_termination_continuation/explicit_geometric_termination_report.json`
- curved-rim figure: `output/demo_explicit_geometric_termination_continuation/curved_rim_explicit_termination_figure.png`
- distance figure: `output/demo_explicit_geometric_termination_continuation/distance_to_termination_ab.png`
- focused test: `6 passed, 1 warning` (pytest cache permission warning)
- actual run: `cases=2`, meeting figure 미생성, root-free secondary를 포함해 정상 종료

## 최종 판정

> **C. EXPLICIT TERMINATION DOES NOT MATERIALLY HELP; THE FAILURE LIES BEYOND THE START-BOUNDARY CONTRACT**

primary에서 termination 위치의 local interface median은 일부 개선됐지만,
full 및 observed-supported attribution population의 first-order geometry가
개선되지 않았다. 따라서 Worklog 129 failure를 arbitrary rectangular
NURBS edge 하나의 문제로 귀속할 근거는 부족하다. 다음 단계에서 curvature나
더 강한 completion prior를 추가하더라도, 그것은 이번 attribution 결과 이후의
별도 연구 질문이며 이 배치에서는 구현하지 않았다.
