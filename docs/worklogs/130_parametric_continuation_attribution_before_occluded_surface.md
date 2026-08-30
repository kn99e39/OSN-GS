# Worklog 130 — Occluded Surface 이전 Parametric Continuation 원인 귀속

## INTENT ALIGNMENT

Worklog 128(`2d87366`)과 Worklog 129(`1ca0da5`)를 그대로 보존한 별도
attribution track이다. 목표는 completion prior를 강화하는 것이 아니라,
Worklog 129의 실패가 NURBS parameterization/경계 계약인지, withheld target
혼합인지, 또는 그 이후의 curvature 문제인지 분리하는 것이었다. 기존 두
모듈, canonical production path, Worklog 127 TSDF, Candidate B는 수정하지
않았다. second-order continuation과 true-occluded prototype은 실행하지
않았다.

## IMPLEMENTATION FIDELITY

- `devtools/demo/parametric_continuation_attribution.py`에만 분석을 추가했다.
- frozen WL128 `observed_points`, reconstructed initial UV, fitter 설정을
  사용해 진단용 refit을 수행했다. withheld XYZ는 fitter/prediction에
  전달하지 않고 interface extraction, target coherence, evaluation,
  visualization에만 사용했다.
- 대형 WL127 `mesh.npz`의 vertices는 ROI 계산에 읽고, `faces.npy`는
  streaming으로 읽었다. 고정 ROI/축/범위/`holdout_u_cut=0.58`은 변경하지
  않았다.
- 수동 선택은 기존 WL128/129의 두 ROI와 좌표축/경계뿐이다. 분석 threshold
  (fit allclose tolerance, support gate, 단일-sheet 90%)와 face-local
  connectivity는 귀속용 operational rule이다.
- Euclidean interface distance, PCA normal, face-local ROI clipping은
  분석용 heuristic이며 최종 논문 방법으로 사용할 수 없다. mesh 전체를
  evaluation oracle처럼 이용하는 것 역시 최종 방법에서는 불허된다.

## FROZEN FIT REPRODUCTION

동일한 관측 fit population과 `8×4`, degree `2/2`,
`smoothness_lambda=1e-4`, `tikhonov_lambda=1e-4`, correction 2회,
projection 2회의 설정으로 replay했다.

| ROI | fit rows | max abs control-grid diff | RMS diff | 판정 |
|---|---:|---:|---:|---|
| `curved_table_rim` | 12,000 | `3.8147e-05` | `8.5705e-06` | sufficiently identical |
| `thin_table_leg_brace` | 3,075 | `1.0610e-05` | `2.1906e-06` | sufficiently identical |

명시 tolerance는 absolute/relative `2e-5`이며 두 replay 모두 통과했다.
따라서 다른 fitted surface를 기준으로 attribution을 수행하지 않았다.

## INITIAL vs FINAL UV PARAMETERIZATION

최종 footpoint UV는 projection rounds 이후 반환값을 직접 캡처했다.

| ROI | u Pearson / Spearman | u inversion fraction | u shift median / p95 | terminal final-u median | terminal `u>=.95` |
|---|---:|---:|---:|---:|---:|
| curved rim | 0.99997 / 0.99994 | 0.411% | 0.00097 / 0.00446 | 0.9512 | 51.6% |
| thin leg/brace | 0.92186 / 0.92940 | 11.412% | 0.05648 / 0.25566 | 0.9638 | 58.2% |

curved rim의 terminal local slope는 `0.9532`, thin structure는 `0.4134`였다.
고정 판정 규칙(terminal median `>=.95`, terminal `u>=.95` 비율 `>=90%`,
전체 inversion `<=1%`)을 적용하면 두 ROI 모두 안정 parameterization을
통과하지 못한다. 따라서 질문에 대한 직접 답은:

> 최종 fitted `u=1`을 manual observed termination과 동일하다고 부를 수 없다.
> 특히 thin structure에서 reparameterization과 terminal support 부족이 크다.

## NURBS `u=1` BOUNDARY SUPPORT

32개 고정 v bin에서 observed fitting points와 observed WL127 mesh만으로
support를 계산했다.

| ROI | fit boundary `<=h / <=2h` | observed mesh `<=h / <=2h` | no `u>=.95` bin | classification |
|---|---:|---:|---:|---|
| curved rim | 47.7% / 73.4% | 60.2% / 74.2% | 21.9% | partial support |
| thin leg/brace | 16.4% / 36.7% | 16.4% / 36.7% | 59.4% | `PARAMETRIC DOMAIN EDGE` |

`u=1` boundary 전체가 observed visible termination으로 지지되지 않는다.
분석에서는 이를 숨기지 않고 `curved rim`도 부분 지지로, thin structure는
명시적인 `PARAMETRIC DOMAIN EDGE`로 분류했다.

## GEOMETRIC TERMINATION AGREEMENT

고정 manual `u_cut`을 가로지르는 WL127 mesh face edges에서 실제 interface를
구성했다. fitted boundary와의 결과는 다음과 같다.

| ROI | boundary→interface median / p95 (`/h`) | coverage `<=h / <=2h` | tangent median / p95 | normal median / p95 |
|---|---:|---:|---:|---:|
| curved rim | 0.589 / 30.110 | 70.3% / 74.2% | 1.71° / 17.70° | 23.04° / 61.37° |
| thin leg/brace | 5.261 / 16.497 | 0.0% / 1.6% | 55.89° / 86.95° | 49.90° / 79.33° |

curved rim도 일부 interface는 맞지만 긴 tail이 있고, thin structure는
fitted `u=1`이 intended geometric termination을 대표하지 않는다.

## WITHHELD TARGET COHERENCE

mesh face connectivity로 fixed interface에 닿는 withheld component를
추적했다. ROI 밖으로 나가는 face는 local ROI contract에 따라 제외했다.

| ROI | withheld components | interface-connected components | connected vertices | connected faces | competing sheet |
|---|---:|---:|---:|---:|---|
| curved rim | 444 | 11 | 76.64% | 85.68% | 있음 |
| thin leg/brace | 92 | 7 | 78.21% | 86.90% | 있음 |

primary의 interface-connected target도 vertices 기준 90% 단일-sheet gate를
통과하지 못했다. competing sheet까지의 nearest separation은 curved rim
median `2.37h` (p05 `1.06h`, p95 `4.39h`), thin structure median `2.47h`
(p05 `0.84h`, p95 `12.05h`)였다. 따라서 원래 holdout은 단일 surface
continuation target으로 완전히 coherent하지 않으며, `TARGET-MIXTURE
CONFOUNDED` 위험이 있다.

Worklog 129 corrected prediction을 두 모집단에 각각 평가했다.

| ROI | population | median / p95 (`/h`) | coverage `<=h / <=2h` | normal median / p95 |
|---|---|---:|---:|---:|
| curved rim | original 12,000 rows | 3.179 / 10.918 | 9.23% / 32.27% | 24.56° / 79.89° |
| curved rim | interface-connected 9,203 rows | 2.710 / 9.931 | 10.94% / 37.18% | 19.16° / 69.33° |
| thin leg/brace | original 2,386 rows | 9.020 / 19.828 | 0.38% / 2.43% | 59.17° / 86.73° |
| thin leg/brace | interface-connected 1,866 rows | 8.185 / 13.806 | 0.48% / 3.11% | 57.68° / 86.72° |

original Worklog 129 evaluation population의 수치는 frozen report와
재계산 결과가 모두 일치했다(`original_population_metric_unchanged=true`).
그러므로 interface-connected target을 별도로 보는 것은 historical metric을
교체한 것이 아니다.

## DISTANCE-TO-TERMINATION ERROR

prediction은 바꾸지 않고, reference point에서 face-derived interface까지의
Euclidean distance를 `/h`로 나눠 고정 bin에 넣었다. geodesic distance가
아니라는 점을 명시한다.

| ROI | bin | count | median / p95 error (`/h`) | coverage `<=h / <=2h` |
|---|---|---:|---:|---:|
| curved rim | 0–1h | 158 | 1.51 / 3.41 | 27.85% / 67.09% |
| curved rim | 1–2h | 168 | 1.51 / 3.36 | 29.76% / 72.62% |
| curved rim | 2–4h | 391 | 1.68 / 3.29 | 23.79% / 64.71% |
| curved rim | 4–8h | 777 | 1.68 / 4.10 | 21.24% / 65.77% |
| curved rim | 8–16h | 1,824 | 2.02 / 4.36 | 13.76% / 49.07% |
| curved rim | >16h | 8,682 | 4.26 / 11.74 | 5.81% / 22.86% |
| thin leg/brace | 0–1h | 129 | 2.63 / 6.77 | 4.65% / 34.11% |
| thin leg/brace | 1–2h | 120 | 3.36 / 6.70 | 2.50% / 6.67% |
| thin leg/brace | 2–4h | 326 | 5.13 / 6.80 | 0.00% / 1.84% |
| thin leg/brace | 4–8h | 611 | 7.35 / 9.53 | 0.00% / 0.00% |
| thin leg/brace | 8–16h | 921 | 11.62 / 14.89 | 0.00% / 0.00% |
| thin leg/brace | >16h | 279 | 19.14 / 23.16 | 0.00% / 0.00% |

고정 0–2h에서도 primary median은 약 `1.5h`이고 coverage `<=h`는
약 28–30%뿐이다. 이는 local continuation behavior를 보여주는 자료이지,
더 짧은 successful horizon을 선택하거나 holdout extent를 재정의한 근거가
아니다.

## CURVATURE ATTRIBUTION

실행하지 않았다. frozen fit/terminal boundary support/geometric interface/
target coherence gate가 통과하지 않았으므로 missing curvature evolution으로
실패를 귀속할 수 없다.

## SECOND-ORDER RESULT

실행하지 않았고 후보 코드도 추가하지 않았다. true-occluded continuation과
canonical Occluded Surface도 실행하지 않았다.

## 최종 판정

> **A. PARAMETERIZATION CONTRACT FAILED**

frozen fit 자체는 재현됐지만, final fitted NURBS parameter scale과 `u=1`
boundary가 의도한 geometric visible termination과 일치한다는 계약이
성립하지 않는다. 동시에 target mixture 위험도 확인됐다. 따라서 현 시점에서
Worklog 129의 실패를 curvature 부족으로 해석하거나 second-order completion을
추가하는 것은 근거가 없다.

## 산출물과 검증

- 분석 모듈: `devtools/demo/parametric_continuation_attribution.py`
- focused tests: `tests/test_parametric_continuation_attribution.py`
- report: `output/demo_parametric_continuation_attribution/parametric_continuation_attribution_report.json`
- attribution figure: `output/demo_parametric_continuation_attribution/curved_rim_attribution.png`
- boundary support: `output/demo_parametric_continuation_attribution/curved_rim_boundary_support.png`
- distance plot: `output/demo_parametric_continuation_attribution/distance_to_termination.png`
- 검증: `11 passed, 1 warning` (pytest cache permission warning)
- 실제 실행: `A_PARAMETERIZATION_CONTRACT_FAILED`, `cases=2`, gate 미통과
