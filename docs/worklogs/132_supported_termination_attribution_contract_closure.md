# Worklog 132 — Supported-Termination Attribution Contract Closure와 Conditional Curvature Attribution

## INTENT ALIGNMENT / 의도 정렬

Worklog 131(`cf027b4c270a383995c7d23e9b25cd9ba99ee3d1`)의 `Gamma`, fixed
physical termination plane, physical first-order direction, ROI, `u_cut=0.58`,
horizon, NURBS control grid, Arm A/Arm B geometry를 모두 read-only로 보존했다.
이번 batch는 기존 continuation을 개선하지 않고, Worklog 131의
"supported-termination attribution population"이 실제로 supported Gamma에
대응하는지를 교정했다.

## IMPLEMENTATION FIDELITY / 구현 충실도

- 신규 모듈은 `devtools/demo/supported_termination_attribution.py`이며,
  출력은 `output/132_demo_supported_termination_attribution/`에 격리했다.
- Worklog 131 보고서와 WL128 frozen control grid를 replay했다. replay가
  일치하지 않으면 attribution 및 curvature 단계로 진행하지 않도록 했다.
- 각 withheld reference row의 frozen manual-normalized `v`에 대해
  `j = argmin_j |v_target-v_gamma_j|`를 한 번만 수행했다. 동률은 가장 작은
  Gamma index이며 tolerance dilation, prediction error 선택, XYZ 거리 선택을
  사용하지 않았다.
- 지원 판정은 기존 규칙 그대로 `nearest observed fitting point to Gamma <= 2h`
  를 사용했다. support threshold, ROI, `Gamma`, horizon은 변경하지 않았다.
- full reference mesh는 WL131과 같은 interface/support 정의 및 평가/보고에만
  사용했다. refit, root, direction, prediction 구성에는 withheld XYZ를 넣지 않았다.
- thin ROI는 새 Gamma를 만들지 않고 fixed `u∈[0,1]` scan의 `min f`, `max f`,
  `min |f|`만 진단했다.
- 첫-order supported 결과가 여전히 poor한 뒤에만 curvature diagnostic을
  실행했다. `q`는 평가용으로만 기록하고 curvature를 scale하지 않았다.
- canonical production code, Candidate B, historical topology, Worklog 131
  source/output은 수정하지 않았다.

## WL131 FROZEN REPRODUCTION / frozen 재현

primary와 secondary 모두 frozen Worklog 131 Arm B replay가 `PASS`였다.
primary에서 32개 root row와 Gamma UV가 frozen report의 root row와 수치적으로
일치했고, fixed plane과 `d local_u_world/dl=1`도 재현했다.

| 항목 | 결과 |
|---|---:|
| primary root coverage | 32/32 |
| primary predicted points | 3,072 |
| primary frozen Arm B median / p95 | 3.275h / 10.807h |
| primary frozen Arm B coverage `<=h / <=2h` | 8.83% / 30.31% |
| secondary root coverage | 0/32 |

frozen WL131 report에는 full predicted-point hash가 저장되어 있지 않았다.
따라서 replay report에는 Gamma UV hash와 Gamma point hash를 남기고, point
count, fixed horizon/extent, root rows, full metrics를 함께 identity basis로
기록했다.

## CORRECT UNIQUE GAMMA CORRESPONDENCE / 유일 대응

curved rim의 valid Gamma 32개에 대해 각 target row를 정확히 한 column에
할당했다. 기존의 `supported Gamma ± one v sample interval` dilation은 사용하지
않았다.

| 항목 | 값 |
|---|---:|
| valid Gamma | 32 |
| observed-supported Gamma (`<=2h`) | 23 (71.875%) |
| 전체 target | 12,000 |
| SUPPORTED_TARGET | 11,640 (97.000%) |
| UNSUPPORTED_TARGET | 360 (3.000%) |
| accounting / intersection | 12,000 = 11,640 + 360 / empty |

target 비율은 Gamma 비율과 같을 필요가 없다. 실제 `v` 분포에 따라
unsupported Gamma column에 배정되는 target row 수가 달라지기 때문이다.
per-Gamma assigned row count는 machine-readable report의 `per_gamma`에
고정 index, `v`, support distance, support state와 함께 기록했다.

## FULL-TARGET HISTORICAL METRIC / 전체 target

Worklog 131 Arm B의 historical free-nearest-surface metric은 동일한 12,000개
target에서 그대로 재현됐다.

| population / metric | median / h | p95 / h | coverage `<=h / <=2h` | normal median / p95 |
|---|---:|---:|---:|---:|
| FULL_TARGET free-nearest | 3.275 | 10.807 | 8.83% / 30.31% | 24.51° / 80.02° |
| FULL_TARGET correspondence-restricted | 4.030 | 10.914 | 5.38% / 17.37% | 24.45° / 80.07° |

첫 행이 Worklog 131 compatibility metric이다. 두 번째 행은 각 target을
자신의 Gamma column에만 제한했을 때의 별도 attribution metric이다.

## SUPPORTED-TARGET METRIC / 지원 종료 target

| metric | free-nearest historical compatibility | correspondence-restricted authoritative |
|---|---:|---:|
| samples | 11,640 | 11,640 |
| median / h | 3.264 | 4.012 |
| p95 / h | 10.663 | 10.739 |
| coverage `<=h / <=2h` | 9.06% / 30.60% | 5.52% / 17.56% |
| normal median / p95 | 24.39° / 80.19° | 24.31° / 80.26° |

authoritative correspondence-restricted 결과는 median `>h`이고 `<=h`
coverage가 5.52%로 낮다. 따라서 unsupported termination contamination을
제거해도 first-order continuation failure는 남는다.

## UNSUPPORTED-TARGET RESULT / 비지원 종료 target

| metric | free-nearest | correspondence-restricted |
|---|---:|---:|
| samples | 360 | 360 |
| median / h | 3.682 | 4.592 |
| p95 / h | 12.467 | 13.192 |
| coverage `<=h / <=2h` | 1.11% / 20.83% | 0.56% / 11.11% |
| normal median / p95 | 50.04° / 68.83° | 50.01° / 68.68° |

각 unsupported target도 unrelated distant Gamma column에서 nearest prediction을
얻지 않도록 correspondence-restricted metric을 별도로 계산했다.

## SUPPORTED DISTANCE-TO-TERMINATION / 지원 target 고정 bin

WL131과 같은 face-derived interface까지의 Euclidean distance를 `/h`로 나눈
bin이며, geodesic distance나 짧아진 horizon을 사용하지 않았다.

| bin | count | restricted median / p95 | coverage `<=h / <=2h` |
|---|---:|---:|---:|
| 0–1h | 154 | 2.208 / 4.691 | 22.08% / 44.81% |
| 1–2h | 159 | 2.041 / 4.140 | 19.50% / 48.43% |
| 2–4h | 374 | 2.262 / 4.514 | 15.24% / 43.32% |
| 4–8h | 757 | 2.266 / 4.689 | 16.91% / 42.27% |
| 8–16h | 1,792 | 2.765 / 5.070 | 10.71% / 28.96% |
| >16h | 8,404 | 4.895 / 11.548 | 2.39% / 10.67% |

near bin median `2.208h`에서 far bin median `4.895h`로 증가해, 정확한
supported termination correspondence 이후에도 long continuation에서 error가
커진다.

## THIN ROOT-FREE AUDIT / thin 구조 진단

thin leg/brace ROI의 32개 fixed `v` row 모두에서 기존 sign-change root는
없었고, 새 Gamma도 만들지 않았다.

| classification | count |
|---|---:|
| definitely no intersection | 32/32 |
| possible tangential near-contact | 0/32 |
| ordinary crossing root | 0/32 |

257개 `u` scan에서 모든 row가 strictly negative였다. `min |f|`는
`0.02338–0.09395` physical local-u 범위로 root tolerance `1e-6` 및 fixed
diagnostic tolerance `1e-4`보다 컸다. 이 결과는 tangent root 가능성도 관측하지
못했다는 뜻이며, thin ROI continuation을 이번 batch에서 새로 정의하지 않았다는
뜻이다.

## CURVATURE ATTRIBUTION / 조건부 curvature 진단

supported correspondence-restricted first-order 결과는 다음 세 조건을
충족하는 poor result로 판정했다.

- median `4.012h > 1h`;
- coverage `<=h = 5.52%`로 낮음;
- fixed distance bin median이 `2.208h → 4.895h`로 증가.

따라서 visible-side derivative를 진단만 했다. 각 supported Gamma에서

`T = a S_u + b S_v`

`A = a² S_uu + 2ab S_uv + b² S_vv`

를 계산했다. `S_uu`만 사용하지 않았고, second-order surface를 먼저 만들지
않았다.

| `l/h` bin | count | median `||R||/h` | median cosine(`R,A`) | `R·A>0` | median `0.5 l²||A||/h` | median residual-along-A / h | median q |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–2h | 354 | 2.304 | -0.309 | 38.14% | 0.004 | -0.291 | -35.487 |
| 2–4h | 376 | 2.279 | -0.390 | 33.78% | 0.054 | -0.525 | -9.983 |
| 4–8h | 760 | 2.298 | -0.300 | 36.18% | 0.226 | -0.386 | -1.707 |
| 8–16h | 1,775 | 2.747 | -0.409 | 36.45% | 0.815 | -0.597 | -0.753 |
| >16h | 8,375 | 4.929 | -0.694 | 11.20% | 8.249 | -3.177 | -0.387 |

전체 valid residual의 median cosine은 `-0.617`, positive dot fraction은
`18.23%`였다. valid Gamma 범위는 한쪽 작은 `v` segment에만 갇히지 않았고
(100%), error growth 자체는 관측됐지만 residual이 curvature 방향과 반대였다.

따라서 curvature candidate gate는 통과하지 않았다.

## SECOND-ORDER RESULT / 2차 결과

실행하지 않았다. `X2 = Gamma + lT + 0.5l²A` 후보를 생성하거나 평가하지
않았으며, `q`를 이용한 scale/damping/clipping/horizon 변경도 없었다. true-
occluded prototype 및 canonical Occluded Surface도 실행하지 않았다.

## TESTING / 검증

- `tests/test_supported_termination_attribution.py`: `11 passed, 1 warning`
- warning은 `.pytest_cache` 생성 권한 경고이며 test failure가 아니다.
- syntax check: new module/test `py_compile` 통과
- actual run: `cases=2`, primary/secondary frozen replay `PASS`
- report: `output/132_demo_supported_termination_attribution/supported_termination_attribution_report.json`

## FINAL REPORT / 최종 판정

### A. intent alignment

이번 batch는 Worklog 131의 continuation을 개선하지 않고, 기존 supported
attribution population의 dilution 오류를 unique nearest-Gamma assignment로
교정했다.

### B. implementation fidelity

수동 선택은 기존 두 ROI와 `u_cut=0.58`뿐이다. nearest-v assignment,
smallest-index tie break, `<=2h` support, fixed bins, curvature gate는
deterministic rule이다. withheld XYZ는 target/evaluation과 WL131에서 이미
정의한 mesh interface report 역할에만 사용했다. target error로 Gamma, branch,
horizon을 선택하지 않았다.

이러한 manual ROI, fixed termination plane, PCA normal, descriptive gate는
meeting attribution demo에는 허용되지만 final paper method의 자동화된
termination/support/confidence 계약으로는 부적절하다.

### C. verdict

> **C. FIRST-ORDER STILL FAILS ON CORRECTLY ISOLATED SUPPORTED TERMINATION; VISIBLE CURVATURE DOES NOT EXPLAIN IT**

정확히 하나의 Gamma column에 대응시킨 `11,640`개 observed-supported target에서도
first-order geometry가 poor했고, curvature residual은 방향 정렬에 실패했다.
따라서 unsupported termination contamination만으로 WL131 failure를 설명할 수
없고, visible curvature를 이번 결과만으로 completion prior로 채택할 근거도
없다. 이 결과는 canonical Occluded Surface가 해결됐다는 주장이 아니며,
다음 architecture 결정 전의 별도 attribution evidence다.
