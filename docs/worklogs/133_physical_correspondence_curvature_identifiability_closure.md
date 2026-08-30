# Worklog 133 — 물리적 대응과 곡률 식별성 폐쇄

## 실행 목적

이번 배치는 최종 `Occluded Surface`나 continuation 개선 실험이 아니다. Worklog 132의 frozen first-order continuation 결과를 그대로 재생하고, 그 실패 귀속이 fitted parametric-v 대응에 의존하는지와 관측 경계의 표현 바닥이 곡률 신호를 가리는지를 감사했다.

실행 모듈은 [physical_correspondence_curvature_identifiability.py](../../devtools/demo/physical_correspondence_curvature_identifiability.py)이며 canonical research path와 분리했다. 산출물은 `output/confirmed/demo_physical_correspondence_curvature_identifiability/`에 기록했다.

## 보존 및 frozen replay

- Worklog 128–132의 ROI, `u_cut=0.58`, NURBS control grid, geometric termination `Gamma`, physical direction, first-order prediction, support rule `distance <= 2h`를 변경하지 않았다.
- curved rim의 WL132 identity check는 `PASS`다. Gamma UV hash, Gamma XYZ hash, support mask, target population, old assignment counts, supported/unsupported counts, restricted metric, curvature diagnostics가 모두 일치했다.
- thin leg/brace는 WL132와 같이 Gamma를 만들지 않았다.
- second-order candidate, third-order continuation, `q` fitting/scaling, true-occluded prototype, canonical production 변경은 실행하지 않았다.

## 물리적 v 대응 감사

Gamma XYZ를 frozen ROI origin/`axis_v`/`v_bounds`에 투영해 physical-v를 계산했다. target XYZ도 평가 population의 physical-v correspondence를 정하는 데만 사용했으며 fitter 입력이나 prediction construction에는 사용하지 않았다. target error는 assignment와 경계 `B` 선택에 사용하지 않았다.

| 항목 | curved table rim |
|---|---:|
| Gamma physical-v vs parametric-v Pearson / Spearman | 0.9983 / 1.0000 |
| Gamma physical-v median / p95 absolute difference | 0.01183 / 0.04803 |
| target assignment 변경 | 3,919 / 12,000 (32.66%) |
| supported target: old → physical correspondence | 11,640 → 11,920 |
| unsupported target: old → physical correspondence | 360 → 80 |

v 자체의 순위 상관은 높지만, nearest-column assignment는 무시할 수 없는 비율로 바뀐다. 따라서 WL132의 old correspondence 기반 curvature attribution은 그대로 physical correspondence의 결론으로 승격할 수 없다.

## 통제된 feasibility 결과 (CONTROLLED FEASIBILITY RESULT)

### Primary — curved table side / rim

- ROI vertex partition: full `39,059`, observed `21,878 (56.01%)`, boundary-attached withheld `17,181 (43.99%)`.
- 평가 target: `12,000` withheld reference rows.
- continuation mechanism: frozen WL132 first-order `Gamma + l*T`; 이번 배치에서는 continuation surface를 재학습하거나 수정하지 않고 correspondence만 physical-v로 재계산했다.
- continuation extent: fixed physical local-u horizon `0.798`; target-selected length가 아니다.
- physical-correspondence supported target: median `3.717h`, p95 `10.866h`, coverage `<=h 6.69%`, `<=2h 21.75%`.
- normal error: median/p95 `24.39° / 80.04°` (`estimated_unoriented_pca_vs_correspondence_polyline`).
- boundary position gap `B`: median `0.660h`, p95 `30.298h`, max `38.289h`.
- boundary tangent-angle discontinuity: median `22.34°`, p95 `69.81°` (deterministic adjacent interface-v estimate). Frozen interface는 vertex normal을 저장하지 않으므로 boundary normal-angle은 `unavailable`로 명시했다.

고정 거리 bin의 supported target 결과는 다음과 같다. 각 행은 `raw median/p95`와 fixed face-interface sample로 정의한 `bias-corrected median/p95`를 `h` 단위로 기록한다.

| continuation distance bin | target 수 | raw median / p95 | bias-corrected median / p95 |
|---|---:|---:|---:|
| `0–2h` | 369 | 1.675 / 4.201 | 1.845 / 5.041 |
| `2–4h` | 389 | 1.647 / 4.218 | 1.762 / 4.923 |
| `4–8h` | 779 | 1.773 / 4.569 | 1.871 / 5.546 |
| `8–16h` | 1,806 | 2.416 / 4.911 | 2.550 / 5.659 |
| `>16h` | 8,577 | 4.616 / 11.896 | 4.855 / 12.283 |

곡률 신호 `0.5 l² ||A||`는 `0–2h`에서 `0.005h`, `2–4h`에서 `0.054h`, `4–8h`에서 `0.230h`, `8–16h`에서 `0.815h`, `>16h`에서 `8.314h`의 median이다. 고정 경계 표현 바닥 median은 `0.660h`다. bias-corrected residual과 curvature의 overall cosine은 `-0.674`, `R·A > 0` fraction은 `22.23%`로 anti-aligned다. 따라서 이 배치에서 곡률 신호는 존재하지만 continuation residual을 예측하는 방향으로 정렬되지 않는다.

### Secondary — thin table leg / brace

- ROI vertex partition: full `5,461`, observed `3,075 (56.31%)`, boundary-attached withheld `2,386 (43.69%)`.
- fixed 32개 v row가 모두 `definitely_no_intersection` (`possible_tangential_near_contact=0`, `ordinary_crossing_root=0`)이었다.
- Gamma와 continuation surface가 없으므로 withheld error/normal metric은 계산하지 않았다. 이는 thin negative control이며 failed case를 숨기지 않은 것이다.

## 경계 표현 바닥

`B[j]`는 frozen face-derived interface의 physical-v 최근접 sample 하나만 사용해 `interface_point - Gamma[j]`로 정의했다. 선택에는 target error, prediction error, withheld target ranking을 사용하지 않았다. `B`는 prediction method가 아니라 raw residual에서 fixed boundary representation floor을 분리하는 진단량이다. interface normal은 frozen trace에 없어 normal-angle discontinuity는 계산 불가로 남겼다.

## TRUE-OCCLUDED PROTOTYPE RESULT

실행하지 않았다. controlled closure에서 physical-v correspondence 자체가 WL132 curvature attribution을 confound하는 것이 확인됐고, 이번 배치의 범위는 귀속 계약 폐쇄까지다. 따라서 Candidate B를 사용한 occluded-space continuation이나 novel-view 결과를 만들지 않았다.

## IMPLEMENTATION FIDELITY

- 수동 선택: 기존 WL132의 두 ROI와 고정 output path만 사용했다. continuation length는 frozen WL132 horizon이다.
- heuristic: physical-v nearest Gamma assignment, argmin tie 시 낮은 Gamma index, fixed interface-v nearest sample로 `B`, 인접 interface-v tangent estimate.
- full-reference 사용: Worklog 127 mesh faces/vertices에서 frozen interface를 재생하고 withheld target evaluation population 및 evaluation-only physical-v correspondence를 정의했다.
- withheld XYZ는 evaluation correspondence의 physical-v 산출과 metric/최종 진단에만 사용했다. fitter input, control grid, prediction construction, `B` sample 선택에는 사용하지 않았다.
- final paper method에서 허용되지 않을 선택: manual ROI/horizon, target-derived evaluation correspondence를 학습/예측 신호로 해석하는 것, heuristic `B` correction을 completion으로 주장하는 것.
- canonical isolation: 새 모듈, 새 focused tests, 새 report/output만 추가했다. WL132 파일과 기존 canonical TSDF, renderer, checkpoint, cameras, Candidate B, topology는 수정하지 않았다.

## MEETING VERDICT

**B. PARTIAL FEASIBILITY DEMO**

이 배치의 정확한 결론은 physical-v correspondence가 바뀌면서 WL132의 curvature attribution이 confounded되었다는 것이다. 현재 결과만으로 `Occluded Surface is solved`, final OSN-GS algorithm, 또는 paper method validation을 주장하지 않는다. 다음 연구 질문은 correspondence/termination/extent/confidence를 principled하게 정의하는 일이며, 이번 결과는 그 계약이 닫히기 전에는 curvature-based negative attribution을 확정할 수 없음을 보인다.

## 산출물 및 검증

- `output/confirmed/demo_physical_correspondence_curvature_identifiability/physical_correspondence_curvature_identifiability_report.json`
- `output/confirmed/demo_physical_correspondence_curvature_identifiability/gamma_v_parametric_vs_physical.png`
- `output/confirmed/demo_physical_correspondence_curvature_identifiability/README.md`
- focused tests: `17 passed, 1 skipped`
