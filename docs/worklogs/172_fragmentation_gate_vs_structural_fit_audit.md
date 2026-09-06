# Worklog 172 — W171 real-scene 실패 원인: fragmentation gate와 structural fit 계약

## 1. W171 기준 결과

W171 complete-support 결과를 그대로 보존한 baseline commit은 `a7d7957`이다. Tabletop은 support 17,965개 / native component 313개, curved/vase는 support 118,030개 / component 1,925개였으며 모두 `multiple_native_tsdf_components`에서 중단했다. Boundary, rank, LSQ residual은 평가되지 않았다.

W172는 W171 loader와 `analyze_case`를 변경 없이 호출해 두 case의 **전체 result JSON이 저장된 baseline과 동일함**을 확인했다. W171 candidate, 모든 W171 output 파일, W154 boundary/fitter 및 NURBS 구현 파일의 실행 전후 SHA-256도 일치했다. W171 산출물은 재생성하거나 수정하지 않았다.

## 2. 진단용 단일-component 대조군

각 case의 기존 native component 중 sample count가 가장 큰 component 하나를 기계적으로 선택했다. 동률일 때는 minimum source-cell key, region ID 순으로 결정한다. 크기 threshold, residual 기반 선택, 시각적 선택, tuning은 없다. 두 실측 case의 largest component 크기는 W171에 기록된 maximum과 일치한다.

선택 후에도 원래 source-cell key, cell index, world XYZ, normal, region ID와 row 순서를 그대로 유지했다. W171 real loader가 비워 두는 corner provenance placeholder도 동일하게 유지했다. `selected_support.npz`의 `complete_row_indices`로 원래 complete support의 row에 대응할 수 있다. 전체 및 선택 support의 실행 전후 hash와 row별 동일성을 검사했다.

Boundary는 기존 `derive_native_support_boundary`, representative는 기존 `fit_boundary_first_region_representative`를 **인수 override 없이** 사용했다. 기존 8x4 control grid, degree 2x2, smoothness/Tikhonov `1e-4`, correction/projection 각 2회가 유지된다. 기존 regularization 외에 geometry smoothing이나 repair는 추가하지 않았다. Mixed/contact case는 W171 loader가 함께 읽지만 W172에서 analyze/fit하지 않았다.

| 항목 | Tabletop | Curved/vase |
| --- | ---: | ---: |
| Complete support | 17,965 | 118,030 |
| Complete native components | 313 | 1,925 |
| 진단 component의 support | 15,189 | 111,601 |
| Complete support 대비 비율 | 84.5477% | 94.5531% |
| 기존 component ID / region ID | 0 / 1 | 0 / 0 |
| 진단 fit에서 제외된 support / components | 2,776 / 312 | 6,429 / 1,924 |
| 선택 후 native component 수 | 1 | 1 |

제외된 support는 원본에 남아 있고 overview에도 전부 그렸다. 이 선택은 attribution control이며 production filtering 제안이나 trusted subset이 아니다.

## 3. Tabletop 결과

15,189개 support의 단일 native component에서 boundary를 시도했지만 **134개 loop**가 나왔다. W154의 `ordered_boundary_valid=False`, `closed=False`이고 정확한 실패 reason은 `native_support_boundary_has_multiple_loops`다. Representative 함수는 이 boundary-invalid 상태에서 `ABSTAIN_REPRESENTATIVE`를 반환했다.

Design rank는 미평가(`null`, 요구 rank 32), LSQ는 미실행, NURBS는 미생성이다. Fit residual의 min / median / mean / p95 / max는 world units와 `/h` 모두 **미정의(`null`)**이며 residual count는 0이다. 미정의를 오차 0으로 해석하지 않는다.

## 4. Curved/vase 결과

111,601개 support의 단일 native component에서도 boundary가 **1,841개 loop**를 만들었다. `ordered_boundary_valid=False`, `closed=False`, 실패 reason은 동일한 `native_support_boundary_has_multiple_loops`이며 `ABSTAIN_REPRESENTATIVE`다.

Design rank는 미평가(`null`, 요구 rank 32), LSQ는 미실행, NURBS는 미생성이다. Fit residual min / median / mean / p95 / max는 world units와 `/h` 모두 `null`, residual count는 0이다.

두 case에서 W154의 `closed`는 **traced loop가 정확히 하나인 유효 단일 경계 상태**를 뜻한다. `closed=False`를 각 개별 loop가 모두 열린 polyline이라는 의미로 읽으면 안 된다. 모든 traced loop를 저장하고 시각화했으며 특정 outer loop만 선택하지 않았다.

## 5. 시각 검토

[산출물 README](../../output/172_fragmentation_gate_vs_structural_fit_audit/README.md), [실측 report](../../output/172_fragmentation_gate_vs_structural_fit_audit/worklog_172_report.json).

각 family에 두 case PNG를 직접 저장했다. PNG **8개**, PPM 0개, root·family·case별 UTF-8 README **9개**다. 모든 README에 input/state, palette, 공통 rendering 조건과 한계를 개별적으로 설명했다.

| Family | 시각화 의미 |
| --- | --- |
| `complete_support_overview` | Complete raw support의 모든 row를 표시. Cyan은 진단 component, gray는 나머지 component이며 제외 fragment도 subsample하지 않는다. |
| `component_boundary` | 선택 support와 orange ordered loops의 world 3D / local chart 비교. 모든 loop를 유지한다. |
| `nurbs_fit_common_world` | 선택 support와 fit 상태. 두 case 모두 NURBS/control net이 없음을 실패 reason과 함께 명시한다. |
| `residual_review` | 두 case 모두 fit residual이 미정의이므로 gray support와 미정의 상태를 표시한다. |

W171의 world 좌표, 기본 3D 시점, equal XYZ scale, white background, 150-dpi PNG 설정을 재사용했다. Boundary는 world와 chart 두 panel을 함께 제공했다. Overview는 W171 context를 포함한 원래 extent를 유지하고, 나머지 view는 표시 geometry의 전체 extent를 사용한다. Stable display stride는 overview 외 그림에만 적용되며 boundary/fit 계산에는 선택 component의 모든 row가 들어간다.

직접 확인한 그림에서 tabletop component는 넓은 sheet 형태지만 chart에 많은 빈 영역과 작은 loop가 있다. Curved/vase로 이름 붙은 component도 넓게 펼쳐진 sheet 형태이며 chart 내부와 가장자리에 다수의 loop가 보인다. **기존 case 이름만으로 vase의 coherent physical sheet가 분리됐다고 확인할 수 없다.** Orange boundary는 canonical chart plane에 놓인 경계이며 actual curved surface edge가 아니다. NURBS가 생성되지 않았으므로 support를 얼마나 잘 따라가는지는 시각적으로도 평가할 수 없다.

## 6. 아키텍처 원인 판정

최종 결과는 **`SINGLE_COMPONENT_CONTROL_STILL_FAILS_PRESERVE_STRUCTURAL_CONTRACT_FAILURE`**다.

W171이 처음 중단된 직접 원인은 complete-support fragmentation gate였다. 그러나 W172에서 이 gate를 진단 목적으로 통과시켜도 두 largest component 모두 **single-boundary/domain 계약에서 추가로 실패**했다. 따라서 “fragmentation precondition만 제거하면 기존 real NURBS fit이 성공한다”는 결론은 이번 대조군으로 성립하지 않는다.

동시에 이번 실패는 LSQ solver나 NURBS approximation capacity의 실측 실패가 아니다. 단일 native connected component가 W154가 요구하는 단일-loop chart를 보장하지 않았고, boundary에서 멈춰 **NURBS 표현 용량 자체는 여전히 미검증**이다. 현재 structural pipeline의 실패를 보존하며 이 지점에서 종료한다.

## 7. 유지 항목과 미해결 사항

- 유지: W171 complete-support baseline과 case selection, raw zero-set geometry, existing region organization, W154 boundary/fitter, production semantics, `BEHIND_ZEROSET`, W161 pause.
- 미채택: largest-component production filtering, component size threshold, trusted subset, loop 선택/통합, geometry repair, parameter tuning, occluded continuation, UNRESOLVED, 자동 architecture 성공 선언.
- 미해결: 현 support의 multi-loop chart에서 LSQ에 도달하지 못했으므로, coherent physical surface에 대한 NURBS capacity와 boundary-domain 계약의 적합성은 이번 결과만으로 확정할 수 없다. 해결책은 구현하거나 자동 제안하지 않았다.

Focused test **5 passed (4.37s)**. W171 파일/결과 보존, deterministic largest identity와 동률 처리, 선택 row의 원본 일치, 기존 함수 호출 및 default 인수 유지, valid PNG/README와 boundary artifact 생성을 확인했다. 전체 regression은 실행하지 않았다. 로컬 CPU에서 기존 배열만 읽었고 replay cache를 temp에 복사하지 않았다. Output은 repository 기존 규약대로 gitignore된 로컬 산출물이며 코드·테스트·Worklog는 commit한다.
