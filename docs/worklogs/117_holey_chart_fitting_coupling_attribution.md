# Worklog 117 — Holey-Chart Fitting-Coupling Attribution

## 상태

**완료 — 실측 있음(전체 161개 뷰). 판정: MIXED/INCONCLUSIVE(스크립트의 조야한 이진 판정 `B2_NOT_SUPPORTED`를 그대로 따르지 않고, 실제 수치의 뉘앙스를 반영해 교정한 결론).** Worklog 116을 정정한 상태로 받아들이고, WL112 한-카메라-blob fitting baseline과 고정 8×4 NURBS(통제용으로만)를 동결한 채, "구멍이 있는 chart는 residual이 나쁘다"는 WL113의 상관관계가 진짜 **fitting-coupling 실패(B2)**인지 단순히 **chart 규모/복잡도의 대리 변수**였는지를 가렸다. **핵심 발견**: (1) B1(materialization) 검증 — 14,900개 chart 전부에서 `uv_support_mask` 부여가 control grid와 fitted 출력을 단 하나도 바꾸지 않음을 실측으로 확인(0/14,900 위반). (2) 일반 chart 모집단에서는 **약하지만 실재하는** hole-근접 residual 상관관계가 있다(중앙값 -0.055, hole 전용 거리는 -0.098, 66-68%가 음의 상관). (3) 그러나 규모로 stratify하면 WL113의 원시 2.6-3배 비율은 **극단적으로 불안정**해진다 — 일부 구간은 무의미하게 부풀려지고(수백~수천 배, 분모가 거의 0인 아티팩트), 가장 큰/가장 복잡한 chart 구간에서는 오히려 **1.3배로 줄어들거나 1 미만(구멍 없는 쪽이 더 나쁨)으로 역전**된다. (4) 가장 중요한 발견: WL113이 지목한 바로 그 **거대 patio chart**(10개 사례 전부, 21만 픽셀급, 전부 구멍 1개)에서 residual은 구멍/미지지 경계에 **가깝지 않은** 곳에서 오히려 더 나쁜 경우가 8/10이었다(예: chart 14745는 경계 근처 0.0187, 먼 곳 0.1015 — 5.4배 더 나쁨). 이는 B2가 예측하는 방향과 **정반대**다. **결론**: 일반 chart 모집단에는 약한 실재 hole-근접 효과가 있지만, WL113의 극단값 스토리를 지배하는 바로 그 거대 chart들에서는 그 효과가 나타나지 않거나 역전된다 — scale/capacity/parameterization이 거대 chart 실패의 더 유력한 설명이다.

## Agent Interpretation of Intent

1. **DIRECTION 이해**: WL116을 "수정과 함께 수용"한다는 것은 이번 배치 시작 전에 WL116의 구체적 오류(§1 참조)를 실측으로 바로잡아야 한다는 뜻으로 이해했다. WL107/109 위상, WL112 렌더러-네이티브 픽셀 기하, WL112 one-blob-one-chart fitting baseline, 고정 8×4는 전부 **통제용으로만** 동결하고 canonical로 재해석하지 않는다. `>=32`를 NURBS의 내재적 요구사항으로, full-rank closure를, WL114의 local extraction을 재도입하지 않는다. `uv_support_mask`는 이미 구현된 materialization 시맨틱으로만 사용한다(fitting에 결합시키지 않는다).
2. **PURPOSE 이해**: WL113이 "구멍 있는 chart의 residual이 없는 chart보다 2.6배 높다"고 보고했지만, 구멍 있는 chart는 대체로 더 크고 복잡하기도 했다 — 따라서 그 상관관계가 "구멍 자체가 fitting을 해친다(B2)"를 증명하지 않는다는 것이 목적이라고 이해했다. multi-patch coupling이나 다른 chart 분해를 구현하기 **전에** 이 귀속을 먼저 해결하라는 지시로 받아들였다.
3. **CENTRAL INTENT 이해**: "chart 규모와 기하적 복잡도를 통제한 뒤에도, 미지지/구멍 영역에 대한 근접성이 독립적으로 fitting 실패를 예측하는가?"에 답하는 것으로 이해했다. YES면 B2가 실재하고 representation-domain 메커니즘이 정당화되고, NO면 hole topology 자체는 아직 chart 재설계의 증거가 아니며 지배적 문제는 다른 곳(capacity/scale)에 있다는 것으로 이해했다. 이번 배치에서 다음 메커니즘을 구현하지 않는다.
4. **동결 유지 사항**: WL107/109 canonical topology, WL112 렌더러-네이티브 픽셀 기하, WL112 one-blob-one-chart baseline, 고정 8×4/degree-2(통제용), WL112 자신의 32-샘플 게이트(재현성을 위해 그대로 재사용하되 NURBS의 내재적 요구사항이라고 주장하지 않음).
5. **의도적으로 구현하지 않은 것**: multi-patch coupled fitting(`fit_coupled_patch_graph_lsq`) 연결, 새 chart 분해, WL114의 local extraction 재도입, 새 architecture 메커니즘 일체.
6. **도입한 조작적 가정**: (a) `uv_support_mask` 계산에 `torch_pipeline.py`의 기존 config 기본값(`surface_trim_resolution=24`, `surface_trim_dilation=1`)을 그대로 재사용했다 — directive가 "existing OSN-GS occupancy-mask semantics"라고만 지시했을 뿐 정확한 resolution/dilation 값을 지정하지 않아서, 새 숫자를 발명하지 않기 위해 기존 코드의 기본값을 그대로 가져왔다. (b) within-chart 상관계수는 Pearson 상관을 사용했다(directive가 구체적 통계량을 지시하지 않음). (c) near/far 분할은 각 chart 자신의 거리 분포 3분위(하위/상위 tercile)를 사용했다(directive의 "quantile strata or another deterministic descriptive grouping" 지시에 부합하는, scene-tuned 아닌 결정론적 선택). (d) 합성 대조군의 hole은 UV 중심 [0.35,0.65]×[0.35,0.65] 사각 영역으로 정의했다(directive가 정확한 hole 모양을 지정하지 않음). (e) 최종 B2 판정은 스크립트가 4가지 신호(중앙값 상관 부호, 다수결 상관 부호, matched-ratio survival, giant-chart far-high) 중 몇 개가 지지하는지로 자동 산출했으나, 실제 수치를 정밀히 읽어보니 이 자동 판정이 각 신호의 **크기**(약한 상관 vs 강한 역전)를 반영하지 못해 지나치게 이분법적임을 발견했다 — 이 worklog는 스크립트의 자동 산출값(`B2_NOT_SUPPORTED`)을 그대로 보고하지 않고, §10에서 그 수치를 직접 재해석해 MIXED/INCONCLUSIVE로 교정했다. 이것이 "prompt를 따랐다고만 말하지 말라"는 지시에 해당하는 지점이다.
7. **prompt의 모호함**: §4가 "canonical component, source view where practical"까지 confound로 통제하라고 요구했는데, 이번 배치는 pixel_count와 representative_count로만 stratify했다(directive가 명시한 목록의 처음 두 항목) — component/view별 완전한 층화는 (a) patio 최대 컴포넌트가 전체 holed-chart 인구를 압도적으로 지배해 별도 층화의 통계적 힘이 약하고, (b) directive 자체가 "as much as possible"이라는 완화된 표현을 썼으므로, 가장 강한 두 confound(크기 관련 지표)로 층화하는 것으로 충분하다고 판단했다. 이는 명시적으로 disclose하는 범위 축소다.

## Implementation Fidelity Statement

이번 배치는 신규 devtools 스크립트(`scripts/devtools/holey_chart_fitting_coupling_attribution.py`)와 신규 테스트(`tests/test_holey_chart_fitting_coupling_attribution.py`)만 추가했다. **다음 production 코드는 읽었을 뿐 전혀 수정하지 않았다**: `osn_gs/surface/torch_camera_induced_visible_adjacency.py`(WL107/109 위상, 무수정 재사용), `osn_gs/surface/torch_camera_observed_chart_domains.py`(WL112 chart 구성, 무수정 재사용), `osn_gs/surface/torch_nurbs.py`(fitter, 무수정 재사용), `osn_gs/core/torch_pipeline.py::TorchOSNGSPipeline._uv_occupancy_mask`(기존 occupancy-mask 시맨틱, staticmethod로 무수정 직접 호출), `scripts/devtools/chart_representation_contract_diagnostic.py`(WL113의 `_bin_by_quantile`/`_distribution`, 무수정 재사용). 새 코드는 순수 진단 함수(`hole_and_edge_masks`, `distance_to_unsupported_grid`, `distance_to_hole_grid`, `sample_uv_to_cell`, `within_chart_distance_correlation`, `near_far_median_split`, 합성 fixture 생성기)와 이들을 오케스트레이션하는 `main()`뿐이다.

## 1. Worklog 116 정정

**A. 기존 capacity 선례**: git 이력을 직접 조회해 확인했다 — `TorchOSNGSPipeline._fit_surface_patches`/`_target_resolution`은 실제로 존재**했다**(commit `13d9f61`의 부모 커밋, "Integrate canonical NURBS and renderer diagnostics" 직전 상태). `_fit_surface_patches`는 voxel-region의 `density`(`region_density.sum()`)와 `boundary_fraction`(`boundary_mask.mean()`)으로 `score = sqrt(density) * (1 + boundary_fraction)`을 계산해 패치 전체 control-point 예산을 분배했고, `_target_resolution`이 그 예산을 각 패치의 PCA aspect ratio에 맞춰 (U,V)로 나눴다. **그러나 이 정책은 정확히 그 통합 커밋에서 제거됐고, 현재 작업 트리(`git grep`)에는 전혀 존재하지 않는다** — voxel/boundary-first 파이프라인 전용 개념(`region_density`, `boundary_mask`)에 묶여 있었기 때문이다. 정정된 분류: **역사적 적응형 용량 선례는 실재했다(현재는 삭제됨); renderer-native 용량 정책은 여전히 미해결이다.** 이 옛 정책을 renderer-native architecture에 그대로 재사용할 수 있다고 주장하지 않는다.

**B. Effective degree**: `_effective_degree`는 control-point **개수**(`n_ctrl`)에만 의존하지, fitting 샘플 개수에 의존하지 않는다(`torch_nurbs.py`의 정의: `max(0, min(degree, n_ctrl-1))`). 8×4 control grid는 샘플이 2개든 2만 개든 항상 degree 2를 유지한다 — "2-샘플 8×4 fit은 degree를 자동으로 낮춘다"는 주장은 하지 않는다(WL116에는 이런 주장이 없었으나, directive의 지시대로 이번 배치에서 명시적으로 재확인한다).

**C. WL113 rank residual 방향**: 그대로 보존한다 — full-rank chart의 중앙값 residual **≈0.0067**, rank-deficient chart의 중앙값 residual **≈0.0031**(full-rank 쪽이 더 나쁨). 뒤집지 않았다.

**D. Per-view 독립성**: "per-view 독립 fitting이 overlap 실패의 필요/직접 원인"이라고 부르지 않는다. 그것은 consistency 제약을 **제거**할 뿐이며, 논리적으로 불일치를 강제하지는 않는다(WL115/116이 이미 이렇게 서술했음을 재확인한다).

## 2. B1 — 기존 Trim 시맨틱스 적용

WL112와 동일하게 fit한 14,900개 chart 각각에 대해, fit이 끝난 **뒤** `TorchOSNGSPipeline._uv_occupancy_mask(uv, resolution=24, dilation=1)`(기존 config 기본값, 새 숫자 없음)로 `uv_support_mask`를 만들어 부여했다. **실측 검증**: mask 부여 전후로 `control_grid`와 `surface.evaluate(uv)`의 결과를 `torch.equal`로 비트단위 비교 — **14,900개 chart 전부 위반 0건(`materialization_violations=0`)**. 이는 코드 읽기로 예측한 대로였지만, 이번 배치는 실측으로 실제 증명했다.

## 3. B2 — Chart 내부 Residual 귀속

각 chart의 fitted UV를 24×24 격자에 bin하고, `scipy.ndimage.distance_transform_edt`로 (a) 가장 가까운 미지지 셀까지의 거리, (b) 가장 가까운 **둘러싸인 hole** 셀까지의 거리(hole이 있는 chart만)를 구해 각 관측 샘플에 부여했다. **결과(13,101개 chart, 거리 분산이 있는 경우만)**: `corr(residual, dist_to_unsupported)` 중앙값 **-0.0545**, 66.0%가 음의 상관. hole 전용 거리(9,598개 chart)는 중앙값 **-0.0981**, 67.9%가 음의 상관 — hole 근접성이 일반 미지지 경계 근접성보다 다소 더 강한 신호를 보인다. 그러나 이 상관관계 자체는 **약하다**(중앙값이 -0.1 근처, 강한 상관이 아님) — B2가 실재한다면 이 정도의 약한 신호가 그 유일한 증거다.

## 4. Matched Chart 분석

pixel_count와 representative_count 사분위로 층화한 hole/no-hole 비교 결과, WL113의 원시 비율(~2.6-3배)은 **전혀 안정적이지 않았다**:

| pixel_count 사분위 | holed/unholed 비율 |
|---|---|
| bin 0 (32-40px) | 1.69배 |
| bin 1 (41-58px) | 14.19배 (분모 아티팩트 — unholed 중앙값 0.0003) |
| bin 2 (59-109px) | 2874배 (분모 아티팩트 — unholed 중앙값 2.3e-6, 거의 완벽한 우연의 fit) |
| bin 3 (110-216,783px, 최대) | **1.30배** |

| representative_count 사분위 | holed/unholed 비율 |
|---|---|
| bin 0 (1-5명) | 489.7배 (분모 아티팩트) |
| bin 1 (6-17명) | 1.33배 |
| bin 2 (18-32명) | 1.21배 |
| bin 3 (33-74,608명, 최대) | **0.85배 (역전 — 구멍 없는 쪽이 오히려 더 나쁨)** |

중간 구간의 극단적 배율(14배, 2874배, 489배)은 "구멍이 해롭다"가 아니라, **unholed 비교군의 중앙값 residual이 우연히 거의 0에 가까워서 생기는 분모 아티팩트**다(예: bin 2의 unholed 중앙값이 2.3e-6 — 사실상 완벽한 fit). 실질적 신호가 있는 두 극단(가장 작은/가장 큰 chart 구간)만 보면, 최대 규모 구간에서는 비율이 **1.30배로 크게 줄고, representative-count 최대 구간에서는 0.85배로 역전**된다. **결론: WL113의 원시 2.6-3배 비율은 규모를 제대로 통제하면 사라지거나 역전되는 구간이 있다 — 균일하게 생존하지 않는다.**

## 5. 거대 Patio Chart 귀속 (핵심 발견)

WL112/113의 residual 극단값을 지배해 온 바로 그 거대 patio chart(component 0) 상위 10개(전부 21만 픽셀급, 전부 hole 1개, 여러 뷰에 걸쳐 반복 관측된 같은 물리적 표면) 각각에 대해 near/far 분할을 직접 확인했다:

| chart_id | view | 근접 residual | 원거리 residual | 배율(원거리/근접) |
|---|---|---|---|---|
| 13670 | 139 | 0.0279 | 0.0487 | 1.75배 (원거리가 더 나쁨) |
| 14745 | 158 | 0.0187 | 0.1015 | **5.43배 (원거리가 훨씬 더 나쁨)** |
| 13618 | 138 | 0.0254 | 0.1287 | **5.07배** |
| 4894 | 28 | 0.0716 | 0.0319 | 0.44배 (근접이 더 나쁨, B2 방향) |
| 7483 | 49 | 0.0424 | 0.0878 | 2.07배 (원거리가 더 나쁨) |
| 14854 | 160 | 0.0233 | 0.0362 | 1.55배 (원거리) |
| 13906 | 142 | 0.0321 | 0.0400 | 1.25배 (원거리) |
| 7772 | 56 | 0.0540 | 0.0936 | 1.73배 (원거리) |
| 13736 | 140 | 0.0484 | 0.0379 | 0.78배 (근접, B2 방향) |
| 14794 | 159 | 0.0302 | 0.0488 | 1.62배 (원거리) |

**10개 중 8개**에서 residual은 미지지/hole 경계에 **가까울수록 오히려 낮고, 멀수록 높다** — B2가 예측하는 방향과 **정반대**다. 상관계수도 대부분 양수(0.01~0.47)로 이 패턴과 일치한다. 이는 WL113의 residual_max=1517 스토리를 만든 바로 그 컴포넌트에서, 실패가 hole 근접성이 아니라 **chart 전체에 걸친, 특히 hole에서 먼 영역에 집중된 무언가**(scale/capacity/parameterization 후보)에서 온다는 강한 증거다.

## 6. 합성 대조군 (A/B/C/D)

무수정 fitter로 4개 fixture를 fit했다(평면 24×24 격자/같은 평면+중심 hole/곡면 격자/같은 곡면+중심 hole, UV는 정확한 ground truth 사용, retained point는 hole 변형과 정확히 동일한 좌표로 매칭):

| | 평면(full, retained 대비) | 평면(hole variant, 자기 점) | 곡면(full, retained 대비) | 곡면(hole variant, 자기 점) |
|---|---|---|---|---|
| 중앙값 residual | 5.96e-8 | 5.96e-8 | 0.00179 | 0.00174 |
| p95 residual | 1.33e-7 | 1.69e-7 | 0.00559 | **0.00779** |
| 최대 residual | 2.55e-7 | 4.77e-7 | 0.00713 | **0.0107** |

**평면**: hole 유무가 residual에 사실상 영향이 없다(둘 다 기계 정밀도 수준) — 평면은 곡률이 0이라 8×4 control grid로 사실상 완벽하게 표현되므로, 중간이 비어도 남은 데이터로 완벽히 복원된다. **이 fixture는 B2에 대해 정보가 거의 없는 negative-control이다** — 평면에서는 애초에 coupling 실패가 일어날 방법이 없다.

**곡면**: 중앙값은 거의 동일(0.00179 vs 0.00174, hole 쪽이 오히려 미세하게 낮음)이지만, **p95(0.00559→0.00779, +39%)와 최대(0.00713→0.0107, +50%)는 hole variant에서 뚜렷이 나빠진다.** 이는 곡면 지오메트리에서는 hole이 꼬리(tail) 품질을 실제로 해친다는 **모호하지 않은, 통제된 증거**다 — 그러나 전형값(중앙값)에는 나타나지 않는다.

## 7. Coupled Fitting 미구현

directive 지시대로 `fit_coupled_patch_graph_lsq`/`SharedBoundaryConstraint`를 이 데이터에 연결하지 않았다. 이번 배치는 귀속(attribution)만 수행했다.

## 8. Capacity 이력 — 감사만

§1.A 참조. `density`/`boundary_fraction` 기반 예산 → 패치별 target → PCA-aspect (U,V)라는 정책은 실재했으나 제거됐고, voxel/boundary 전용 수량에 묶여 있었다. 이번 배치는 이를 재사용하지도, renderer-native 용량 정책을 새로 구현하지도 않았다 — 고정 8×4가 역사적으로 근본적이었다는 잘못된 결론을 막기 위한 기록일 뿐이다.

## 9. Persistent Binding 용어

Gaussian-surface provenance가 "일괄 폐기됐다"고 말하지 않는다. **역사적 ownership 시맨틱**(`cluster_ids`, `surface_owner_kind`, 옛 `torch_pipeline.py` 학습 루프)과 **향후 renderer-native representation provenance**(이 계보의 `subset_ids`, 아직 구현되지 않음)를 구분한다. 역사적 `cluster_id`/owner 시맨틱을 canonical로 상속하지 않으며, 이번 배치에서 대체물을 구현하지도 않는다.

## 10. B2 SUPPORTED / NOT SUPPORTED / MIXED 판정

스크립트의 자동 판정 로직은 4개 이진 신호(중앙값 상관 부호, 다수결 부호, matched-ratio 생존, giant-chart 원거리-여전히-높음)를 세어 `B2_NOT_SUPPORTED`를 산출했다. 그러나 각 신호의 **크기**를 직접 검토한 결과, 이 이진 집계는 실제 증거의 결을 지나치게 뭉갠다. 정확한 재해석:

- **일반 chart 모집단**: 약하지만 방향이 일관된 hole-근접 상관관계가 실재한다(중앙값 -0.05~-0.10, 66-68% 다수). 이것만 보면 B2 쪽으로 약간 기운다.
- **규모 매칭 비교**: 원시 2.6-3배 비율은 통계적 아티팩트(분모가 0에 가까운 구간)를 걷어내면 최대 규모 구간에서 1.3배로 줄거나(pixel_count) 0.85배로 **역전**된다(representative_count) — B2에 불리한 증거다.
- **거대 patio chart(WL113 극단값의 실제 근원)**: 10개 사례 중 8개가 residual이 hole/경계에서 **멀수록 나쁘다** — B2와 정반대 방향. 이는 가장 중요한(가장 큰 영향을 미치는) 실패 사례에서 B2가 지배적 원인이 아님을 강하게 시사한다.
- **합성 대조군**: 평면은 무정보, 곡면은 **꼬리(p95/max)에서만** hole이 유의미하게 해로움을 보여준다(중앙값에는 없음).

**판정: MIXED/INCONCLUSIVE, 방향성 있는 특성화와 함께.** 일반적이고 작은/중간 규모의 chart에서는 약한 실재 B2 신호가 있고(특히 곡면 지오메트리의 꼬리 품질에서), 이것이 fitting-coupling이 어느 정도 실재함을 시사한다. 그러나 WL112/113의 residual 극단값 스토리를 실제로 지배하는 **가장 크고 가장 영향력 있는 chart**에서는 이 신호가 나타나지 않거나 역전되며, 그 실패는 scale/capacity/parameterization으로 더 잘 설명된다. **"hole이 있으니 chart를 재설계해야 한다"는 결론은 아직 정당화되지 않는다** — 특히 최대-영향 사례에서는 그렇다. 동시에 "hole은 전혀 문제가 아니다"라고도 말할 수 없다 — 일반 모집단과 곡면 꼬리 품질에서는 실재하는 (약한) 효과가 있다.

## 11. 검토용 export 경로

`output/117_osn_gs_holey_chart_attribution/` 아래 7개 뷰 폴더(`iteration_0000001/point_cloud.ply`, `render.ppm`, `README.md`): `ORIGINAL_2DGS_SCENE`, `WL112_UNTRIMMED_SURFACE`, `WL112_TRIMMED_SUPPORT`, `UNSUPPORTED_DOMAIN_REMOVED`, `RESIDUAL_VS_DIST_UNSUPPORTED`, `RESIDUAL_VS_DIST_HOLE`, `GIANT_CHART_ATTRIBUTION`. 미리보기 PNG는 `preview_png/<뷰이름>.png` 한 폴더에 통합. `GIANT_CHART_ATTRIBUTION` 시각 검토에서 상위 10개 거대 chart 전부가 patio 지면 전체와 정확히 일치함을 확인했다(테이블·헤지는 어두운 회색으로 제외됨). 전체 JSON 리포트: `output/117_osn_gs_holey_chart_attribution/holey_chart_fitting_coupling_attribution_report.json`.

## 12. 테스트

`osn_gs/` production 코드는 무수정(읽기전용 재사용만).

- `tests/test_holey_chart_fitting_coupling_attribution.py`(신규, 15개): hole/edge 마스크 분해(4: 꽉 찬 사각형, 링, 모서리 gap은 hole 아님, 2개의 분리된 hole 각각 검출), 거리 회계(4: 경계에서 거리 0, hole 없으면 None, hole에서 멀수록 거리 증가, cell 매핑이 기존 occupancy mask 규약과 일치), within-chart 통계(3: 완전 음의 상관 검출, 상수 거리에서 NaN, near/far 분할 방향), 합성 fixture 기하(3: 평면 hole이 중심만 제거, 곡면 hole이 나머지 지오메트리 보존, `run_synthetic_contracts` 정상 산출), **실제 fitter로 support-mask 부여가 control grid/evaluate를 전혀 바꾸지 않음을 증명**(1, `torch.equal` 비트단위 검증).
- `.venv/Scripts/python.exe -m pytest tests/test_holey_chart_fitting_coupling_attribution.py -q` → **15 passed**.
- 실측 스크립트는 `--max-views 6` smoke test로 파이프라인 전체(합성 대조군 → 실측 sweep → 위상 재생 → chart-fit → B1 검증 → B2 회계 → matched 분석 → giant-chart 귀속 → 판정 → export/렌더)가 오류 없이 끝까지 도는 것을 먼저 확인한 뒤, 전체 161개 뷰로 재실행했다(런타임 952.2초). 위상 재생 수치와 `fitted_chart_count=14,900`이 WL112/113과 정확히 일치함을 확인해 동일 chart 구성이 재현됐음을 검증했다. 전체 pytest는 재실행하지 않았다(directive 지시: production 동작 무변경).
