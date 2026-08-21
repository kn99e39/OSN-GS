# Worklog 105 — Renderer Contribution Diagnostics: Worklog 104 Branch A 기각

## 상태

**완료 — 실측 있음. Worklog 104의 Branch A는 실제 renderer-grounded 증거 앞에서 기각된다(Case B).** 벤더된 2DGS CUDA 커널을 전혀 수정하지 않고, 그 커널의 **공식 backward 경로**(모든 학습 스텝이 이미 실행하는 그 경로, 0줄 수정)를 진단 목적으로만 재사용해 surfel별 실제 렌더링 기여(`omega_i = alpha_i * T_i`, 커널 자신의 alpha-acceptance/transmittance-termination 컷오프)를 측정했다. 결과: **Worklog 104가 "한 번도 positively observed 안 됨"(713,540개)으로 분류한 surfel의 95.4%(680,527개)가 실제로는 렌더러의 공식 alpha-compositing에 실제로 기여하고 있었다** — 중앙값 20개 뷰, 평균 29.4개 뷰에 걸쳐. Phase-C의 point-sample CENTER 질의는 학습된 2DGS 표현(다수의 겹치는 planar surfel이 alpha-compositing으로 하나의 픽셀을 구성)에 대해 **부적절한 primitive-level visibility 정의**임이 확인됐다. Directive 지시대로 여기서 멈춘다 — 새 threshold도, 새 adjacency도 만들지 않는다.

## 아키텍처

```
Worklog 103/104 (수정 없음, 그대로 재실행)
    -> 신규: 진단 전용 renderer-contribution 신호
       (osn_gs/render/torch_surfel_contribution_diagnostics.py)
       -- 벤더 CUDA 커널의 official backward pass를 torch.autograd.grad로
          호출(.backward() 아님, 어떤 파라미터의 .grad도 건드리지 않음)
    -> A. PHASE_C_CENTER_POSITIVE vs C. RENDERER_CONTRIBUTING 2x2 cross-tab
    -> Branch A 재평가
```

## 1. 커널에서 직접 확인한 실제 contribution 시맨틱

벤더 소스(`osn_gs/render/vendor/diff_surfel_rasterization/cuda_rasterizer/forward.cu`의 `renderCUDA`, 라인 344-424)를 직접 읽어 확인했다: 한 (surfel, pixel) 쌍이 "accepted contributor"가 되려면 정확히 다음 5개 조건을 순서대로 통과해야 한다 —

1. `p.z != 0` (ray-splat 교차 평면이 well-defined)
2. `depth >= near_n`
3. `power <= 0` (즉 `rho >= 0`)
4. `alpha = min(0.99, opacity * exp(power)) >= 1/255` (커널 자신의 alpha-acceptance floor)
5. `test_T = T * (1 - alpha) >= 0.0001` (커널 자신의 transmittance-termination floor)

이 5개를 전부 통과한 surfel만 `last_contributor = contributor`(forward.cu:423)를 실행하고 `w = alpha * T`를 픽셀 색에 누적한다. 이것이 정확히 Worklog 104가 "Python에 반환되지 않는다"고 보고한 `omega_i = alpha_i * T_i`다.

## 2. 진단 계측 방법 — CUDA 재빌드 대신 공식 backward pass 재사용

Directive가 선호한 방법은 벤더 확장의 진단 COPY(새 atomic 출력 버퍼를 forward.cu/rasterizer_impl.cu/rasterize_points.cu/ext.cpp에 추가하고 별도 빌드)였다. 이 경로를 조사했지만, **`backward.cu`(라인 144-443)를 직접 읽어보니 이미 벤더된, 전혀 수정하지 않은 backward 커널이 정확히 같은 5-조건 테스트를 역순으로 재현하고(`backward.cu:279,291,302,311-312,315-317`), accepted contributor에 대해서만 `atomicAdd(&dL_dcolors[global_id*C+ch], (alpha*T)*dL_dpixel[ch])`(backward.cu:339)를 실행한다는 것을 확인했다** — 이것은 모든 학습 스텝이 이미 실행하는, 완전히 official한 계산이다.

진단 loss를 `L = render_unclamped.sum()`으로 두면 `dL_dpixel[ch]=1`이 균일하므로, `dL_dcolors[global_id]`는 정확히 "이 surfel이 accepted-contribute한 모든 픽셀에 걸친 `alpha*T`의 합"이 된다 — 근사가 아니라 커널 자신의 산술 그대로다. 한 SH 밴드 더 전파하면(`computeColorFromSH`, backward.cu:20-56) `dL_dsh[0] = SH_C0 * dL_dRGB`(SH_C0은 0이 아닌 상수, view-direction 무관)이므로 `model._features_dc.grad`가 동일한 0/비0 패턴을 그대로 갖는다.

새 모듈 `osn_gs/render/torch_surfel_contribution_diagnostics.py`는 `torch.autograd.grad(loss, (model._features_dc,))`만 호출한다 — `.backward()`가 아니므로 **어떤 파라미터의 `.grad`도 절대 건드리지 않는다**. 벤더 파일, `OSNSurfelRasterizer`, `TorchGaussianSurfelModel` 어디도 수정하지 않았다. `torch_pipeline.py`/`torch_trainer.py`는 이 모듈을 import하지 않는다(AST 기반 테스트로 고정).

**한계도 정직하게 기록한다**: 이 방법으로는 "contributing pixel count"와 "픽셀별 최대 weight"를 복원할 수 없다 — 커널의 atomicAdd가 이미 픽셀 단위 정보를 집계해버리기 때문에, 이 두 값을 얻으려면 정말로 커널 사본을 만들어야 한다. 이번 배치는 이 두 필드 없이 진행했고, directive가 명시한 최소 필수 신호("contributed YES/NO")와 "accumulated compositing weight"는 정확히 얻었다.

## 3. Rendering 불변성 증명 (지시 §5)

`test_canonical_forward_outputs_unchanged_by_diagnostic_call`: 동일 카메라/모델에 대해 (a) 순수 canonical render, (b) 진단 `torch.autograd.grad` 호출이 낀 render, (c) 진단 호출 이후 다시 canonical render — 세 번의 `render`/`depth` 출력이 전부 `torch.testing.assert_close`로 정확히 일치함을 확인했다. `test_diagnostic_call_does_not_mutate_parameter_grad`로 `_features_dc`/`_opacity`/`_xyz`의 `.grad`가 진단 호출 전후로 계속 `None`임도 확인했다.

## 4. Cross-tabulation (전체 1,190,469개)

| | contribution+ | contribution- |
|---|---|---|
| **center+** | 455,357 | 21,572 |
| **center-** | **680,527** | 33,013 |

## 5. Cross-tabulation (Worklog 103 singleton, 754,988개)

| | contribution+ | contribution- |
|---|---|---|
| **center+** | 39,525 | 1,923 |
| **center-** | **680,527** | 33,013 |

(`center-` 행의 두 값이 전체 표와 정확히 동일 — WL104의 A 카테고리(한 번도 center-positive 아님)는 구조적으로 전부 singleton이라는 사실이 다시 교차검증됐다.)

## 6. 핵심 수치 — "한 번도 center-positive 아니었던" 713,540개 중

- **실제로 기여함: 680,527개 (95.37%)**
- contributing view count 분포: min 0, median **20**, mean 29.4, p95 82, max 161
- accumulated weight 분포: min 0, median 0.988, mean 4.93, p95 17.06, max 18,444 (매우 큰 배경 surfel 하나로 추정)

이는 미세한 noise가 아니다 — 중앙값 20개 뷰에 걸쳐 실제로 렌더러의 공식 alpha-compositing에 기여하고 있다.

## 7. Branch 재평가

**Case B: Branch A 기각.** 압도적 다수(95.4% >> "substantial fraction")가 실제로 기여하므로, Phase-C의 point-sample CENTER 질의는 학습된 2DGS surfel의 primitive-level visibility를 판정하기에 부적절한 proxy임이 실증됐다. Directive 지시대로: epsilon을 조정하지도, 새 contribution threshold를 만들지도, 새 adjacency를 구현하지도 않고 여기서 멈춘다.

## 8. 실제 scene 리뷰

- **테이블/패티오**: `CENTER_VS_RENDERER_CONTRIBUTION_VIEW`에서 대부분 초록(CENTER_POSITIVE) — Phase-C와 renderer 둘 다 일관되게 인정.
- **hedge/배경**: 압도적으로 **주황**(CENTER_NEGATIVE_CONTRIBUTING) — Worklog 104의 `SINGLETON_CAUSE_VIEW`에서 짙은 빨강(한 번도 관측 안 됨)으로 보였던 바로 그 영역이, 실제로는 렌더러의 공식 alpha-compositing에 활발히 기여하고 있다. `RENDERER_CONTRIBUTION_VIEW`(view-count ramp)에서도 hedge/배경이 어둡지 않고 상당한 주황빛을 띤다 — 다수 뷰에 걸친 실질적 기여를 시각적으로 재확인.

## 9. Primitive accounting

1,190,469개 전량 그대로 (총량/도메인 변화 없음, WL103/104와 동일).

## 10. Review export

- `output/osn_gs_renderer_contribution_diagnostics/{ORIGINAL_2DGS_SCENE, WL103_PAIRWISE_POSITIVE_COMPONENTS, CENTER_VS_RENDERER_CONTRIBUTION_VIEW, RENDERER_CONTRIBUTION_VIEW}/`
- PNG: `output/osn_gs_renderer_contribution_diagnostics/preview_png/`
- 전체 리포트: `output/osn_gs_renderer_contribution_diagnostics/renderer_contribution_diagnostics_report.json`

## 11. 테스트

`tests/test_surfel_contribution_diagnostics.py` (9 tests, 실제 CUDA로 실행): canonical/진단 렌더 출력 일치, `.grad` 비변경, 실제 가시 surfel 검출, **진짜로 가려진 surfel이 오탐되지 않음**(3개의 완전 불투명 근접 surfel 뒤에 놓인 far surfel이 커널 자신의 transmittance-termination만으로 실제 미기여 처리됨을 실측 확인), frustum 밖 surfel 미기여, 반복 실행 결정론(불리언 신호는 bit-identical, 누적 weight는 CUDA atomicAdd 순서 의존 허용 오차 내), primitive tensor 불변, 학습 경로 미import(AST). 전체 regression: WL104의 1216 + 신규 9 = 1225 passed 1 skipped(실행 결과는 커밋 메시지에 기록).

## 12. 결론

**Worklog 104의 Branch A는 살아남지 못한다.** Phase-C의 point-sample CENTER 질의로 "한 번도 관측 안 됨"으로 분류된 surfel의 95.4%가 실제 renderer의 공식 alpha-compositing에 (평균 29개 뷰에 걸쳐) 기여하고 있다. 이는 그 surfel들이 "존재하지 않는 evidence"가 아니라 "Phase-C가 잘못된 granularity로 질문했기 때문에 인정받지 못한 실제 evidence"임을 의미한다. Directive 지시대로 여기서 멈춘다: 새 adjacency도, 새 threshold도 만들지 않았다. **다음 architecture는 renderer-grounded surfel evidence(이번 배치가 새로 노출한 `RendererContributionEvidence`)를 사용해 "무엇이 Visible Surface primitive인가"와 "그 primitive들이 어떻게 Visible Surface Component를 구성하는가"를 다시 정의해야 한다 — 그러나 그 결정은 이 배치의 범위 밖이다.**

## 참고

- 새 모듈: `osn_gs/render/torch_surfel_contribution_diagnostics.py`
- 테스트: `tests/test_surfel_contribution_diagnostics.py`
- Export 스크립트: `scripts/devtools/renderer_contribution_diagnostics_export.py`
- 관련: [[project_node_level_observability_accounting]] (WL104, 이번 배치가 재평가한 Branch A)
