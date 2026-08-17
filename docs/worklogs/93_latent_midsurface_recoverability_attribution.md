# Worklog 93 — thick/curved center band의 latent midsurface 회수 가능성 귀속

## 상태

**완료 — Decision A. LATENT_SURFACE_RECOVERABLE.** Worklog 89 boundary reconstruction, Worklog 82 relation semantics, ADC, visible Gaussian training, NURBS fitting은 모두 미변경이다. 이번 배치는 새 boundary 실험도, position-PCA-normal 대체 실험도 아니다. Worklog 92가 발견한 `LOCALLY_THICK_UNIMODAL_SHEET`(62.2~70.2%)와 `LOCALLY_SINGLE_CURVED_SHEET`(25.1~31.7%) — Worklog 90 `MULTILAYER_OR_VOLUMETRIC` evidence의 대부분 — 이 recoverable latent 2D midsurface를 갖는지를 읽기 전용으로 측정했다.

## 제약

- Gaussian center position만 사용한다. Covariance normal/tangent/scale은 이 모듈 어디에서도 읽지 않는다(AST 검사로 import/함수 시그니처를 직접 검증, Worklog 92와 동일 관례).
- Gaussian xyz는 수정하지 않고, 계산된 projection을 model에 저장하지 않는다(`tests/test_chart_unit_latent_midsurface_attribution.py::test_never_mutates_input_positions`가 입력 tensor 불변을 직접 검증).
- 새 production boundary나 NURBS path를 만들지 않는다. 사용된 threshold(normal alignment 0.85·mutual residual 0.35)는 Worklog 82의 기존 기본값을 위치 전용 analog에 그대로 재사용한 것이며, 성공 방향으로 조정하지 않았다. `flattening_curvature_ratio_threshold=0.1`은 "diagnostic 곡률이 raw 곡률의 10% 미만으로 붕괴하면 flattening으로 간주"하는 고정 disclosure 하한이며 결과를 좋게 보이려는 튜닝이 아니다.

## 측정 방법

신규 `osn_gs/surface/torch_chart_unit_latent_midsurface_attribution.py`. Worklog 92의 local kNN(k=8) 및 diagnostic-only local 평면 적합(`_local_plane_normal`, `_knn_indices`)을 그대로 재사용하고, 추가로 각 local neighborhood에 quadratic height field(`z = au²+buv+cv²+du+ev`)를 적합해 local curvature(Hessian trace의 절반)를 추정한다.

1. **LOCAL_THICKNESS**: 각 node의 signed normal residual std를 local in-plane spacing으로 정규화, unit 전체에 evidence-weighted 평균.
2. **LATENT_MIDSURFACE_CONSISTENCY**: 이웃한 local patch끼리 position(이웃 center가 내 평면에서 얼마나 벗어나는지), tangent(평면 법선 정렬도), curvature(Hessian trace 상대 차이) 일치도를 측정한다.
3. **PROJECTION_DISPLACEMENT**: 각 center를 자신의 local 평면에 diagnostic으로만 투영(저장 안 함)하고, local spacing 및 unit extent 대비 displacement를 report한다.
4. **TOPOLOGY_RECOVERABILITY**: chart membership과 threshold를 그대로 둔 채, 같은 position-only same-surface adjacency test(Worklog 82의 normal alignment/mutual residual 검사 shape을 covariance 대신 position-fit normal에 적용)를 raw center와 diagnostic-projected center 양쪽에 동일하게 실행해 open/non-manifold 비율과 valid local face incidence(닫힌 삼각형에 참여하는 node 비율, Worklog 85의 boundary-curve 전용 degree-2 기준과는 별개의 interior-mesh 기준)를 비교한다.
5. **CURVATURE_PRESERVATION**: raw/diagnostic 양쪽에서 추정한 curvature를 비교해, diagnostic curvature가 raw curvature의 10% 미만으로 떨어지면 `curvature_preserved=False`로 fail한다(global planarization으로만 회수가 이뤄진 경우를 감지).
6. **OBSERVED-EVIDENCE FIDELITY**: 모든 projection displacement가 unit 자신의 관측 normal residual spread(지어낸 값 아님) 이내인지 비율로 report한다.

## 검증

`tests/test_chart_unit_latent_midsurface_attribution.py` 7개: 평평한 sheet(thickness 0, 완전 회수), thick unimodal sheet(collapse 후 valid face incidence가 raw보다 같거나 좋아짐), curved bowl(collapse 후에도 curvature가 substantially 보존됨, ratio>0.5), bounded thickness의 projection이 관측 지지대역 안에 머무름, tiny member count의 non-recoverable 기본값, 입력 position 불변, covariance 미사용(AST 검사) 전부 통과. Worklog 79~92 관련 focused 70개 통과. 전체 회귀 **962 passed, 1 skipped**(351.0초).

## 실측: 4개 checkpoint 재분류

`baseline_compatible` checkpoint 2900/3000/3100/final, cap=2048, 동일 7-region 파이프라인. Worklog 90의 `MULTILAYER_OR_VOLUMETRIC` unit 중 Worklog 92가 `LOCALLY_THICK_UNIMODAL_SHEET`/`LOCALLY_SINGLE_CURVED_SHEET`로 분류한 member만(6점 미만 subset은 quadratic fit 최소 요건 미달로 제외) 대상으로 했다. 산출물: `output/extent_ab/val93/chart_unit_latent_midsurface_attribution_replay.json`.

| iteration | 대상 evidence | manifold 개선 비율 | curvature 보존 비율 | valid face(raw→diagnostic) | support band fidelity | thickness/spacing | curvature(raw→diagnostic) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2900 | 2609 | **100.0%** | 90.9% | 36.3%→72.7% | 88.0% | 0.721 | 7.29→4.69 |
| 3000 | 8349 | **99.8%** | 86.6% | 36.1%→69.5% | 86.0% | 0.730 | 8.50→5.30 |
| 3100 | 10066 | **99.8%** | 91.6% | 32.5%→66.8% | 90.3% | 0.723 | 8.24→5.20 |
| final | 5775 | **100.0%** | 91.9% | 30.5%→63.0% | 91.5% | 0.751 | 10.11→6.91 |

4개 checkpoint 전부에서 manifold 개선 비율은 99.8~100%, curvature 보존 비율은 86.6~91.9%로 안정적이다. Valid local face incidence는 raw 30.5~36.3%에서 diagnostic thickness-collapse 후 63.0~72.7%로 거의 두 배가 되고, open/non-manifold 비율은 63.9~69.5%에서 27.3~37.0%로 절반 가까이 줄어든다. Support band fidelity 86.0~91.5%는 회수된 latent surface가 관측된 evidence support band를 크게 벗어나 지어내지 않았음을 보인다. Neighbor tangent agreement(0.77~0.80)와 neighborhood preservation(0.92~0.94)도 이웃 local patch들이 서로 일관된 평면 방향을 공유하고, thickness collapse가 topology 자체를 흩트리지 않았음을 뒷받침한다.

## 결정

**A. LATENT_SURFACE_RECOVERABLE.** Worklog 92의 thick/curved 대다수 evidence(2900~final 4개 checkpoint 전부)는 curvature를 보존하면서(90% 내외, global planarization 아님) manifold topology가 substantially 개선되는(raw 대비 valid face incidence 거의 2배, open/non-manifold 비율 거의 절반) 안정적인 latent 2D 표면을 갖는다는 실측 근거가 있다. 회수된 표면은 관측 evidence support band 안에 머문다(지어낸 geometry 아님).

**raw Gaussian center 자체가 visible surface topology를 위한 잘못된 geometry representation이다.** 다음 architecture target은 boundary 추출 이전에 명시적인 latent-surface evidence representation이어야 한다. 이 배치는 그 representation 자체를 만들지 않았다 — position-only diagnostic quadratic patch 적합과 disclosure만 수행했다. 지시대로 새 boundary 실험이나 position-PCA-normal 대체 실험으로 결론짓지 않는다.
