# Worklog 128 — 실제 장면 Parametric Surface Continuation Feasibility Demo

## 작업

- canonical OSN-GS 경로를 변경하지 않고 `devtools/demo/`에 격리된 meeting-demo 모듈을 추가했다.
- Worklog 127의 `output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz`를 읽기 전용 reference geometry로 사용했다.
- 고정 수동 ROI 두 개를 만들었다.
  - `curved_table_rim`: curved table side/rim
  - `thin_table_leg_brace`: thin table leg/brace
- 각 ROI에서 `u <= u_cut`만 fitting input으로 보존하고 `u > u_cut`은 boundary-attached withheld reference로 분리했다. 중앙 interior hole은 만들지 않았다.
- 기존 `osn_gs.surface.torch_nurbs.fit_torch_visible_surface_lsq`를 그대로 호출하고, 마지막 control-column과 직전 column의 finite difference를 고정 continuation rule로 사용했다.
- withheld geometry는 evaluation target, final visualization, quantitative error 계산에만 사용했다.
- four-panel PNG, no-completion sanity baseline, error/normal backup map, WL127 TSDF context backup, JSON report를 `output/128_demo_parametric_surface_continuation/`에 생성했다.

## 결과

### Controlled feasibility result

`h = 0.0121054854`는 WL127 `field.npz`에서 읽었고 재선정하지 않았다.

| ROI | visible | withheld | continuation extent | median / h | p95 / h | coverage ≤ h / ≤ 2h | normal median |
|---|---:|---:|---:|---:|---:|---:|---:|
| curved table rim | 56.01% | 43.99% | 0.798 local-u | 22.63 | 55.65 | 2.71% / 7.93% | 24.26° |
| thin table leg/brace | 56.31% | 43.69% | 0.252 local-u | 10.19 | 20.34 | 0.00% / 0.67% | 59.15° |

두 case 모두 interface는 construction상 연속이다. position gap은 각각 약 `1e-6`, normal-angle discontinuity p95는 각각 `0.020°`, `0.023°`지만, withheld reference와의 거리는 크고 coverage가 낮다. 따라서 경계가 붙어 보인다는 사실을 geometry recovery 성공으로 해석하지 않았다.

### Meeting verdict

**C. NEGATIVE FEASIBILITY RESULT**

이 고정된 non-planar/ thin real-scene holdout에서 단순 NURBS tangent continuation은 withheld real geometry를 정량적으로 회복하지 못했다. Figure 2는 실패한 mandatory primary case를 숨기지 않고 그대로 보여준다.

True-occluded prototype은 controlled holdout가 성공하지 않았으므로 실행하지 않았다. 따라서 Figure 3 end-to-end Occluded-Surface feasibility prototype도 생성하지 않았다.

## 평가

- 평가 모집단은 withheld reference rows뿐이다. observed fitting residual을 성공 근거로 사용하지 않았다.
- normal은 withheld sample의 local PCA normal과 predicted NURBS normal의 unoriented angle이다.
- `NO COMPLETION`은 withheld 영역을 비워두는 sanity baseline으로 각 case에 함께 저장했다.
- Figure 1의 두 수치는 WL127 cache에서 재계산했다: renderer evidence coverage `≤ h = 89.981%`, ray-hit coverage `99.880%`.

## 구현 fidelity와 누출 고지

- 수동 선택: ROI box, affine axes, numeric bounds, `holdout_u_cut`.
- heuristic: boundary finite-difference tangent sweep과 local PCA normal estimate.
- full reference 사용: 고정 mask 적용, observed/withheld 분리, Figure A 및 evaluation target. full reference의 XYZ는 fitter input으로 들어가지 않았다.
- final paper method에서 허용할 수 없는 shortcut: 수동 ROI/extent 선택, withheld error를 보고 continuation 길이 또는 파라미터를 재선택하는 것, full reference oracle 사용.
- canonical TSDF, checkpoint, 161 cameras, Candidate B, historical topology, production behavior, NURBS implementation은 변경하지 않았다.

## 검증

- `tests/test_parametric_surface_continuation_demo.py`: **4 passed**
- 검증 계약: deterministic ROI/mask, one-sided boundary holdout, deterministic continuation, withheld row의 fitter input 교집합 0, withheld-only metrics.

## 남은 위험

- WL127 mesh의 local point density와 ROI 내 surface mixture가 작고 단순한 demo fitter의 extrapolation 품질을 제한했다.
- 본 결과는 canonical Occluded Surface의 negative proof가 아니라, 이 bounded heuristic의 feasibility negative result다.
