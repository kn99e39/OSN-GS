# Worklog 138 — scale-separated Visible Surface representative closure

## 의도 정렬 (INTENT ALIGNMENT)

이 배치는 canonical OSN-GS Occluded Surface architecture를 진행하지 않고,
Worklog 127의 raw Visible Surface Evidence와 smooth Surface Representative를
분리해 점검하는 비정규 feasibility/diagnostic track이다. 질문은 다음 하나로
고정했다.

> 관측된 실제 Visible Surface만으로 맞춘 parametric representation이
> boundary-attached missing region으로 계속되어 withheld real geometry를
> 회복할 수 있는가?

두 ROI 모두 중앙 interior hole이 아니라 고정된 `u_cut` 뒤쪽을 withheld하는
boundary-attached holdout이다. true-occluded prototype은 조건부 gate를 실행하지
않았고, 결과는 continuation feasibility에서 멈췄다.

## 구현 충실도 (IMPLEMENTATION FIDELITY)

- 구현은 `devtools/demo/scale_separated_visible_surface_representative.py`에만
  추가했다. 기존 `osn_gs/surface/torch_nurbs.py`, canonical renderer,
  Candidate B, WL127/136/137 결과는 수정하지 않았다.
- 입력 raw geometry는
  `output/confirmed/127_osn_gs_evidence_bounded_projective_tsdf/_cache/mesh.npz`의
  Worklog 127 mesh이며, `field.npz`에서 `h`와 `mu`만 read-only로 읽었다.
- 기존 NURBS fitter의 고정 설정을 full evaluation-only fit과 retained-only fit에
  동일하게 적용했다: `resolution_u=8`, `resolution_v=4`, `degree_u=2`,
  `degree_v=2`, `smoothness_lambda=1e-4`, `tikhonov_lambda=1e-4`,
  `correction_rounds=2`, `projection_iterations=2`.
- 수동 선택은 두 ROI의 의미적 위치와 고정 continuation extent뿐이다. leg/brace는
  WL136 H1 box와 `u_cut=0.75`, curved rim은 기존 fixed ROI와 world-space
  `u_cut=-5.498`을 사용했다. 이 ROI/extent 선택은 demo-only이며 withheld
  geometry를 보고 사후 선택하지 않았다.
- full representative는 withheld 영역을 포함하므로 evaluation-only다. retained
  representative, physical frontier mapping, continuation construction에는
  retained points와 frozen frontier만 전달했다.
- withheld XYZ는 target visualization과 metric 계산에만 사용했다. continuation
  길이, fitter 설정, ROI, threshold를 평가 후 변경하지 않았다.

## Raw 표면 스케일 감사 (RAW SURFACE SCALE AUDIT)

`h=0.012105485424399376`은 WL127 field cache의 고정 값이다. 아래 raw 통계는
각 ROI의 retained raw points에서만 계산했다.

| case | retained / withheld point | PCA normal dispersion median / p95 | NN median / p95 (h) | k8 local scale median / p95 (h) | plane residual median / p95 (h) | coarse diagonal (h) |
|---|---:|---:|---:|---:|---:|---:|
| `wl136_leg_brace` | 3,524 / 6,319 | 53.6563° / 86.3379° | 0.44095 / 1.00015 | 1.22126 / 1.84023 | 3.98145 / 9.65395 | 53.7079 |
| `curved_table_rim` | 13,443 / 10,557 | 22.8014° / 71.3381° | 1.03161 / 1.67994 | 2.33547 / 3.29033 | 1.70158 / 5.39151 | 129.1864 |

leg/brace raw geometry는 local scale 대비 normal/plane noise가 매우 크고,
curved rim은 macro curvature가 보이지만 raw normal은 여전히 거칠다. 따라서
raw PCA를 직접 continuation frame으로 승격하지 않고, 별도의 smooth
representative를 진단 가설로 사용했다.

## Full evaluation-only representative

각 full ROI 전체에 기존 NURBS fitter를 적용해 withheld 쪽을 포함한 macro
reference를 만들었다. 이 fit은 retained construction에 사용하지 않았다.

| case | fit residual median / p95 (h) | representative sample |
|---|---:|---:|
| `wl136_leg_brace` | 3.57023 / 8.44543 | 3,840 |
| `curved_table_rim` | 1.14249 / 3.37143 | 3,840 |

## Retained representative

retained raw points만으로 같은 고정 fitter를 실행했다. retained raw와의 fit
residual 및 representative analytic normal variation은 다음과 같다.

| case | raw-to-representative median / p95 (h) | analytic normal variation median / p95 | fit residual median / p95 (h) |
|---|---:|---:|---:|
| `wl136_leg_brace` | 1.66165 / 5.19840 | 20.9909° / 75.7133° | 1.85163 / 5.57568 |
| `curved_table_rim` | 1.51547 / 3.30843 | 4.75013° / 16.1135° | 1.11129 / 3.15944 |

## Visible macro-shape 감사 (VISIBLE MACRO-SHAPE AUDIT)

고정 기준은 representative가 retained raw와 비교해 median `<=3h`, p95
`<=12h`이고 유한한 값을 갖는지였다. 이 기준은 continuation이나 withheld
metric을 입력으로 받지 않는다.

- `wl136_leg_brace`: `PASS` — median `1.66165h`, p95 `5.19840h`.
- `curved_table_rim`: `PASS` — median `1.51547h`, p95 `3.30843h`.

따라서 두 case 모두 representative가 관측된 macro shape을 설명하는 데는
유용하지만, 이것만으로 continuation 성공을 주장하지 않는다.

## 물리 frontier에서 representative mapping (PHYSICAL FRONTIER -> REPRESENTATIVE MAPPING)

WL136 방식의 frozen world-space frontier를 사용하고, NURBS rectangular UV edge를
termination으로 사용하지 않았다. raw frontier를 retained representative에
projection한 뒤 representative의 analytic frame을 계산했다.

| case | frontier samples | projection gap median / p95 (h) | fixed physical extent |
|---|---:|---:|---:|
| `wl136_leg_brace` | 13 | 1.53330 / 2.65515 | 0.3300 world unit |
| `curved_table_rim` | 19 | 0.57317 / 2.53924 | 0.7980 world unit |

frontier 위치와 target extent는 각 fixed ROI contract에서 결정되며 withheld XYZ로
맞추지 않았다. 특히 leg/brace의 raw frontier projection 자체가 불안정한 점은
해석 시 별도 위험으로 남긴다.

## Raw normal과 representative normal (RAW NORMAL vs REPRESENTATIVE NORMAL)

raw PCA normal과 representative analytic normal의 frontier 비교는 다음과 같다.

| case | angle median / p95 |
|---|---:|
| `wl136_leg_brace` | 18.1969° / 77.4787° |
| `curved_table_rim` | 22.0331° / 72.0808° |

representative analytic normal을 continuation frame으로 사용했으며 raw PCA normal은
진단용으로만 유지했다. observed/predicted boundary position gap은 두 case 모두
`0 / 0h`, boundary normal discontinuity는 leg/brace `0° / 0.01717°`, curved rim
`0.00568° / 0.01576°` (median / p95)였다. 이는 interface에서의 local
representative continuity만 보이는 것이며 withheld geometry 회복을 의미하지
않는다.

## Representative-frame continuation

고정된 하나의 규칙을 두 case에 적용했다.

1. physical frontier를 retained NURBS에 projection한다.
2. representative analytic tangent plane에서 physical `u` 방향을 tangent
   plane으로 투영한다.
3. 그 방향으로 first-order tangent-plane continuation을 fixed physical extent만큼
   생성한다. curvature fitting, second-order term, collision rejection, target
   feedback은 사용하지 않았다.
4. normal은 representative `dv` tangent와 continuation direction으로 계산하고
   representative normal orientation에 맞춘다.

이 규칙은 heuristic이며 Worklog 138의 feasibility audit용이다. 아래 평가는
construction 이후에만 계산된 frozen post-evaluation interpretation이다. median
`<=5h` 및 coverage `<=2h >=20%`를 두 case 모두 만족하는 경우만 continuation
pass로 해석하도록 미리 고정했으며, 현재 두 case 모두 통과하지 못했다.

## Raw held-out metric

metric은 withheld raw reference에 대해서만 계산했다.

| case | median / p95 (h) | coverage <=h / <=2h | normal median / p95 |
|---|---:|---:|---:|
| `wl136_leg_brace` | 14.94517 / 26.35577 | 0.2849% / 1.8199% | 60.4998° / 86.9083° |
| `curved_table_rim` | 4.07889 / 10.25458 | 3.8837% / 19.4563% | 24.8971° / 79.2736° |

curved rim은 representative가 macro shape을 설명하지만 first-order continuation은
held-out raw geometry와 충분히 가깝지 않다. leg/brace는 raw scale/noise와
continuation error가 모두 크게 나타난다.

## Macro held-out representative metric

별도로 full evaluation-only representative의 withheld-side sample을 target으로
계산했다. 이 결과는 raw target과 구분해 macro-shape extrapolation을 확인하는
참조이며 construction에 사용하지 않았다.

| case | median / p95 (h) | coverage <=h / <=2h | normal median / p95 |
|---|---:|---:|---:|
| `wl136_leg_brace` | 19.50888 / 28.38561 | 0% / 0% | 55.5706° / 86.0698° |
| `curved_table_rim` | 91.11986 / 92.74604 | 0.3906% / 1.3021% | 11.0859° / 20.3535° |

curved rim에서 macro representative의 withheld-side position error가 특히
크게 나타나므로, raw held-out 결과만이 아니라 representative-to-representative
비교도 continuation의 한계를 보여준다. 이를 숨기거나 raw fit residual로
대체하지 않았다.

## Raw visualization paths

출력 root는 `output/confirmed/138_scale_separated_visible_surface_representative/`이다. 각 case에
다음 raw fixed-view PNG가 생성된다.

- `raw_visible_surface.png`
- `raw_vs_full_representative.png`
- `retained_raw_vs_retained_representative.png`
- `raw_normals_vs_representative_normals.png`
- `representative_continuation.png`
- `continuation_vs_raw_reference.png`
- `continuation_vs_macro_reference.png`

또한 retained/withheld/full representative/continuation geometry의 PLY와 NPZ를
저장했다. PNG의 raw/reference point는 `alpha=0.98/0.99`, point size `2.35`
(reference는 필요 시 `2.7`)로 near-opaque하게 그렸다. display-only voxel thinning
(`0.02` world unit)은 렌더링 성능을 위한 것이며 저장 geometry, metric population,
fit input을 변경하지 않는다. 회색 retained/raw, 녹색 withheld reference, 파란
full representative, 주황 retained representative, cyan prediction, 노란 frontier로
구분해 raw surface의 거칠기와 smooth representative를 동시에 직접 확인할 수
있도록 했다.

## 승격·유지·거부·미해결 (PROMOTED / RETAINED / REJECTED / OPEN)

- **PROMOTED**: raw Evidence와 smooth Surface Representative를 별도 계층으로
  보고하는 scale-separated diagnostic contract. 이는 canonical architecture
  승격이 아니다.
- **RETAINED**: Worklog 127 raw mesh, fixed ROI/holdout, 기존 NURBS fitter,
  physical-frontier mapping, fixed first-order representative-frame rule.
- **REJECTED**: raw mesh PCA를 직접 structural continuation frame으로 사용하는
  방식. raw local normal dispersion이 높고 case 간 일관된 frame으로 보기 어렵다.
- **OPEN**: representative scale 선택의 원리, 장거리 shape prior, termination/
  occlusion extent, confidence, junction transfer 및 publishable canonical method.

## 판정

**B. PARTIAL FEASIBILITY DEMO — representative useful but continuation insufficient.**

두 mandatory ROI에서 retained-only NURBS representative는 visible macro shape
감사 기준을 통과해 raw surface를 읽는 smooth diagnostic representation으로는
유용했다. 그러나 같은 고정 first-order continuation은 curved rim과 thin
leg/brace의 withheld real geometry를 quantitatively close하게 회복하지 못했다.
따라서 `A. STRONG FEASIBILITY DEMO`를 선택하지 않는다. true-occluded prototype,
second-order continuation, parameter sweep, canonical Occluded Surface 구현은
수행하지 않았다.
