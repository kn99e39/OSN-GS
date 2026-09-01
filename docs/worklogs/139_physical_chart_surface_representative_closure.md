# Worklog 139 — Physical-chart-constrained Surface Representative and controlled continuation closure

## 의도 정렬 (INTENT ALIGNMENT)

이 배치는 Worklog 127–138을 수정하지 않고, WL138의 낮은 raw-to-surface
거리가 macroscopic Surface Representative의 유효성을 충분히 보장하지 못한다는
문제를 닫는 비정규 architecture batch다. 핵심 가설은 graph-like visible surface의
representative가 frozen world-space physical `(u,v)` chart를 보존해야 한다는 것이다.

다음 두 질문만 검증했다.

1. WL138의 병적인 NURBS geometry는 free-3D parameterization이 physical chart를
   왜곡할 수 있었기 때문인가?
2. chart를 고정하면 한 장의 coherent visible macro sheet와 controlled continuation
   frame을 얻을 수 있는가?

whole-scene Occluded Surface, production fitter 변경, second-order continuation은
수행하지 않았다.

## 구현 충실도 (IMPLEMENTATION FIDELITY)

- 신규 경로는 `devtools/demo/physical_chart_surface_representative.py`와
  `output/confirmed/139_physical_chart_surface_representative/`에만 격리했다.
- canonical renderer, Candidate B, TSDF, `osn_gs/surface/torch_nurbs.py`, WL138
  module과 confirmed artifact를 수정하지 않았다. 실행 전후 confirmed WL138 file
  manifest SHA가 동일함을 report에서 확인했다.
- 삭제된 staging `_cache`를 재생성하지 않았다. 대신
  `output/confirmed/scale_separated_visible_surface_representative/`의 exact WL138
  retained/withheld populations, control grid, sampled representative, frontier,
  continuation NPZ/JSON을 read-only로 사용했다.
- curved exact population은 retained `13,443`, withheld `10,557`; leg/brace는
  retained `3,524`, withheld `6,319`다.
- historical settings는 `8×4`, degree `2/2`, smoothness/tikhonov `1e-4/1e-4`로
  고정했다. control resolution 또는 regularization sweep은 없다.
- 최소 operational difference는 물리 UV를 immutable하게 유지하기 위해 WL138의
  3D foot-point reprojection/correction round를 금지하고 normal-coordinate scalar
  controls만 한 번 선형 solve한 것이다.
- retained fit과 full evaluation-only fit은 role로 분리했다. continuation은
  `retained_construction` role만 받으며 full role은 assertion으로 거부한다.
- full/withheld geometry는 qualitative PASS 이후에만 평가에 사용했고, 어떤 metric도
  geometry construction에 feedback하지 않았다.

## WL138 접힘 baseline (WL138 FOLDED BASELINE)

confirmed WL138 geometry를 그대로 A/B baseline으로 재생했다. historical CUDA와 CPU
analytic replay의 최대 차이는 curved `7.23e-5` world unit 이하였으며, 비교에 사용한
geometry는 replay 값이 아니라 exact confirmed NPZ 배열이다.

curved rim baseline:

- raw → representative median/p95: `1.5155h / 3.3084h`
- representative → raw median/p95: `1.0533h / 32.9299h`
- analytic normal variation median/p95: `4.7501° / 16.1135°`
- area / physical chart footprint / inflation: `1.7020 / 1.3224 / 1.2871`
- physical frontier projection gap median/p95: `0.5732h / 2.5392h`

낮은 raw → representative 거리와 큰 reciprocal p95가 동시에 존재하므로 proximity
하나만으로 macro validity를 선언할 수 없음을 다시 확인했다.

## 접힘 원인 귀속 (FOLDING ATTRIBUTION)

curved rim WL138 baseline에서는 physical-u/v reversal, Jacobian flip, disconnected
multi-valued occupancy, near self-crossing이 모두 `0`이었다. 따라서 curved ROI에서
보였던 모든 shape 차이를 discrete chart fold 탓으로 귀속할 수 없다.

반면 leg/brace WL138 baseline은 다음의 명확한 pathology를 보였다.

- physical-u reversal: `592`
- Jacobian/orientation flip: `629`
- disconnected duplicate chart bins: `3`
- nonlocal near-crossing pairs within `h`: `5,007`

즉 free-3D NURBS가 실제 chart pathology를 허용한다는 점은 leg/brace에서 확인됐다.
그러나 아래 graphness gate가 leg raw evidence 자체를 multi-valued로 판정했으므로,
이 ROI는 scalar graph family의 positive validation case가 될 수 없다. 최종 귀속은
**partial attribution**이다.

## Raw graphness 감사 (RAW GRAPHNESS AUDIT)

retained raw geometry만 사용했다. physical chart bin width는 사전에 고정한 `4h`,
clearly separated mode는 양쪽에 최소 3점이 있고 adjacent normal-coordinate gap이
`>=3h`인 경우로 정의했다. graph gate는 eligible bin 중 multimode 비율 `<=10%`다.

| case | occupied / total bins | multimode / eligible | multimode fraction | within-bin n spread median / p95 | coverage | 판정 |
|---|---:|---:|---:|---:|---:|---|
| curved rim | 434 / 575 | 16 / 423 | 3.7825% | 4.6446h / 8.3296h | 75.4783% | PASS_GRAPH_LIKE |
| leg/brace | 28 / 84 | 5 / 25 | 20.0000% | 21.0671h / 30.7353h | 33.3333% | FAIL_MATERIALLY_MULTIVALUED |

leg/brace에는 `CURRENT GRAPH REPRESENTATIVE NOT APPLICABLE TO THIS STRUCTURE`를
적용하고 chart-constrained fit/continuation을 강제하지 않았다.

## Physical-chart-constrained representative

clamped uniform B-spline basis의 Greville abscissae를 physical u/v control coordinates로
사용했다. unit weights에서 linear precision으로 `u(s),v(t)`가 frozen affine chart를
정확히 재생하며, normal-coordinate `n(s,t)` control만 retained raw values에 fit한다.

curved retained representative 결과:

- physical-u affine precision median/p95: `4.77e-7 / 9.54e-7` world unit
- physical-v affine precision median/p95: `2.38e-7 / 4.77e-7` world unit
- fixed-UV fit residual median/p95: `1.3239h / 3.7515h`
- raw → representative median/p95: `1.4961h / 3.2988h`
- representative → raw median/p95: `1.0089h / 27.8454h`
- symmetric median / mean accounting: `1.2525h / 3.5315h`
- area / footprint / inflation: `1.5695 / 1.3224 / 1.1869`

reciprocal p95 tail은 rectangular chart의 sparse outer support를 드러내며 숨기지 않았다.
이는 후속 trimmed/support-aware graph representative의 open 문제다.

## 정성 macro-shape gate (QUALITATIVE MACRO-SHAPE GATE)

near-opaque raw points와 동일 local physical-chart viewpoint에서
`baseline_folded_representative.png`,
`raw_vs_chart_constrained_representative.png`,
`unconstrained_vs_constrained_representative.png`를 직접 검사했다.

curved chart-constrained representative는 obvious fold/self-crossing 없이 raw macro
shape을 따라가는 한 장의 coherent sheet였으므로 **PASS**로 고정했다. 낮은 거리
metric이 아니라 렌더 검사와 topology contract가 gate의 근거다.

leg/brace는 graphness fail이므로 qualitative graph gate 대상이 아니다.

## Unconstrained 대 chart-constrained A/B (UNCONSTRAINED vs CHART-CONSTRAINED A/B)

curved에서는 두 arm 모두 discrete reversal이 `0`이지만 chart-constrained arm은
physical affine error를 numerical precision까지 제거하고 area inflation을
`1.2871→1.1869`로 줄였다. raw proximity 차이는 작아 이 결과를 mere fit-distance
개선으로 해석하지 않는다.

## Topology/Jacobian/self-overlap accounting

curved chart-constrained representative:

- physical-u reversal: `0`
- physical-v reversal: `0`
- Jacobian/orientation flip: `0`
- duplicate multi-valued chart bins: `0`
- nonlocal near self-crossing pairs within `h`: `0`
- Jacobian median/p95: `1.322400 / 1.322403`

physical chart orientation은 fitted normal controls와 무관하게 construction으로
고정된다.

## Normal-field accounting

curved chart-constrained representative의 analytic normal variation median/p95는
`4.7579° / 13.3379°`, adjacent local normal angle median/p95는
`0.4744° / 1.3373°`다. raw PCA와 frontier representative normal 차이는
median/p95 `23.9410° / 71.7474°`이므로 raw normal은 계속 diagnostic/support
quantity로만 유지한다.

## Physical frontier mapping

WL138과 동일한 frozen world-space frontier 19점을 physical chart에 mapping했다.
NURBS UV edge는 termination semantics로 사용하지 않았다.

- frontier → chart representative gap median/p95: `0.8860h / 3.5707h`
- frontier representative normal variation median/p95: `3.1590° / 6.1427°`
- prediction boundary position gap median/p95: `0 / 0h`
- boundary normal discontinuity median/p95: 약 `1.21e-6° / 2.25e-6°`

## Controlled continuation A/B

qualitative/topology PASS 이후 같은 physical frontier, same first-order tangent-plane
rule, same historical full extent `0.798` world unit를 사용했다. curvature, second-order,
withheld bending은 없다.

| arm | median / p95 | coverage <=h / <=2h | normal median / p95 |
|---|---:|---:|---:|
| WL134 raw frame | 6.7088h / 13.2860h | 1.5629% / 8.5820% | 26.6395° / 80.8154° |
| WL138 unconstrained frame | 4.0789h / 10.2546h | 3.8837% / 19.4563% | 24.8971° / 79.2736° |
| WL139 physical-chart frame | **3.3492h / 8.7900h** | **4.0068% / 20.9529%** | 24.4497° / 79.3127° |

frozen WL138 interpretation criterion인 median `<=5h`, coverage `<=2h >=20%`를
WL139 arm만 통과했다. p95와 normal tail은 여전히 크므로 whole extent completion
성공으로 일반화하지 않는다.

## Frontier 거리별 결과 (DISTANCE-FROM-FRONTIER RESULT)

WL139 physical-chart continuation의 raw withheld error:

| physical distance | samples | median / p95 |
|---|---:|---:|
| 0–2h | 325 | 2.1718h / 4.2656h |
| 2–4h | 343 | 2.1569h / 3.8692h |
| 4–8h | 687 | 2.2405h / 4.0596h |
| 8–16h | 1,590 | 2.3339h / 4.2464h |
| >16h | 7,612 | 4.1868h / 9.3523h |

따라서 positive signal은 우선 `0–16h`, 즉 `0.1936878` world unit의 bounded local
range로 해석한다.

## Full macro reference validity

full chart-constrained representative는 qualitative PASS 이후 동일 설정으로만 fit했고,
held-out side는 sampled XYZ의 physical `u > fixed u_cut`으로 선택했다. parameter-u는
selection에 사용하지 않았다.

- raw withheld → macro median/p95: `1.6057h / 3.6525h`
- macro → raw withheld median/p95: `0.9406h / 23.6884h`

reciprocal p95가 fixed `12h` 기준을 넘으므로 **INVALID_MACRO_REFERENCE**다. 따라서
frozen prediction을 이 full representative와 비교한 metric은 최종 report에서
생성하지 않았고, 이를 physical ground truth로 부르지 않는다.

## Conditional true-occluded micro demo

controlled `0–16h` signal은 있지만 필수 read-only Candidate B archive
`output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz`
가 현재 workspace에 없다. output 전체 NPZ 검색에서도 발견되지 않았다.

판정은 **NOT_EXECUTED_CANDIDATE_B_ARCHIVE_UNAVAILABLE**다. persistent occlusion을
발명하거나 marker geometry로 대체하지 않았고 true-occluded candidate를 만들지
않았다.

## 시각화 및 geometry

output root는 `output/confirmed/139_physical_chart_surface_representative/`이다. near-opaque
raw/reference points(`alpha=0.98/0.99`, marker size `3.4`)와 동일 local physical-chart
view를 사용했다. display-only voxel thinning `0.02`는 fit/metric/PLY/NPZ에 영향을
주지 않는다.

필수 PNG와 `chart_constrained_continuation.ply`, representative/continuation NPZ를
각 case 폴더에 저장했다. continuation overlay에서 gray retained, green withheld,
cyan WL139 continuation, yellow frontier를 직접 구분할 수 있다.

## 승격 (PROMOTED)

- Visible Surface Evidence와 Surface Representative는 별도 layer다.
- graph-like broad surface에서는 physical-chart preservation이 representative의
  topology contract다.
- fixed-chart 8×4 degree-2 B-spline graph는 curved rim의 viable retained macro
  representative다.
- analytic chart frame은 bounded local first-order continuation에 사용할 수 있다.

## 유지 (RETAINED)

- NURBS/B-spline representation family
- WL127 raw TSDF mesh와 confirmed WL138 populations를 immutable evidence carrier로 유지
- first-order continuation을 local diagnostic propagation으로 유지
- raw normals를 diagnostic/support quantity로 유지

## 거부 (REJECTED)

- free-3D unconstrained NURBS를 topology audit 없이 macro representative로 승인하는 방식
- raw → surface nearest distance만으로 macro shape을 승인하는 방식
- NURBS parameter semantics로 physical world-space semantics를 대체하는 방식
- multi-valued leg/brace에 scalar graph representative를 강제하는 방식

## 미해결 (OPEN)

- non-graph/multi-sheet 및 thin-structure representative
- sparse chart support trimming
- long-range completion과 junction/closure prior
- confidence와 principled continuation extent
- Candidate-B-supported true-occluded micro validation

## 최종 architecture 판정

**A. PHYSICAL-CHART REPRESENTATIVE PROMOTED**

curved rim의 retained chart-constrained representative는 visually coherent하고
physical chart topology를 construction으로 보장하며, frozen controlled holdout에서
WL134/WL138보다 나은 bounded continuation signal을 제공했다. 이 판정은
`Occluded Surface solved` 또는 final OSN-GS algorithm을 의미하지 않는다.

folding 원인은 부분적으로 닫혔다. free-3D parameterization은 leg/brace에서 실제
reversal/crossing을 허용했지만 curved baseline에는 discrete reversal이 없었다.
따라서 chart preservation은 graph-like representative의 필수 방어 계약으로
승격하되 모든 historical shape error의 단일 원인으로 주장하지 않는다.

## 검증

- WL139 focused tests와 보존 대상 WL138/WL136 adjacent tests: `22 passed`
- 실데이터 final run: 2 cases, failures `[]`
- confirmed WL138 manifest: unchanged
- full repository regression: production code를 변경하지 않아 실행하지 않음