# Worklog 90 — full-region surface topology 실패의 upstream root-cause attribution

## 상태

**완료 — 원인은 중심점 graph의 단독 표현보다 upstream visible-Gaussian evidence distribution이다.** Worklog 89 boundary algorithm은 변경하지 않았다. 이번 배치는 boundary ordering/closure/patch fitting 실험이 아니라, Worklog 89의 `chart_unit_cut_non_manifold` 167개 coherent chart unit이 왜 full-region local surface face topology를 만들지 못하는지 covariance footprint로 읽기 전용 귀속한 진단이다.

## 고정한 범위

- visible Gaussian training, region ownership, Worklog 82 micro-component, Worklog 83 assembly, Worklog 84 coherence, Worklog 89 boundary 및 Worklog 79 coverage→PCA-UV→6×6 NURBS 체인은 모두 미변경이다.
- Worklog 82의 kNN=8, degree cap=12, normal alignment=0.85, mutual residual=0.35와 typed crease/frontier veto를 그대로 재실행해 relation 결과를 읽었다. threshold sweep이나 relation 재분류는 하지 않았다.
- 각 Gaussian은 center뿐 아니라 기존 covariance frame의 tangent ellipse와 normal thickness를 **1σ footprint 측정 규약**으로 사용했다. 이는 새 admission rule이나 production edge가 아니다.
- footprint pair를 새 graph/face/boundary로 만들지 않았다. Worklog 89가 이미 실패한 unit을 그대로 분류만 했다.

## 귀속 규약

각 failed unit 안에서 center-pair와 covariance footprint pair를 비교하고, node-level evidence mass가 가장 큰 원인을 primary cause로 하나만 기록했다. 동률은 presentation만 결정한다.

- `CENTER_UNDERSAMPLING`: 1σ tangent/depth/normal-compatible footprint가 있지만 bounded center graph가 local face-complex를 위한 triangle support를 제공하지 못한다.
- `RELATION_FALSE_NEGATIVE`: compatible footprint pair가 기존 Worklog 82 candidate였으나 `ambiguous` relation으로 거부됐다. typed crease/frontier veto는 false negative로 세지 않고 별도 disclosure한다.
- `TRUE_SUPPORT_GAP`: compatible footprint continuation도 competing layer overlap도 없다.
- `MULTILAYER_OR_VOLUMETRIC`: tangent footprint가 공간적으로 겹치지만 normal/depth compatibility가 깨지는 competing layer가 있다.
- `GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH`: compatible accepted same-surface triangle support는 있으나 fixed bounded graph의 face topology가 여전히 유효한 local 2-manifold가 되지 못한다.

`valid_local_surface_complex_plausible`은 center/footprint edge를 실제로 추가한다는 뜻이 아니다. primary evidence가 center undersampling, relation false negative 또는 graph-to-surface mismatch일 때만 “continuous footprint support를 representation으로 쓴다면 추가 조사 가치가 있음”으로 기록한 진단 flag다.

## 검증

`tests/test_chart_unit_surface_topology_attribution.py`는 다섯 primary class를 각각 hand-built covariance fixture로 검증한다. 신규 5개와 Worklog 79/82~89 관련 86개를 함께 실행한 focused 결과는 **91 passed in 6.52s**다.

## 7개 region replay

Checkpoint `output/extent_ab/val64/baseline_compatible/2900`, cap 2048, 전체 owned evidence 3526점. Worklog 89의 full-face topology non-manifold/open failure는 **167 unit / 3073 evidence**다. 산출물: `output/extent_ab/val90/chart_unit_surface_topology_attribution_replay.json`.

| region | failed unit | failed evidence | center under | relation FN | true gap | multilayer/vol. | graph mismatch | plausible footprint complex |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 83 | 0 | 0 | 0 | 83 | 0 | 0 |
| 1 | 23 | 461 | 8 | 0 | 9 | 431 | 13 | 21 |
| 2 | 20 | 486 | 18 | 0 | 12 | 456 | 0 | 18 |
| 3 | 9 | 74 | 0 | 0 | 9 | 65 | 0 | 0 |
| 4 | 55 | 893 | 57 | 0 | 12 | 814 | 10 | 67 |
| 5 | 7 | 350 | 0 | 0 | 23 | 327 | 0 | 0 |
| 6 | 51 | 726 | 58 | 0 | 30 | 634 | 4 | 62 |
| **전체** | **167** | **3073** | **141** | **0** | **95** | **2810** | **27** | **168** |

Primary evidence-weighted attribution:

- `MULTILAYER_OR_VOLUMETRIC`: **2810/3073 = 91.44%** (98 unit)
- `CENTER_UNDERSAMPLING`: **141/3073 = 4.59%** (38 unit)
- `TRUE_SUPPORT_GAP`: **95/3073 = 3.09%** (28 unit)
- `GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH`: **27/3073 = 0.88%** (3 unit)
- `RELATION_FALSE_NEGATIVE`: **0** primary unit/evidence

Primary classes는 상호 배타적이다. unit 내부의 겹치는 secondary node evidence도 따로 계산하면 center undersampling 949, relation false negative 211, true gap 343, multilayer 2547, graph mismatch 563 evidence-node mass다. 이는 한 unit 안에 여러 현상이 공존할 수 있음을 보이지만, 우세 원인을 바꾸지는 않는다.

## 요청한 footprint/graph 지표

failed evidence 가중 평균:

- center nearest-neighbor spacing / equivalent tangent footprint scale: **1.247**
- compatible 1σ footprint-overlap node coverage: **49.20%**
- footprint continuation이 있는데 same-surface edge가 없는 fraction: **11.60%**
- relation false-negative node fraction: **6.87%**
- local layer normal/depth ambiguity fraction: **82.88%**
- continuous footprint representation에서 valid local surface complex를 추가로 조사할 가치가 있는 evidence: **168/3073 = 5.47%**

raw compatible footprint pair는 1852개다. 그중 accepted same-surface 1494, bounded center graph에 없는 pair 177, Worklog 82 `ambiguous` 거부 181, typed provenance veto 0이다. Tangentially overlapping데 normal/depth layer가 incompatible한 pair는 6789개였다. 따라서 “compatible support가 있지만 relation이 주로 막는다”는 B 가설보다, 다층 footprint conflict가 먼저 발생한다는 C 가설이 더 직접적으로 지지된다.

## 대표 사례

- `MULTILAYER_OR_VOLUMETRIC`: region 2 / unit 10 / 17 evidence. footprint-compatible coverage 47.1%이지만 layer ambiguity 100%, competing layer pair 40개다. accepted same-surface pair 6개가 있어도 한 tangent sheet가 아니다.
- `CENTER_UNDERSAMPLING`: region 4 / unit 9 / 6 evidence. compatible coverage 100%, spacing/footprint scale 0.682, local face triangle 부족 node fraction 100%다. 이 소수 집합은 footprint-support representation을 별도로 검토할 근거가 있다.
- `TRUE_SUPPORT_GAP`: region 6 / unit 21 / 5 evidence. spacing/footprint scale 4.655, compatible footprint pair 0, true-gap node 5다.
- `GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH`: region 4 / unit 12 / 10 evidence. compatible coverage 90%, accepted same-surface pair 21개와 triangle-support node 90%가 있으나 fixed graph face complex가 실패했다. 단 27 evidence뿐이다.
- `RELATION_FALSE_NEGATIVE`: primary 사례는 없다. secondary로는 compatible pair 181개/failed evidence-node mass 211이 있으나 어느 unit에서도 multilayer 또는 다른 원인보다 크지 않았다.

## 결정

**Decision C.** true support gap 또는 특히 multilayer/volumetric ambiguity가 footprint support를 포함해도 압도적이다. covariance footprint로 center graph의 부족을 설명할 수 있는 primary evidence는 4.59%, graph representation mismatch는 0.88%이고, relation semantics의 primary false negative는 0%다. 따라서 지금은 covariance-footprint/surfel graph로 production constructor를 교체하거나 Worklog 82 threshold를 조정할 근거가 없다.

다음 upstream 조사에서는 boundary를 다시 만지지 말고 다음 training/ADC 분포 통계를 stable Gaussian ID와 iteration별로 추적해야 한다.

1. competing layer의 opacity-weighted rendered depth/visibility ordering, screen-space overlap 및 depth separation
2. ADC clone/split/prune 이후 tangent major/minor scale, normal thickness, planarity 및 center-spacing/footprint-scale 분포
3. local normal-depth layer multiplicity와 `uncertain_confidence`/uncertain mask/ownership의 상관
4. competing layer 위치의 photometric gradient, opacity, screen-space radius, prune 이유 및 ADC birth lineage

이는 다음 구현 지시가 아니라 upstream evidence-distribution의 원인을 확인하기 위한 관측 목록이다. Worklog 89 boundary algorithm과 canonical Region→Charts 경로는 계속 고정한다.
