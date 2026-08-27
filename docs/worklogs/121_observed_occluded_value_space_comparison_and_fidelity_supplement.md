# Worklog 121 -- Worklog 120 Value-Space Comparison and Fidelity Supplement

브랜치: `arch/2dgs-coverage-first-surface`
역사적 기준 커밋: `fdfb8ad60b6233ea8364a09ea3467c18e600a246` (Worklog 120)

---

## 1. Agent Interpretation of Intent

**DIRECTION**: WL120의 네 후보 state-decision 함수(A. Surface-Hit / B. Median-Depth / C. Geometric Visibility / D. Renderer Reachability)와 frozen global aggregation을 **정확히 보존**한다. 최적화도, 결합도, 다섯 번째 후보도, aggregation 변경도 하지 않는다. 대신 그 결정들 **아래에 깔린 실제 surface/depth/blocker/transmittance 값**을 (1) WL120의 원본 query bank 그대로, (2) R5 같은-영역 중점 근사가 아니라 **실제 frozen visible-topology fragmentation**에 묶인 새 보충 bank에서 측정하는 진단 계층을 추가한다.

**PURPOSE**: 독립 소스 리뷰가 찾아낸 다섯 가지 해석/충실도 문제(C의 `nearest_blocker_t`가 MAX(t)라는 점, C의 rho3d 타원이 렌더러의 완전한 기여 support가 아니라는 점, D의 probe가 center-depth 정렬 traversal을 따른다는 점, D의 `T_pre`가 1e-4로 분리되지 않는다는 점, R5가 실제 fragmentation을 재생하지 않았다는 점)를 **보고 계층에서** 교정하고, 교정된 원시 값 증거가 WL120의 candidate ranking을 강화하는지 약화하는지 판정한다.

**CENTRAL INTENT**: 질문은 "B/C/D를 어떻게 더 잘 작동시킬까"가 **아니다**. "실제 밑값들을 올바르게 비교했을 때 WL120의 아키텍처 순위가 여전히 유효한가"이다. **B와 D 사이의 threshold를 찾지 않는다.** "진짜 경계는 median T=0.5와 termination T=1e-4 사이 어딘가에 있다"고 **주장하지 않는다** -- 그 진술은 확립되지 않았다. B와 D는 서로 다른 렌더러 수준의 질문(B = renderer-defined visible-surface frontier, D = canonical traversal termination/reachability)에 답하고 있을 수 있으며, 이번 배치의 목적은 그 구분을 **측정하는 것이지 지우는 것이 아니다**. 부정적 결과도 유효하다.

### PRESERVE
학습된 checkpoint, 161 학습 카메라, WL107/109 visible topology, canonical renderer, **WL120의 A/B/C/D 결정 함수(파일 무수정)**, WL120 global aggregation, 원본 4,712-query bank, 원본 candidate state 배열, WL120 S1-S7 synthetic contracts.

### CHANGE ONLY
보충 value/provenance 진단, 교정된 provenance 이름/계산, 보충 real topology-gap query bank.

### DO NOT (전부 준수)
A/B/C/D 결정 변경 / Candidate E 생성 / B+D hybrid / T threshold 탐색 / 0.5~1e-4 사이 sweep / confidence weighting / percentage vote / SparseObserved / geometric blocker tolerance / same-component blocker 무시 / opacity hardening / opaque proxy / occlusion용 mesh 재구성 / visible topology 수정 / component 병합·분할 / NURBS fitting 수정 / occluded NURBS / Trust / occluded surface 구축 / downstream 자동 진행 -- **하나도 하지 않았다.**

### PROMPT-REQUIRED DECISION
- 네 결정 함수와 aggregation의 완전 보존, 그리고 baseline replay gate가 실패하면 **값 해석을 중단**한다는 규칙(section 2).
- C의 `camera_nearest_blocker_t = MIN(t)` / `query_nearest_blocker_t = MAX(t)` 분리 보고, world gap·opacity·blocker count·same-component 귀속(section 6).
- D의 resolution reason 4분류, `termination_test_T = T_pre * (1 - alpha)` 노출과 `test_T < 1e-4` 계약 검증, `T_pre < 1e-4` 주장 금지(section 7).
- accepted-event depth inversion 및 `late_front_event_count` 측정(section 8).
- 실제 frozen fragmentation 기반 보충 bank: 관측된 raster-local 인접이 최종 component 분리를 가로지르는 컨텍스트, endpoint A/B + world midpoint(section 11).
- 검증된 zero-relevant-view control(section 12), S-D1/S-C1/S-B1(section 13), 축별 분리 보고(section 14), 용어 교정(section 16).

### AGENT-INTRODUCED OPERATIONAL CHOICE (전부 공개)
1. **CUDA 진단 필드 5개 추가**(`_qdepth` sibling, worklog 121 표시): `query_resolution_depth`, `query_termination_alpha`, `query_late_front_count`, `pixel_inversion_count`, `pixel_max_backward_jump`. 전부 순수 additive이며 기존 출력의 비트 단위 불변을 테스트로 강제했다.
2. **`late_front_count` 이벤트 정의**: 슬롯이 해소된 **뒤에** 처리되는 accepted 이벤트 중 자기 per-pixel depth가 여전히 query depth보다 **앞**인 것의 개수. 해소 이벤트 자신은 절대 포함되지 않는다(측정이 resolution 루프보다 먼저 실행된다).
3. **`pixel_inversion_count` 정의**: accepted 이벤트의 depth가 지금까지의 accepted 최대 depth보다 작아지는 횟수. `pixel_max_backward_jump`는 그 최대 낙폭.
4. **C의 world gap 정의**: 광선 방향 거리 `(1 - t) * ||x - camera||`. `blocker_region_thickness = (t_max - t_min) * ||x - camera||`. 정규화된 거리를 새로 발명한 것이 아니라 기존 t를 world 단위로 환산한 것이다.
5. **보충 bank 선택 규칙**(전부 stride/개수, 결정에 영향 없음): 뷰당 raster-order 균등 stride 상한 2,000, 영역당 컨텍스트 60개, 컨텍스트당 query 3개(endpoint A / endpoint B / world midpoint -- 튜닝된 보간 파라미터 없음, midpoint가 유일한 a-priori 내부 위치).
6. **검증된 control 구성**: scene extent의 `{12, 16, 20, 24}`배를 6축으로 바깥으로 걸어나가며, **candidate 평가 이전에** relevant-view count == 0을 확인한 점만 채택(목표 16개).
7. **S-D1/S-C1/S-B1 fixture 상수**: tilt 80도, tilted scale 3.0, flat centre z=0.05, query depth 4.10; S-C1은 8겹 opacity-0.15 spacing 0.01; S-B1은 pixel-centre / half-pixel-offset 두 경우. 결과 방향을 보기 전에 확정했다.
8. **C 값 패스와 결정 패스의 이중 실행**: 값을 재계산한 뒤 frozen `candidate_c.classify_view`의 blocked 집합과 **일치하는지 매 뷰마다 검사**했다(실측 불일치 0건). 런타임 2배 비용은 결정 불변성을 코드로 증명하기 위한 의도적 선택이다.
9. **새 numerical epsilon 없음**: C의 값 패스는 frozen `candidate_c.SEGMENT_EPSILON`과 `GeometricSceneSupport`를 그대로 import해서 쓴다.

---

## 2. Historical Baseline Replay

**GATE: PASS** (`baseline_replay_gate`). 값 해석은 이 게이트를 통과한 뒤에만 산출된다 -- 실패 시 스크립트는 리포트만 쓰고 `SystemExit`으로 중단하도록 구현되어 있다.

| 항목 | 결과 |
|---|---|
| query count | 4,712 (일치) |
| query 순서·positions | **비트 단위 동일** |
| kind / source_view / source_surfel / region / ladder_step / support_radius | **전부 동일** |
| A/B/C/D per-view state 배열 (4712 x 161) | **전부 동일** |
| A/B/C/D global state 배열 | **전부 동일** |
| relevance code / relevant-view count | **동일** |
| query_depth (4712 x 161) | **동일** |
| C 값 패스 vs frozen 결정 blocked 집합 불일치 | **0건** |
| D termination 계약 위반(`test_T < 1e-4`) | **0건** |

부수적 동결 지문:
- median surface representative 합집합 = **785,937** (WL119/120과 정확히 일치).
- WL107/109 topology 재생 = **component 559,989 / singleton 535,910 / largest fraction 0.36771306** -- WL107/109/112-120과 정확히 일치.

state 재생 수치도 WL120 그대로다: A global (OBS 2,676 / OCC 648 / UNRES 1,388), B (4,054 / 654 / 4), C (767 / 3,941 / 4), D (**4,708 / 0** / 4). relevant pair 383,322 / 758,632.

WL120의 S1-S7 synthetic contract도 그대로 재실행했고 **구현 충실도 실패 0건**이다.

---

## 3. Historical Decision Invariance Audit

| 결정 | 정확한 소스 | 이번 배치의 변경 |
|---|---|---|
| A per-view state | `scripts/devtools/observed_occluded/candidate_a_surface_hit.py::classify_view` | **없음(파일 무수정)** |
| B per-view state | `candidate_b_median_depth.py::classify_view` | **없음** |
| C per-view state | `candidate_c_geometric_visibility.py::classify_view` | **없음** |
| D per-view state | `candidate_d_renderer_reachability.py::classify_view` | **없음** |
| global aggregation | `shared.py::aggregate_global` | **없음** |
| C blocker 기하 상수 | `candidate_c.SEGMENT_EPSILON`, `candidate_c.GeometricSceneSupport`, `shared.canonical_geometric_support_rho_max` | **없음(값 패스가 그대로 import)** |
| D termination 조건 | vendored `forward.cu`의 `if (test_T < 0.0001f)` | **없음(관측만 추가)** |

`git status` 기준 이 배치가 수정한 결정 관련 파일은 **0건**이다. `value_diagnostics.evaluate_with_values`는 네 `classify_view`를 직접 호출하며, 테스트 `TestHistoricalDecisionInvariance`가 (a) 각 후보 모듈이 `classify_view`를 유지하는지, (b) 값 계층이 스스로 상태를 만들지 않는지, (c) topology-gap 모듈이 read-only 재생 함수 집합 바깥을 import하지 않는지, (d) `subset_ids`가 canonical connected-component 결과에서만 한 번 대입되는지를 AST로 강제한다.

`_qdepth` sibling 확장은 section 0의 세 조건을 모두 만족한다: 기존 D 출력 비트 동일(`TestQDepthWorklog121Additivity`), canonical 출력 비트 동일(WL107 `_diag` 빌드와 12개 필드 비교), WL120 state 배열 비트 동일(section 2 게이트).

---

## 4. Value-Diagnostic-to-Code Ownership Map

| path | function | ownership | 계산하는 정확한 양 | state 변경 가능? | 테스트 |
|---|---|---|---|---|---|
| `observed_occluded/value_diagnostics.py` | `evaluate_with_values` | SHARED | 한 스윕으로 frozen state + 값 테이블 동시 산출 | **NO**(state는 frozen 함수 호출 결과) | `TestReplayGateHelpers`, section 2 게이트 |
| " | `candidate_c_blocker_values` | C-VALUE | min-t/max-t blocker, world gap, thickness, opacity, surfel/component id, same-component count | **NO**(결정 blocked 집합과 매 뷰 일치 검사) | `TestCorrectedBlockerProvenance`(6) |
| " | `d_resolution_reason` | D-VALUE | probe flag -> 4개 resolution reason | **NO** | `TestDResolutionReasonMapping`(2) |
| " | `assert_historical_state_replay` / `bank_replay_check` | SHARED | 역사적 재생 게이트 | **NO** | `TestReplayGateHelpers`(2) |
| `observed_occluded/topology_gap_bank.py` | `replay_frozen_topology` | TOPOLOGY-GAP | WL107/109 재생(read-only 함수만) | **NO** | section 2 지문 일치 |
| " | `collect_cross_component_contexts` | TOPOLOGY-GAP | 최종 component 분리를 가로지르는 관측 raster 인접 | **NO** | `TestTopologyGapProvenance`(2) |
| " | `attribute_gating` | TOPOLOGY-GAP | 재생 산출물에서 읽은 gating 사유 | **NO** | `TestTopologyGapProvenance` |
| " | `build_verified_out_of_frustum_controls` | TOPOLOGY-GAP | 평가 전 relevant-view == 0 확인된 control | **NO** | 실측 8/8 UNRESOLVED |
| " | `build_supplemental_bank` | TOPOLOGY-GAP | 컨텍스트당 endpoint A/B + midpoint | **NO** | `TestTopologyGapProvenance` |
| `observed_occluded/synthetic_value_contracts.py` | `build_s_d1` / `build_s_c1` / `build_s_b1` | D-/C-/B-VALUE | 통제 fixture 값 | **NO** | `TestAcceptedDepthInversion`(3), `TestSyntheticValueContracts`(4) |
| `osn_gs/render/vendor/diff_surfel_rasterization_qdepth/cuda_rasterizer/forward.cu` | worklog 121 probe 블록 | D-VALUE | resolution depth / termination alpha / late-front / inversion 카운터 | **NO**(순수 additive) | `TestQDepthWorklog121Additivity`(3), `TestDTerminationContract`(4) |
| `scripts/devtools/observed_occluded_value_space_comparison.py` | `main` 및 metric 함수 | SHARED | 오케스트레이션·집계·export | **NO** | 실 scene 재생 |

**하나의 함수가 여러 후보의 결정 의미를 담은 경우: 없음.**

---

## 5. Implementation Fidelity Statement

| OUR INTENT | AGENT INTERPRETATION | ACTUAL IMPLEMENTATION | MEASURED VALUE | ARCHITECTURE INTERPRETATION |
|---|---|---|---|---|
| C의 nearest blocker 모호성 교정 | min-t(카메라측)/max-t(질의측) 분리 보고 | `candidate_c_blocker_values` | camera-nearest t 중앙값 **0.9692** vs query-nearest **0.9993** | WL120이 보고한 0.99935는 **질의측**이었다. blocker 영역은 WL120 숫자가 시사한 것보다 훨씬 두껍다 |
| C primitive 명명 교정 | rho3d footprint ≠ 완전한 렌더러 support | 리포트 `candidate_C_values.primitive(_caveat)` | -- | C의 기하는 렌더러가 실제로 합성하는 것의 **엄격한 부분집합** |
| D를 traversal-order로 명명 | 물리 depth prefix로 자동 해석 금지 | `pixel_inversion_count` / `late_front_count` | relevant pair의 **99.998%**가 inversion 있는 픽셀, REACHED 해소의 **81.2%**가 late-front ≥ 1 | D의 OBSERVED-by-reaching 측은 traversal order에 **material하게 영향받음** |
| `T_pre`는 1e-4로 분리되지 않음 | `test_T = T_pre*(1-alpha)`를 별도 노출 | CUDA `query_termination_alpha` | termination 97,676건 중 **`T_pre < 1e-4`는 0건**, `test_T < 1e-4`는 100% | WL120 문구가 부정확했음이 **완전 데이터로 확정** |
| R5 근사 대체 | 실제 component 분리를 가로지르는 관측 인접 | `topology_gap_bank` | 관측된 cross-component 인접 **5,713,235건**, 300 컨텍스트 선택 | 근사 없이도 WL120 순위 재현 |

### PROMPT-REQUIRED vs AGENT-INTRODUCED
section 1에 전부 열거했다. 새 threshold·새 tolerance·새 numerical epsilon은 **하나도** 도입하지 않았다(C는 frozen epsilon을 import, D는 canonical 상수만 검증용으로 참조).

### INABILITY TO REALIZE REQUESTED DIAGNOSTIC
**1건 있다.** section 8이 요구한 "late front event" **개수**는 전수 측정했지만, 그 이벤트들을 물리 depth 순서로 합성했을 때의 **교정된 prefix transmittance**는 계산하지 **않았다** -- 그러려면 각 late-front 이벤트의 `alpha`를 슬롯별로 누적해야 하고, 이는 요청된 진단(개수)을 넘어서는 새 커널 상태다. 따라서 "traversal order가 물리 순서였다면 D의 OCCLUDED가 몇 건 늘었을 것"이라는 양은 **이 배치의 증거로 산출되지 않으며, 추정치로 대체하지도 않았다.** 방향만 말할 수 있다: 물리 순서였다면 prefix에 이벤트가 더 들어가므로 T는 **감소만** 하고, 따라서 OCCLUDED는 **늘어날 수만** 있다(줄 수 없다).

---

## 6. Original Query-Bank Reproduction

원본 bank 4,712 query x 161 view = 758,632 쌍, relevant 383,322(50.53%). relevance 분해: 투영 무효 95,750 / near 미만 4,738 / 이미지 밖 274,822 -- WL120과 동일.

state space(전부 WL120과 비트 동일):

| | per-view OBSERVED / OCCLUDED / UNRESOLVED | global OBSERVED / OCCLUDED / UNRESOLVED |
|---|---|---|
| A | 2,676 / 245,102 / 135,544 | 2,676 / 648 / 1,388 |
| B | 137,708 / 245,614 / 0 | 4,054 / 654 / 4 |
| C | 39,706 / 343,616 / 0 | 767 / 3,941 / 4 |
| D | 285,646 / 97,676 / 0 | 4,708 / **0** / 4 |

---

## 7. Supplemental True-Fragmentation Gap Bank

frozen WL107/109 topology를 read-only 함수로 재생한 뒤(section 2에서 지문 일치 확인), 각 뷰에서 렌더러 자신의 per-pixel representative map을 WL107과 **동일한** 4-연결(right/down)로 훑어, 두 representative가 **서로 다른 surfel이면서 서로 다른 최종 visible component**에 속하는 raster 이웃을 관측 컨텍스트로 수집했다.

- 161뷰 전체에서 관측된 cross-component raster 인접: **5,713,235건**.
- 영역별 가용 컨텍스트: table_top 58,864 / table_side_curved 59,015 / table_legs 23,696 / patio 124,984 / hedge 55,441.
- 결정론적 stride로 영역당 60개씩 **300 컨텍스트** 선택 -> endpoint A / endpoint B / world midpoint **900 query** + 검증된 control 8개 = **908 query** (relevant pair 70,563).

**gating 귀속**(재생 산출물에서 읽음, 새 규칙 없음):

| 사유 | 컨텍스트 |
|---|---|
| REJECTED_BY_3D_LOCALITY_FILTER | **288** |
| REJECTED_BY_SECONDARY_GEOMETRIC_GATE | 12 |
| POSITIVE_EDGE_YET_DIFFERENT_COMPONENTS | 0 |

즉 이 지점들의 fragmentation은 압도적으로 **3D-locality 제한**(image-space 인접 쌍이 애초에 spatial candidate edge가 아니었음)에서 온다.

**검증된 out-of-frustum control**(section 12): 24개 후보를 바깥으로 걸어 검사해 **8개**가 relevant-view count == 0으로 확인되었고(평가 전 확인), 네 후보 모두 그 8개에 대해 **전부 UNRESOLVED**를 반환한다. WL120의 R6는 12개 중 8개가 실제로는 프러스텀 안이었으므로 이 control이 그 결함을 대체한다. 원본 R6는 역사적 재생을 위해 그대로 두었다.

---

## 8. Candidate A Value Results

결정 무변경. 값:

| state | signed_depth_delta_A = query_depth − event_depth (중앙값 / min / max) | pairs |
|---|---|---|
| OBSERVED | 0.0 / −3e-6 / +3e-6 | 2,676 |
| OCCLUDED | +0.2508 / 0.0 / +738.72 | 245,102 |
| UNRESOLVED | **−0.0283 / −36.54 / 0.0** | 135,544 |

**핵심 결과**: A의 UNRESOLVED 135,544쌍은 signed delta가 **예외 없이 ≤ 0**이다(최댓값이 정확히 0.0). 즉 A가 판정하지 못하는 질의는 전부 **surface event보다 카메라 쪽**, 곧 표면 앞의 노출된 자유 공간이다. 산발적 association 실패가 아니라 **정확히 한 종류의 기하 영역**이다.

그리고 그 **135,544쌍 전부에서 B는 OBSERVED**다(OCCLUDED 0건, UNRESOLVED 0건). B의 signed median margin은 중앙값 −0.0283. 즉 A가 침묵하는 곳은 전부 B의 frontier 카메라 쪽이다.

disagreement 클래스: A UNRESOLVED & B OBSERVED 135,544 / A UNRESOLVED & D OBSERVED 135,544 / A OCCLUDED & B OCCLUDED 245,102 / A OCCLUDED & D OBSERVED 147,426.

branch 구성(메타데이터): OBSERVED는 rho3d 1,320 + rho2d 1,356, OCCLUDED는 rho3d 194,335 + rho2d 50,767, UNRESOLVED는 rho3d 97,918 + rho2d 37,626 -- **branch가 A의 실패를 설명하지 않는다**. association radius는 어디에도 추가하지 않았고, event 앞 자유 공간을 OBSERVED로 바꾸지 않았다.

---

## 9. Candidate B Value Results

결정 무변경. primitive는 **"canonical pre-update T > 0.5 규칙 아래 renderer가 정의한 median-surface event"** 이며, 이 문서 어디에서도 physical first hit이라고 부르지 않는다.

signed_median_margin = query_depth − median_depth:

| state | 중앙값 | min | p95 | max | pairs |
|---|---|---|---|---|---|
| OBSERVED | −0.0271 | −36.536 | −0.00076 | **0.0** | 137,708 |
| OCCLUDED | +0.2485 | **+2.38e-07** | +4.403 | +738.72 | 245,614 |
| UNRESOLVED | -- | -- | -- | -- | 0 |

부호가 정확히 상태를 가른다(경계에 걸친 값이 부호 0에서 OBSERVED로 감). UNRESOLVED가 한 건도 없다.

**R1 source anchor float32 경계 분석**(WL120에서 보존, 수리하지 않음):

| | 값 |
|---|---|
| anchors | 2,640 |
| delta == 0 (정확히) | **1,653** |
| delta > 0 -> B가 OCCLUDED | **507** |
| delta < 0 | 480 |
| \|delta\| max | **1.907e-06** |
| relative delta max | **2.15e-07** |

즉 B의 507건 "source-view 모순"은 전적으로 float32 반올림이며, B의 분할면이 **두께 0**이라 그 위의 점이 양쪽으로 갈리는 것이다. **이 배치는 B에 tolerance를 넣지 않았다.**

---

## 10. Candidate C Corrected Blocker Results

결정 무변경. primitive는 **"canonical alpha cutoff에서 유도된 rho3d 기하 footprint"** 이며 **렌더러의 완전한 기여 support가 아니다** -- canonical acceptance는 `rho = min(rho3d, rho2d)`이고 rho2d low-pass 이벤트는 rho3d footprint 바깥에서도 accept될 수 있으므로, C의 기하는 렌더러가 실제로 합성하는 것의 **엄격한 부분집합**이다.

전체 OCCLUDED 쌍(343,616)에 대한 **교정된** 분포:

| 양 | p05 | 중앙값 | p95 | max |
|---|---|---|---|---|
| **camera**_nearest_blocker_t (MIN t) | 0.4729 | **0.9692** | 0.9971 | 1.0000 |
| **query**_nearest_blocker_t (MAX t) | 0.8990 | **0.9993** | 1.0000 | 1.0000 |
| camera_nearest_blocker_world_gap | 0.0146 | **0.2010** | 4.196 | 739.46 |
| query_nearest_blocker_world_gap | 0.00018 | **0.00403** | 0.6558 | 738.05 |
| blocker_region_thickness | 0.0089 | **0.1651** | 3.946 | 19.65 |
| blocker_count | 3 | **19** | 103 | 249 |
| max_blocker_opacity | 0.333 | **0.99997** | 1.0 | 1.0 |
| opacity of **camera**-nearest blocker | 0.065 | **0.2363** | 1.0 | 1.0 |
| opacity of **query**-nearest blocker | 0.132 | **0.5885** | 1.0 | 1.0 |

**WL120이 보고한 `nearest_blocker_t` 중앙값 0.99935는 위 표의 query-nearest 열이다.** camera-nearest는 0.9692로, blocker 영역이 WL120의 단일 숫자가 시사한 것보다 훨씬 두껍다(world 두께 중앙값 0.165).

**source view에서 OCCLUDED로 판정된 R1 anchor(2,605 / 2,640 = 98.67%)**:

| 양 | p25 | 중앙값 | p75 | p95 |
|---|---|---|---|---|
| camera_nearest_blocker_world_gap | 0.0417 | **0.0922** | 0.2172 | 1.162 |
| query_nearest_blocker_world_gap | 0.00107 | **0.00353** | 0.0105 | 0.0527 |
| blocker_region_thickness | 0.0351 | **0.0834** | 0.1965 | 1.065 |
| blocker_count | 6 | **9** | 13 | 22 |
| **same_component_blocker_count** | 4 | **8** | 12 | 20 |
| **same-component 비율** | 0.80 | **0.95** | 1.00 | 1.00 |

**이것이 이번 배치의 C 관련 핵심 교정 증거다**: anchor를 가리는 blocker의 **중앙값 95%가 anchor 자신과 동일한 frozen visible component에 속한다**(camera-nearest blocker 자체도 2,605건 중 1,763건, 67.7%가 동일 component). 즉 C의 자기 차폐는 "우연히 앞에 낀 다른 표면"이 아니라 **같은 표면을 이루는 겹친 footprint**가 지배한다는 것이 topology 기반으로 확정되었다. 그 blocker 영역의 실제 world 두께는 중앙값 **0.083** 단위이며, 이는 WL120이 측정한 surfel canonical support 반경 중앙값(0.100)과 같은 자릿수 -- **대략 footprint 하나 두께**다.

동일 component blocker는 결정에서 **무시하지 않았고**, self-occlusion tolerance도 도입하지 않았다. 이 귀속은 진단 전용이다.

---

## 11. Candidate D Traversal-Value Results

결정 무변경. primitive는 **"canonical traversal-order reachability"** 이며, 추가 진단이 물리 depth prefix 동등성을 독립적으로 확립하기 전에는 그렇게 해석하지 않는다.

**resolution reason**(relevant 383,322쌍, UNRESOLVED 0건):

| reason | pairs | T_pre 중앙값 | T_pre min |
|---|---|---|---|
| REACHED_ACCEPTED_EVENT | **273,964** | 0.9033 | 1.000e-4 |
| TERMINATED_BEFORE_QUERY | **97,676** | 1.428e-4 | 1.000e-4 |
| CONTRIBUTOR_LIST_EXHAUSTED | **11,682** | 8.671e-3 | 1.003e-4 |
| UNRESOLVED | 0 | -- | -- |

**termination 이벤트 97,676건의 교정된 계약 검증**:

| 양 | min | 중앙값 | p95 | max |
|---|---|---|---|---|
| termination_alpha | 0.00393 | **0.6029** | 0.9758 | 0.9900 |
| termination_T_pre | **1.000e-4** | 1.428e-4 | 1.006e-3 | 9.993e-3 |
| **termination_test_T = T_pre·(1−alpha)** | 1.004e-6 | **7.115e-5** | 9.875e-5 | **1.000e-4** |

- `test_T < 1e-4` 계약 **위반 0건**(전수 검증).
- **`T_pre` 자신이 1e-4 미만인 termination 이벤트: 0건 / 97,676건 (0.0%).** `T_pre`의 최솟값은 정확히 1.000e-4다.

즉 WL120이 "OCCLUDED 쌍의 T 중앙값 1.43e-4로 canonical termination 상수 1e-4를 경계로 깨끗이 갈린다"고 쓴 문구는 **부정확했다**: `T_pre`는 1e-4 **아래**가 아니라 **위**에 있고(그것이 canonical traversal의 불변식이다), 1e-4와 비교되는 양은 `test_T`다. 이번 배치는 그 양을 커널에서 직접 노출해 전수 확인했다.

또 하나 주목할 구조: D의 OBSERVED 버킷은 서로 다른 두 가지를 섞고 있다 -- **여전히 기여 가능**(REACHED, 273,964)과 **애초에 더 볼 geometry가 없음**(EXHAUSTED, 11,682). 후자는 transmittance와 무관한 사유다.

---

## 12. Candidate D Depth-Order Fidelity Audit

**전수 회계**(deterministic subset 불필요 -- 진단이 커널에서 전 픽셀·전 슬롯에 대해 산출된다). canonical renderer는 재정렬하지 않았고 D도 바꾸지 않았다.

| 양 | 값 |
|---|---|
| relevant pairs | 383,322 |
| **inversion이 1건 이상인 픽셀 위의 pair** | **383,314 (99.998%)** |
| pixel_inversion_count 중앙값 / p95 / max | **31** / 75 / 193 |
| pixel_max_backward_jump 중앙값 / p95 / max | **0.1942** / 1.464 / 11,353.2 |
| **late_front_event_count** (REACHED로 해소된 273,964쌍) | 중앙값 **6**, 평균 8.21, p95 26, max 147 |
| **late-front가 1건 이상인 비율** | **222,435 / 273,964 = 81.19%** |

해석: canonical tile list는 surfel **중심**의 camera-space z로 정렬되므로, accepted 이벤트의 per-pixel 교차 depth는 traversal을 따라 단조가 아니다. 실측상 이것은 예외가 아니라 **거의 보편적**이다(픽셀의 99.998%). 그리고 D가 "query depth에 도달했다"고 해소한 뒤에도, **그 해소의 81.2%에서** 아직 물리적으로 query 앞에 있는 accepted 이벤트가 평균 8건 더 처리된다.

따라서 D가 기록하는 `T_pre`는 **traversal-order prefix transmittance**이지 physical-depth prefix transmittance가 아니다. 이 구분은 D의 두 측에 다르게 작용한다:
- **OCCLUDED 측(TERMINATED)은 견고하다.** termination은 누적된 prefix에 대한 사실이고, 순서가 무엇이든 이미 T·(1−alpha) < 1e-4에 도달했다는 뜻이며, late-front 이벤트는 termination 이후 존재하지 않는다(루프 종료).
- **OBSERVED-by-reaching 측이 영향을 받는다.** 물리 순서였다면 prefix에 이벤트가 더 들어가므로 T는 **감소만** 하고, 따라서 OCCLUDED는 **늘어날 수만** 있다. **그 증가량은 이 배치가 산출하지 않았다**(section 5의 INABILITY 참조) -- 추정치로 대체하지 않는다.

**S-D1 통제 fixture**가 이 메커니즘을 단독으로 재현한다: 중심 depth 4.00의 거의 edge-on surfel(per-pixel 교차 depth 4.1337)과 중심·교차 모두 4.05인 정면 surfel. tile 정렬은 전자를 먼저 합성한다. depth 4.10 probe는 `accepted_prefix_count = 0`, `T_pre = 1.0`, `resolution_event_depth = 4.1337`로 해소되지만 `late_front_count = 1` -- 물리적으로 query 앞(4.05)의 accepted 이벤트가 해소 뒤에 처리된다. `pixel_inversion_count = 1`, `max_backward_jump = 0.0837`.

---

## 13. B-vs-D Value-Space Comparison

**state 관계 정확 재현**: global B=OCCLUDED & D=OBSERVED **654건**, 역방향 **0건** (WL120과 동일).
per-view 쌍에서도 B=OCCLUDED & D=OBSERVED **147,938건**, 역방향 **0건** -- 383,322 relevant 쌍 전체에서 **예외 없는 포함 관계**다.

654개 disagreement query(45,692 relevant 쌍)의 값:

| 양 | min | 중앙값 | p95 | max |
|---|---|---|---|---|
| query_depth | 2.114 | 6.003 | 13.881 | 742.97 |
| B_signed_median_margin | 1e-06 | **0.3155** | 4.361 | 738.72 |
| D_traversal_T_pre | 1.000e-4 | **0.00296** | 0.3804 | 1.0 |
| D_accepted_prefix_count | 0 | 32 | 74 | 157 |
| D_resolution_event_depth | 1.278 | 4.650 | 13.497 | 625.00 |
| D_late_front_event_count (REACHED만) | 0 | 5 | 30 | 121 |
| pixel_inversion_count | 2 | 29 | 69 | 142 |

**D resolution reason 분해가 결정적이다**: 45,692쌍 중 REACHED 18,007 / **TERMINATED 16,083** / EXHAUSTED 11,602. 즉 **D도 이 쌍들의 35.2%에서는 per-view로 OCCLUDED라고 말한다.** 두 후보를 갈라놓는 것은 per-view 의미론만이 아니라 **frozen global aggregation**("어느 한 뷰라도 OBSERVED면 OBSERVED")이며, 161뷰에서는 그 조건이 D의 OCCLUDED를 전부 흡수한다.

query kind 분해: R3 behind-surface probe **620** / R5 region gap 20 / R6 8 / R1 anchor 6(=B의 float32 건). 영역: patio 315 / table_side_curved 101 / hedge 96 / table_legs 77 / table_top 57.

**해석 가드(지시 준수)**: 위 값들은 B와 D를 **서로 다른 두 렌더러 수준 질문**으로 보고한다 -- B는 renderer-defined visible-surface frontier, D는 canonical traversal termination/reachability. 둘 사이의 threshold를 탐색하지 않았고, T=0.5와 test_T=1e-4 사이에 진짜 경계가 있다고 **주장하지 않는다**. 이번 배치가 산출한 것은 (a) 포함 관계가 예외 없이 성립한다는 사실, (b) D의 per-view OCCLUDED가 실재하며 global 규칙에서 소멸한다는 사실, (c) D의 OBSERVED-by-reaching 측이 traversal order에 영향받는다는 사실이며, 이 셋 중 어느 것도 "사이의 경계"를 확립하지 않는다.

---

## 14. A-vs-B / C-vs-B Comparisons

**A vs B**: A UNRESOLVED 135,544쌍 -> B는 **전부 OBSERVED**(OCCLUDED 0, UNRESOLVED 0). 그 지점들의 B signed margin은 중앙값 −0.0283, 최댓값 정확히 0.0. A가 표현하지 못하는 영역은 곧 **B frontier의 카메라 쪽 전체**다.

**C vs B**(B=OBSERVED이면서 C=OCCLUDED인 98,274쌍):

| 양 | 중앙값 | p95 | max |
|---|---|---|---|
| B_signed_median_margin | −0.0124 | −0.00045 | 0.0 |
| C camera_nearest_blocker_t | 0.9891 | 0.9985 | 1.0000 |
| C query_nearest_blocker_t | 0.9995 | 1.0000 | 1.0000 |
| C camera_nearest_blocker_world_gap | 0.0618 | 0.6055 | 15.96 |
| C query_nearest_blocker_world_gap | 0.00307 | 0.0679 | 15.96 |
| C blocker_region_thickness | 0.0512 | 0.4756 | 15.51 |
| C blocker_count | 7 | 20 | 135 |
| **C same_component_blocker_count** | **6** | 18 | 97 |

즉 B가 "frontier 앞(관측됨)"이라고 부르는 자유 공간에서도 C는 blocker를 7개(중앙값) 찾고 그 중 6개가 **같은 component**다. 두 비교를 하나의 정확도 점수로 합치지 않는다.

---

## 15. Synthetic Supplement

WL120의 S1-S7은 그대로 재실행했고 **구현 충실도 실패 0건**(9개 계약 x 4후보). 추가 3개:

**S-D1 (accepted-event depth inversion)** -- section 12에 수치 인용. 통제 조건에서 D probe의 해소 지점이 엄격한 물리 depth 순서와 **다름**을 단독 시연했다.

**S-C1 (하나의 표면을 이루는 겹친 footprint)** -- 8겹 opacity-0.15, spacing 0.01. renderer 자신의 median event(5번째 층, depth 4.04)를 query로 두면:
`blocker_count = 4`, `camera_nearest_blocker_t = 0.9901` / world gap **0.0400**(= 4 x 0.01 ✓), `query_nearest_blocker_t = 0.99752` / world gap **0.0100**(= 1 x 0.01 ✓), `blocker_region_thickness = 0.0300` ✓. camera-nearest / query-nearest 분리가 **정확히 fixture 기하와 일치**하며, WL120의 단일 숫자가 query-nearest였음을 통제 조건에서 확인한다.

**S-B1 (median event 왕복)** -- G2로 3D 재구성 후 같은 카메라로 재투영. pixel-centre와 half-pixel-offset 두 경우 모두 `absolute_delta = 0.0`, 재투영 픽셀도 원래 픽셀과 동일. 즉 **왕복 자체는 정확하며**, 실 scene의 4.8e-7급 불일치(section 9)는 왕복의 구조적 결함이 아니라 실제 scene의 크기·기울기·부동소수 조건에서만 나타난다. **tolerance는 어디에도 추가하지 않았다.**

---

## 16. Region-Level and Query-Kind Quantitative Results

**보충 true-fragmentation bank, query kind별 global 상태**:

| kind | n | A (OBS/OCC/UNR) | B | C | D |
|---|---|---|---|---|---|
| ENDPOINT_A | 300 | **300 / 0 / 0** | 290 / 10 / 0 | 12 / 288 / 0 | 300 / 0 / 0 |
| ENDPOINT_B | 300 | **300 / 0 / 0** | 296 / 4 / 0 | 15 / 285 / 0 | 300 / 0 / 0 |
| **MIDPOINT** | 300 | **0 / 0 / 300** | **300 / 0 / 0** | 34 / 266 / 0 | **300 / 0 / 0** |
| verified control | 8 | 0 / 0 / **8** | 0 / 0 / **8** | 0 / 0 / **8** | 0 / 0 / **8** |

**실제 fragmentation gap의 중점에서 A는 300/300 전부 UNRESOLVED, B는 300/300 OBSERVED, C는 266/300 OCCLUDED, D는 300/300 OBSERVED.** WL120의 R5 근사가 아니라 실제 component 분리를 가로지르는 관측 컨텍스트에서 **WL120의 순위가 그대로 재현**된다.

gating 사유별 중점 상태: 3D-locality 거부 288개(A 288 UNRES / B 288 OBS / C 254 OCC / D 288 OBS), 기하 게이트 거부 12개(C 12/12 OCC).

**영역별**(endpoint+midpoint 통합):

| 영역 | n | A UNRES | C OCC | B margin 중앙값 | C thickness 중앙값 | D T_pre 중앙값 |
|---|---|---|---|---|---|---|
| table_top | 184 | 60 | 174 | 0.0188 | 0.0893 | 0.787 |
| table_side_curved | 182 | 60 | 166 | 0.0343 | 0.1758 | 0.545 |
| table_legs | 172 | 60 | 159 | 0.0458 | 0.1767 | 0.287 |
| patio | 181 | 60 | 171 | 0.0436 | 0.3377 | 0.478 |
| hedge/background | 181 | 60 | 169 | 0.0236 | 0.5068 | 0.583 |

네 후보의 상대 패턴이 영역 간에 균일하다 -- 실패 양상은 영역 특이적이지 않고 가설 수준의 성질이다. C의 blocker 두께만 hedge(0.507) > patio(0.338) > table_legs/curved(0.177) > table_top(0.089) 순으로 뚜렷이 층화되며, 이는 표면이 얇고 정연할수록 blocker 수프가 얇다는 해석과 일치한다.

원본 bank의 B margin/영역/kind별 분해, A/C/D의 상태별 값 분포는 리포트 JSON(`candidate_A_values`, `candidate_B_values`, `candidate_C_values`, `candidate_D_values`)에 전부 있으며, 각 항목은 min/median/mean/p95/max에 더해 p01/p05/p25/p75/p99 quantile 표를 포함한다. 비유한 값은 `non_finite_excluded`로 명시되고 분모에서 숨기지 않는다.

---

## 17. Qualitative Review Exports

`output/121_osn_gs_observed_occluded_value_space/` 아래 6개 view(장면 전체를 near-black으로 깔고 보충 gap query를 색으로 표시), PNG는 공유 `preview_png/`에 모았다.

- `ORIGINAL_2DGS_SCENE` -- 표준 export.
- `TOPOLOGY_GAP_BY_GATING_REASON` -- 주황=3D-locality 거부, 파랑=기하 게이트 거부, 자홍=positive edge인데 분리, 흰=검증된 control.
- `TOPOLOGY_GAP_CANDIDATE_A` ~ `_D` -- 녹색=OBSERVED / 적색=OCCLUDED / 회색=UNRESOLVED, 동일 query·동일 카메라(`DSC07957.JPG`).

A의 export에서 endpoint(녹색)와 midpoint(회색)가 육안으로 분리되어 보이며, 이는 section 16의 300/300 수치와 일치한다.

**정성 사례 30건**을 `annotated_topology_gap_cases`에 사이드카로 기록했다. 각 사례는 컨텍스트 인덱스, source camera(인덱스+이름), 양쪽 pixel 좌표, representative id, **component id**, gating 사유, 영역, 그리고 endpoint A/B/midpoint 세 query 각각에 대해 query id, world position, 네 후보의 global 상태, source camera 안에서의 상태와 **B median frontier 위치(median depth·signed margin), C camera-nearest / query-nearest blocker(t·world gap)·blocker count·same-component count, D resolution reason·T_pre·resolution event depth·termination test_T·late-front count, A hit distance**를 담는다. **모든 시각 사례는 query id와 정량 레코드로 되짚을 수 있으며, 외형만으로는 어떤 주장도 하지 않았다.**

---

## 18. Updated Candidate Verdicts

### Candidate A -- SURFACE-HIT
- INTENT ALIGNMENT: **PASS**
- IMPLEMENTATION FIDELITY: **PASS** (state 비트 동일, OBSERVED signed delta가 ±3e-6 float 동일성 범위)
- VALUE-EVIDENCE QUALITY: **STRONG** -- 교정된 값이 WL120보다 **더 강한** 증거를 준다
- ARCHITECTURE RESULT: **NOT VIABLE**

> **"교정된 값 증거가 A가 근본적으로 부피 증거-기아임을 확인하는가?"** -- **그렇다, 그리고 더 정확하게.** A의 UNRESOLVED 135,544쌍은 signed depth delta가 **예외 없이 ≤ 0**(최댓값 정확히 0.0)이고, 그 **전부가 B에서 OBSERVED**다. 즉 A의 침묵 영역은 산발적 association 실패가 아니라 **정확히 "표면 앞의 관측된 자유 공간" 전체**다. 실제 fragmentation gap 중점에서는 **300/300 전부 UNRESOLVED**. branch(rho3d/rho2d)는 이 실패를 전혀 설명하지 않는다. association radius를 추가하지 않고 확인한 결과다.

### Candidate B -- MEDIAN-DEPTH
- INTENT ALIGNMENT: **PASS**
- IMPLEMENTATION FIDELITY: **PASS**
- VALUE-EVIDENCE QUALITY: **STRONG**
- ARCHITECTURE RESULT: **ADVANCE TO DEDICATED FRONTIER VALIDATION** (단, 아래 조건부)

> **"median frontier가 실제 visible-fragmentation query에서 수치적으로 일관되고 유용한가?"** -- **그렇다, 수치적 일관성은 확인된다.** signed margin의 부호가 상태를 정확히 가르고, UNRESOLVED가 원본 bank 383,322쌍 중 **0건**이며, 실제 fragmentation 중점 300/300에서 판정을 낸다. 유일한 수치 결함인 R1 anchor 경계는 **|delta| ≤ 1.9e-6, 상대 ≤ 2.15e-7의 순수 float32 반올림**임이 전수 확인되었다(1,653건은 정확히 0). 이는 tolerance로 고칠 문제가 아니라 **두께 0 분할면의 구조적 성질**이며 이번 배치는 고치지 않았다.
>
> **자동 승격은 하지 않는다.** advance는 "median depth가 physical first hit이다"를 주장하지 않으며, WL120의 S6(첫 contributor 4.00 / median crossing 4.05 / termination 5.25가 서로 다른 깊이)는 여전히 유효하다. 승격의 의미는 **"renderer-defined visible-surface frontier"라는 이름 그대로의 대상에 대해 전용 검증 배치를 할 근거가 값 수준에서 충분해졌다**는 것뿐이다. 그 검증이 답해야 할 것은 여전히 "그 frontier가 광도 감독 도달 경계로 쓸 수 있는가"이며, 이번 배치는 그 질문에 답하지 않았다.

### Candidate C -- GEOMETRIC VISIBILITY
- INTENT ALIGNMENT: **PASS**
- IMPLEMENTATION FIDELITY: **PASS with CAVEAT** (primitive 명명 교정: rho3d footprint는 렌더러의 완전한 기여 support가 아니다)
- VALUE-EVIDENCE QUALITY: **STRONG**
- ARCHITECTURE RESULT: **NOT VIABLE**

> **"교정된 blocker provenance가 경성 기하 가시성이 치명적으로 자기 차폐한다는 결론을 보존하는가?"** -- **보존할 뿐 아니라 원인을 topology로 확정한다.** source view에서 OCCLUDED로 판정된 R1 anchor는 2,605/2,640(98.67%)로 WL120과 동일하고, 교정된 값은 그 blocker들의 **중앙값 95%가 anchor 자신과 같은 frozen visible component**임을 보인다(camera-nearest blocker 자체도 67.7%가 동일 component). blocker 영역의 실제 world 두께는 중앙값 **0.083** 단위로, canonical support 반경 중앙값(0.100)과 같은 자릿수 -- **대략 footprint 하나 두께의 같은-표면 수프**다.
>
> 교정이 바꾼 것: WL120이 인용한 `nearest_blocker_t` 0.99935는 **query-nearest**였고, camera-nearest는 0.9692다. 따라서 "차단자가 질의 바로 앞에 붙어 있다"는 WL120의 서술은 **한쪽 끝만 본 것**이며, 실제로는 카메라 쪽으로 상당히 두꺼운 구간에 걸쳐 있다. 이 교정은 결론을 뒤집지 않고 **강화**한다.

### Candidate D -- RENDERER REACHABILITY
- INTENT ALIGNMENT: **PASS**
- IMPLEMENTATION FIDELITY: **CAVEAT** -- probe는 canonical termination을 정확히 관측하지만(계약 위반 0/97,676), 그 해소 지점은 **traversal order**를 따르며 물리 depth 순서가 아니다
- VALUE-EVIDENCE QUALITY: **LIMITED** (OCCLUDED 측은 STRONG, OBSERVED-by-reaching 측은 LIMITED)
- ARCHITECTURE RESULT: **NOT VIABLE AS STATED** (WL120과 동일)

> **"D의 행동 중 얼마가 진짜 canonical termination 의미론이고, 얼마가 traversal-order / physical-depth 불일치의 영향인가?"** -- 두 측이 **다르다**.
> - **OCCLUDED 측(97,676쌍, TERMINATED)은 진짜 canonical termination 의미론이다.** `test_T < 1e-4` 계약 위반 0건, termination 이후 이벤트가 존재하지 않으므로 순서 불일치의 영향을 받지 않는다. 다만 WL120의 문구는 교정되어야 한다: **`T_pre < 1e-4`인 termination 이벤트는 0건**이며(`T_pre` 최솟값이 정확히 1.000e-4), 1e-4와 비교되는 양은 `test_T = T_pre·(1−alpha)`다(중앙값 7.11e-5, alpha 중앙값 0.603).
> - **OBSERVED-by-reaching 측(273,964쌍)은 material하게 영향받는다.** relevant pair의 **99.998%**가 accepted-event depth inversion이 있는 픽셀 위에 있고(중앙값 31회, 최대 backward jump 중앙값 0.194), REACHED 해소의 **81.2%**에서 해소 후에도 물리적으로 query 앞인 accepted 이벤트가 평균 8건 더 처리된다.
> - D의 OBSERVED 버킷은 또한 **두 가지 다른 사유를 섞는다**: 여전히 기여 가능(REACHED, 273,964)과 더 볼 geometry 없음(EXHAUSTED, 11,682).
> - global OCCLUDED가 여전히 **0**이라는 WL120의 결과는 변하지 않는다. 다만 B-vs-D disagreement 쌍의 **35.2%에서 D는 per-view로 OCCLUDED라고 말한다** -- 소멸시키는 것은 per-view 의미론이 아니라 frozen global aggregation이다.
> - 순서 교정이 D의 OCCLUDED를 **얼마나** 늘릴지는 이 배치가 산출하지 않았다(section 5의 INABILITY). 방향만 확정된다: 늘어날 수만 있고 줄 수 없다.

### B vs D -- 하나의 threshold 축인가, 두 개의 다른 질문인가?
**이 배치의 증거는 "두 개의 다른 렌더러 수준 질문" 쪽을 지지한다.**
- 포함 관계는 383,322 relevant 쌍 전체에서 **예외 0건**으로 성립한다(B_OBS ⊂ D_OBS). 이것만 보면 하나의 축처럼 보인다.
- 그러나 disagreement 쌍의 **35.2%에서 D는 per-view OCCLUDED**이고, 이는 B의 per-view OCCLUDED와 **다른 사유**(termination vs frontier 통과)로 발생한다. 두 후보가 같은 축 위의 두 눈금이라면 per-view 수준에서도 한쪽이 다른 쪽을 포함해야 하지만, D의 per-view OCCLUDED 97,676건 중 상당수는 B가 이미 OCCLUDED라고 부른 지점의 **훨씬 뒤**에 있다(B margin 중앙값 0.3155 vs D resolution event depth 중앙값 4.65 대 query depth 6.00).
- 결정적으로 D의 OBSERVED 측은 `EXHAUSTED`(geometry 없음)와 `REACHED`(기여 가능)를 섞고 있고, 이는 B의 frontier 개념에 대응물이 **없다**.

**따라서 "진짜 경계가 둘 사이에 있다"고 진술하지 않는다.** 이번 배치는 그 주장에 대한 직접 증거를 새 threshold 없이 산출하지 못했고, 산출을 시도하지도 않았다.

---

## 19. Remaining Architecture Question

**"B의 renderer-defined median-surface frontier는 'canonical pre-update T > 0.5' 사건이 정의하는 면이다. 이 면이 광도 감독 도달 경계로서 물리적 의미를 갖는가 -- 즉 이 면 뒤의 점이 실제로 학습 신호를 받지 못하는가 -- 아니면 이 면은 단지 투과도의 절반이 소진된 통계적 지점이며 감독 도달 여부와는 별개의 양인가?"**

이번 배치가 이 질문으로 좁힌 근거: (a) B는 커버리지·수치 일관성 두 축을 실제 fragmentation query에서도 통과한 **유일한** 후보다, (b) B의 유일한 수치 결함은 상대 2e-7급 float32 반올림이며 tolerance 문제가 아니다, (c) 그러나 B가 그리는 면이 물리적 first hit이 아니라는 WL120 S6 증거는 그대로 유효하고, 이번 배치는 그 면의 **물리적 의미**를 검증하지 않았다. A는 그 면의 카메라 쪽 전체를 표현하지 못하고(구조적), C는 같은-표면 footprint 수프로 자기 차폐하며(topology로 확정), D는 순서 의존성과 global 소멸을 갖는다.

**이 배치는 여기서 멈춘다.** 새 threshold, hybrid classifier, visible-topology 수정, Occluded Surface 구축, NURBS continuation 중 어느 것으로도 진행하지 않았다.

---

## 20. Exact Branch / Commit / Commands / Output Paths

**브랜치**: `arch/2dgs-coverage-first-surface`
**역사적 기준 커밋(보존)**: `fdfb8ad60b6233ea8364a09ea3467c18e600a246`
**이 배치의 부모 커밋**: `fdfb8ad` (작업 시작 시 working tree 청결)
**이 배치의 커밋 SHA**: `fafade1` -- WL122와 함께 하나의 커밋에 담겼다(사용자 지시 "작업 할 때마다 커밋" 이전에 두 배치가 누적된 결과이며, 이후 배치부터는 배치당 1커밋으로 진행한다). 동시 진행 중이던 다른 에이전트의 worklog 119-4 변경분도 같은 커밋에 포함했다.

**추가한 파일**:
```
scripts/devtools/observed_occluded/value_diagnostics.py
scripts/devtools/observed_occluded/topology_gap_bank.py
scripts/devtools/observed_occluded/synthetic_value_contracts.py
scripts/devtools/observed_occluded_value_space_comparison.py
tests/test_observed_occluded_value_space_comparison.py
docs/worklogs/121_observed_occluded_value_space_comparison_and_fidelity_supplement.md
```

**수정한 파일(2개, 둘 다 순수 additive)**:
```
osn_gs/render/vendor/diff_surfel_rasterization_qdepth/   (cuda_rasterizer/{config.h 제외} forward.h/forward.cu/rasterizer.h/rasterizer_impl.cu, rasterize_points.h/.cu)
osn_gs/render/torch_surfel_query_depth_diagnostics.py    (새 5개 필드 언팩/반환 + 정확한 semantics 문서화)
```
**A/B/C/D 결정 함수와 `shared.aggregate_global`은 수정하지 않았다.**

**진단 CUDA 재빌드**(신규 필드 추가 후 stale object 링크 오류를 피하려면 build 디렉터리를 비우고 재빌드해야 한다):
```
rmdir /s /q %TEMP%\osn_gs_diff_surfel_rasterization_qdepth
scripts\build_surfel_extension_qdepth.bat 12.0
```

**테스트**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_value_space_comparison.py -q
  -> 38 passed
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_volumetric_audit.py tests\test_observed_occluded_value_space_comparison.py tests\test_surfel_representative_diagnostics.py tests\test_surfel_rasterization_cuda.py -q
  -> 129 passed
```
변경분이 전부 diagnostic/devtools 코드와 진단 sibling 렌더러 안에 머무르므로 focused regression으로 충분하다(section 19). production/shared 아키텍처 경로는 변경하지 않았다.

**실 scene 재생**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe scripts\devtools\observed_occluded_value_space_comparison.py ^
  --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt ^
  --out output\121_osn_gs_observed_occluded_value_space ^
  --device cuda --source-path DATASET --images images_8
```
전체 실행 시간 **430.0초**. `--allow-replay-failure`는 smoke test 전용이며 **이 보고 실행에서는 사용하지 않았다** -- baseline gate가 자력으로 PASS했다.

**출력 경로**:
```
output/121_osn_gs_observed_occluded_value_space/observed_occluded_value_space_comparison_report.json
output/121_osn_gs_observed_occluded_value_space/value_space_original_bank.npz         (4712x161 state + 전체 값 배열)
output/121_osn_gs_observed_occluded_value_space/value_space_supplemental_bank.npz     (908 query + 컨텍스트 provenance)
output/121_osn_gs_observed_occluded_value_space/topology_gap_contexts.json
output/121_osn_gs_observed_occluded_value_space/<VIEW_NAME>/iteration_0000001/point_cloud.ply
output/121_osn_gs_observed_occluded_value_space/<VIEW_NAME>/render.ppm
output/121_osn_gs_observed_occluded_value_space/preview_png/<VIEW_NAME>.png
output/confirmed/_run_logs/121_observed_occluded_value_space_run.log
```
WL120의 export는 규약대로 `output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/`로 이동했으며, 이 배치의 baseline gate가 그 npz를 참조 아티팩트로 읽는다.
