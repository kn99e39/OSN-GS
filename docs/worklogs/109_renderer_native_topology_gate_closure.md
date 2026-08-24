# Worklog 109 — Renderer-Native Surface Representative Graph: Gate Closure (CAVEAT 1/2)

## 상태

**완료 — 실측 있음. GATE PASS(단, 명시적 caveat 동반).** Worklog 108의 두 caveat를 이번 배치에서 직접 실측으로 닫았다: (1) 36,051 불일치를 부동소수점 추측이 아니라 같은 forward 실행 안에서 직접 확인했고, (2) "패티오/hedge 경계" 중첩(WL108: 20.4%/26.2%)이 취약한 단일 다리(bridge)가 아니라 다중 뷰로 뒷받침되는 견고한 다중 경로 연결임을 실측으로 확인했다. Worklog 107의 adjacency 알고리즘은 이번에도 한 줄도 수정하지 않았다.

## CAVEAT 1 — 같은 forward 실행에서의 회계 해소

WL107의 진단 CUDA 빌드(`diff_surfel_rasterization_diag`, canonical과 별개, canonical은 이번에도 무수정)에 새 per-primitive 출력 `out_forward_accepted`를 추가했다. 위치는 `forward.cu`의 `float w = alpha * T;` 바로 앞 — 즉 원래 커널이 이미 수행하던 5가지 forward acceptance 체크(depth>=near, power<=0, alpha>=1/255, test_T>=0.0001)를 통과한 바로 그 지점이며, WL107의 `median_surfel_id`(representative) 캡처와 **동일한 forward 실행, 동일한 루프 반복**에서 기록된다. 새 threshold 없음, 새 CUDA 실행 경로 없음.

실측 cross-tab(전 씬, 161개 학습 카메라, 동일 체크포인트 iteration 30000):

| | FORWARD_ACCEPTED+ | FORWARD_ACCEPTED- |
|---|---|---|
| **REPRESENTATIVE+** | 785,937 | **0** |
| **REPRESENTATIVE-** | 395,676 | 8,856 |

`representative_and_not_forward_accepted = 0` — 전수 조사로 확인. MEDIAN_SURFACE_REPRESENTATIVE는 같은 forward 실행에서 FORWARD_ACCEPTED_CONTRIBUTOR 없이는 결코 발생하지 않는다(구조적으로 보장되고, 이제 실측으로도 확인됨).

이 결과로 WL108의 36,051(representative지만 WL105 backward 기준 non-contributing) 불일치를 직접 해소했다: 이 36,051개 전부가 forward_accepted=1이었다(100%, 부분 아님). 즉 두 신호는 같은 forward 실행 안에서는 완전히 일관되며, WL108의 discrepancy는 **WL105의 별도 backward-pass 진단(다른 CUDA 빌드, `T`를 나눗셈으로 역산)이 실제로 일어난 기여를 놓친 것**이지, WL107 자체의 forward 신호가 내적으로 모순되는 게 아니다 — 이번에는 부동소수점 추측이 아니라 같은 실행에서의 직접 비교로 확정했다.

Fixture 테스트에서도 흥미로운 확인 사항 하나: 완전 불투명이 아닌(opacity 0.99) 근접 surfel에 가려진 "occluded" surfel도 잔여 투과율(T≈0.02)이 1/255보다 크면 여전히 forward_accepted=1이 될 수 있다(`test_occluded_but_weakly_transmitted_surfel_can_still_be_forward_accepted`) — representative가 되지는 못해도(같은 fixture에서 WL107이 이미 확인) 약하게 알파-합성에는 기여한다. forward_accepted는 representative보다 명백히 더 약한 조건이며, 그 방향으로만 비대칭이 성립함을 fixture로도 실측 확인했다.

렌더링 불변성도 재검증했다(`test_diagnostic_rendering_still_matches_canonical_after_forward_accepted_addition`) — canonical 렌더와 `torch.testing.assert_close`로 완전 일치.

## CAVEAT 2 — 패티오/hedge 프론티어와 고영향 bridge

WL107의 그래프를 그대로 재실행(`build_candidate_graph` → `accumulate_image_space_pairs` → `filter_by_3d_locality` → `apply_secondary_geometric_gate`, 전부 무수정)해 재현성부터 확인했다: 최대 component 비율 36.77%, singleton 45.02% — WL107/WL108과 완전 일치. Representative-only backbone도 연결 83.3%로 WL108과 일치.

### Hedge-region 분해 (WORKING INTERPRETATION ONLY — 하드 임계값 없음, 순수 3D 최근접 anchor 거리)

| 항목 | 값 |
|---|---|
| hedge-region 표면 수 | 342,085 |
| 최대 component에 속한 hedge-region 수 | **89,502** (hedge 전체의 26.16%) |
| 높이(axis-1) 분포 | 중앙값 -3.43, p95 -0.14 (대체로 원점 아래 — 지면 근접 식생) |
| planarity-proxy(shape-operator norm) | 중앙값 8.37, 평균 12.43, 최대 543.9 (매끈한 표면 대비 훨씬 큰 곡률/불연속 — 식생 특유 구조) |
| representative-view-count | 중앙값 **1**, 평균 5.58, p95 28 (hedge 내 representative의 상당수가 단 1개 뷰에서만 representative — 약하고 산발적인 지위) |

### 패티오-hedge 그래프 프론티어 (실제 엣지 전수, 샘플링 없음)

최대 component 내부에서 hedge-region 멤버와 non-hedge-region 멤버를 직접 잇는 엣지 **1,110개** 전수 추출:

| 항목 | 값 |
|---|---|
| 거리 중앙값 | 0.0568 (씬 단위) |
| distance/local-spacing 중앙값 | 1.148 (기하 게이트 통과 범위) |
| residual 중앙값 | 0.1036 |
| normal-offset-ratio 중앙값 | 0.263 (게이트 임계값 1.0 대비 여유 있게 통과) |
| 뷰 지지 수 중앙값 | 5, 평균 8.29 |
| 단일 뷰 지지 비율 | 19.3% (즉 80.7%는 다중 뷰로 뒷받침) |

### 고영향 bridge 분석 (Tarjan bridge-finding + DFS 서브트리 크기로 정확한 split-impact 계산)

최대 component(437,751 노드, 1,034,050 엣지) 전체에서 **56,816개 bridge**를 발견했다 — 백본 자체가 상당히 취약한 트리형 구조다.

| impact bin (진단용, cut 기준 아님) | bridge 수 |
|---|---|
| ≥1% of component (≥4,378개 분리) | 10 |
| ≥1,000 | 17 |
| ≥100 | 136 |
| ≥10 | 1,432 |
| <10 | 55,221 |

**핵심 실측**: 전역 top-20 고영향 bridge(최대 46,701개/10.7% 분리) 중 **패티오-hedge frontier 엣지는 0개**다 — 전부 hedge-region 내부 체인이거나 non-hedge 내부(추정 patio/table) 체인이다. Frontier와 교차하는 bridge만 따로 뽑으면(전체 56,816개 중 벡터화 전수 검사) **67개(0.118%)**뿐이고, 그 중 최대 분리 크기는 **56개(컴포넌트의 0.013%)**다. 즉 1,110개 frontier 엣지 중 1,043개(93.9%)는 bridge조차 아닌 중복 경로다.

**결론**: WL108이 발견한 89,502개(26.2%) 패티오/hedge 중첩은 취약한 단일 관측(하나의 우연한 다리)에 의존하지 않는다. 다중 뷰로 뒷받침되는 두터운 다중 경로로 얽혀 있다 — 임의로 컷할 수 있는 "숨겨진 버그성 다리"가 아니라, 렌더러가 실제로 반복 관측한 구조적 연결이다. 반대로, 최대 component 자체의 진짜 취약점(56,816개 bridge, 상위 10개가 각각 1%+ 분리)은 hedge/patio 경계가 아니라 각 영역 **내부**에 있다.

고영향 bridge 표본(전역 top-20 + frontier top-20, 총 40개) 중 view-support 분포: 전역 top-20은 single-view 5/20, frontier top-20은 single-view 6/20 — 대다수(75%, 70%)가 다중 뷰 지지다. 즉 고영향 bridge가 단일 뷰 노이즈에 편중돼 있다는 가설은 기각된다 — 증거로만 보고, cut 규칙으로는 쓰지 않았다(directive 지시).

## Bridge 출처 추적 (예시)

frontier bridge 중 최대 영향(56개 분리)인 엣지는 좌표 `(-11.06, 0.91, 3.66)` / `(-11.09, 0.85, 3.78)` 사이이며, 18개 뷰에서 각각 독립적으로 픽셀-인접 representative 쌍으로 관측됐다(카메라별 픽셀 좌표는 JSON `high_impact_bridge_provenance`에 전수 기록). 이는 "우연히 한 뷰가 만든 노이즈"가 아니라 여러 각도에서 반복 관측된 실제 렌더러 표면 연속성이다.

## GATE 판정

**GATE: PASS** — Renderer-Native Surface Representative Graph(WL107, WL108 감사)를 canonical Visible Surface Topology Backbone으로 정식 채택한다.

근거:
1. CAVEAT 1이 같은 forward 실행 안에서 실측으로 완전히 닫혔다(representative → forward_accepted가 예외 없이 성립, 36,051 discrepancy 100% 설명).
2. WL107 재실행이 완전히 재현 가능하고 결정론적이다(모든 수치가 WL107/WL108과 일치).
3. 패티오/hedge 중첩은 알고리즘 버그나 숨겨진 취약한 단일 다리가 아니라, 다중 뷰로 뒷받침되는 실제 renderer-observed 연속성이다 — 렌더러 기반 가시 표면 위상(purely geometric/photometric visible-surface topology)이 원래 하도록 설계된 일 그대로다.
4. 최대 component의 진짜 구조적 취약점(56,816개 bridge)은 patio-hedge 경계가 아니라 각 영역 내부에 있다는 것도 실측으로 명확히 분리됐다.

**단, 명시적 caveat**: 이 topology backbone은 "관측된 연속 표면"을 포착할 뿐, "의미론적으로 하나의 물체"를 보장하지 않는다. 최대 component가 실제로는 패티오-인접 구조와 hedge 식생을 다중 뷰로 뒷받침되는 방식으로 함께 포함하고 있다는 사실은 이 그래프 구성 방식의 **정직한 한계**로 기록하고 넘어간다 — 이를 "patio"라고 부르지 않고 최대 component로만 지칭한 이번 배치의 표현이 옳았다. 의미론적(patio vs hedge) 분리는 이 순수 geometric 위상 구성 단계의 책임이 아니며, 이후 Trust/의미 분리 단계의 과제로 남긴다. Non-representative contributor는 이번에도 어떤 역할도 부여하지 않았다(unattached support evidence로만 유지).

## 시각/재현 자료

`scripts/devtools/renderer_native_topology_gate_closure.py`, 실행 로그 `output/renderer_native_topology_gate_closure_run2.log`, 전체 JSON `output/osn_gs_renderer_native_topology_gate_closure/renderer_native_topology_gate_closure_report.json`. 요구된 10개 리뷰 export(ORIGINAL_2DGS_SCENE, SAME_FORWARD_CONTRIBUTOR_VS_REPRESENTATIVE, REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS, LARGEST_COMPONENT_MEMBERSHIP, LARGEST_COMPONENT_HEDGE_REGION_MEMBERS, PATIO_HEDGE_GRAPH_FRONTIER, HIGH_IMPACT_BRIDGES, HIGH_IMPACT_BRIDGE_SOURCE_VIEW, HEDGE_REPRESENTATIVE_BACKBONE, HEDGE_LARGEST_COMPONENT_OVERLAP) 전부 `output/osn_gs_renderer_native_topology_gate_closure/<VIEW>/`에 `point_cloud.ply` + `render.ppm` + `preview_png/render.png` + 색상-범례 `README.md`로 존재.

## 테스트

집중 진단 테스트만 실행(production/runtime 공유 코드 변경 없음 — 진단 전용 CUDA 빌드만 확장, canonical 및 학습 경로 무수정이므로 전체 pytest 재실행 불필요, directive 지시):

- `tests/test_surfel_representative_diagnostics.py` — 10개 전부 통과(기존 5개 + 신규 5개: 렌더 불변성 재검증, visible surfel forward_accepted=1, occluded-but-weakly-transmitted forward_accepted=1, representative→forward_accepted 무예외 성립, 기존 5개 유지).

## 미완료/다음 단계

Trust, latent surface, NURBS fitting/decomposition, occluded surface construction, uncertain Gaussian proposal, non-representative contributor 부착 — 전부 이번 배치에서 구현하지 않았다. OSN-GS 표면 재구성이 완료됐다고 주장하지 않는다.
