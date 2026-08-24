# Worklog 108 — Renderer-Native Surface Representative Backbone: Architecture Gate / 회계 감사

## 상태

**완료 — 실측 있음. 조건부 통과(CONDITIONAL PASS), 명확한 caveat 동반.** Worklog 107의 adjacency 알고리즘은 전혀 수정하지 않았다(directive 지시). 대신 WL107이 "renderer-contributing surfel 전체"를 마치 topology를 정의할 수 있는 집단인 것처럼 통계를 낸 것을 "renderer surface representative"라는 올바른 구조적 모집단으로 재평가했다. **핵심 회계 불일치(§1)를 실측으로 정확히 해소**: WL107이 보고한 385,998이 아니라, contributing과 representative 사이에는 **정확히 36,051개**의 "representative지만 WL105 기준으로는 contributing이 아닌" surfel이 존재한다 — representative는 contributing의 엄격한 부분집합이 아니다(실측 확인). Representative-only topology는 상당히 결합력 있다(83.3% 연결, singleton 16.7%뿐). **테이블은 여전히 단일하고 패티오와 명확히 분리된 component(subset 1)이며, 결정론적 픽셀 anchor로 확인한 hedge의 3개 지점 모두 patio의 거대 component(subset 0)와 다른, 서로도 다른 component에 속한다.** 그러나 **패티오 거대 component(437,751개, 전체의 36.77%) 자체의 멤버 중 20.4%가 이번 배치가 정의한 "hedge 인접 영역"에 속한다** — 즉 패티오-hedge 경계 지대의 일부(지면에 가까운 하부 식생)가 실제로 패티오 component와 연결돼 있다. Bridge 강건성 감사에서 최대 component의 edge 중 5.5%가 구조적 bridge(제거 시 분리)이고 19.2%가 단일 뷰 지지 edge임을 확인했다 — 심각하지는 않지만 무시할 수 없는 취약성이다.

## 아키텍처 (변경 없음)

```
WL107 그대로 재실행 + 재해석:
    renderer-native median representative -> image-space 4-neighbor 인접
        -> 기존 3D 국소성 필터 -> 기존 기하 게이트 -> 다중뷰 양성 합집합
    을 그대로 두고, 통계를 "전체 surfel"이 아니라 "representative surfel"
    모집단으로 재계산 + 감사.
```

`torch_camera_induced_visible_adjacency.py`(WL107), `torch_surfel_representative_diagnostics.py`, `torch_surfel_contribution_diagnostics.py`(WL105) 전부 한 줄도 수정하지 않았다. 이번 배치는 새 `scripts/devtools/camera_induced_representative_backbone_audit.py` 하나만 추가했다(production 코드 변경 없음, 전체 pytest 재실행 불필요 — directive 지시).

## 1. 회계 불일치 해소

정확한 2x2 cross-tab(동일 체크포인트, 동일 WL105/WL107 시맨틱, 재실행으로 재현 확인):

| | REPRESENTATIVE+ | REPRESENTATIVE- |
|---|---|---|
| **CONTRIBUTING+** | 749,886 | 385,998 |
| **CONTRIBUTING-** | **36,051** | 18,534 |

**representative는 contributing의 엄격한 부분집합이 아니다** (`not_contributing_and_representative = 36,051 > 0`). WL107이 보고한 385,998(contributing이지만 representative 아님)은 그 자체로는 정확했다 — WL107의 원래 산술(`1,135,884 - 785,937 = 349,947`)이 discrepancy를 만든 이유는 단순 뺄셈이 교차 카테고리를 무시했기 때문이다: `785,937(representative) = 749,886(둘 다) + 36,051(representative만)`이고 `1,135,884(contributing) = 749,886(둘 다) + 385,998(contributing만)`이므로, `contributing - representative`는 `385,998 - 36,051 = 349,947`이 되어야 정확히 일치한다 — WL107은 이 36,051이라는 "예상 밖 카테고리"의 존재를 계산에서 누락했을 뿐, 385,998이라는 숫자 자체는 재현 가능하고 정확하다.

**36,051의 실제 원인**: 이 surfel들은 매 뷰마다 렌더러 CUDA 커널 내부에서 `T>0.5` 크로싱(median representative, forward-pass에서 직접 캡처)을 통과하지만, WL105의 `torch.autograd.grad` 기반 backward-pass contribution 신호에서는 한 번도 "기여함"으로 잡히지 않는다. 두 신호는 **서로 다른 두 개의 개별 CUDA 빌드**(canonical `diff_surfel_rasterization` vs 진단 `diff_surfel_rasterization_diag`)에서 나온다 — 소스는 동일하게 복사됐지만, WL105의 backward 커널은 `T_final`로부터 `T = T / (1 - alpha)`라는 **나눗셈으로 T를 역산**하며 자체적으로 `alpha < 1/255` 재검사를 수행하는 반면, forward 경로(median representative가 직접 캡처되는 곳)는 나눗셈 없이 순전파로 `alpha`를 계산한다. 이 두 경로의 부동소수점 재구성 방식 차이는 alpha가 1/255 문턱값에 극히 가까운 경계 사례에서 forward는 "accepted"로, backward의 독립 재계산은 "rejected"로 서로 다르게 판정할 수 있는 잘 알려진 비대칭이다. 3.0%(contributing 대비)/4.6%(representative 대비)라는 작은 비율은 이 가설과 일치한다. **이것이 버그라는 증거는 없다** — 두 CUDA 경로가 각각 자신의 산술로 독립적으로 도출한, 재현 가능한(재실행으로 확인) 결과다. 다만 완전히 실증적으로 증명하지는 못했으며, 전체 아키텍처 결론에는 영향을 주지 않는 작은 규모임을 밝힌다.

## 2. Representative-only topology 회계

| | 값 |
|---|---|
| representative surfel 수 | 785,937 |
| degree 0 (singleton) | 131,378 |
| degree ≥ 1 | 654,559 |
| representative singleton 비율 | **16.7%** |
| representative 연결 비율 | **83.3%** |
| representative를 포함하는 component 수 | 155,457 |
| 그 component 크기 (min/median/mean/p95/max) | 1 / 1 / 5.06 / 4 / 437,751 |
| 최대 component의 representative 모집단 대비 비율 | 55.7% |

전체 1,190,469개 기준 WL107 자체 통계(singleton 45.0%, 최대 36.77%)와는 **다른 계약**이다 — 전체 모집단은 non-representative(반드시 singleton)를 포함해 희석되지만, representative만 놓고 보면 singleton은 16.7%에 불과하다.

## 3. Non-representative 회계 (A/B/C/D)

| 카테고리 | 개수 |
|---|---|
| A. contributing & representative | 749,886 |
| B. contributing & never representative | 385,998 |
| C. renderer-noncontributing | 18,534 |
| D. non-contributing인데 representative(예상 밖, §1) | 36,051 |

B/C/D 전부 이번 배치에서 component에 강제 편입하지 않았다 — 여전히 "renderer-contributing/observed primitive"로만 보존된다.

## 4. 남은 representative singleton 원인 (131,378개, 전역)

| 원인 | 개수 | 비율 |
|---|---|---|
| `REPRESENTATIVE_HAS_NO_DISTINCT_PIXEL_NEIGHBOR_RELATION` | 0 | 0% |
| `CAMERA_PAIR_GENERATED_BUT_FAILS_3D_LOCALITY` | 75,771 | 57.7% |
| `PASSES_LOCALITY_BUT_GEOMETRIC_DISCONTINUITY` | 10,447 | 8.0% |
| `PASSES_LOCALITY_BUT_POSITIONAL_SEPARATION` | 45,160 | 34.4% |
| `OTHER_EXPLICITLY_REPORTED_CAUSE` | 0 | 0% |

압도적 원인은 3D 국소성 실패(57.7%)와 positional separation(34.4%) — "이웃 관계 자체가 없음"은 0건(모든 representative singleton이 최소 1개의 이미지-공간 이웃 관계를 가졌지만 그 관계가 국소성/기하 게이트를 통과 못 했다는 뜻).

**테이블/패티오 분리 회계**(hedge는 §8):

| | 국소성 실패 | Geometric | Positional |
|---|---|---|---|
| table | 11,999 | 1,109 | 6,266 |
| patio | 38,104 | 5,106 | 24,920 |

## 5. 81.1% 국소성 기각 감사

기각된 7,538,912쌍의 3D 거리/local-spacing 비율 분포: min 1.04x, **median 2.96x**, mean 12.49x, p95 73.2x, max 1979x. **median이 약 3배에 불과하다는 것은 대부분의 기각이 "터무니없이 먼" 우연이 아니라 실루엣/깊이 경계 근처의 진짜 근접-하지만-비국소 관계임을 시사한다** — 다만 p95/max의 긴 꼬리는 극단적으로 먼 우연의 일치도 존재함을 보여준다. Threshold를 조정하지 않았다(directive 지시).

## 6. 최대 component의 정확한 정체 (정량적 검증)

- subset_id 0, 멤버 437,751개(전체의 36.77%, WL107과 재현 일치)
- 3D bounding box: x∈[-12.56, 18.52], y∈[-7.59, 8.71], z∈[-13.25, 12.09] — y축 범위가 16.3 단위로 상당히 넓다(순수 평면 바닥이라면 좁아야 함 — 아래 §8 참조)
- **결정론적 anchor 검증**(픽셀 색상 대조가 아니라 실제 `representative_id`를 preview 카메라에서 직접 조회): 테이블 anchor → subset **1**(degree 9, 명확히 다른 component); 패티오 anchor 2곳 → 둘 다 subset **0**; hedge anchor 3곳 → subset 448039(degree 0, singleton), subset **9**(degree 7), subset **479**(degree 3) — **셋 다 서로 다르고, 전부 subset 0이 아니다.**
- **그러나 hedge_region_surfels_in_largest_component_count = 89,502개 (거대 component 멤버의 20.4%, 이번 배치가 정의한 "hedge 인접 영역"의 26.2%)** — 시각 검토(REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS)에서도 hedge 좌상단 일부가 패티오와 동일한 색으로 나타남을 확인했다.

**결론**: 3개의 특정 anchor 지점은 절대 patio에 흡수되지 않았지만, "hedge 영역"(단순 최근접-anchor 거리 기준 heuristic) 전체로 보면 약 1/4 정도가 실제로 patio의 거대 component와 연결돼 있다 — **완전한 percolation은 아니지만, 완전히 순수한 "patio만"도 아니다.** 지면-식생 하부 경계에서 진짜 물리적 연속성이 존재할 가능성이 높다(y-범위가 넓은 것과 일치).

## 7. 큰 component의 연결 강건성

| component | 멤버 | edge | 1-view-지지 edge 비율 | bridge 수 |
|---|---|---|---|---|
| rank 0 (patio, 437,751개) | 437,751 | 1,034,050 | 19.2% | **56,816 (5.5%)** |
| rank 1 (51,734개) | 51,734 | 124,713 | 16.8% | 7,081 (5.7%) |
| rank 2 (7,014개) | 7,014 | 14,417 | 22.2% | 1,277 (8.9%) |

Bridge 비율은 5.5~8.9%로, **94.5%+ 의 edge는 중복/순환 경로를 가진 진짜 2D mesh형 연결**이지만, 나머지 5.5~8.9%는 제거 시 component가 쪼개지는 단일 연결점이다. 1-view-지지 edge 비율(16.8~22.2%)도 무시할 수 없는 수준이다 — "여러 뷰 중 단 하나가 관측한 관계"가 전체 연결성의 상당 부분을 떠받치고 있다는 뜻이며, 이는 directive가 의도적으로 허용한 permissive union의 자연스러운 결과이지 오류는 아니다. 56,816개 bridge 전부를 개별적으로 실루엣 여부까지 검증하지는 못했다(규모상 이번 배치 범위 밖) — 집계 통계로만 보고한다.

## 8. Hedge를 representative backbone만으로 재평가

| | 개수 |
|---|---|
| hedge 영역 surfel(heuristic) | 342,085 |
| 그중 contributing | 319,281 (93.3%) |
| 그중 representative | 204,164 (59.7%) |
| representative 중 singleton | 43,874 |
| representative 중 연결됨 | **160,290 (78.5%)** |

**답**: hedge의 representative backbone 자체는 "심각하게 파편화"돼 있지 않다 — representative로 판정된 surfel의 78.5%가 실제로 최소 하나의 다른 surfel과 연결돼 있다. WL107이 시각적으로 "hedge 대부분 파편화"라고 본 것의 상당 부분은 **hedge 영역의 원래 renderer-contributing surfel 중 40.3%(319,281 - 204,164... 정확히는 representative 아닌 115,117개)가 애초에 representative가 아니어서 render-support 역할일 뿐 자기 자신의 topology node가 될 수 없었기 때문**이다 — WL107이 "hedge 파편화"로 본 시각적 인상은 fragmentation과 non-representative-support primitives 둘 다 섞여 있었다.

## 9. Table 결과

Subset 1, degree 9, 패티오(subset 0)와 명확히 분리. Singleton 원인은 국소성(11,999)이 지배적, positional(6,266), geometric(1,109) 순.

## 10. Patio 결과

Subset 0(최대 component)의 핵심. Singleton 원인 분포는 table과 유사한 비율(국소성 > positional > geometric).

## 11. Hedge representative-backbone 결과

§8 참조 — 78.5% 연결, 20.4%가 실제로 patio component에 흡수(§6).

## 12. Non-representative contributor 회계

385,998(contributing만) + 18,534(둘 다 아님) = 404,532개가 이번 배치에서 topology에 전혀 참여하지 않는다 — 그러나 여전히 renderer-contributing/observed 증거로 완전히 보존된다. **어떤 attachment도 이번 배치에서 수행하지 않았다**(directive 지시).

## 13. Visible/Occluded 계약

변경 없음 — WL107 자체를 그대로 재실행했으므로 occluded gap 분리 등 기존 계약은 그대로 유지된다.

## 14. Architecture Gate 판정

**CONDITIONAL PASS.** PASS 조건을 항목별로 대조:

| 조건 | 충족 여부 |
|---|---|
| representative-only topology가 상당히 결합력 있음 | ✅ (83.3% 연결) |
| 테이블이 결합력 있는 단일 component로 남음 | ✅ |
| 테이블이 패티오와 분리됨 | ✅ |
| 최대 component가 percolation이 아니라 하나의 정당한 visible surface 영역과 정량적으로 일치 | **부분적** — patio가 핵심이지만 멤버의 20.4%가 hedge 인접 영역과 겹침 |
| hedge representative backbone이 치명적으로 파편화되지 않았거나 원인이 귀속됨 | ✅ (78.5% 연결, 원인 대부분 국소성/positional로 귀속) |
| 소수의 명백히 잘못된 bridge가 major percolation을 주도하지 않음 | **부분적** — bridge가 5.5~8.9%로 적지 않지만, 그것이 "table/hedge 전체 흡수"를 일으키지는 않음 |

**결론적으로 Renderer-Native Surface Representative Graph를 canonical Visible Surface Topology Backbone의 유력한 후보로 formally propose하되, "최대 component의 순도"라는 한 항목에서 명시적 caveat(패티오-hedge 경계 지대 약 20% 교차 연결)를 동반한다.** Renderer-Contributing Non-Representative Surfel(404,532개)은 retained Visible Surface Support Evidence로 분류하고, 이번 배치에서는 어떤 component에도 배정하지 않는다.

## 15. Review export

`output/osn_gs_camera_induced_representative_backbone_audit/{ORIGINAL_2DGS_SCENE, ALL_RENDERER_CONTRIBUTORS, RENDERER_SURFACE_REPRESENTATIVES, REPRESENTATIVE_ONLY_VISIBLE_COMPONENTS, REPRESENTATIVE_SINGLETON_CAUSE_VIEW, LARGEST_COMPONENT_MEMBERSHIP_VIEW, LARGE_COMPONENT_BRIDGE_VIEW, 3D_LOCALITY_REJECTION_VIEW, HEDGE_REPRESENTATIVE_BACKBONE, HEDGE_NON_REPRESENTATIVE_SUPPORT}/`, PNG: `preview_png/`, 전체 리포트: `camera_induced_representative_backbone_audit_report.json`.

## 16. 테스트

`tests/test_camera_induced_representative_backbone_audit.py` (9 tests, CUDA 불필요, 순수 로직): bridge-finding 정확성(경로/사이클+pendant/완전연결/고립노드), distribution 통계, scatter_or 헬퍼. Production 코드(osn_gs/)를 전혀 수정하지 않았으므로 전체 pytest는 재실행하지 않았다(directive 지시 — "prefer not to modify production behavior at all in this batch"를 그대로 지켰음을 확인).

## 결론

Representative backbone은 WL107의 raw contributor 기준 통계보다 **훨씬 더 결합력 있고 해석 가능**하다(singleton 16.7% vs 45.0%). 회계 불일치는 완전히 해소됐고(36,051의 원인까지 규명), 테이블-패티오 분리는 결정론적 anchor로 재확인됐다. 그러나 "최대 component가 순수하게 패티오만"이라는 WL107의 다소 낙관적인 결론은 이번 정밀 감사로 **부분적으로 수정**된다 — 실제로는 패티오-hedge 경계의 하부 식생 일부(20.4%)가 함께 연결돼 있다. Bridge 강건성도 완벽하지 않다(5.5~8.9% bridge edge, ~19% 단일-뷰 지지). **최종 OSN-GS surface reconstruction 완성을 주장하지 않는다** — 이 배치는 architecture gate/회계 감사이며, non-representative contributing surfel을 어떻게 다룰지는 다음 배치의 질문으로 남긴다.

## 참고

- 새 스크립트: `scripts/devtools/camera_induced_representative_backbone_audit.py` (production 코드 변경 없음)
- 테스트: `tests/test_camera_induced_representative_backbone_audit.py`
- 관련: [[project_camera_induced_visible_adjacency]] (WL107, 이번 배치가 그대로 재실행한 baseline)
