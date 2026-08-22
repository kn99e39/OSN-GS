# Worklog 106 — Renderer-Grounded Visible Adjacency: 통제된 대조 실험, 부정적 결과

## 상태

**완료 — 실측 있음. 명확히 부정적인 결과이며, 정직하게 보고한다.** Worklog 105가 확인한 "Phase-C center 질의는 2DGS primitive-level visibility에 부적절하다"는 원칙을 살려, WL103의 endpoint eligibility를 Phase-C center에서 **실제 renderer 공식 contribution**(WL105, 수정 없음)으로 교체한 새 아키텍처(`torch_renderer_grounded_visible_adjacency.py`)를 구현·실측했다. WL103(A)과 정확히 동일한 candidate graph·corridor 테스트·기하 게이트로 통제된 대조를 실행한 결과: **singleton 비율이 63.4%(A) → 83.8%(B)로 악화됐고, 최대 component 비율도 10.50%(A) → 2.9%(B)로 더 작아졌다** — percolation이 재발한 게 아니라 훨씬 심한 파편화가 발생했다. WL103의 singleton이면서 실제로는 renderer-contributing인 720,052개 중 겨우 11.4%(81,974개)만 이번 배치에서 edge를 얻었고, 나머지 88.6%(638,078개)는 여전히 singleton으로 남았다 — 그 압도적 원인(86.3%)은 "co-contributing 이웃이 없어서"가 아니라 **다중 뷰 관측 모순**(`OBSERVATION_CONFLICT` 57.9% + `HARD_OBSERVATION_CONTRADICTION` 28.4%)이었다. Directive 지시대로 여기서 threshold를 조정하지 않고 멈춘다 — 이 결과는 **다음 아키텍처가 "3D candidate edge에 카메라가 승인" 방식이 아니라 "카메라가 실제 가시 표면 지지로부터 인접성을 생성" 방식(camera-induced adjacency)으로 이동해야 함을 시사**하지만, 그 구현은 이번 배치 범위 밖이다.

## 아키텍처

```
공유 candidate graph (build_candidate_graph, 1회만 빌드, A/B 공유)
    A. WL103 재실행(수정 없음): endpoint eligibility = Phase-C center on_observed_surface
    B. 신규(이번 배치): endpoint eligibility = WL105 renderer-contributing(같은 뷰)
        -> corridor RANGE 테스트(WL103과 동일 코드/파라미터)
        -> 기하 게이트(WL103과 동일 코드/파라미터)
        -> connected components
```

`torch_positive_visible_adjacency.py`(WL103), `torch_node_level_observability_accounting.py`(WL104), `torch_surfel_contribution_diagnostics.py`(WL105) 전부 한 줄도 수정하지 않았다. Corridor RANGE 테스트, depth_epsilon, interior sample 수, residual MAD multiplier, positional ratio는 전부 WL103과 동일한 값 — directive §6이 요구한 "controlled replay"를 만족한다.

## 1. Renderer-grounded primitive evidence 정의

WL105의 `compute_renderer_contribution_for_view`를 그대로 재사용(재구현 없음) — 매 뷰마다 각 surfel이 official alpha-compositing에 실제로 기여했는지(bool)를 얻는다. `radii>0`도, Phase-C center도 endpoint eligibility로 쓰지 않았다(AST 기반 테스트로 고정).

## 2. Same-view co-contribution 관계

candidate edge (i,j)가 corridor 테스트를 받으려면 **같은 뷰에서 i와 j 둘 다 renderer-contributing**이어야 한다. 이 조건이 WL103의 "둘 다 on_observed_surface"를 대체한 유일한 변경점이다.

## 3. 유지된 image-space continuity 테스트

WL103/102의 RANGE 기반 화면-경로 corridor 테스트(직선 chord도 선형보간도 아님)를 코드 그대로 재사용했다 — corridor 자체는 co-contribution 조건과 무관하게 카메라 자신의 렌더 depth 필드를 그대로 읽는다.

## 4. 다중 뷰 취합

WL103과 동일 — 퍼센트/과반 threshold 없음. `>=1` 유효 양성 관계 + 모순 없음 => 연결.

## 5. WL103 vs Renderer-grounded 실측 비교 (동일 체크포인트, 161 카메라, 공유 graph)

| | A. WL103 (center-grounded) | B. Renderer-grounded (신규) |
|---|---|---|
| visible component 수 | 768,829 | **1,004,080** |
| 최대 component 비율 | 10.50% | **2.91%** |
| singleton 비율 | 63.4% | **83.8%** |
| 최종 positive edge 수 | 1,043,908 (20.3%) | **409,620 (8.0%)** |
| `CUT_OCCLUDED_DOMAIN` | 1,380 | **1,175,611** |
| `CUT_KNOWN_FREE_SPACE` | 1,616 | 167,341 |
| `UNRESOLVED_OBSERVATION_CONFLICT` | 18,107 | **2,834,719** |
| `UNKNOWN_NO_...RELATION` | 3,836,371 | 384,093 |

같은 spatial candidate edge(5,132,180개) 중 co-eligible(같은 뷰에 둘 다 등장) edge가 renderer-contribution 기준으로 훨씬 많아졌지만(카메라당 평균 백만 개대, WL103의 평균 14만 개 대비), 그중 압도적 다수가 **positive가 아니라 모순(occluded/conflict)으로 귀결**됐다.

## 6. WL103-singleton-and-renderer-contributing (720,052개) 회복 실측 — 핵심 인과 테스트

- edge를 얻어 singleton을 벗어남: **81,974개 (11.4%)**
- 여전히 singleton: **638,078개 (88.6%)**

## 7. 남은 singleton 원인 분해 (638,078개)

| 원인 | 개수 | 비율 |
|---|---|---|
| `NO_SAME_VIEW_COCONTRIBUTING_SPATIAL_NEIGHBOR` | 870 | 0.14% |
| `HARD_OBSERVATION_CONTRADICTION` | 203,752 | 31.9% |
| `GEOMETRIC_DISCONTINUITY` | 2,277 | 0.36% |
| `POSITIONAL_SHEET_SEPARATION` | 16,340 | 2.56% |
| `OBSERVATION_CONFLICT` | 414,839 | **65.0%** |

**"co-contributing 이웃이 없어서"는 사실상 무시할 수준(0.14%)이다.** 압도적 원인은 다중 뷰 관측 모순(conflict + hard contradiction = 96.9%) — 즉 3D 공간적으로 가깝고 둘 다 렌더러에 실제로 기여하는 surfel 쌍이라도, 여러 카메라에 걸쳐 실제로 관측하면 서로 다른 depth layer(겹치는/중복된 primitive 층)에 속한다는 신호가 자주 나온다.

## 8. 최종 visible component 통계

1,004,080개, 최대 2.91%(34,588개), median 크기 1, mean 1.186. Coverage identity 유지(1,190,469개 전량).

## 9-11. 테이블 / 패티오 / hedge-배경

`RENDERER_GROUNDED_VISIBLE_COMPONENTS` 뷰에서 **테이블, 패티오, hedge/배경 전부** 미세한 무지개 스페클로 나타난다 — WL103에서는 테이블이 단일 색으로 깨끗이 분리돼 있었지만, 이번 배치에서는 테이블조차 파편화됐다. `OCCLUDED_FREE_SPACE_TERMINATION_VIEW`는 장면 거의 전체(테이블·패티오 포함)가 붉게 물들어 있다 — occlusion cut이 hedge뿐 아니라 장면 전반에서 대량 발동한다.

## 12. Percolation 결과

percolation은 재발하지 않았다(최대 component가 오히려 더 작아짐, 2.9%) — 이번 배치의 실패 모드는 percolation의 반대인 **극단적 과소-연결**이다.

## 13. Visible/Occluded 역할 분리 회귀

없음 — `test_co_contributing_endpoints_separated_by_occluder_do_not_connect`로 실제 occluder가 여전히 두 component를 분리함을 확인했다(WL103과 동일 계약 유지).

## 14. Primitive/topology accounting

| | 개수 |
|---|---|
| 전체 학습 surfel | 1,190,469 |
| renderer-contributing | 1,135,884 (95.4%) |
| renderer-noncontributing | 54,585 |
| adjacency로 연결된 surfel | 193,222 |
| renderer-contributing이지만 위상적으로 고립 | **942,662** |

## 15. Review export

`output/osn_gs_renderer_grounded_visible_adjacency/{ORIGINAL_2DGS_SCENE, WL103_CENTER_GROUNDED_COMPONENTS, RENDERER_CONTRIBUTING_PRIMITIVES, SAME_VIEW_CO_CONTRIBUTION_RELATIONS, RENDERER_GROUNDED_VISIBLE_ADJACENCY, RENDERER_GROUNDED_VISIBLE_COMPONENTS, REMAINING_SINGLETON_CAUSE_VIEW, OCCLUDED_FREE_SPACE_TERMINATION_VIEW}/`, PNG: `preview_png/`, 전체 리포트: `renderer_grounded_visible_adjacency_report.json`.

## 16. 테스트

`tests/test_renderer_grounded_visible_adjacency.py` (14 tests): 모듈이 Phase-C center 분류에 의존하지 않음(AST), center-negative지만 contributing인 endpoint가 adjacency를 형성함, non-contributing endpoint는 형성 못 함, occluder로 분리된 co-contributing pair가 연결 안 됨, known-free-space도 동일, 뷰 부재는 모순이 아님(연결 유지), 진짜 모순은 unresolved 유지, 곡면 co-contributing 표면 연결 유지, 인접 시트 positional gate로 분리 유지, primitive visibility와 graph degree 독립성, coverage/determinism. 전체 regression: WL105의 1225 + 신규 14 = 1239 passed 1 skipped(실행 결과는 커밋 메시지에 기록).

## 17. 결론

**부정적 결과.** Renderer-grounded primitive evidence(WL105)는 "무엇이 실제로 존재하는 evidence인가"라는 질문에는 올바른 답을 줬지만, WL103의 pairwise 3D-candidate-edge + per-view corridor 아키텍처에 그대로 얹으면 **훨씬 심한 파편화**(singleton 63.4%→83.8%, 최대 component 10.5%→2.9%)를 낳는다. 원인은 압도적으로(96.9%) 다중 뷰 관측 모순이지 co-contributing 이웃의 부재(0.14%)가 아니다 — 즉 **"이 exact WL103 pairwise corridor relation" 자체가 지배적 실패 원인**임을 실측이 명확히 보여준다. Directive의 완결 조건에 따르면 이는 다음 아키텍처가 "3D candidate edge를 카메라가 승인"하는 방식에서 "카메라가 실제 가시 표면 지지로부터 인접성을 생성"하는 방식(camera-induced adjacency)으로 이동해야 함을 정당화하는 결과이지만, **이 배치에서는 그 구현을 하지 않는다** — directive가 명시적으로 다음 배치로 미뤘다.

## 참고

- 새 모듈: `osn_gs/surface/torch_renderer_grounded_visible_adjacency.py`
- 테스트: `tests/test_renderer_grounded_visible_adjacency.py`
- Export 스크립트: `scripts/devtools/renderer_grounded_visible_adjacency_export.py`
- 관련: [[project_renderer_contribution_diagnostics]] (WL105, 이번 배치가 그대로 재사용한 primitive evidence), [[project_positive_visible_adjacency]] (WL103, baseline A로 재실행)
