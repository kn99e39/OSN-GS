# Worklog 110 — Non-Representative Renderer Evidence: Role Attribution

## 상태

**완료 — 실측 있음. 건축 결정: AMBIGUOUS/LAYERED SUPPORT (귀속 유예).** Worklog 109가 GATE PASS로 확정한 Renderer-Native Surface Representative Graph(WL107/109)는 이번 배치에서 한 줄도 수정하지 않았다. 이번 배치의 목표는 오직 "representative가 되지 못한 채 forward-accept된 395,676개 서펠"의 역할을 귀속(attribution)하는 것이었으며, 지시대로 **어떤 서펠도 어떤 컴포넌트에도 attach하지 않았다.** 측정 결과 이 인구는 하나의 canonical 표면 컴포넌트에 일관되게 대응하는 소수(26.2%)와, 다중 컴포넌트에 걸쳐 있거나(48.0%) median 뒤쪽 증거가 지배적인(63.3%가 최소 1회 POST_MEDIAN) 다수로 나뉘며, 게다가 측정 자체가 픽셀당 K=16 슬롯 캡으로 인해 **전체 픽셀·뷰 이벤트의 97.4%에서 truncation**이 발생했다 — 이 truncation은 표본을 항상 "더 단순하게"(단일 컴포넌트·PRE_MEDIAN 쪽으로) 편향시키는 방향이므로, 실측된 다중-컴포넌트/POST_MEDIAN 비율은 진짜 값의 하한(lower bound)이다. 이 두 가지(다수가 이미 모호함 + truncation이 그 모호함을 과소평가하는 방향으로만 작용) 때문에 "IDENTIFIABLE SUPPORT"를 선언할 근거가 없다.

## 1. 같은-forward 순회(traversal) 시맨틱스

WL108/109가 이미 노출한 `out_forward_accepted`(같은 forward 실행, canonical 커널의 5가지 accept 체크를 통과한 지점)에 더해, 이번 배치는 같은 지점에서 이미 커널이 읽고 있던 running transmittance `T`(이 contributor 자신의 알파 업데이트 **이전** 값 — median-crossing 체크 `if (T > 0.5)`가 읽는 바로 그 값)를 포착했다:

- `T > 0.5` → 이 accept 이벤트는 이 픽셀의 median crossing **이전(또는 그 지점)**에 일어났다 (`contrib_post_median=0`)
- `T <= 0.5` → 이미 어떤 앞선 contributor가 T=0.5를 넘은 **이후**에 일어났다 (`contrib_post_median=1`)

이번 배치가 다루는 인구(어느 뷰에서도 representative가 된 적 없는 서펠)에 한해서는, 이 자신은 결코 median crossing 자체가 될 수 없다는 population-membership 정의상 `contrib_post_median=0`은 항상 순수한 PRE_MEDIAN을 의미한다 — 새 CUDA 로직 없이 기존 신호만으로 pre/median/post 분류가 성립한다(모듈 docstring에 상세 증명, `osn_gs/surface/torch_nonrepresentative_evidence_attribution.py`).

## 2. Contributor↔representative provenance 메커니즘

각 accept 이벤트에서 "같은 렌더된 픽셀"의 `representative_id`(WL107이 이미 노출)를 co-occurring evidence로 사용한다. **픽셀×서펠 전체 행렬은 절대 만들지 않는다** — directive 지시대로, 픽셀당 bounded slot 배열 `(H, W, K=16)`(`out_contrib_ids`/`out_contrib_post_median`, `config.h`의 `OSN_GS_MAX_CONTRIB_SLOTS`)와 별도의 uncapped 진짜 카운트 `out_contrib_count`(H, W)를 새 CUDA 진단 필드로 추가해, 뷰별로 distinct (contributor, component) 쌍을 뽑고(`view_contributor_component_pairs`), 161개 뷰 전체를 `key = contributor * subset_count + component`(int64) 인코딩 후 단일 `torch.unique`로 전역 중복 제거한다(`finalize_component_co_support`) — sparse/streamed 표현이며 truncation은 `contrib_count`로 항상 감지 가능하다.

**실측된 truncation은 매우 크다**: 전체 161개 뷰 × 648×420 픽셀 = 43,817,760개 픽셀·뷰 슬롯 중 **42,660,905개(97.4%)**가 `contrib_count > 16`이었다. 슬롯은 depth 오름차순(가까운 것부터)으로 채워지므로, truncation은 항상 "더 먼(더 POST_MEDIAN에 가까운, 더 다중-컴포넌트일 가능성이 높은)" contributor를 기록에서 누락시키는 방향으로만 작용한다 — directive Section 5의 "정확한 provenance를 노출할 수 없다면 그 한계를 보고하고 STOP하라"는 지시에 해당하는 상황이다. 휴리스틱으로 대체하지 않고 이 한계를 그대로 보고한다: **이번 배치의 pre/post-median 및 component co-support 수치는 진짜 인구의 정확한 값이 아니라, "가까운 K=16개 이내" 증거만 반영한 편향된(단순화 쪽으로 편향된) 표본이다.**

## 3. Representative / accepted 회계

| | 개수 | 비율 |
|---|---|---|
| 전체 학습된 서펠 | 1,190,469 | 100% |
| MEDIAN_SURFACE_REPRESENTATIVE | 785,937 | 66.0% |
| SAME-FORWARD ACCEPTED NON-REPRESENTATIVE | 395,676 | 33.2% |
| NEVER FORWARD-ACCEPTED | 8,856 | 0.7% |

WL109와 정확히 동일 — 이번 배치는 이 회계를 재확인만 했다(같은 진단 forward 실행 재사용).

## 4. Pre/post-median 분포 (accepted non-representative 395,676개 중)

| 항목 | 개수 | 비율 |
|---|---|---|
| 최소 1회 PRE_MEDIAN | 173,276 | 43.8% |
| 최소 1회 POST_MEDIAN | 250,526 | 63.3% |
| PRE_MEDIAN만 | 43,177 | 10.9% |
| POST_MEDIAN만 | 120,427 | 30.4% |
| PRE와 POST 혼합 | 130,099 | 32.9% |
| 둘 다 없음(예상대로 = ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION 인구와 정확히 일치, 101,973) | 101,973 | 25.8% |

POST_MEDIAN 쪽(63.3%)이 PRE_MEDIAN 쪽(43.8%)보다 크다 — 그리고 위 2절의 truncation 방향성 때문에 이 격차는 실제로는 더 클 가능성이 높다.

## 5. Component co-support 분포

| 카테고리 | 개수 | 비율 |
|---|---|---|
| SUPPORTS_ONE_REPRESENTATIVE_COMPONENT | 103,776 | 26.2% |
| SUPPORTS_MULTIPLE_REPRESENTATIVE_COMPONENTS | 189,927 | 48.0% |
| ACCEPTED_BUT_NO_MEDIAN_REPRESENTATIVE_ASSOCIATION | 101,973 | 25.8% |

다중-컴포넌트 접촉 수 분포(189,927개 대상): 중앙값 4개, 평균 7.30개, p95 24개, 최대 609개. "정확히 하나의 컴포넌트만 일관되게 지지"하는 인구는 전체의 4분의 1 수준이며, 나머지 4분의 3은 무연관이거나 다중 컴포넌트에 걸쳐 있다.

## 6. 국소 기하 적합성 분포 (SINGLE_COMPONENT_COSUPPORT, 기존 3D candidate graph만 재사용, 새 휴리스틱 없음)

`build_candidate_graph`(WL96 계열, 무수정)로 이미 존재하는 3D 근접 후보 엣지 중, single-component-cosupport 관계와 독립적으로 일치(corroborated)하는 엣지 247,044개(single-component-cosupport 인구의 238.1% — 한 contributor가 같은 컴포넌트의 여러 representative 멤버와 corroborated edge를 가질 수 있으므로 100% 초과 가능):

| 항목 | 중앙값 | 평균 | p95 | 최대 |
|---|---|---|---|---|
| 거리(씬 단위) | 0.0434 | 0.0483 | 0.1024 | 0.4211 |
| 거리/local-spacing | 1.147 | 1.114 | 1.493 | 1.997 |
| shape-operator residual | 0.2062 | 0.2876 | 0.8221 | 7.6470 |
| normal-offset-ratio | 0.7649 | 1.2659 | 3.7274 | 515.83 |

거리/local-spacing 중앙값(1.147)과 normal-offset-ratio 중앙값(0.765)은 WL100 계열 게이트 임계값(≈1.0) 근처거나 그 이상 — "SINGLE_COMPONENT_COSUPPORT로 분류됐다"는 사실이 기하적으로도 깨끗하게 가깝다는 것을 자동으로 보장하지 않는다. residual/normal-offset-ratio의 긴 꼬리(최대 7.65 / 515.8)는 기하적으로 먼 corroborated 사례도 존재함을 보여준다 — attachment 판단 시 이 분포를 참고해야 한다는 진단 정보일 뿐, 이번 배치에서 필터링에 쓰지 않았다.

## 7. 테이블 결과

| 항목 | 값 |
|---|---|
| accepted non-representative | 46,313 |
| PRE_MEDIAN만 | 3,627 (7.8%) |
| POST_MEDIAN만 | 10,863 (23.5%) |
| MIXED | 15,007 (32.4%) |
| 무연관 | 16,816 (36.3%) |
| 단일-컴포넌트 co-support | 13,103 (28.3%) |
| 다중-컴포넌트 co-support | 16,394 (35.4%) |

세 영역 중 무연관 비율이 가장 높다 — 테이블은 이미 WL107-109에서 잘 분리된 컴팩트한 표면이라, non-representative 증거 자체가 적고(다른 두 영역 대비 population도 가장 작음) 애매한 채로 남는 비중이 크다.

## 8. 파티오 결과

| 항목 | 값 |
|---|---|
| accepted non-representative | 215,083 (세 영역 중 최대) |
| PRE_MEDIAN만 | 24,827 (11.5%) |
| POST_MEDIAN만 | 69,272 (32.2%) |
| MIXED | 68,436 (31.8%) |
| 무연관 | 52,548 (24.4%) |
| 단일-컴포넌트 co-support | 62,105 (28.9%) |
| 다중-컴포넌트 co-support | 100,430 (46.7%) |

세 영역 중 절대 개수가 가장 크고, 다중-컴포넌트 co-support 비율(46.7%)도 헤지(54.4%)보다는 낮지만 테이블(35.4%)보다 뚜렷이 높다. `cross_component_examples_sample`(30개 무작위 표본) 중 table↔patio를 동시에 접촉하는 사례는 0건 — 테이블과 파티오 사이는 non-representative 증거 수준에서도 깨끗이 분리돼 있다(표본 30개 한정).

## 9. 헤지/배경 결과

| 항목 | 값 |
|---|---|
| accepted non-representative | 134,280 |
| PRE_MEDIAN만 | 14,723 (11.0%) |
| POST_MEDIAN만 | 40,292 (30.0%) |
| MIXED | 46,656 (34.7%) |
| 무연관 | 32,609 (24.3%) |
| 단일-컴포넌트 co-support | 28,568 (21.3%) |
| 다중-컴포넌트 co-support | 73,103 (54.4%) |

세 영역 중 POST-계열(POST_MEDIAN만 + MIXED = 64.7%)과 다중-컴포넌트 비율(54.4%)이 모두 가장 높다 — WL105/108/109가 반복 확인한 "헤지/배경은 volumetric-dense하고 layered하다"는 특성과 일치한다. `largest_component_hedge_cross_examples_in_sample`(30개 표본 중)은 0건이었으나 표본이 작아 결론적 증거는 아니다. directive Section 9 지시대로, POST_MEDIAN_ONLY를 자동으로 "가려진 표면"이라 해석하지 않았다 — 이는 renderer 순회 순서 사실일 뿐이다.

## 10. 교차-컴포넌트(cross-component) 지지 거동

189,927개(48.0%)가 2개 이상의 canonical 컴포넌트에 걸쳐 co-support 관계를 가진다(중앙값 4개, 최대 609개). 무작위 30개 표본(`cross_component_examples_sample`, JSON에 전수 좌표·컴포넌트ID·pre/post 플래그 포함) 중 table↔patio 동시 접촉 0건, 최대컴포넌트↔hedge 동시 접촉 0건 — 표본 수준에서는 "서로 다른 물체" 간 교차보다 "같은 물체 표면 위의 여러 파편화된 WL96 계열 컴포넌트" 간 교차가 대부분으로 보인다(예: `contributor_visible_id=207`이 접촉한 21개 컴포넌트가 모두 patio 표면 위에 있음). 다수결로 애매함을 해소하지 않았다 — directive 지시대로 원자료 그대로 보고한다.

## 11. IDENTIFIABLE SUPPORT인가, AMBIGUOUS/LAYERED SUPPORT인가

**AMBIGUOUS/LAYERED SUPPORT.** 근거:

1. 정확히 하나의 컴포넌트를 일관되게 co-support하는 인구(26.2%)는 소수다 — 나머지 74%는 무연관(25.8%)이거나 다중 컴포넌트(48.0%)다.
2. POST_MEDIAN 증거(63.3%가 최소 1회)가 PRE_MEDIAN 증거(43.8%)보다 크고, 순수 POST_MEDIAN만 가진 인구도 30.4%로 상당하다.
3. 측정 자체가 K=16 슬롯 캡으로 97.4%의 픽셀·뷰 이벤트에서 truncation됐고, 그 편향은 항상 "더 단순한 쪽"(단일 컴포넌트, PRE_MEDIAN)으로 표본을 밀어낸다 — 즉 실측된 모호성(다중 컴포넌트/POST_MEDIAN 비율)은 진짜 값의 **하한**이다. 이 방향성 때문에 "모호함이 측정 오류일 뿐 실제로는 더 단순할 것"이라고 볼 근거가 전혀 없다 — 오히려 반대다.
4. 국소 기하 적합성(6절)도 SINGLE_COMPONENT_COSUPPORT라는 라벨이 자동으로 "기하적으로도 깨끗함"을 보장하지 않음을 보여준다(긴 residual/normal-offset-ratio 꼬리).

따라서 directive Section 16의 지침대로, 모든 non-representative 증거를 Visible Surface Support Evidence라고 부르지 않는다. 렌더러 역할(PRE_MEDIAN/POST_MEDIAN/MIXED, 단일/다중/무연관 컴포넌트)을 계속 별도로 보고해야 하며, 다음 아키텍처는 이 레이어드/모호한 증거를 강제로 하나의 컴포넌트 소유로 밀어넣지 않고 그대로 보존해야 한다.

## 12. 검토용 export 경로

`output/osn_gs_nonrepresentative_evidence_attribution/` 아래 10개 뷰(각 `iteration_0000001/point_cloud.ply`, `render.ppm`, `preview_png/render.png`, `README.md`):

`ORIGINAL_2DGS_SCENE`, `CANONICAL_REPRESENTATIVE_BACKBONE`, `FORWARD_ACCEPTED_NON_REPRESENTATIVES`, `PRE_MEDIAN_ACCEPTED`, `POST_MEDIAN_ACCEPTED`, `POST_MEDIAN_ONLY`, `SINGLE_COMPONENT_COSUPPORT`, `MULTI_COMPONENT_COSUPPORT`, `TABLE_PATIO_NONREP_RELATIONS`, `HEDGE_NONREP_RELATIONS`.

전체 JSON 리포트: `output/osn_gs_nonrepresentative_evidence_attribution/nonrepresentative_evidence_attribution_report.json`.

## 13. 집중 테스트

- CUDA 진단 확장(`diff_surfel_rasterization_diag`)에 `out_contrib_ids`/`out_contrib_post_median`/`out_contrib_count` 3개 필드 추가(픽셀당 K=16 bounded slot, `config.h`의 `OSN_GS_MAX_CONTRIB_SLOTS`) — canonical 벤더 코드(`diff_surfel_rasterization/`)는 이번에도 무수정.
- `tests/test_surfel_representative_diagnostics.py`: 렌더 불변성 재검증 4개(`test_diagnostic_rendering_still_matches_canonical_after_contrib_provenance_addition` 신규 포함), 단일 표면 서펠은 항상 PRE_MEDIAN(`test_single_visible_surfel_is_pre_or_at_median_never_post`), 약하게 투과된 가려진 contributor는 POST_MEDIAN(`test_occluded_weakly_transmitted_contributor_is_post_median`), true uncapped count 검증(`test_contrib_count_matches_number_of_distinct_accepted_contributors`) — 총 14개, 모두 통과.
- `tests/test_nonrepresentative_evidence_attribution.py`(신규, CUDA 무관 순수 로직): pre/post 분류(3개), contributor-component 쌍 추출(3개), 전역 co-support 집계(4개), 카테고리 분류(1개) — 총 11개, 모두 통과.
- `.venv/Scripts/python.exe -m pytest tests/test_nonrepresentative_evidence_attribution.py tests/test_surfel_representative_diagnostics.py -q` → **25 passed**.
- 전체 pytest suite는 재실행하지 않았다(directive 지시: canonical training/topology 동작이 바뀌지 않았으므로 불필요) — canonical 벤더 CUDA, `torch_camera_induced_visible_adjacency.py`, 트레이닝 경로는 모두 무수정으로 유지됨을 코드로 확인.

## 다음 배치로 넘길 사항

이 배치는 attribution만 수행했고 attachment는 전혀 하지 않았다(directive Section 15 명시). Section 16 판정이 AMBIGUOUS/LAYERED SUPPORT이므로, 다음 아키텍처가 있다면 이 레이어드 증거를 강제로 하나의 컴포넌트에 귀속시키지 않는 방식으로 설계해야 한다는 것이 이번 배치의 결론이다 — 구체적 다음 단계 제안은 Master 문서 addendum에서만 다룬다(worklog 자체에는 넣지 않음, [[feedback_no_next_step_suggestions_in_worklogs]]).
