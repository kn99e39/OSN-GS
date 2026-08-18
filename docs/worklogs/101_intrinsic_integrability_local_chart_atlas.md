# Worklog 101 — intrinsic-integrability-driven local chart atlas

## 상태

**완료 — Decision B: VALID_CHARTS_EXIST_BUT_CURRENT_PATCH_MODEL_FAILS.** Worklog 100은 Worklog 98의 coherent component 전체를 하나의 전역 chart로 강제하는 것 자체가 다수 component에게 잘못된 scale임을 실측으로 확인했다(global differential integration이 domain-valid를 15→18/46로만 늘렸고, 그 위의 local-injectivity 보정은 단 하나도 추가로 구제하지 못했다). 이번 배치는 "하나의 component = 하나의 chart"라는 가정 자체를 버리고, 같은 continuously-supported source graph 위에서 결정론적으로 성장시킨 **여러 개의 겹칠 수 있는 local chart로 이루어진 atlas**로 대체했다. 실측 결과는 뚜렷하다 — **chart 단위 domain validity는 100%(115/115)**로, 기존 단일-전역-chart 방식의 39.1%(18/46, Worklog 100 candidate B와 동일 수치)를 압도한다. 하지만 그렇게 만들어진 chart의 **92.2%(106/115)는 고정된 6×6/degree-2 control grid가 필요로 하는 최소 evidence(36점)에도 못 미치는** 너무 작은 크기(median 9점)였고, 실제로 fit이 시도된 나머지 9개(7.8%)는 전부 unsafe(8개) 또는 extrapolative(1개)로 끝나 **valid_supported는 여전히 0%**다. Domain 구성 문제는 chart 단위에서 사실상 해소됐지만, 그 위에 얹힌 고정 patch representation(6×6 tensor-product NURBS)이 병목이 됐다 — 다음 architecture gate는 patch representation/fitting model 자체를 다뤄야 한다. Visible Gaussian training, ADC, region ownership, Worklog 95 latent-surface estimator, continuous support contract, Worklog 98 synchronized tangent-frame field, Worklog 100 symmetric edge differential·global differential integration·수정된 source-graph validator, held-out evaluation, 기존 6×6/degree-2 NURBS fitter는 모두 미변경이다. Chart 생성·성장·종료 어디에도 fit 오차·held-out 오차·extrapolative/unsafe 분류를 사용하지 않는다(AST로 검증).

## 구현

### 1~4. Chart atlas 구성

신규 `osn_gs/surface/torch_intrinsic_chart_atlas.py`: chart는 component의 **동일한** continuously-supported source graph(Worklog 100의 `_source_graph_adjacency`가 이미 검증한 tree edge ∪ holonomy edge)의 부분집합이다 — Euclidean bounding box, PCA rectangle, convex hull, 물리적 경계 폐곡선을 전혀 쓰지 않는다. 구성 절차:

1. **결정론적 anchor**: 첫 chart는 centroid-최근접 node(field 자신의 un-anchored 관례와 동일), 이후 chart는 현재 coverage로부터 BFS graph-거리가 가장 먼 미커버 node(동률은 최소 index로 확정).
2. **BFS hop-count ring 성장**: anchor로부터의 홉 거리로 ring 0, 1, 2, …를 구성(튜닝된 Euclidean/PCA 반경이 아니라 순수 그래프 위상).
3. 매 ring마다 그 ring 내부의 **모든** 지원 edge(스패닝 트리가 아니라 전부)로 Worklog 100의 `integrate_global_differential_uv`를 다시 풀고, 수정된 `assess_parametric_domain_validity`로 검증한다.
4. 유효한 동안 계속 성장하고, 처음으로 무효가 되는 ring 직전(마지막 유효 ring)을 그 anchor의 maximal chart로 확정한다. Validator를 느슨하게 하거나 NURBS/held-out 결과로 반경을 찾지 않는다.
5. 모든 chartable node가 최소 1개의 유효 chart에 덮이거나, 남은 연결된 미커버 evidence가 최소 chart 크기(3)보다 작아질 때까지 반복한다. 커버되지 않은 evidence는 명시적으로 report하며, 100% 커버리지를 맞추려고 가짜 chart를 만들지 않는다.

### 4. 겹침(overlap)

Chart 성장은 이미 커버된 node를 배제하지 않으므로 인접 chart는 그래프가 허용하는 만큼 자연스럽게 겹친다(강제 hard partition 아님). 두 chart를 가르는 edge, 또는 chart와 미커버 node를 가르는 edge는 `seam_edges`로만 report되며, 물리적 termination/crease/observation boundary 의미는 전혀 부여하지 않는다(테스트로 확인 — `AtlasResult`의 필드명에 boundary/crease/feature가 없음을 직접 검사).

### 5. Synchronized frame 보존

`_restrict_component`는 chart에 남는 모든 node에서 부모 component의 `e_u`/`e_v`/`normals`를 그대로 복사한다 — 새 tangent field를 독립적으로 추정하지 않는다(PCA 재추정 없음, 테스트로 확인). Chart 고유의 것은 자신의 candidate-B 적분으로 얻는 `(u,v)` gauge뿐이다.

### 6. Chart별 intrinsic parameterization

각 chart는 자신의 internal supported edge **전부**(스패닝 트리 아님)로 Worklog 100의 `integrate_global_differential_uv`를 다시 풀고, 수정된 validator로 검증한다. Hard gate는 local injectivity/non-degeneracy뿐이며, 이번 배치에서 새로운 replay-tuned distortion threshold는 도입하지 않는다.

### 7. Chart-restricted curve lattice

신규 `osn_gs/surface/torch_chart_curve_lattice.py`: 새 curve-seeding 규칙을 만들지 않고 Worklog 98의 `build_curve_lattice`를 chart로 제한된 component view에 **그대로** 호출한다. 유일한 추가 동작은 사후 truncation이다 — tracer가 (chart가 아닌) 전체 latent-surface support field를 통해 연속적으로 걷기 때문에 chart 경계를 넘어갈 수 있다; curve의 각 지점이 원본 component에서 가장 가까운 node가 더 이상 그 chart의 멤버가 아니게 되는 순간부터 truncate하고 `PARTITION_SEAM`으로 표시한다. **구현 중 발견한 버그**: bidirectional curve(`[…backward, anchor, forward…]`로 이어붙여짐)를 배열 왼쪽부터 한 방향으로 스캔해 첫 위반에서 자르면, backward 절반이 먼저 chart를 벗어났을 때 완전히 유효한 forward 절반까지 통째로 버려진다 — anchor의 실제 배열 위치를 3D 좌표로 역으로 찾아 forward/backward를 독립적으로 바깥쪽으로 truncate하도록 수정했다.

### 8. NURBS는 이번 배치에서 downstream probe로만 사용

Domain-valid chart마다 기존 `fit_torch_visible_surface_from_uv`(degree=2, 6×6, 기존 정규화 solver, 기존 안전 기준)를 그대로 사용한다. 6×6 grid 자체의 제어점 개수(36)를 밑도는 chart는 `CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT`로 별도 report하고 `PARAMETER_DOMAIN_INVALID`와 절대 혼동하지 않는다(테스트로 확인) — 이 임계값(36)은 고정 6×6 grid 자체의 구조적 사실이지 replay로 조정한 값이 아니다.

### 9~11. 비교 + real replay

신규 `scripts/devtools/intrinsic_chart_atlas_gate_replay.py`가 동일 checkpoint 2900/final, 동일 7-region evidence·synchronized field에 대해 A(SINGLE_COMPONENT_GLOBAL_UV = Worklog 100 candidate B, 무변경)와 B(LOCAL_INTRINSIC_CHART_ATLAS, 이번 배치)를 fallback 없이 비교한다. Component-count 기준과 evidence-weighted 기준을 모두 report하고, chart별 겹침(overlapping chart)에 대해서는 별도 ownership/reconciliation 로직을 새로 만들지 않고 각 chart의 결과를 독립적으로 report한다(최우수 겹침 patch를 골라 headline 수치를 부풀리지 않음).

## 검증

신규 focused 테스트 18개: `test_intrinsic_chart_atlas.py` 13개(평면 component→chart 1개 전체 커버, 국소적으로 injective한 완만한 곡면→chart 1개, 전역적으로 접히는 component→다중 유효 chart, partition 일관성(covered/uncovered/unchartable이 정확히 분할), 그래프 기반 ring 성장, farthest-uncovered anchor 결정론성, ring 겹침 허용, NURBS/held-out 오차 비의존(AST, docstring 제외 후 검사), PCA 미사용, synchronized frame이 restriction을 통해 보존됨, spanning tree가 아닌 chart 내부 전체 edge 사용, 새 curve-seed 생성 없음, seam edge가 물리적 경계 의미를 갖지 않음), `test_chart_curve_lattice.py` 3개(각 chart에서 lattice 유효, 기존 builder 재사용 확인, truncate된 curve의 모든 점이 실제로 그 chart 멤버임), `test_intrinsic_chart_atlas_gate_replay.py` 2개(작은 domain-valid chart가 PARAMETER_DOMAIN_INVALID로 잘못 분류되지 않고 CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT로만 분류됨, 충분한 evidence는 실제 fit 시도에 도달함). 전체 회귀 실행함(아래).

구현 중 발견/수정한 버그 2건: (1) `_farthest_uncovered`가 실패한 anchor(`unchartable_seeds`)를 제외하지 않아 같은 실패 anchor를 반복 선택 → coverage guard 한계까지 공회전하다 조기 종료 — `excluded` 인자로 수정. (2) 위의 bidirectional curve truncation 버그.

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(held-out evidence 1915, 실행 530초), final(held-out evidence 3173, 실행 798초). 산출물: `output/extent_ab/val101/intrinsic_chart_atlas_gate_replay_2900.json`, `intrinsic_chart_atlas_gate_replay_final.json`.

### Domain validity: 단일 전역 chart(A) vs atlas(B)

| checkpoint | A: coherent component | A domain-valid | B: 총 chart 수 | B domain-valid chart |
|---|---:|---:|---:|---:|
| 2900 | 29 | 11 (37.9%) | 73 | **73 (100%)** |
| final | 17 | 7 (41.2%) | 42 | **42 (100%)** |
| combined | 46 | 18 (39.1%) | 115 | **115 (100%)** |

Atlas는 combined 39.1%였던 domain-valid 비율을 chart 단위 100%로 끌어올렸다 — 도달하지 못한 evidence는 아예 chart를 만들지 못한(anchor 자체가 최소 ring에서도 무효) `unchartable_seed`로 명시적으로 분리 보고됐을 뿐, "유효하지 않은 chart"는 단 하나도 없다.

### Atlas 구조

| 지표 | 2900 | final | combined |
|---|---:|---:|---:|
| 1개 chart로 표현된 component | 14 | 8 | 22 |
| 다중 chart가 필요한 component | 14 | 9 | 23 |
| chart를 하나도 못 만든 component | 1 | 0 | 1 |
| 총 source node | 734 | 507 | 1241 |
| 유효 chart 하나 이상으로 커버된 node | 618 | 471 | 1089 (87.8%) |
| 2개 이상 chart에 겹쳐 커버된 node | 203 | 100 | 303 |
| 어떤 chart로도 커버 안 된(unchartable) node | 116 | 36 | 152 (12.2%) |
| seam edge(순수 parametric, 물리적 경계 아님) | 809 | 271 | 1080 |
| chart 크기 median / p95 | 9 / 32 | 8 / 40 | 9 / 40 |

Node 기준으로 미커버(uncovered) evidence는 0건이다 — 모든 node가 최종적으로 "어떤 유효 chart에 커버됨" 또는 "그 어떤 anchor로도 chart를 못 만드는 unchartable seed"로 분류됐다.

### Downstream NURBS(고정 6×6/degree-2, downstream probe로만 사용)

| 범주 | combined chart 수 | 비율 |
|---|---:|---:|
| CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT | 106 | 92.2% |
| FIT_SUCCEEDED_BUT_UNSAFE | 8 | 7.0% |
| FIT_SUCCEEDED_BUT_EXTRAPOLATIVE | 1 | 0.9% |
| VALID_SUPPORTED | 0 | 0.0% |

Domain-valid 115개 chart 중 **92.2%(106개)는 6×6 grid 자체가 요구하는 최소 evidence(36점)에도 못 미치는** median 9점짜리 소형 chart였다. 실제로 fit이 시도된 나머지 9개(7.8%, 크기 36~84점)는 8개가 unsafe, 1개가 extrapolative로 끝났다 — valid_supported는 combined 0%다. Evidence-weighted 기준으로도 CHART_DOMAIN_VALID_BUT_INSUFFICIENT_PATCH_SUPPORT가 2900에서 94.67%, final에서 78.12%를 차지한다.

## 결정

**B. VALID_CHARTS_EXIST_BUT_CURRENT_PATCH_MODEL_FAILS.** Chart-domain validity는 39.1%(단일 전역 chart)에서 100%(atlas)로 광범위하게 해소됐다 — intrinsic parameterization 자체의 문제는 chart scale에서 사실상 풀렸다. 하지만 그 결과의 92.2%가 고정 6×6 NURBS grid에 필요한 최소 evidence보다 작은 chart였고, evidence가 충분했던 나머지 9개마저 전부 unsafe/extrapolative로 끝나 valid_supported는 여전히 0%다. **Intrinsic parameterization 문제는 operationally 해소된 것으로 선언하고, 다음 architecture gate는 local/adaptive parametric patch representation을 다뤄야 한다.** 6×6 fitting을 만족시키기 위해 chart를 인위적으로 더 키우지 않는다 — chart 크기는 이미 intrinsic integrability가 결정한 결과이며, 다음 단계는 그 크기에 맞는 patch representation을 찾는 것이다.
