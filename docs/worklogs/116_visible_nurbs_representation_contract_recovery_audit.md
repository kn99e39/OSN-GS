# Worklog 116 — Visible-NURBS Representation Contract Recovery Audit

## 상태

**완료 — 감사(audit)만 수행, 구현/튜닝 없음(directive 지시).** Worklog 114(disjoint local rank-closed 추출)를 그 정확한 후보에 대한 유효한 NEGATIVE 결과로 받아들이되, 이를 "LOCAL NURBS REPRESENTATION IS NOT VIABLE"로 일반화하지 않는다. `osn_gs/surface/torch_nurbs.py`와 직접 소비자를 정밀 감사한 결과, **기존 NURBS fitter는 애초에 full-rank 관측 지지를 요구하지 않으며**(Tikhonov 항이 항상 시스템을 solvable하게 만듦), **`uv_support_mask`(UV trimming)는 이미 존재하지만 이번 계보(WL107-114)에서 한 번도 쓰이지 않았고**, **`fit_coupled_patch_graph_lsq`(shared-boundary 공동 fitting)라는 진짜 multi-patch 메커니즘도 이미 존재하지만 마찬가지로 미사용**임을 확인했다. WL111-114가 동결한 제약(blob 하나=chart 하나, 고정 8×4, ≥32 샘플, full-rank closure, disjoint 픽셀 소비)은 fitter 자체의 요구사항이 아니라 통제 실험을 위해 도입된 조작 선택이었다.

## Agent Interpretation of Intent

1. **DIRECTION 이해**: WL114의 특정 후보(disjoint rank-closed 추출)를 "local NURBS 자체가 실패"로 일반화하지 말고, 기존 `torch_nurbs.py`가 이미 제공하는 표현 시맨틱을 먼저 회수하라는 지시로 이해했다. WL111-114가 동결한 제약 중 어느 것이 fitter의 진짜 요구사항이고 어느 것이 통제 실험용 제약이었는지 구분하는 것이 핵심 작업이다.
2. **PURPOSE 이해**: WL114가 남긴 모순(국소화는 전형적 fit 품질을 크게 개선했으나, 특정 rank-closed disjoint 추출 규칙은 커버리지 손실·chart 폭증·cross-chart normal 불일치를 유발함)을 해소하는 것이 목적이라고 이해했다. 이 trade-off가 모든 local decomposition에 내재하는지, 아니면 이 특정 decomposition 규칙이 fitter가 이미 가진 능력(정규화·trimming·coupled fitting)을 전혀 쓰지 않아서 생긴 결과인지를 가리는 것.
3. **CENTRAL INTENT 이해**: "어떤 새 분할 휴리스틱을 구현할까"가 아니라 "이 코드베이스에서 NURBS patch가 원래 무엇을 의미하도록 설계됐고, 그 의미를 renderer-native visible surface evidence 위에서 표현할 기존 메커니즘이 있는가"를 묻는 것으로 이해했다. 새 메커니즘을 제안하려는 충동을 이번 배치 내내 억제했다.
4. **보존해야 할 것**: WL107/109 canonical topology(동결, 읽기전용), WL110의 AMBIGUOUS/LAYERED 판정(non-representative 증거 미부착 유지), WL114가 그 정확한 후보에 대해 유효한 negative 결과라는 사실(재실행하지 않음), 코드 동작 무변경.
5. **의도적으로 구현하지 않은 것**: 새 chart 성장 규칙, 분할 휴리스틱, merge/stitch, adaptive threshold, non-representative attachment, Trust, latent surface, visible termination, occluded surface. 이 보고서에는 새 메커니즘이 전혀 없다.
6. **유지돼야 한다고 믿는 semantic invariant**: (a) NURBS patch 경계는 representation seam이지 자동으로 물리적/위상적 경계가 아니다, (b) visible topology evidence와 NURBS-materialized evidence는 항상 별도로 보고되고 절대 혼동되지 않는다, (c) 모호한 증거는 강제로 하나의 소유로 밀어넣지 않는다, (d) canonical topology는 감사 배치에서 동결·읽기전용으로 유지된다.
7. **directive가 명시적으로 요구하지 않았지만 도입한 가정**: (a) "이 renderer-native 계보(WL107-114)에 실제로 연결된 것"과 "코드베이스에 존재하지만 옛 boundary-first architecture용으로 만들어진 것"을 구분하는 것이 중요하다고 판단했다 — 이 구분 없이는 "existing mechanism"이 살아있는 코드와 유물 코드를 섞어버린다. (b) `fit_torch_visible_surface_lsq`의 함수 시그니처 기본값(8×4, degree 2)을 WL111이 그대로 재사용했을 뿐 이 architecture용으로 재유도한 적이 없다는 것은 WL111 자신의 워크로그 텍스트에 근거했으며, git blame으로 독립 검증하지는 않았다. (c) `construct_visible_nurbs_from_gaussians`(train.py의 live NURBS 경로)가 이 브랜치의 실제 학습에서 정말 실행되는지 여부는 완전히 추적하지 못했다 — 단정하지 않고 미해결로 명시한다(§5 참조).
8. **prompt의 모호함**: §7이 "실제 existing NURBS implementation을 사용해 WL113/114 실패 양상을 재분류하라"고 지시하는데, WL113/114의 A/B/C/D는 애초에 이미 실제 정규화 fitter(`fit_torch_visible_surface_lsq`, 가상의 비정규화 fitter가 아님)로 측정된 수치다 — 따라서 "실제 구현으로 재분류"는 "측정된 수치가 보여주는 것"과 "만약 fitter가 실제로 비정규화·full-rank-only였다면 보여줬을 것"을 구분하는 작업이 됐다. 측정된 정규화-fit 수치 자체는 그대로 두고, WL111-114가 그 수치에 붙인 **개념적 프레이밍**(예: "32 샘플이 필요하다"는 프레이밍이 실제로는 fitter의 solve 자체가 아니라 WL111-114 자신의 외부 게이팅 로직의 요구사항이었음)만 재해석했다.

## Implementation Fidelity Statement

이번 배치는 진단 전용이다. 다음을 **읽었을 뿐 수정하지 않았다**: `osn_gs/surface/torch_nurbs.py`(전체 약 1337줄), `osn_gs/core/torch_pipeline.py`(`_assign_uv_support_masks`, `_uv_occupancy_mask`와 그 호출부), `osn_gs/surface/torch_trimmed_component_fitter.py`(전체), `osn_gs/surface/torch_patch_boundary.py`, `osn_gs/surface/torch_visible_surface_construction.py`(옛 boundary-first 계보임을 확인 — `OrderedBoundaryComponent`/`RegionFormationResult`/"worklog 129" 참조, `arch/2dgs-coverage-first-surface` 소속 아님; `fit_torch_visible_surface_lsq`나 `TorchNURBSSurface(...)`를 직접 호출하지 않음을 확인), `osn_gs/surface/torch_annulus_chart.py`/`torch_boundary_reconciliation.py`/`torch_occluded_chart.py`/`torch_region_owned_full_evidence.py`/`torch_visible_boundary_materialization_adapter.py`(`resolution_u` 기본값만 grep), `train.py`(`primitive=surfel_2d`도 `TorchPipelineConfig`를 거쳐 `construct_visible_nurbs_from_gaussians`를 호출함을 확인했으나, 이 브랜치의 실제 데이터셋에서 그 경로가 끝까지 실행되는지 vs `review_required`로 fail-closed하는지는 **완전히 추적하지 못했다 — 명시적으로 미해결로 남긴다**), `nurbs_constructor_benchmark/*.py`(grep만, `uv_support_mask` 소비자 확인용). **어떤 소스 파일도 수정하지 않았다.** 구체적 모호함을 해소하기 위한 새 테스트도 필요 없었다(코드 자체가 이번 감사가 필요로 한 모든 질문에 명확했다).

## 1. Worklog 114 Closure Rule 폐기

**명시적으로 기각한다**: `full column rank == valid local NURBS chart boundary`. `_solve_control_grid_lsq`(`torch_nurbs.py:730-769`)의 정확한 시스템:

```
system = normal_matrix/scale + smoothness_lambda*penalty + tikhonov_lambda*eye(n)
rhs    = normal_rhs/scale + tikhonov_lambda*seed
solution = torch.linalg.solve(system, rhs)   # 실패 시 lstsq로 폴백
```

`tikhonov_lambda*eye(n)`(기본값 1e-4) 항이 `normal_matrix`의 rank와 무관하게 `system`을 **항상 strictly positive-definite**로 만든다 — `normal_matrix=0`(어떤 control point에 대해 관측 기여가 0인 극단적 경우)에서도 마찬가지다. 이 시스템은 **항상 풀린다**. 게다가 Tikhonov 항은 0이 아니라 **IDW seed 자체에 anchoring**하도록 설계됐다(함수 자체 docstring: "sparsely covered control points follow the smooth seed rather than collapsing to origin", 줄 743-744) — 우연이 아니라 명시적 설계다.

세 개념을 정확히 분리한다: **data-matrix rank**(0~32 어디든 가능) / **정규화 시스템의 solvability**(rank와 무관하게 항상 보장) / **기하적 chart 타당성**(둘 중 어느 것도 보장하지 않음, WL115 §7이 지적했고 WL114의 D-outlier chart 자체가 실측으로 확인함 — full rank이면서도 기하적으로 병적). **결론**: full column rank는 좋은 정규화 fit의 필요조건이 아니다. WL111-114가 fitter를 호출하기 **전에** 외부적으로 부과한 게이트였을 뿐, fitter 자신은 그런 요구를 한 적이 없다.

## 2. 기존 NURBS Fitter 계약 (`osn_gs/surface/torch_nurbs.py`)

| 메커니즘 | 동작 | 분류 |
|---|---|---|
| IDW seed(`fit_torch_visible_surface`) | PCA 파라미터화 후 k=min(16,N) 최근접 관측점의 역거리가중 평균으로 채움. N=1/N=0 명시적 예외 처리. docstring 목적: "sparse COLMAP samples 완충." | **CORE** |
| 정규화 LSQ data term(`_lsq_normal_system`) | 청크 단위 정규방정식 조립, rank 체크 없음 | **CORE** |
| Smoothness 정규화(`_second_difference_penalty`, λ=1e-4) | control grid의 이산 2차미분 페널티(thin-plate류), 데이터 밀도와 무관 | **CORE** |
| Tikhonov anchoring(λ=1e-4) | seed에 anchor하는 ridge 항 — rank와 무관하게 solvability 보장 | **CORE — WL111-114가 전혀 관여하지 않은 정확한 그 메커니즘** |
| UV correction(foot-point projection) | grid init + damped Gauss-Newton, monotonic-improvement-only 채택 | **CORE** |
| Sparse/underdetermined 거동 | 최소 샘플 체크 전무. N=1/N=0만 특수 처리, 나머지는 32와 무관하게 동일 경로 | **32-샘플 하한이 fitter 계약이 아님을 확인** |
| Effective-degree(`_effective_degree`) | `degree=max(0,min(요청degree, control수-1))` — 작은 grid에서 자동 강등, 에러 아님 | **CORE — graceful degradation 설계** |
| Multi-patch: `fit_coupled_patch_graph_lsq`(877-1044) | 임의 patch graph + `SharedBoundaryConstraint`, 제약된 control point 쌍을 union-find로 solve **이전에** 하나의 공유 unknown으로 병합(사후 평균 아님) | **OPTIONAL CAPABILITY, EXPERIMENTAL LEGACY**(옛 annulus/boundary-first용, WL107-114 미사용) |
| Multi-patch: `fit_coupled_wedge_ring_lsq`(1046+) | 위의 cyclic-ring 특화판, "patch 경계 컬럼만 공유, G1/tangent 연속성은 별도의 필요시-단계(Step 5-B)로 명시적으로 미룸"을 자기 docstring에서 선언 | **HISTORICAL/UNUSED**, 그러나 "seam≠물리적 경계"를 이미 한 번 풀어본 직접적 선례 |

## 3. UV Support/Trimming — Materialization vs Fitting Coupling

**질문 A(MATERIALIZATION SUPPORT)**: **그렇다, 이미 해결됐다.** `TorchNURBSSurface.support(uv)`(`torch_nurbs.py:276-289`)는 순수 사후 boolean 조회이며, `torch_pipeline.py::_assign_uv_support_masks`(1410-1445)가 현재 할당된 점들의 occupied cell(dilate)로 계산한다. 자기 docstring: "sampling the untrimmed corners draws surface where there is no data... so downstream consumers can restrict the surface to its supported footprint."

**질문 B(FITTING COUPLING)**: **아니다, 해결되지 않았고 코드베이스 스스로 이미 안다.** `torch_trimmed_component_fitter.py::fit_trimmed_component`에서 mask는 fit이 끝난 **뒤**(130번째 줄) 할당된다(148번째 줄, `surface.uv_support_mask = refined_mask`). fit 자체는 항상 전체 `[0,1]²` tensor-product basis를 쓴다 — mask를 근거로 basis function/control point를 제거하거나 재가중하는 코드 경로는 어디에도 없다. 이 모듈 자신의 docstring이 정확히 명시한다: *"the control grid spans the component's whole rectangular UV domain and MAY cross the hole — topology is carried entirely by the trim mask, not by control-grid structure"*, 그리고 스스로를 *"a correctness baseline, not the final architecture"*라고 부른다.

**결론**: 현재 trimming은 **B1(비지지 도메인 materialization)**만 해결하고 **B2(구멍/오목 도메인을 가로지르는 fitting coupling)**는 해결하지 않는다 — 이는 이번 감사의 새 발견이 아니라, `torch_trimmed_component_fitter.py`를 쓴 사람들이 WL113가 독립적으로 재발견한(그리고 이 모듈을 참조하지 않은) 실패 B와 정확히 같은 간극을 이미 수년 전에 코드 주석으로 명시해 둔 것이다.

## 4. 용량(Capacity) 시맨틱스

고정 8×4는 **최종 architecture 계약이었던 적이 없다.** 코드베이스 전체에서 실제로 쓰인 서로 다른 resolution 기본값: `fit_torch_visible_surface(_lsq)` 8×4, `fit_coupled_patch_graph_lsq` **12×12**, `fit_coupled_wedge_ring_lsq`/`torch_annulus_chart.py` 8×4, `torch_boundary_reconciliation.py` 8, `torch_occluded_chart.py` **7**, `torch_region_owned_full_evidence.py` **6**, `torch_trimmed_component_fitter.py` **12×12**, `torch_visible_boundary_materialization_adapter.py` **6×6**(호출부에서 명시). 6개의 서로 다른 값이 등장하며, 데이터로부터 resolution을 유도하는 메커니즘은 어디에도 없다 — 전부 작성자가 그때그때 고른 상수다. **결론**: capacity는 항상 호출부별 선택이었지 architecture적 법칙이었던 적이 없다. WL111이 8×4를 고른 이유는 `fit_torch_visible_surface_lsq`의 "가장 적게 코드를 써서 호출할 수 있는" 함수 시그니처 기본값이었기 때문이지 특별한 지위를 가져서가 아니다.

## 5. Multi-Patch/Seam 시맨틱스

OSN-GS는 이미 "하나의 표면/컴포넌트 → 여러 representation patch"이며 patch 경계가 물리적 종결로 자동 해석되지 않는 architecture를 가지고 있다: `fit_coupled_wedge_ring_lsq`의 docstring이 정확히 이 구분을 이름으로 명시한다(공유 boundary 컬럼은 공동 unknown, 물리적 불연속 아님; 경계를 가로지르는 완전한 G1/tangent 연속성은 별도로 범위 지정된 "Step 5-B"로 명시적으로 미뤄져 한 번도 구현되지 않음). 이것은 WL114의 central intent가 새롭게 재진술한 바로 그 semantic invariant("patch 경계는 representation seam")를 이미 한 번 풀어본 실제 선례다.

**단, 이 메커니즘(`fit_coupled_patch_graph_lsq`, `fit_coupled_wedge_ring_lsq`, `SharedBoundaryConstraint`)은 옛 boundary-first/annulus architecture용으로 만들어졌고, WL107-114 renderer-native 계보에서 한 번도 호출된 적이 없다.** directive §5 지시대로, 이번 배치에서 이를 새로 연결하지 않는다 — 존재하며 이 계보에 대해 현재 휴면 상태라는 사실만 기록한다.

## 6. 현재 증거 → 기존 NURBS 요구사항 호환성 행렬

| 현재 증거 | 기존 NURBS 요구/능력 | 분류 |
|---|---|---|
| median representative(WL107, 대표당 3D 점 1개) | IDW seed/LSQ data term은 3D 점만 필요 | **직접 호환** |
| per-pixel median surface position(WL112, 조밀한 픽셀당 3D) | 동일 — 점 개수 무관 | **직접 호환**(오히려 fitter에 샘플 하한이 없으므로 더 적합할 수 있는 입력) |
| image-space adjacency(WL107, blob/component 연결성 정의) | fitter는 adjacency 개념이 전혀 없음 — 순서 없는 점 집합 + 선택적 initial UV만 소비 | **표현 선택에 종속된 호환** — adjacency는 fitter 이전 단계에서 어떤 형태로든 UV로 번역돼야 함 |
| visible component ID(WL107/109 canonical topology) | fitter는 component ID를 직접 쓰지 않음(어떤 점들을 한 번의 fit 호출에 넣을지는 상위 결정) | **표현 선택에 종속된 호환** |
| per-view image 좌표를 UV로(WL111 §4) | fitter는 `initial_uv`로 PCA 파라미터화를 대체하는 것을 이미 지원 | **직접 호환** — 정확히 의도된 확장점 |
| ambiguous non-representative contribution(WL110, AMBIGUOUS/LAYERED) | fitter의 선택적 `point_weights` 인자(nan/inf 정리, ≥0 clamp)는 강제 소유 없이 *부드러운* 비이진 포함을 표현할 수 있을지도 모름 | **미해결** — 이론적으로 가능하지만 WL110-114 어디서도 시도된 적 없음; WL110의 판정은 *소유권*에 관한 것이지 *fitting weight*에 관한 것이 아니므로 둘이 동등하다고 증명되지 않았다. 열린 질문으로만 남긴다. |

## 7. A/B/C/D 재분류

**A. SUPPORT-LIMITED** — WL113은 "*비정규화·≥32-샘플* 8×4 control에 불충분"(자신이 외부적으로 부과한 게이트)을 측정했다. 실제 *정규화* fitter는 그런 하한이 없다(§1/§2) — 관측 2개짜리 컴포넌트도 fit 가능하다(퇴화적이지만, `_effective_degree`가 자동으로 낮춤; 1개면 상수 grid). **재분류**: "불충분한 관측"이라는 실측 자체는 진짜지만, "32"라는 숫자와 이진 통과/실패 게이트는 **WL111의 외부 게이팅 로직의 산물**이지 fitter가 실제로 요구하는 것이 아니다. 그 "불충분한" 컴포넌트에 정규화 fit을 시도하면 (불확실하더라도) 어떤 답이든 나온다 — 그 답이 *유용한지*는 이번 감사가 답하지 않는 별개의, 여전히 열린 실증적 질문이다.

**B. RECTANGULAR-DOMAIN FAILURE** — **B1(materialization)뿐**임을 확인, B2(fitting coupling)는 아님(§3). 기존 `uv_support_mask`를 `torch_trimmed_component_fitter.py`와 동일하게 연결하면 B1의 *가시적* 증상(데이터 없는 곳에 표면이 그려짐/측정됨)은 새 메커니즘 없이 없앨 수 있을 개연성이 있다. 그러나 진짜로 이분된/비연결된 지지 영역에 하나의 공유 control grid가 적절한지라는 더 깊은 질문(B2)은 그대로 남는다.

**C. FIXED CAPACITY FAILURE** — **의도적으로 동결된 Worklog 용량**임을 확인, 어떤 기존 variable-capacity 시맨틱의 산물도 아니다(§4 — 그런 메커니즘 자체가 없다). WL113의 좁은 신호(full-rank chart의 residual이 오히려 낮음)라는 실측 자체는 그대로 유지되며, 이번 감사는 그 해석만 바꾼다 — "C"는 "NURBS 용량이 architecture적으로 틀렸다"가 아니라 "이 특정 동결된 숫자가 재유도된 적이 없다"로 읽어야 한다.

**D. NUMERICAL/GRAZING FAILURE** — 지시대로 분리 유지. fitter 감사 결과 WL115의 D 분류(주로 렌더러 데이터 현상 V, 부차적으로 adjacency의 내부 depth-연속성 부재 II)를 뒤집을 근거는 없다. fitter 자신의 시맨틱과는 무관하다 — 병적인 점이 주어져도 fitter는 충실하게 그것에 fit할 뿐이다.

**새로운 "E" 메커니즘은 도입하지 않았다.**

## 8. Worklog 114 재해석

"LOCAL CHART UNIT NOT VIABLE"을 무조건적으로 다시 보고하지 않는다.

| 증거 | 뒷받침하는 주장 |
|---|---|
| residual 중앙값/p95 ~9배/~8배 개선; 구멍 있는 chart 비율 46.1%→15.7%; aspect ratio p95 3.36→1.22 | **LOCALITY IS USEFUL** — 더 작고 기하적으로 컴팩트한 영역이 실제로 더 잘 fit되고 도메인 모양도 좋아진다, *어떻게* 뽑아냈는지와 무관하게 |
| 커버리지 11.7% 감소(다섯 영역 전부); 79,317개 `TOO_FEW_PIXELS`, 2,418개 `INSUFFICIENT_RANK_CLOSURE`로 좌초 | **RANK-CLOSED DISJOINT EXTRACTION IS NOT VIABLE** — (a) 정확히 32-샘플 full-rank closure를 요구(§1: fitter가 요구한 게 아님)하고 (b) 픽셀을 *disjoint하게* 소비해 남은 경계 조각을 다시 방문하거나 이웃 patch에 포함시킬 수 없게 만든 것의 직접적·기계적 결과. 둘 다 기존 fitter가 요구하지 않는다. |
| chart 수 15.9배; 한 컴포넌트에 최대 10,776개 집중 | 그 자체로는 모호함 — 진짜 국소 기하 복잡도를 반영할 수도, disjoint 소비 규칙이 (병합/완화된 멤버십이었다면 더 적고 더 잘 지지되는 patch로 남았을 것을) 파편화시킨 결과일 수도 있다. **이번 감사로 해결되지 않음, 열린 질문으로 남김.** |
| overlap 법선 불일치 크게 악화(중앙값 5.8°→18.2°, p95 59.3°→96.5°) | **자동으로 "seam이 늘어서"가 아니다.** Directive가 명시적으로 이 귀속을 경고했다. 실제 측정이 보여주는 것: 15.9배 많은 chart가 **서로 완전히 독립적으로** fit됐다(shared-boundary coupling은 전혀 쓰이지 않았다 — §5의 `fit_coupled_patch_graph_lsq` 메커니즘은 존재하지만 WL114 구현이 호출한 적이 없다). 더 정확하고 더 잘 뒷받침되는 인과 주장은 **독립적(비결합) per-chart fitting이, chart 밀도가 높아지면서 인접 chart끼리 불일치할 기회를 늘렸다**는 것이다 — 이것은 "coupling의 부재"에 관한 주장이지 일반적인 "seam 개수"에 관한 주장이 아니다. coupling이 이를 고칠지는 미검증(이번 감사 범위 밖). |

**교정된 결론**: Worklog 114는 그 정확한 후보(disjoint rank-closed 추출)에 대한 유효한 negative 결과다, directive가 말한 그대로. 그러나 이를 일반적인 locality에 대한 반증으로도, 더 미세한 분해가 본질적으로 cross-chart consistency를 해친다는 증거로도 읽어서는 안 된다 — **기존 coupled-fitting 메커니즘을 전혀 쓰지 않은 채 다수의 작은 chart를 fit한 것**이 일관성을 해쳤다는 증거다. 이것은 서로 다른 주장이며, 더 좁은 쪽만 뒷받침된다.

## 9. 단순성 감사 — 개념적 architecture에서 삭제 가능한 실험적 제약

| 제약 | 진짜 NURBS 요구사항인가? | 판정 |
|---|---|---|
| `MIN_CHART_MEMBERS=32`/`MIN_PIXEL_SAMPLES=32` | 아니오(§1/§2 — 최소 샘플 하한 없음) | **architecture 법칙으로서는 삭제.** "이 아래에서는 fit의 분산이 크다"는 진단 임계값으로는 남을 수 있으나, fitting 시도 자체를 막는 하드 게이트로는 아니다. |
| full-rank closure | 아니오(§1) | **삭제.** 이번 감사가 명시적으로 폐기. |
| `_RANK_CHECK_STEP` | 해당 없음 — 이제 폐기된 closure 규칙의 부산물 | **함께 삭제.** |
| disjoint local 픽셀 소비 | 아니오 — 픽셀이 정확히 하나의 patch에만 속해야 한다는 요구는 어디에도 없음; `uv_support_mask`와 coupled fitting 둘 다 patch가 겹치거나 경계를 공유할 수 있음을 전제로 함 | **재검토** — fitter 요구사항이었던 적 없음, WL114 자신의 추출 편의였을 뿐 |
| 고정 8×4를 architecture 법칙으로 | 아니오(§4 — 코드베이스에 6개의 서로 다른 기본값 존재) | **법칙으로서는 삭제; 다음 배치의 통제 비교를 위한 합리적 출발 기본값으로는 유지 가능.** |
| blob 하나=chart 하나 | 아니오 — WL111 자신이 "가장 단순한 첫 테스트"로 명시 | **WL114 스스로 대안을 테스트함으로써 이미 사실상 폐기됨; 기본 가정으로 되살리지 말 것.** |
| architecture 법칙으로서의 per-view patch 정체성 | fitter는 요구하지 않지만(chart 내용에 무관함), 매 배치가 보고해 온 overlap 불일치의 직접적·필연적 원인(WL115 §6) | **재검토** — `fit_coupled_patch_graph_lsq`가 정확히 "독립 fit 후 불일치"를 피하기 위해 만들어진 shared-boundary 공동 unknown 메커니즘을 이미 제공하며, 이 계보에 단 한 번도 연결된 적이 없다 |

## 10. 표현 계약 표

| 기존 메커니즘 | 원래 의미론적 역할 | 현재 구현 상태 | Renderer-Native Topology와 호환? | Keep/Reuse/Retire/Reconsider | 근거 |
|---|---|---|---|---|---|
| 정규화 LSQ(data+smoothness+Tikhonov) | full rank 없이도 부드러운 표면 fit | 살아있음, 무수정, WL111-114 전 배치가 동일하게 사용 | 예 — 이미 사용 중인 메커니즘 | **Keep** | `torch_nurbs.py:730-769` |
| IDW seed | sparse/불규칙 COLMAP류 밀도 허용 | 살아있음, 무수정 | 예 | **Keep** | `torch_nurbs.py:590-650` |
| UV correction(foot-point projection) | 초기 추정 이후 UV 파라미터화 개선 | 살아있음, 무수정 | 예 | **Keep** | `torch_nurbs.py:1203+` |
| `uv_support_mask` | 비지지 UV 코너가 표면으로 그려지는/측정되는 것 방지 | 옛 boundary-first 파이프라인(`torch_pipeline.py`, `torch_trimmed_component_fitter.py`)에서 살아있음; **WL107-114에서는 한 번도 사용 안 됨** | 직접 적용 가능 — 적응 불필요, 연결만 필요 | **Reuse**(materialization만 — §3의 B1/B2 유보 참고) | `torch_nurbs.py:225,276-289`; `torch_pipeline.py:1410-1445`; `torch_trimmed_component_fitter.py:118-148` |
| Variable control-grid capacity | 적응적 메커니즘으로 존재한 적 없음 | 코드베이스 전체에 6개의 서로 다른 손수 고른 상수, 데이터 유도 없음 | 해당 없음 — 재사용할 것 자체가 없음 | **Reconsider**(재사용이 아니라 진짜 열린 설계 질문으로) | §4 표 |
| Multiple patches(`fit_coupled_patch_graph_lsq`) | 하나의 컴포넌트 → 진짜 shared-boundary unknown을 가진 여러 결합 patch | 옛 annulus/boundary-first 계보에 살아있음; **WL107-114에서 한 번도 호출 안 됨** | 개연적으로 호환 — topology-무관 함수(임의 점/UV 리스트 + 제약 리스트를 받음) | **Reconsider**(WL114가 결여했던 coupling의 진짜 후보, renderer-native 증거로 미검증) | `torch_nurbs.py:877-1044` |
| Shared-boundary/coupled fitting(`SharedBoundaryConstraint`, `fit_coupled_wedge_ring_lsq`) | seam≠물리적 경계, 이미 한 번 풀어 명시적으로 코드화 | 옛 annulus 전용, 이 계보에서 미사용 | 개념적으로 호환; wedge-ring 특화판은 topology-특정적(cyclic) | **Reconsider**(graph 버전) / **Retire**(wedge-ring 특화판, 이 계보에 너무 특정적) | `torch_nurbs.py:1046+` |
| Persistent Gaussian-surface binding(`cluster_ids`, `surface_uv`, `surface_owner_kind`) | 어떤 Gaussian이 어떤 patch에 속하는지 추적, ownership-gated | `torch_pipeline.py`(옛 파이프라인의 live 학습 루프)에 살아있음 | 직접 적용 불가 — 이 계보의 소유권 개념은 canonical topology `subset_ids`, `cluster_ids`와는 다른(renderer-native) 메커니즘 | **Retire**(이 계보에 한해 — 병렬적이지 공유되지 않는 소유권 개념) | `torch_pipeline.py:1436-1445` |

## 11. 최소 현재 Visible-NURBS Architecture

**증명된 시맨틱**: fitter는 정규화돼 있으며 답을 내기 위해 full rank나 어떤 최소 샘플 수도 요구하지 않는다 / `uv_support_mask`는 materialization trimming(B1)을 해결하지만 fitting coupling(B2)은 해결하지 않는다(코드 자체의 실행 순서와 저자 자신의 docstring으로 증명됨) / control-grid capacity는 이 코드베이스 역사상 단일 architecture 상수였던 적이 없다 / patch seam을 물리적 경계와 별개로 다루는 진짜 coupled multi-patch fitting이 이미 존재하며, 이미 한 번(다른 계보에서) 풀렸으나 이 계보에 연결된 적은 없다.

**미해결 표현 선택**: `uv_support_mask` 재사용(B1)만으로 충분한지, renderer-native 증거를 잘 표현하려면 B2(오목/이분 도메인의 fitting coupling)도 다뤄야 하는지 / `fit_coupled_patch_graph_lsq`의 shared-boundary 메커니즘을 renderer-native 증거에 적용하면 WL114의 overlap-normal 문제를 실제로 해결하는지, 아니면 새 문제를 만드는지(전혀 검증된 바 없음) / 하나의 canonical topology 컴포넌트를 multi-patch로 분해할 때 어떤 UV 파라미터화를 써야 하는지 — image 좌표(WL111의 canonical 선택)는 한 컴포넌트가 여러 뷰에 걸쳐 있을 때 하나의 per-view chart처럼 자명하게 확장되지 않는다 / point-weight 기반의 모호한(WL110) 증거의 부드러운 포함이 AMBIGUOUS/LAYERED "강제 소유 금지" 판정과 양립하는지, 모순되는지 / renderer-native 증거 전용으로 임의 고정 resolution을 대체할 어떤 양이 있는지 — 기존 메커니즘 중 답하는 것이 없다, 발견이 아니라 결정돼야 한다.

## 12. 다음 구현 배치를 이끌어야 할 단 하나의 미해결 architecture 질문

**기존 `uv_support_mask`(materialization trimming)를 현재의 blob-하나=chart-하나 WL112 베이스라인에 다른 변경 없이 그대로 적용하는 것만으로 Worklog 113의 실패 B가 충분히 해소돼 chart-단위 재설계가 더 필요 없는가, 아니면 B2 fitting-coupling 간극(§3) 때문에 trimming만으로는 고칠 수 없어서 coupled multi-patch fitting(`fit_coupled_patch_graph_lsq`)이 또 다른 local-growth 휴리스틱 대신 다음에 필요한 메커니즘이 되는가?**

이 질문은 새 chart-분해 메커니즘을 결정하기 **전에** 작고 좁게 범위 지정된 통제 실험(WL112의 기존 blob에 `uv_support_mask`를 적용하고 B의 증상을 재측정)으로 답할 수 있으며, 튜닝이나 메커니즘 중첩 없이 이번 감사의 증거 기반이 다음 방향을 정당화하기에 충분한 유일한 질문이다.

## 테스트

정적/의미론적 architecture 감사이며 전체 regression을 실행하지 않았다(directive 지시). 구체적 모호함을 해소하기 위한 기존 focused test 실행도 필요하지 않았다 — 코드 자체가 이번 감사가 필요로 한 모든 질문에 명확했다. 어떤 코드도 수정하지 않았다.
