# Worklog 56: Phase 5 Step 5-A — Coupled Patch-Boundary Fitting Production 채택

날짜: 2026-07-22

상태: **PRODUCTION 채택 완료 (사용자 승인).** `build_annulus_chart`의 `coupled_boundary_fit` 기본값을 `True`로 전환. Step 5-B(soft G1) 진행 여부는 별도 지시 대기.

## 배경

Worklog 55에서 Step 5-A(coupled patch-boundary fitting)를 구현·평가한 결과, 4씬 x 5시드 전체에서 orientation flip이 정확히 0으로 사라지고(region별 분리 카운트도 전부 0), chamfer_rms는 모든 씬에서 동일하거나 개선, false_fill은 거의 변화 없는 매우 깨끗한 결과를 얻었다. 사용자가 이를 즉시 production으로 채택하기로 결정했다("일단 지금 Production으로 채택하는 것으로 하고, Step-5B의 진행 여부에 대해선 추후 지시할 때까지 대기하고 있어").

## 변경 사항

- `osn_gs/surface/torch_annulus_chart.py`: `build_annulus_chart`의 `coupled_boundary_fit` 기본값 `False` → **`True`**. Docstring을 "PRODUCTION ADOPTED, 2026-07-22, user-approved"로 갱신하고, 이전 hard-C0 post-hoc-overwrite 패턴과의 구조적 차이 설명은 유지.
- `nurbs_constructor_benchmark/boundary_first.py`: `construct_boundary_first`의 `annulus_coupled_boundary_fit` 기본값도 동일하게 `True`로 전환.
- `nurbs_constructor_benchmark/runner.py`: CLI 플래그를 opt-in(`--bf-coupled-boundary-fit`, 기본 꺼짐)에서 **opt-out**(`--bf-disable-coupled-boundary-fit`, 기본 꺼짐 = coupled fit 사용)으로 전환 — worklog 49가 eligibility filtering을 채택할 때 CLI 토글 자체를 추가하지 않았던 것과 달리, 이번엔 비교/ablation 목적으로 되돌릴 수 있는 스위치를 유지했다(기존 `--no-stage1-boundary-refinement` 네이밍 관례를 따름).
- 이전 독립 per-wedge fitting 경로는 `coupled_boundary_fit=False`로 여전히 사용 가능(삭제하지 않음, 검증된 fallback/ablation 경로로 유지).

## 테스트 영향 및 수정

기본값 전환으로 인해 "기본값 = 독립 fitting"을 가정하던 기존 테스트 2개가 실패했다(사전에 예상된 영향, 새 값으로 갱신):

- `test_coupled_boundary_fit_default_off_is_unchanged` → `test_coupled_boundary_fit_is_now_the_default`(기본값이 이제 coupled임을 확인) + `test_coupled_boundary_fit_false_recovers_independent_fit`(명시적 `False`로 이전 경로가 여전히 동작하고, 공유 컬럼이 더 이상 강제로 일치하지 않음을 확인)로 교체.
- `test_known_bad_seed_reproduces_inner_corner_degeneracy` (seed=14 고정 시나리오, 8개 flip 재현 검증) → `test_known_bad_seed_reproduces_inner_corner_degeneracy_under_independent_fit`(명시적 `coupled_boundary_fit=False`로 예전 실패 모드를 그대로 detection guard로 유지) + 신규 `test_known_bad_seed_is_resolved_by_default_coupled_fit`(동일 seed=14 픽스처가 새 기본값에서는 flip=0으로 해소됨을 직접 확인 — worklog 55의 멀티씬 결과를 이 unit fixture에서도 재현).

`--bf-coupled-boundary-fit`을 코드 전체에서 `--bf-disable-coupled-boundary-fit`으로 rename하면서 관련 docstring/주석/help 문자열도 함께 갱신했다.

## 검증

- `python -m unittest discover -s tests -p "test_*.py"` → **145 passed, 1 skipped** (기존 143+1에서 테스트 2개 교체 + 2개 신규 추가로 순증가, 회귀 없음).
- 직접 스모크 확인: `construct_boundary_first(scene, annulus_coupled_boundary_fit=True/False)` 양쪽 실행 결과 `topology_checks["shared_boundary_constraint"]`가 각각 `True`/`False`로 정확히 반영되고, `planar_hole` seed 0에서 disable 시 flip=10(독립 fitting의 알려진 값과 일치), 기본값에서는 flip=0으로 확인됨.

## 남은 사항

- **Step 5-B(soft tangent-plane/G1 continuity)는 구현하지 않음** — 사용자가 별도 지시할 때까지 대기. Worklog 55에서 이미 tangent/normal mismatch가 근접 0으로 나타났으므로 필요성이 낮아 보이지만, 이는 4개 테스트 씬에 한정된 관찰이며 Step 5-B의 필요 여부는 사용자 판단 사항이다.
- Phase 5 본편(§5.1-5.4, boundary segment 분류/local frame/`S_ext`/`C_ext`)은 아직 시작하지 않음 — `OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md`에 순서 명시되어 있다.
