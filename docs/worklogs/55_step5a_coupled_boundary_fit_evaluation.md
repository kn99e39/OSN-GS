# Worklog 55: Phase 5 Step 5-A — Coupled Patch-Boundary Fitting 구현 및 다중 시드 검증

날짜: 2026-07-22

상태: **구현 완료, 4씬 x 5시드 검증 완료. 결과 매우 우수 — production 채택은 사용자 승인 대기.**

## 배경

Step 4-D 계열(`worst_wedge_optimized`, `profile_constrained`) seam 각도 재배치 탐색이 worklog 52/53에서 모두 canonical default 자격 미달로 종료되고, 사용자가 `OSN_GS_Phase4_Hardening_Plan.md`를 완료 처리(삭제)한 뒤 Phase 5(`OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md`)를 승인했다. Phase 5 본편(occluded extension chart) 이전에, 사용자가 GPT 리뷰를 거쳐 지시한 선행 작업이 Step 5-A: **coupled patch-boundary fitting**이다 — "seam을 봉합해서 보기 좋게 만드는 것"이 아니라, 독립적으로 fitting된 wedge들을 shared-boundary constrained system으로 바꿨을 때 **patch-interior fold까지 완화되는지** 검증하는 것이 핵심 목적이었다.

## 구현

### `osn_gs/surface/torch_nurbs.py`

- `_solve_control_grid_lsq`를 리팩터링해 정규방정식 조립 로직을 `_lsq_normal_system(...)`으로 분리(단일 surface 솔브는 그대로 유지, 동작 변화 없음).
- 신규 `fit_coupled_wedge_ring_lsq(wedge_points, wedge_initial_uv, ...)`: `segments`개 wedge surface를 원형(cyclic ring)으로 배치하고, 인접 wedge 쌍의 공유 seam 경계 컬럼(각 wedge의 `u=0`/`u=1` 컬럼, `resolution_v`개 control point)을 **하나의 joint 변수**로 묶어 단일 선형계를 조립·솔브한다.
  - 전역 unknown 배열: `segments * resolution_v`개의 공유 경계 변수 + `segments * (resolution_u-2) * resolution_v`개의 wedge-private interior 변수.
  - 각 wedge의 로컬 정규방정식(데이터 항 + second-difference smoothness + Tikhonov, 기존 단일-surface 솔브와 동일한 정규화)을 `local_to_global` 인덱스 매핑으로 전역 행렬에 scatter-accumulate한 뒤 **한 번** 솔브하고, 각 wedge로 다시 gather.
  - Interior 컬럼과 smoothness/Tikhonov 정규화는 wedge-private으로 유지(seam을 가로지르는 smoothness 항은 추가하지 않음) — Step 5-B(soft G1)는 이번에 구현하지 않았고, 결과가 아래처럼 이미 매우 좋아 필요 여부가 불투명해졌다.
  - 이전에 이미 reject된 hard-C0(각 wedge를 독립적으로 최적 fitting한 뒤 경계 컬럼을 사후에 덮어쓰는 방식, `build_annulus_chart`의 "NOT hard-enforced" 주석 블록)와의 구조적 차이: 이번 방식은 공유 경계가 **처음부터** joint fitting 변수이므로, 양쪽 interior가 실제로 공유하게 될 경계에 맞춰 함께 최적화된다.

### `osn_gs/surface/torch_annulus_chart.py`

- 기존 per-slice 단일 루프를 2-pass 구조로 리팩터링: 1차 패스에서 각 wedge의 `slice_points`/`initial_uv`/경계 정보를 먼저 준비하고, 2차 패스에서 fit 결과(독립 fit 또는 coupled fit)를 사용해 잔차/Jacobian/좌표 등 기존 지표를 그대로 계산. 순서만 바뀌었을 뿐 독립 fit 경로(`coupled_boundary_fit=False`, 기본값)는 완전히 byte-identical.
- 신규 opt-in `coupled_boundary_fit: bool = False` 파라미터. `True`일 때 전체 wedge를 `fit_coupled_wedge_ring_lsq`로 한 번에 조인트 fitting.
- `topology_checks["shared_boundary_constraint"]`가 `coupled_boundary_fit` 값을 그대로 반영(이전엔 하드코딩된 `False`).

### CLI

`nurbs_constructor_benchmark/boundary_first.py`/`runner.py`에 `annulus_coupled_boundary_fit`/`--bf-coupled-boundary-fit` 배선(기본 꺼짐). 진단용 `annulus_collect_diagnostic_samples` 파라미터도 함께 노출(기본 꺼짐, 이번 평가 스크립트에서 사용).

## 검증

- 단위 테스트: `tests/test_nurbs_surface.py::CoupledWedgeRingFitTest`(3개 — 공유 경계 컬럼이 정확히 동일한 값인지, 최소 크기(2-wedge) 링에서도 안전한지, diagnostics 수집이 동작하는지) + `tests/test_annulus_chart.py`(3개 — 기본값 byte-identical, 실제 `_annulus()` 픽스처에서 인접 wedge의 공유 컬럼이 정확히 일치하는지, 유한하고 건강한 fit인지).
- 전체 스위트: `python -m unittest discover -s tests -p "test_*.py"` → **143 passed, 1 skipped**(기존 137+1 대비 신규 6개 테스트만 추가, 회귀 없음).
- 4씬 x 5시드 실제 production 경로 A/B(`segment_placement="uniform_angle"` 고정, `coupled_boundary_fit` False/True만 비교): 스크래치 스크립트로 `build_annulus_chart`를 직접 호출해 raw per-sample Jacobian 데이터(`collect_diagnostic_samples=True`)를 수집하고, flip을 seam-adjacent/inner/outer/patch-interior로 분류. GT 지표(chamfer_rms/false_fill/Phase-2 conformance)는 `construct_boundary_first`/`score_state`를 그대로 호출해 별도로 확보.

## 결과 (5시드 평균, independent → coupled)

| scene | flips | outer flips | cond p95/p99/max | mean seam gap | chamfer_rms | false_fill |
|---|---:|---:|---:|---:|---:|---:|
| planar_hole | 3.8 → **0.0** | 0.0 → 0.0 | 2.63/4.06/26.10 → 2.45/2.88/3.30 | 0.01204 → **0.00000** | 0.00556 → 0.00551 | 0.1363 → 0.1418 |
| planar_hole_offcenter | 49.2 → **0.0** | 2.8 → 0.0 | 5.80/17.59/1010.38 → 4.76/7.31/10.89 | 0.01368 → **0.00000** | 0.00614 → 0.00608 | 0.3655 → 0.3776 |
| planar_hole_elliptical | 2.8 → **0.0** | 0.0 → 0.0 | 3.03/4.21/193.08 → 2.86/3.44/3.96 | 0.00944 → **0.00000** | 0.00511 → 0.00506 | 0.1281 → 0.1317 |
| planar_hole_density_gradient | 33.0 → **0.0** | 4.8 → 0.0 | 6.39/22.07/675.73 → 4.39/6.63/21.68 | 0.02388 → **0.00000** | 0.01460 → 0.01413 | 0.3511 → 0.3517 |

**4씬 모두, 5시드 모두에서 orientation flip이 정확히 0으로 사라졌다** — flip을 seam-adjacent/inner/outer/patch-interior로 분리한 카운트도 전 구간에서 전부 0으로, "fold가 interior로 옮겨갔을 뿐"이 아니라 **실제로 사라졌다**는 것을 직접 확인했다. Jacobian condition의 max/p99도 모든 씬에서 극적으로 개선(예: offcenter cond_max 1010→10.9, density_gradient 675→21.7). `chamfer_rms`는 4씬 모두 동일하거나 개선(악화 없음). `false_fill`은 거의 그대로거나 아주 소폭(수 % 수준) 나빠졌으나, 어떤 시드에서도 심각한 역행은 없었다. Seam 위치 gap은 정의상 정확히 0(공유 변수이므로). Seam tangent/normal mismatch도 거의 0으로 떨어졌는데, 이는 Step 5-B(soft G1)를 구현하지 않았음에도 나타난 **부수 효과**로 보인다 — 공유 경계 컬럼 자체가 하나의 곡선이므로 그 곡선을 따르는 미분(along-seam tangent)은 양쪽에서 자동으로 일치하고, cross-seam normal 쪽도 이번 4씬에서는 우연히 거의 일치했다(별도로 강제하지 않았으므로 다른 씬/조건에서는 재현되지 않을 수 있음 — 과대 해석하지 않는다).

## 플랜의 4가지 핵심 질문에 대한 답

1. **`planar_hole_offcenter` seed 1의 wedge 전체 fold 해소 여부**: 완전히 해소됨(219 flips → 0, cond_max 3109 → 35.4). 재배치(relocate)가 아니라 제거(remove).
2. **`planar_hole_offcenter` seed 3의 반복 회귀 해소 여부**: 완전히 해소됨(18 flips → 0, cond_max 1556 → 5.1).
3. **`planar_hole_density_gradient`의 condition 악화 재발 여부**: 재발하지 않음 — 오히려 independent 기준선(worklog 52/53의 uniform_angle 자체)보다도 훨씬 건강해짐(cond_max 675→21.7; worklog 52/53에서 나빴던 두 optimizer의 cond_max 254/451과는 비교가 안 될 정도로 개선).
4. **continuity 이후 patch-interior fold 잔존 여부**: 잔존하지 않음 — region별 flip 카운트가 seam-adjacent/inner/outer/patch-interior 전부 0.

## 결론과 다음 단계

Step 5-A는 이번 세션에서 시도한 모든 continuity/seam 개입 중 **가장 깨끗하고 광범위한 개선**을 보였다. 이전 Step 4-D 계열이 "한 지점을 고치면 다른 지점이 나빠지는" 트레이드오프에 계속 갇혀 있었던 것과 달리, coupled fitting은 4씬 전체에서 accuracy 저하 없이 Jacobian 건강성을 회복시켰다. 이는 애초에 문제의 본질이 "seam 각도를 어디에 두는가"가 아니라 "인접 wedge가 서로 다른 경계값을 향해 독립적으로 fitting되면서 그 불일치가 경계 부근 좁은 영역의 parameterization을 붕괴시켰다"는 것이었음을 시사한다.

**Production 미적용.** 계획대로 결과만 보고하고 멈춘다. `coupled_boundary_fit`은 `--bf-coupled-boundary-fit`으로만 선택 가능한 opt-in이며 기본값은 여전히 `False`(byte-identical). Step 5-B(soft G1)는 이 결과만 보면 필요성이 낮아 보이지만, 그 판단과 production 채택 여부는 사용자 결정 사항이다.

## 검증 명령

- `python -m unittest discover -s tests -p "test_*.py"` → 143 passed, 1 skipped.
- 4씬 x 5시드 A/B: 스크래치 스크립트(레포 외부, 세션 scratchpad)로 `build_annulus_chart`(raw per-sample 진단용) + `construct_boundary_first`/`score_state`(GT 지표용)를 직접 호출 — 커밋된 CLI 동작 변경 없음(`--bf-coupled-boundary-fit`/`annulus_collect_diagnostic_samples`는 기본 꺼짐인 opt-in 플래그 추가만 있음).
