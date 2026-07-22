# Phase 1 경계 복셀 plane-AABB polygon 과대추정 — 분석 + 최소 프로토타입 (production 미적용)

작성일: 2026-07-21
상태: 분석·prototype 완료, 보고. **`torch_voxel_hierarchy.py`/`torch_component_boundary.py`(production estimator)는 전혀 수정하지 않았다 — 사용자 재확인 대기.**
관련 문서: `OSN_GS_Phase4_Hardening_Plan.md`, `docs/worklogs/45_phase2_outer_boundary_bias_analysis.md`

## 배경

worklog 45가 outward bias의 주범을 Phase 1의 `coarse_mask`(voxel plane-AABB polygon union)로 지목했다. 사용자 지시: Phase 1 전체 재설계 없이, **boundary voxel에서 plane-AABB polygon이 실제 Gaussian support보다 과대평가되는 문제만 한정적으로** 다룬다. interior voxel과 voxel hierarchy/component 로직은 불변, KDE 재튜닝·global offset·morphology erosion·GT 기반 보정 전부 금지. 분석과 최소 prototype만 만들고 production 교체 전 다시 승인받는다.

## 코드 확인

`osn_gs/surface/torch_voxel_hierarchy.py:466`의 `plane_aabb_intersection_polygon(leaf.plane.centroid, leaf.plane.normal, leaf.aabb_min, leaf.aabb_max)`는 leaf의 **AABB 전체**를 평면과 교차시킨다 — 그 안에 점이 얼마나 있는지는 전혀 안 본다. `compute_leaf_face_adjacency`(`torch_voxel_hierarchy.py:345`)가 이미 leaf별 `is_boundary_leaf`를 계산해두고 있었고(exterior/unresolved face contact 기준), 각 leaf는 `gaussian_indices`로 자기 소속 점을 이미 갖고 있어서 — 사용자가 요청한 interior/boundary 분리와 "실제 Gaussian support"를 위한 재료가 전부 이미 있었다. 새 분류 로직 없이 바로 활용했다.

**중요한 부수 발견**: flat(z=0) 합성 씬에서는 모든 leaf의 z축 두께가 0이라 root AABB의 z 경계에 항상 닿는다 — `compute_leaf_face_adjacency`의 root-boundary 체크(`abs(lo[axis]-root_lo[axis])<=eps`)가 z축에 대해 무조건 걸리므로, **이 프로젝트의 모든 flat 합성 씬에서 사실상 전체 leaf가 100% boundary leaf로 분류된다** (x/y 위치와 무관하게). 실제 4개 annulus 씬으로 재확인: `planar_hole`(leaves=10, boundary=10), `offcenter`(7/7), `elliptical`(10/10), `density_gradient`(10/10) — 전부 100%. 이는 버그가 아니라 flat 씬의 구조적 특성이고, curved/3D 씬이라면 진짜 interior leaf가 생길 것이다. 이번 prototype이 "boundary leaf만 clip"이라고 해도, 지금 이 프로젝트의 flat 합성 벤치마크 씬들에서는 사실상 전체 leaf에 적용되는 셈이다 — interior leaf 불변 invariant는 코드로는 지켜지지만 flat 씬에서는 그 경로가 거의 실행되지 않는다는 점을 밝혀둔다.

## Step A/B: convex-hull 기반 보수적 clipping 프로토타입

`nurbs_constructor_benchmark/boundary_bias_analysis.py`에 추가(production 코드 미변경, 분석 모듈 내부에서만 동작):

- `_convex_hull_2d`(Andrew's monotone chain), `_sutherland_hodgman_clip`(두 convex polygon의 교집합) — 새 의존성 없이 순수 torch/python.
- `build_boundary_leaf_records`: 컴포넌트의 모든 member leaf에 대해 기존 `plane_aabb_intersection_polygon`(변경 없음)과, **boundary leaf에 한해서만** 그 leaf 소속 Gaussian을 shared UV frame에 투영한 convex hull로 교집합 clip한 결과를 함께 기록. Interior leaf는 무조건 원본 polygon 그대로(테스트로 고정: `test_interior_leaf_polygon_never_clipped`).
- Clip은 **교집합**이라 원본보다 절대 넓어질 수 없다 — "outward bias를 줄인다"는 목표와 정확히 방향이 맞고, 새로운 outward 실패 모드를 만들 수 없는 구조.
- 후보 우선순위는 계획대로 convex hull(candidate 1)만 먼저 구현·평가했다. alpha-shape/occupancy-grid contour(candidate 2), sparse-voxel fallback(candidate 3)은 결과를 보고 필요성 판단 후 진행하기로 미뤘다.

## Step C 결과: 7개 analytic-GT 씬 전/후

`threshold_field`(KDE, **변경 없음**)는 그대로 두고 `coarse_mask`만 clip 전/후로 비교(`refined_mask = threshold_field & coarse_mask`이므로 최종 결과에 미치는 실제 영향까지 측정):

| 씬 | signed_mean (전→후) | chamfer (전→후) | false_fill (전→후) | coverage (전→후) | under_coverage (전→후) |
|---|---|---|---|---|---|
| circle | +0.022→**-0.020** | 0.029→**0.020** | 0.126→**0.000** | 0.958→0.618 | 0.042→0.382 |
| ellipse | +0.046→**-0.001** | 0.047→**0.017** | 0.217→**0.034** | 0.965→0.854 | 0.035→0.146 |
| ellipse_high_ecc | +0.055→+0.029 | 0.034→**0.024** | 0.213→**0.122** | 0.986→0.910 | 0.014→0.090 |
| ellipse_offcenter | +0.034→**-0.003** | 0.036→**0.012** | 0.129→**0.014** | 0.979→0.896 | 0.021→0.104 |
| ellipse_density_gradient | +0.022→-0.037 | 0.046→0.040 | 0.158→**0.012** | 0.833→0.521 | 0.167→0.479 |
| ellipse_sparse_outer_rim | +0.019→-0.051 | 0.049→0.050 | 0.162→**0.011** | 0.743→0.458 | 0.257→0.542 |
| ellipse_anisotropic | +0.033→**-0.005** | 0.036→**0.017** | 0.164→**0.030** | 0.944→0.771 | 0.056→0.229 |

**5/7 씬(circle, ellipse, offcenter, anisotropic, 그리고 부분적으로 high_ecc)에서 outward bias가 거의 사라지거나 절반 이하로 줄고, false-fill이 큰 폭(70~100%)으로 감소한다.** 이건 명백한 개선이다.

**하지만 density_gradient와 sparse_outer_rim(둘 다 경계 근처 샘플이 sparse한 씬)에서는 예상대로 문제가 생겼다**: signed_mean이 양수(+0.02)에서 오히려 더 큰 음수(-0.04, -0.05)로 넘어가고, coverage가 0.83→0.52 / 0.74→0.46으로 크게 떨어지고 under_coverage가 그만큼 늘었다. **convex hull은 경계 근처 점이 sparse하면 실제 경계보다 한참 안쪽에서 hull이 끝나기 때문에, 과대추정을 고치려다 과소추정으로 넘어간다** — plan에서 이미 예상했던 정확히 그 실패 모드다.

## 기존 4개 annulus 씬 확인 (가벼운 체크)

전부 100% boundary leaf(위의 부수 발견과 일치)이고, clip 적용 시 `coarse_mask` cell 수가 30~40% 줄고 `refined_mask`도 함께 줄어든다(elliptical 포함, Step 4-D가 회귀했던 바로 그 씬):

| 씬 | coarse cells (전→후) | refined cells (전→후) |
|---|---|---|
| planar_hole | 3603→2503 | 3005→2455 |
| planar_hole_offcenter | 3644→2680 | 3187→2657 |
| planar_hole_elliptical | 4096→2410 | 2946→2393 |
| planar_hole_density_gradient | 3347→2320 | 2863→2246 |

정성적으로 outer 쪽이 눈에 띄게 tighten되는 걸 확인했다. **다만 이번 패스에서는 실제 NURBS LSQ refit까지 다시 돌려서 chamfer_rms 같은 최종 geometry-accuracy 지표를 재계산하진 않았다** — production 코드를 안 건드리기로 한 지시를 지키기 위해 `coarse_mask`/`refined_mask` 셀 단위 비교까지만 했다. inner(hole) 쪽 정확도까지 포함한 완전한 end-to-end 검증은 실제로 이 방향을 채택하기로 결정된 뒤, 별도 패스에서 하는 게 맞다.

## 결론 및 제안 (구현 안 함, 승인 대기)

1. **convex hull clip(candidate 1)은 유효한 방향이다** — 5/7 analytic 씬과 4개 annulus 씬 모두에서 outward bias/false-fill을 크게 줄인다.
2. **하지만 이대로 production에 넣으면 안 된다** — sparse-boundary 씬 2개에서 새로운 under-coverage 문제를 만든다. plan에 이미 있던 candidate 2(occupancy-grid contour, concave 경계도 다룸)나 candidate 3(sparse-voxel fallback, 점이 너무 적으면 원본 polygon 유지)을 추가로 구현해서 이 트레이드오프를 없애야 한다.
3. 다음으로 시도할 것을 제안한다: **점 밀도가 낮은 boundary leaf에서는 hull 대신 원본 polygon을 유지하는 fallback(candidate 3, 이미 `min_hull_points` 파라미터로 최소 골격은 있음 — 임계값을 점 개수가 아니라 "hull이 leaf AABB를 얼마나 채우는지" 같은 좀 더 직접적인 기준으로 정교화)**부터 시도. 이건 "GT 기반 보정"이 아니라 순수하게 로컬 데이터 충분성 기준이라 금지 목록에 안 걸린다.
4. Step 4-D 재평가는 여전히 보류 — 이 개선을 실제로 채택한 뒤에 진행하는 게 맞다.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"  # 122 passed, 1 skipped
```

새 테스트(`tests/test_boundary_bias_analysis.py`, 8개 추가): convex hull이 알려진 도형(정사각형+내부점, 삼각형)에서 정확한 hull/면적을 내는지, Sutherland-Hodgman clip이 두 정사각형 교집합과 "clip이 subject보다 크면 확장 안 됨" 불변식을 지키는지, interior leaf가 있으면 절대 안 건드리는지(이번엔 flat 씬 특성상 skip으로 처리됐지만 로직 자체는 고정해뒀다), ellipse 씬에서 실제로 편향이 줄어드는지(trade-off인 under_coverage 증가도 같이 확인).

`torch_voxel_hierarchy.py`, `torch_component_boundary.py`, `torch_boundary_refinement.py`, `torch_annulus_chart.py` — **전부 이번 패스에서 수정하지 않았다.**
