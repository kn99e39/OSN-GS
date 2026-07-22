# Stage 1 복셀-패치별 지지 모드 (`stage1_support_mode`)

2026-07-19. Stage 1-F(경계 density 정제)까지 구현된 시점의 support mask 모드 정리.
각 모드가 무엇을 하고, 언제 쓰고, 어떤 비용을 갖는지 기록한다. 수치는 constructor
benchmark(points=600, seed=0, min=10/max=150/depth=6) 기준이며 전체 표는
`nurbs_constructor_benchmark/results/stage1_ablation/summary.md`에 있다.

`voxel_patch_stage1` constructor는 active leaf voxel마다 독립 NURBS patch를
만들고(내부 raw Gaussian에 피팅), patch의 rectangular UV chart를
`uv_support_mask`로 잘라낸다. `stage1_support_mode`는 그 mask의 출처를 정한다.

## `none` — 미트리밍

- mask 없음. patch chart가 [0,1]² 전체를 그대로 덮는다.
- 용도: mask 효과 자체를 분리 측정하는 ablation 대조군.
- 특성: patch끼리 크게 겹침(union overlap 0.17~0.97). hole은 inactive voxel
  자리에서만 생긴다. seam성 가짜 hole은 가장 적다(chart가 서로 덮어버리므로).
- planar_hole: false-fill 0.342, hole IoU 0.610.

## `voxel` — 정확한 plane-AABB polygon

- 각 leaf의 local PCA plane과 voxel AABB의 정확한 교차 polygon(사각~육각형)을
  patch UV frame으로 투영해 rasterize한 mask. 데이터 분포는 보지 않는다.
- 특성: patch가 자기 voxel 영역 밖으로 나가지 않음(overlap ≤0.07). Stage 1의
  기하학적 기본형. 대신 polygon끼리 world에서 1-cell 어긋나는 지점마다 seam성
  tiny hole이 생긴다(plane 17개 등) — Stage 2 boundary refinement 대상으로 보류.
- 한계: **voxel 내부의 데이터 없는 영역**(hole 경계 voxel의 hole 쪽 절반 등)도
  polygon이 덮는다. planar_hole: false-fill 0.338, hole IoU 0.572.

## `voxel_density` — polygon + 밀도 보정 경계 (기본값, Stage 1-F)

- leaf face adjacency로 boundary leaf(외부/미해결 face 접촉)를 검출하고,
  **boundary leaf에만** density 정제를 적용한다. interior leaf는 `voxel`과 동일.
- density: leaf 내부 raw Gaussian UV의 무가중 KDE. bandwidth는
  `stage1_boundary_density_bandwidth`(기본 2.0) × **각 샘플 자신의 UV NN spacing**
  (per-sample adaptive). 커널을 1/h 정규화하지 않으므로 density 값은 "유효 이웃
  수" 단위가 되어 지역 밀도에 불변이고, threshold
  `stage1_boundary_density_threshold`(기본 2.0)는 그 절대 레벨이다.
- sub-voxel contour는 marching squares(edge 보간)로 추출해
  `boundary_refinement.{json,svg}` 및 provenance로 export.
- active-active shared face 비침식 보장 2중 장치: (1) interior face 너머
  이웃 leaf의 Gaussian을 face 주변 margin에서 빌려와 KDE에 포함(표면이 이어지는
  곳의 density 붕괴 방지), (2) interior face 근처 cell은 polygon 값을 유지하는
  deterministic 보호 스트립. morphology closing/global hole fill/GT 참조는 없다.
- 결과(기본값 bw=2.0/th=2.0): planar_hole false-fill 0.338→**0.210**,
  hole IoU 0.572→**0.665**, union IoU 0.867→0.879. plane/sine은 polygon-only와
  사실상 동일(0.957/0.927, tiny false hole 동일 수준). density_gradient는 희소
  배경이 약간 후퇴(union IoU 0.824→0.795, uncovered 0.088) — raw-count Stage 1의
  한계로, Stage 2 confidence-weighted support mass의 대상.
- 공격 설정 bw=1.5/th=1.5: false-fill 0.130까지 내려가지만 density_gradient가
  0.769로 더 후퇴. th≤1.0은 polygon-only로 수렴.

## 폐기된 모드: `voxel_data` (점유 영역 AND)

Stage 1-F 이전에 잠시 있던 중간 구현. polygon에 leaf 자신의 UV 이진
occupancy(count 적응 coarse grid)를 그대로 AND했다. boundary/interior 구분이
없어 interior까지 깎았고, 고정 grid라 밀도 변화에 취약했다
(density_gradient union IoU 0.824→0.611). Stage 1-F의 adaptive KDE + boundary
한정 정제가 이를 대체하며 코드에서 제거됨. 교훈: 이진 occupancy에 closing을
걸면 실제 hole까지 메워지고(측정으로 확인), 고정 해상도 occupancy는
비균일 밀도에서 희소 지역을 오판한다 — 이 두 실패가 per-sample adaptive
bandwidth와 절대 유효-이웃 threshold 설계의 직접적 근거다.

## 관련 파일

- 구현: `osn_gs/surface/torch_voxel_hierarchy.py`(hierarchy, adjacency, polygon),
  `osn_gs/surface/torch_boundary_refinement.py`(KDE, marching squares),
  `osn_gs/core/torch_pipeline.py` `_initialize_stage1`/`_refine_boundary_leaf_support`.
- config parity: `TorchPipelineConfig` ↔ `add_stage1_constructor_arguments`
  (`osn_gs/interop/colab_args.py`, 양쪽 CLI 공유) ↔ notebook `OSN_STAGE1_*` 상수.
- metrics: `nurbs_constructor_benchmark/metrics.py` `patch_union_metrics`
  (global union raster; coarse-vs-refined는 `mask_override`로 동일 기계 재사용).
