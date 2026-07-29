# Worklog 111 — NURBS Construction Synthetic 3D Gaussian Dataset 교체

## 상태

구현·검증 완료. 이 기록은 benchmark 입력 dataset과 covariance 계약의 변경 근거이며, constructor 품질 승인이나 production integration 승인을 뜻하지 않는다.

## 수행 내용

- 기본 `SCENE_NAMES`를 평면 중심 scene에서 `saddle_shell`, `spherical_cap`, `folded_roof`, `wave_annulus`의 depth-bearing 3D shell set으로 교체했다.
- 이전 `plane`, `sine`, `crease` 등은 focused compatibility test를 위해 `LEGACY_SCENE_NAMES`로만 유지했다. 기본 benchmark population에는 포함하지 않는다.
- `SyntheticGaussianScene`에 `covariance_scales`, `covariance_rotations`(wxyz), `covariance_normals`를 추가했다.
- 실제 baseline 3DGS 결과 `output/graphdeco_ab_3k/point_cloud/iteration_3000/point_cloud.ply`를 20,033개 표본으로 분석했다. Gaussian별 major/minor anisotropy는 median 5.44, p25/p75 3.14/10.09, 2배 이상 비율 90.9%였다.
- synthetic covariance는 위 분포를 국소 nearest-neighbor spacing으로 scale하고, local covariance z축을 analytic surface normal에 정렬했다. 따라서 dataset 해상도에 따라 절대 크기는 변하지만 표면 접선 방향의 납작한 3DGS 패턴은 유지한다.
- `TorchGaussianModel.initialize()`와 `TorchOSNGSPipeline.initialize()`에 optional covariance scale/rotation 입력을 추가했다. benchmark는 이 값을 pipeline에 전달하므로 renderer export도 synthetic covariance를 보존한다.

## 검증

- baseline PLY covariance 표본 분석: 1,822,948 vertices 중 20,033개 균일 표본.
- `tests/test_synthetic_gaussian_dataset.py`, `tests/test_gt_nurbs.py`, `tests/test_boundary_first_support_pipeline.py`: 11 passed.
- `tests/test_torch_pipeline_smoke.py`, `tests/test_stage1_pipeline.py`, `tests/test_synthetic_gaussian_dataset.py`: 14 passed, 기존 tensor-to-scalar warning 1건.
- 기본 3D dataset smoke: `python -m nurbs_constructor_benchmark --points 180 --skip-renderer-export --output C:\tmp\osn_gs_3d_benchmark_smoke` 성공.

## 결과 해석과 남은 위험

새 dataset은 기존 XY projection/planar support 가정의 한계를 의도적으로 노출한다. smoke에서 `spherical_cap`의 false annulus, `wave_annulus`의 component split 및 hole mismatch가 관측됐다. 이는 quality pass가 아니라 Boundary-first hardening이 해결해야 할 실제 입력 조건이다. 기본 dispatcher, trainer, renderer 및 production integration은 변경하거나 승인하지 않았다.