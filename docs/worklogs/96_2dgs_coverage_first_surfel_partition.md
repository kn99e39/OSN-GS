# Worklog 96 — 2DGS Coverage-first Surfel Subset partition (신규 canonical 방향, 1단계)

## 상태

**부분 완료 — 실측은 아직 없다.** 이 배치는 branch/구현/테스트까지 완료했지만, **실행에 필요한 학습된 2DGS surfel checkpoint를 이 머신에서 찾지 못했다.** §5(checkpoint 식별)를 먼저 읽는다. Architecture 성공/실패 판단은 이 배치에서 내리지 않는다(실측 자체가 아직 없으므로 원천적으로 불가능하다).

## 1. 방향 전환 요약

`voxel-surface-regions`에서 진행하던 Worklog 105/106(volumetric 3DGS covariance-minor-axis normal 기반 coverage-first partition, 이어서 계획됐던 "Worklog 106" 형태의 `partition_normal_reliability`(lambda2/lambda3 기반) 계속 확장)을 **중단**한다. 대신 이미 구현·검증된 `exp/2dgs-nurbs-surface-evidence` 브랜치의 실제 2DGS surfel primitive를 canonical surface evidence로 채택한다:

    trained 2DGS planar surfel scene
        -> intrinsic surfel tangent plane / normal (t_u, t_v, t_w -- READ, not decomposed)
        -> Coverage-first Surfel Subset partition
        -> (이후 배치) subset-local surface-evidence Trust
        -> latent surface
        -> 1 final subset : 1 NURBS Patch
        -> (이후) reconciliation / occluded continuation

이 배치의 목적은 **§105의 심각한 병리 현상이 주로 volumetric 3DGS normal evidence(고윳값 기반 축 정렬 모호성) 때문이었는지를 판정할 수 있는 실측을 준비하는 것**이다. 목표 subset 개수를 미리 정해두고 거기에 맞추지 않는다 — 입력 orientation만 바꾸고 나머지 partition 메커니즘은 그대로 둔 채 무엇이 자연히 나오는지 관찰한다.

## 2. 브랜치/통합 전략

| 항목 | 값 |
|---|---|
| 신규 브랜치 | `arch/2dgs-coverage-first-surface` |
| base | `origin/exp/2dgs-nurbs-surface-evidence` @ **`54b72c2`**(Worklog 95: epsilon-regularized sensitivity check completed) |
| `voxel-surface-regions`로의 병합 | **하지 않음** |
| `exp/2dgs-nurbs-surface-evidence` 자체 | **변경 없음** — 과거 실험 증거로 그대로 보존 |

`voxel-surface-regions`에 절대 병합하지 않는다는 branch 규칙에 대한 명시적 예외로(사용자가 이번 지시에서 직접 지정), `arch/2dgs-coverage-first-surface`는 `exp/2dgs-nurbs-surface-evidence`를 기반으로 새로 만들었다. `voxel-surface-regions`의 Worklog 95-104(selection-first constructor 가정)는 이 브랜치로 가져오지 않았다 — `exp/2dgs-nurbs-surface-evidence`의 base는 애초에 Worklog 94이므로 애초에 없다.

Worklog 105/106의 coverage-first partition 구현(`torch_coverage_first_subset_partition.py`, `torch_gaussian_surface_orientation.py`와 각 테스트)만 최소한으로 이식했다 — `voxel-surface-regions`의 파일 그대로 복사, 코드 변경 없이 그대로 25개 focused 테스트 전부 통과함을 재확인했다(§8).

## 3. 2DGS를 재구현하지 않았다는 증거

기존 구현(`osn_gs/gaussian/torch_surfel_model.py::TorchGaussianSurfelModel`, `osn_gs/render/vendor/diff_surfel_rasterization`, `osn_gs/render/surfel_rasterizer.py`, `osn_gs/gaussian/torch_surfel_density_control.py`, `osn_gs/losses/torch_surfel_losses.py`)를 **전혀 수정하지 않았다.** `git diff`로 확인:

```
git diff origin/exp/2dgs-nurbs-surface-evidence..arch/2dgs-coverage-first-surface -- osn_gs/gaussian/torch_surfel_model.py osn_gs/render/
```

는 빈 결과다(신규 파일 추가만 있고 기존 2DGS 구현 파일은 손대지 않았다). `TorchGaussianSurfelModel.scale_dim == 2`이고 `_scaling`은 정확히 2개 열이며, 세 번째 scale을 위한 텐서 슬롯 자체가 없다(§4에서 실제로 `IndexError`가 나는 것으로 재확인).

## 4. Canonical normal의 정확한 소스

신규 `osn_gs/surface/torch_surfel_surface_orientation.py::derive_surface_orientation_from_surfel`.

```python
surface_normal = model.get_normal        # t_w = t_u x t_v, R의 3번째 열
tangent_axis_u = model.get_tangent_u     # t_u, R의 1번째 열
tangent_axis_v = model.get_tangent_v     # t_v, R의 2번째 열
```

**이 모듈에는 `eigh` 호출도, covariance 구성도, 축 재정렬도 전혀 없다.** 정적 테스트(`test_module_contains_no_eigendecomposition_or_covariance_construction`, docstring/문자열 리터럴을 AST로 제거한 뒤 code token만 검사)로 `eigh`/`eigenvalue`/`covariance`/`Sigma`/`_batched_eigh` 토큰이 코드에 전혀 없음을 강제한다.

Volumetric 3DGS 경로(`torch_gaussian_surface_orientation.py`)와의 구조적 차이: 3DGS는 세 주축이 **정렬되지 않은** eigenvalue 세 개이므로 어느 것이 normal인지 정렬로 결정해야 한다(Worklog 105 §1). 2DGS surfel은 `t_u`/`t_v`/`t_w`가 학습된 rotation quaternion의 1/2/3번째 열로 **이미** 고정된 순서이며, `s_u`/`s_v`의 상대 크기와 무관하다 — `test_orientation_is_independent_of_tangent_scale_magnitude`로 실제 검증했다(같은 rotation, `s_v`를 0.001↔0.20으로 200배 바꿔도 `t_u`/`t_v`/`t_w`가 완전히 동일).

이런 이유로 `GaussianSurfaceOrientation`의 `axis_separability`(lambda2/lambda3 기반 진단) 개념을 이식하지 않았다 — 정렬 자체가 없는 표현에 "정렬이 신뢰할 만한가"라는 진단은 성립하지 않는다. 지시 §8이 금지한 `partition_normal_reliability`(lambda2/lambda3, covariance-axis separability, isotropic volumetric Gaussian detection)도 이 배치에 전혀 없다.

## 5. Historical checkpoint 식별 — **미확보**

| 항목 | 확인된 값 |
|---|---|
| 학습 스크립트 | `scripts/experiments/run_2dgs_vs_vanilla_30k.sh 2dgs <output_dir>` |
| Iteration | 30,000(공식 스케줄, 재조정 없음) |
| Densification 창 | 500..15000, interval 100, opacity reset 3000마다 |
| Dataset | `DATASET`(COLMAP 185장) `--eval --llffhold 8` → train 161 / test 24 |
| 해상도 | 648×420(`images_8`) |
| Primitive 예산 | `--adc_max_gaussians 2000000`(3DGS arm과 동일, 이 예산에 도달하지 않고 완주함 — final 1,193,268 / peak 1,407,281) |
| 2DGS 전용 옵션 | `--lambda_dist 100 --lambda_normal 0.05 --depth_ratio 0 --dist_from_iter 3000 --normal_from_iter 7000 --adc_prune_opacity_threshold 0.05` |
| 학습 호스트 | RTX 3080 Ti(12GB), CUDA 11.8, torch 2.1.2(`docker/Dockerfile.2dgs`) — **이 문서 작성 중인 로컬 머신(Windows, RTX 5080, CUDA 13, torch 2.12)과 다른 환경** |
| `checkpoint.pt` 경로 | **이 로컬 머신에 없음** — `output/` 트리 전체와 임의 검색(`*2dgs*`, `*surfel*`) 전부 빈 결과 |

`docs/worklogs/95_2dgs_surface_evidence_branch.md`(`exp/2dgs-nurbs-surface-evidence` 브랜치)가 이 학습 run의 실측 결과(§12: 최종 1,193,268 surfel, held-out PSNR 28.24, normal coherence 0.966 등)를 이미 기록하고 있으므로 **학습 자체는 실제로 완료됐다** — 다만 그 `checkpoint.pt` 파일이 어느 호스트/스토리지에 있는지가 이번 로컬 세션에는 인계되지 않았다.

**지시 §3에 따라, 이 사실을 새 30k 학습을 시작하기 전에 먼저 보고한다.** 재학습을 시작하지 않았고, volumetric checkpoint로 조용히 대체하지도 않았다.

## 6. Coverage-first partition 이식

신규 `osn_gs/surface/torch_coverage_first_subset_partition.py`는 `voxel-surface-regions`(Worklog 105/106)의 구현을 **한 글자도 바꾸지 않고** 복사했다 — 단, orientation 인자의 타입 힌트만 `GaussianSurfaceOrientation`이라는 구체 타입에서 구조적 `Protocol`(`SurfaceOrientationEvidence`: `positions`/`surface_normal`/`gaussian_ids` 세 필드만 요구)로 느슨하게 바꿔, 3DGS의 eigen-decomposition 기반 orientation과 2DGS의 intrinsic orientation을 **둘 다** — 어느 쪽도 import하지 않고 — 받아들일 수 있게 했다. 파티션 로직(kNN spatial adjacency, normal compatibility, connected component, deterministic ownership) 자체는 원자 단위로 동일하다.

## 7. 실행 준비된 review export 스크립트

신규 `scripts/devtools/coverage_first_surfel_partition_export.py`. Worklog 105/106의 3DGS export 스크립트와 **동일한 파티션 파라미터 기본값**(`neighbor_count=8`, `spatial_connect_spacing_multiplier=2.0`, `normal_compatibility_min_alignment=0.85`)을 재사용하며, 재조정하지 않는다(지시 §6). Fail-closed: `checkpoint_primitive(payload) != "surfel_2d"`이거나 `model.scale_dim != 2`이면 즉시 `ValueError`로 중단하고 volumetric checkpoint로 대체하지 않는다 — 합성 volumetric checkpoint로 실제 검증함(§9).

생성 예정 view(checkpoint 확보 후):

- `2DGS_ORIGINAL_SCENE`, `2DGS_INTRINSIC_NORMAL_VIEW`(unsigned `|t_w|`), `2DGS_COVERAGE_FIRST_SUBSET_PARTITION`, `2DGS_NORMAL_CUT_VIEW` — 전부 전체 scene, crop 없음, Worklog 105/106과 동일한 결정론적 팔레트·카메라 선택 규칙(`TorchOSNGSTrainer._preview_camera`와 동일한 name-sorted 첫 train camera) 재사용 — 같은 카메라라 `OPTIONAL_COMPARISON`으로 Worklog 105/106의 3DGS render.ppm과 나란히 놓일 수 있다.
- `local_normal_coherence_over_spatial_neighborhood`(§10.D), `per_surfel_cut_ratio` 등 지시 §9/§10이 요구하는 모든 지표를 report JSON에 담는다.

render는 `osn_gs.render.surfel_rasterizer.OSNSurfelRasterizer`를 쓴다 — 공식 벤더링된 CUDA kernel이며 fallback이 없다(`exp` 브랜치 자체 설계). **이 rasterizer가 로컬 머신(CUDA 13/RTX 5080)에서 빌드되는지 아직 검증하지 않았다** — `docker/Dockerfile.2dgs`가 명시하는 `TORCH_CUDA_ARCH_LIST=8.6`(RTX 3080 Ti/3090)은 이 머신의 arch와 다르다. Checkpoint 확보와 별개로 확인이 필요한 지점이다. (§8의 전체 회귀가 이를 실측으로 확인한다 — `test_surfel_rasterization_cuda.py`/`test_surfel_regularization_cuda.py` 21개 테스트 전부 "CUDA and the vendored diff_surfel_rasterization extension are required"로 skip됐다.)

## 8. 검증

**Focused 테스트 38개**(신규 13개 + 이식된 25개), 전부 통과:

- `tests/test_surfel_surface_orientation.py`(신규 13개): `scale_dim==2`이고 세 번째 scale이 없음(`IndexError`로 실증) / 기존 volumetric 모델(`scale_dim==3`)이 영향받지 않음 / 유도된 orientation이 모델 자신의 `get_tangent_u`/`get_tangent_v`/`get_normal`과 정확히 일치 / normal이 이 모델 자신의 학습된 `t_u x t_v`와 일치(재계산이 아니라 읽기) / `s_u`/`s_v` 크기와 무관하게 축 배정 불변 / **모듈이 eigen-decomposition·covariance 구성을 전혀 포함하지 않음(AST, docstring/문자열 제거 후 code token 검사)** / volumetric 모델에 대해 fail-closed(예외 메시지에 `scale_dim` 포함) / gaussian_ids가 모델 자신의 `stable_gaussian_ids`로 기본 설정됨 / 명시적 override 가능 / 원본 텐서 불변 / 부호 뒤집힌 normal이 unsigned alignment로 호환 처리됨 / **유도된 orientation이 coverage-first partition을 end-to-end로 구동**(평평한 두 sheet → subset 2개, coverage 계약 충족) / 파티션이 surfel orientation을 받을 때도 eigen-decomposition을 import하지 않음(AST).
- `tests/test_coverage_first_subset_partition.py`(이식 25개, `voxel-surface-regions`와 코드 동일): 전부 재통과.
- `tests/test_gaussian_surface_orientation.py`(이식 8개): 전부 재통과.

Export 스크립트는 합성(비-학습) surfel checkpoint(2열 scaling, `scale_dim=2`, `primitive_class=TorchGaussianSurfelModel`)로 CPU에서 end-to-end smoke test했다 — partition 실행, 4개 PLY view 기록, report JSON 생성까지 성공(`{"subset_count": 2, "coverage_identity_holds": true}`). 같은 스크립트를 합성 volumetric checkpoint(`scale_dim=3`)로 실행하면 render 이전 단계에서 즉시 `ValueError`로 중단됨을 확인했다 — fail-closed 계약이 실제로 동작한다.

**전체 회귀(`arch/2dgs-coverage-first-surface` 브랜치)**: `1077 passed, 22 skipped, 1 warning, 18 subtests passed in 252.65s`. Skip 22개 전부 `tests/test_surfel_rasterization_cuda.py`/`tests/test_surfel_regularization_cuda.py`("CUDA and the vendored diff_surfel_rasterization extension are required")다 — **§7이 미검증이라고 밝힌 지점을 이 회귀 자체가 실증한다**: 벤더링된 surfel CUDA 확장이 이 머신(CUDA 13/RTX 5080)에서 아직 빌드되지 않았다. 다른 실패·회귀는 없다.

## 9. 아직 실행하지 않은 것 (지시 준수)

- 실제 2DGS checkpoint에 대한 partition 실행 — checkpoint 없음.
- `2DGS_*` render.ppm 생성 — checkpoint 없음, surfel CUDA rasterizer가 이 머신에서 빌드되는지도 미검증.
- WL105 3DGS vs 2DGS partition 비교표(지시 §9) — 2DGS 쪽 실측이 없어 작성 불가.
- §10 질문(A~E: singleton 비율, percolated giant subset 존재 여부, zero-neighbor 비율, normal agreement 분포, 대형 subset의 공간적 일관성) — 전부 미답. 실측 없이 추측하지 않는다.
- Trustable surfel 추정, latent surface, 신규 production NURBS — 지시대로 착수하지 않음.
- Hyperparameter search, threshold 재조정 — 하지 않음.

## 10. 다음 단계 제안 (판단 아님, 사실 보고)

checkpoint를 확보하는 경로는 최소 두 가지다:

1. **재학습**: `scripts/experiments/run_2dgs_vs_vanilla_30k.sh 2dgs <output_dir>`를 이 로컬 머신(RTX 5080)에서 실행. 그전에 `osn_gs/render/vendor/diff_surfel_rasterization`이 이 CUDA/torch 조합에서 빌드되는지 먼저 확인해야 한다(§7). 학습 자체는 원 실행에서 19:34(RTX 3080 Ti)였으므로 이 머신에서는 비슷하거나 더 빠를 것으로 예상되지만 실측 전이므로 추정일 뿐이다.
2. **원본 checkpoint 회수**: Worklog 95(`exp` 브랜치)를 실제로 학습시킨 호스트/스토리지에서 `checkpoint.pt`를 가져온다. 위치는 이 세션에 인계되지 않았다.

**둘 중 어느 것을 선택할지, 혹은 다른 방법을 쓸지는 사용자 결정 사항이다.** 이 worklog는 그 결정을 내리지 않는다.

## 결론 없음

이 worklog는 2DGS coverage-first partition이 Worklog 105의 병리를 해소하는지, 새 architecture가 옳은 방향인지 어떤 것도 판단하지 않는다 — 실측이 아직 없다. Checkpoint를 확보한 뒤 §7의 export를 실행하면 사용자가 직접 `2DGS_COVERAGE_FIRST_SUBSET_PARTITION`을 시각적으로 검토할 수 있다.
