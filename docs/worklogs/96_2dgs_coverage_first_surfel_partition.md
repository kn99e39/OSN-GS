# Worklog 96 — 2DGS Coverage-first Surfel Subset partition (신규 canonical 방향, 1단계)

## 상태

**완료 — 실측 있음.** §5에 기록한 대로 historical checkpoint를 이 머신에서 찾지 못했으나, 사용자가 이 머신(RTX 5080)에서 재학습하기로 결정해 실제로 재학습했다(§5-A). 그 결과 checkpoint로 §9~§11의 모든 실측·비교·review export를 완료했다. **Architecture 성공/실패 판단은 여전히 이 배치에서 내리지 않는다** — 지시 §10/§12대로 사실만 보고하고 판단은 사용자 시각 검토에 맡긴다.

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

**지시 §3에 따라, 이 사실을 새 30k 학습을 시작하기 전에 먼저 보고했다.** 재학습을 시작하지 않았고, volumetric checkpoint로 조용히 대체하지도 않았다 — 사용자에게 직접 질의(재학습 / 원본 checkpoint 회수 / 보류 3택)했고, **사용자가 "이 머신에서 재학습"을 선택**했다.

### 5-A. 재학습 실행 결과

**CUDA 확장 빌드**: `docker/Dockerfile.2dgs`의 VS + `TORCH_CUDA_ARCH_LIST` 패턴을 이 Windows/RTX 5080 머신에 맞게 이식한 신규 `scripts/build_surfel_extension.bat`(기존 `scripts/build_baseline_extensions.bat`과 동일 구조, arch만 `12.0`)로 `osn_gs/render/vendor/diff_surfel_rasterization`을 빌드·설치했다. 빌드 직후 `tests/test_surfel_rasterization_cuda.py`/`test_surfel_regularization_cuda.py`(2개 파일, 21개 테스트)만 직접 실행해 전부 **skip에서 pass로** 바뀌었음을 확인했다 — perspective-correct ray-splat intersection, depth distortion, normal consistency가 이 머신에서 실제로 검증됐다.

**(2026-08-20 정정, Worklog 97 §16)** 위 문장의 원래 표현("§8에서 재확인")은 부정확했다. §8의 `1077 passed, 22 skipped ... in 252.65s` 전체 회귀는 이 CUDA 확장 빌드 **이전**(최초 구현 커밋 `8082336` 시점)에 실행된 것이고, 빌드 이후에는 위 2개 CUDA 테스트 파일만 개별 실행했을 뿐 **전체 회귀를 다시 돌리지 않았다** — 즉 §8에 적힌 "22 skipped"는 빌드 이전 상태를 정확히 기록한 것이지 빌드 이후 상태와 모순되는 오류가 아니며, 두 숫자는 서로 다른 시점을 가리킨다. 다만 "(§8에서 재확인)"이라는 문구가 마치 §8이 빌드 이후 pass를 재확인한 것처럼 읽혀 오해의 소지가 있었다. Worklog 97에서 CUDA 확장이 빌드된 상태로 전체 회귀를 처음 실행했고, 그 결과를 Worklog 97 §16/§17에 정확한 현재 상태로 기록한다.

**학습 실행**: `scripts/experiments/run_2dgs_vs_vanilla_30k.sh`의 `2dgs` arm과 **동일한 파라미터**(§5의 표, 재조정 없음)로 `train.py`를 직접 실행했다(`-s DATASET --images images_8 --eval --llffhold 8 --primitive surfel_2d --lambda_dist 100 --lambda_normal 0.05 --depth_ratio 0 --dist_from_iter 3000 --normal_from_iter 7000 --adc_prune_opacity_threshold 0.05`, 나머지 densification/예산/해상도 설정도 원본 스크립트와 동일). 실행 로그: `output/arch_2dgs_coverage_first_surface/2dgs_run1/`.

| 지표 | 이번 재학습(RTX 5080) | Worklog 95 원본(RTX 3080 Ti) |
|---|---:|---:|
| 최종 iteration | 30,000 | 30,000 |
| 최종 surfel 수 | 1,197,331 | 1,193,268 |
| train PSNR | 30.258 | 30.28 |
| **held-out PSNR**(24 cam) | **28.256** | 28.24 |
| **held-out SSIM** | **0.8997** | 0.899 |
| Wall clock | 약 9분(58 it/s 시작, densification 후 감속) | 19:34 |

**최종 surfel 수(1,197,331 vs 1,193,268)와 held-out PSNR/SSIM(28.256/0.8997 vs 28.24/0.899)이 사실상 일치한다** — 다른 GPU/CUDA/torch 조합에서도 학습이 결정론적이지는 않지만 재현 가능한 범위에서 동일한 결과에 수렴함을 확인했다. Checkpoint는 `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`(`primitive=surfel_2d`, `scale_dim=2` 자체 기록 확인).

## 6. Coverage-first partition 이식

신규 `osn_gs/surface/torch_coverage_first_subset_partition.py`는 `voxel-surface-regions`(Worklog 105/106)의 구현을 **한 글자도 바꾸지 않고** 복사했다 — 단, orientation 인자의 타입 힌트만 `GaussianSurfaceOrientation`이라는 구체 타입에서 구조적 `Protocol`(`SurfaceOrientationEvidence`: `positions`/`surface_normal`/`gaussian_ids` 세 필드만 요구)로 느슨하게 바꿔, 3DGS의 eigen-decomposition 기반 orientation과 2DGS의 intrinsic orientation을 **둘 다** — 어느 쪽도 import하지 않고 — 받아들일 수 있게 했다. 파티션 로직(kNN spatial adjacency, normal compatibility, connected component, deterministic ownership) 자체는 원자 단위로 동일하다.

## 7. 실행 준비된 review export 스크립트

신규 `scripts/devtools/coverage_first_surfel_partition_export.py`. Worklog 105/106의 3DGS export 스크립트와 **동일한 파티션 파라미터 기본값**(`neighbor_count=8`, `spatial_connect_spacing_multiplier=2.0`, `normal_compatibility_min_alignment=0.85`)을 재사용하며, 재조정하지 않는다(지시 §6). Fail-closed: `checkpoint_primitive(payload) != "surfel_2d"`이거나 `model.scale_dim != 2`이면 즉시 `ValueError`로 중단하고 volumetric checkpoint로 대체하지 않는다 — 합성 volumetric checkpoint로 실제 검증함(§9).

생성된 view(§5-A checkpoint, `output/osn_gs_2dgs_coverage_first_subset_partition/`):

- `2DGS_ORIGINAL_SCENE`, `2DGS_INTRINSIC_NORMAL_VIEW`(unsigned `|t_w|`), `2DGS_COVERAGE_FIRST_SUBSET_PARTITION`, `2DGS_NORMAL_CUT_VIEW` — 전부 전체 scene, crop 없음, Worklog 105/106과 동일한 결정론적 팔레트·카메라 선택 규칙(`TorchOSNGSTrainer._preview_camera`와 동일한 name-sorted 첫 train camera) 재사용.
- `local_normal_coherence_over_spatial_neighborhood`(§10.D), `per_surfel_cut_ratio` 등 지시 §9/§10이 요구하는 모든 지표를 report JSON(`surfel_partition_report.json`)에 담았다.

render는 `osn_gs.render.surfel_rasterizer.OSNSurfelRasterizer`를 쓴다 — 공식 벤더링된 CUDA kernel이며 fallback이 없다(`exp` 브랜치 자체 설계). §5-A에서 신규 `scripts/build_surfel_extension.bat`(`docker/Dockerfile.2dgs`의 VS + `TORCH_CUDA_ARCH_LIST` 패턴을 이 머신용 `12.0`으로 이식, 기존 `scripts/build_baseline_extensions.bat`과 동일 구조)로 실제 빌드해 `installed package` backend로 렌더까지 성공했다.

## 8. 검증

**Focused 테스트 38개**(신규 13개 + 이식된 25개), 전부 통과:

- `tests/test_surfel_surface_orientation.py`(신규 13개): `scale_dim==2`이고 세 번째 scale이 없음(`IndexError`로 실증) / 기존 volumetric 모델(`scale_dim==3`)이 영향받지 않음 / 유도된 orientation이 모델 자신의 `get_tangent_u`/`get_tangent_v`/`get_normal`과 정확히 일치 / normal이 이 모델 자신의 학습된 `t_u x t_v`와 일치(재계산이 아니라 읽기) / `s_u`/`s_v` 크기와 무관하게 축 배정 불변 / **모듈이 eigen-decomposition·covariance 구성을 전혀 포함하지 않음(AST, docstring/문자열 제거 후 code token 검사)** / volumetric 모델에 대해 fail-closed(예외 메시지에 `scale_dim` 포함) / gaussian_ids가 모델 자신의 `stable_gaussian_ids`로 기본 설정됨 / 명시적 override 가능 / 원본 텐서 불변 / 부호 뒤집힌 normal이 unsigned alignment로 호환 처리됨 / **유도된 orientation이 coverage-first partition을 end-to-end로 구동**(평평한 두 sheet → subset 2개, coverage 계약 충족) / 파티션이 surfel orientation을 받을 때도 eigen-decomposition을 import하지 않음(AST).
- `tests/test_coverage_first_subset_partition.py`(이식 25개, `voxel-surface-regions`와 코드 동일): 전부 재통과.
- `tests/test_gaussian_surface_orientation.py`(이식 8개): 전부 재통과.

Export 스크립트는 합성(비-학습) surfel checkpoint(2열 scaling, `scale_dim=2`, `primitive_class=TorchGaussianSurfelModel`)로 CPU에서 end-to-end smoke test했다 — partition 실행, 4개 PLY view 기록, report JSON 생성까지 성공(`{"subset_count": 2, "coverage_identity_holds": true}`). 같은 스크립트를 합성 volumetric checkpoint(`scale_dim=3`)로 실행하면 render 이전 단계에서 즉시 `ValueError`로 중단됨을 확인했다 — fail-closed 계약이 실제로 동작한다.

**전체 회귀(`arch/2dgs-coverage-first-surface` 브랜치)**: `1077 passed, 22 skipped, 1 warning, 18 subtests passed in 252.65s`. Skip 22개 전부 `tests/test_surfel_rasterization_cuda.py`/`tests/test_surfel_regularization_cuda.py`("CUDA and the vendored diff_surfel_rasterization extension are required")다 — **§7이 미검증이라고 밝힌 지점을 이 회귀 자체가 실증한다**: 벤더링된 surfel CUDA 확장이 이 머신(CUDA 13/RTX 5080)에서 아직 빌드되지 않았다. 다른 실패·회귀는 없다.

## 9. WL105/106 3DGS vs 2DGS partition 비교표

**3DGS 쪽은 Worklog 106의 정정된 checkpoint(`output/osn_gs_scene/3000`, PSNR 23.92 — `feedback_correct_replay_checkpoint` memory가 지목한 올바른 baseline) 수치를 쓴다.** Worklog 105 자체는 잘못된(opacity-reset 직후, PSNR 20.1) checkpoint였으므로 비교 기준으로 쓰지 않는다. `local_normal_coherence_over_spatial_neighborhood`는 Worklog 106 원본 export 스크립트가 계산하지 않았던 지표라, **이번에 정확히 같은 정의**(spatial edge 위의 unsigned `|dot(n_i,n_j)|`)로 같은 checkpoint에 대해 별도 재계산해 채웠다(파티션 파라미터·코드 미변경, 순수 보고 완결성 목적).

| 지표 | A. 3DGS(WL106, `osn_gs_scene/3000`) | B. 2DGS(이번 배치, `2dgs_run1/30000`) |
|---|---:|---:|
| Primitive 총수 | 1,033,693 | 1,197,331 |
| assigned / unassigned / multiply-owned | 1,033,693 / 0 / 0 | 1,197,331 / 0 / 0 |
| Subset 수 | 29,944 | 58,646 |
| Subset 크기 min/median/mean/p95/max | 1 / 1 / 34.52 / 7 / 857,342 | 1 / 1 / 20.42 / 9 / 894,378 |
| **최대 subset 비율** | **82.94%** | **74.70%** |
| Singleton subset 수(subset 비율) | 21,612(72.17%) | 40,410(68.90%) |
| Singleton이 소유한 primitive 비율 | 2.09% | 3.38% |
| 크기 ≤8 subset 수(subset 비율) | 28,644(95.66%) | 55,390(94.45%) |
| 크기 ≤8이 소유한 primitive 비율 | 4.19% | 7.61% |
| Spatially disconnected subset | 0 | 0 |
| Candidate spatial edge | 5,346,738 | 6,048,719 |
| Spatial edge(거리 기준 통과) | 4,464,080 | 5,156,342 |
| **Normal-incompatible cut edge**(spatial 대비) | 1,121,675(**25.13%**) | 1,141,017(**22.12%**) |
| Accepted edge | 3,342,405 | 4,015,325 |
| Fallback ownership(비율) | 21,612(**2.09%**) | 40,410(**3.38%**) |
| Local spacing min/median/mean/p95/max | 0.00191 / 0.03740 / 0.04774 / 0.11456 / 2.45926 | 0.00033 / 0.03641 / 0.04431 / 0.10471 / 11.21493 |
| **Local unsigned normal agreement**(spatial neighborhood) mean/median/p05/p95 | 0.8693 / 0.9531 / 0.3890 / 0.9984 | 0.8774 / 0.9670 / 0.3561 / 0.9990 |

절대값과 비율을 둘 다 실었다(primitive 총수가 15.8% 다르므로). **참고로 원래 잘못된 checkpoint(WL105)와 비교하면** 2DGS의 fallback ownership(3.38%)·최대 subset 비율(74.70%)·cut edge 비율(22.12%) 전부 WL105(6.40% / 33.2% / 46.1%)보다 낮다 — 다만 WL105는 opacity-reset 직후의 손상된 checkpoint였으므로 이 비교는 참고용일 뿐, 위 표의 WL106 기준이 유효한 비교다.

## 10. §10 질문에 대한 사실 답변 (판단 아님)

**A. 대량 singleton 병리가 비슷한 비율로 남아 있는가?** 남아 있다. 2DGS singleton은 primitive의 3.38%(40,410/1,197,331) — 정정된 3DGS 기준(2.09%)보다 오히려 **약간 높다.** WL105(잘못된 checkpoint, 6.40%)보다는 낮다.

**B. WL105/106급 percolated 거대 subset이 여전히 나타나는가?** 나타난다. 2DGS 최대 subset은 894,378개(전체의 **74.70%**) — WL106(82.94%)보다 낮지만 여전히 scene 대부분을 잠식하는 단일 거대 subset이다. 2번째로 큰 subset은 88,954개(7.43%)로 그 다음 격차가 크다.

**C. 정확히 몇 %의 surfel이 normal-compatible spatial neighbor를 하나도 못 얻는가?** `fallback_ownership_fraction` = **3.38%**(40,410/1,197,331: normal 비호환 이웃만 39,787 + 공간 이웃 자체 없음 623).

**D. Local unsigned normal agreement 분포는?** mean 0.8774 / median 0.9670 / p05 0.3561 / p95 0.9990(3DGS: 0.8693/0.9531/0.3890/0.9984). **분포가 3DGS와 거의 동일하다** — 중앙값은 2DGS가 근소하게 높지만 p05(가장 나쁜 5%)는 오히려 2DGS가 더 낮다. Worklog 95가 보고한 kNN 기준 normal coherence 개선폭(0.748→0.966, 다른 정의·다른 checkpoint)만큼 이 **spatial-adjacency-conditioned** 지표에서는 나타나지 않는다.

**E. 대형 subset이 공간적으로 일관된 하나의 표면인가, 아니면 여러 orientation을 가로질러 chaining됐는가?** `2DGS_COVERAGE_FIRST_SUBSET_PARTITION`(§11) 시각 검토 결과, 최대 subset(74.70%)은 **평평한 patio 바닥(단일 orientation, `2DGS_INTRINSIC_NORMAL_VIEW`에서 균일한 cyan/초록)과 뒤쪽의 굴곡진 산울타리 배경(다양한 orientation, 같은 뷰에서 다채로운 noise) 양쪽을 모두 포함한다.** 즉 겉보기엔 방향이 뚜렷이 다른 두 영역이 한 subset으로 이어져 있다 — 국소적으로는 매번 threshold를 아슬아슬하게 통과하는 완만한 normal 변화를 타고 넘어가며 이어지는(chaining) 양상으로 보인다. 이는 volumetric 3DGS 파티션이 보였던 것과 동일한 실패 유형이다. (시각적 관찰이며 정량적 chaining 지표를 별도로 계산하지는 않았다.)

## 11. Review export 경로

`output/osn_gs_2dgs_coverage_first_subset_partition/`(전부 원본 scene 좌표계, WebRenderer 폴더 규약 준수):

| view | 경로 |
|---|---|
| A. 2DGS_ORIGINAL_SCENE | `2DGS_ORIGINAL_SCENE/iteration_0000001/point_cloud.ply`, `2DGS_ORIGINAL_SCENE/render.ppm` |
| B. 2DGS_INTRINSIC_NORMAL_VIEW | `2DGS_INTRINSIC_NORMAL_VIEW/iteration_0000001/point_cloud.ply`, `.../render.ppm` |
| C. 2DGS_COVERAGE_FIRST_SUBSET_PARTITION | `2DGS_COVERAGE_FIRST_SUBSET_PARTITION/iteration_0000001/point_cloud.ply`, `.../render.ppm` |
| D. 2DGS_NORMAL_CUT_VIEW | `2DGS_NORMAL_CUT_VIEW/iteration_0000001/{point_cloud.ply,nurbs_surface.json}`, `.../render.ppm` |
| 회계 | `surfel_partition_report.json` |
| E. OPTIONAL_COMPARISON | 구현하지 않음 — 3DGS(648×420 다른 checkpoint 좌표계)와 2DGS는 서로 다른 scene reconstruction이라 같은 카메라를 쓰더라도 좌표계가 정합하지 않는다(정정된 3DGS는 `osn_gs_scene/3000`, 2DGS는 이번에 새로 학습한 checkpoint). 억지로 나란히 놓지 않았다. |

**시각 확인(정성적 관찰, §12 계약 — accounting 100%를 품질로 착각하지 않기 위해 직접 렌더를 봤다)**: `ORIGINAL_SCENE`은 checkpoint의 원본 render.ppm과 일치하는 선명한 scene(테이블·화분·산울타리)이다. `INTRINSIC_NORMAL_VIEW`는 바닥·테이블 상판이 각각 균일한 색으로 뚜렷이 구분되고 테이블 다리도 일관된 색이다 — 3DGS normal view보다 시각적으로 더 매끈하다. `COVERAGE_FIRST_SUBSET_PARTITION`은 테이블 상판+화분이 하나의 뚜렷한 파란 subset으로 완전히 매끈하게(3DGS 버전에 있던 다리 부분의 초록 잡음이 사라짐) 분리됐지만, 바닥+산울타리 배경 대부분은 여전히 하나의 거대한 갈색 subset이다. `NORMAL_CUT_VIEW`는 평평한 표면(바닥·상판)에서 어둡고(cut 적음) 산울타리·고주파 식생에서 밝다(cut 많음) — cut ratio 통계(mean 0.225, p95 0.875)와 일치한다.

## 12. 재현 명령

```
# 1) surfel CUDA 확장 빌드(최초 1회)
scripts\build_surfel_extension.bat 12.0

# 2) 학습 (WL95의 2dgs arm과 동일 설정)
.venv\Scripts\python.exe train.py -s DATASET --sparse_dir sparse/0 --images images_8 --eval --llffhold 8 ^
  -m output\arch_2dgs_coverage_first_surface\2dgs_run1 --iterations 30000 --save_iterations 7000 15000 30000 ^
  --densify_from_iter 500 --densify_until_iter 15000 --densification_interval 100 ^
  --densify_grad_threshold 0.0002 --adc_percent_dense 0.01 ^
  --adc_max_screen_size 20.0 --adc_max_scale_ratio 0.1 --adc_split_samples 2 ^
  --opacity_reset_interval 3000 --screen_size_prune_from_iter 3000 --adc_max_gaussians 2000000 ^
  --position_lr_extent_mode scene --surface_update_interval 0 --skip_cuda_build_preflight ^
  --primitive surfel_2d --lambda_dist 100 --lambda_normal 0.05 --depth_ratio 0 ^
  --dist_from_iter 3000 --normal_from_iter 7000 --adc_prune_opacity_threshold 0.05

# 3) partition + review export
.venv\Scripts\python.exe scripts\devtools\coverage_first_surfel_partition_export.py ^
  --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000 ^
  --out output\osn_gs_2dgs_coverage_first_subset_partition ^
  --device cuda --source-path DATASET --images images_8
```

파티션 런타임: 72.0초(1,197,331 surfel, RTX 5080). 학습 런타임: 약 9분(30,000 iteration, RTX 5080 — 원본 RTX 3080 Ti의 19:34보다 빠름).

## 결론 없음

이 worklog는 2DGS coverage-first partition이 volumetric 3DGS의 병리(거대 percolated subset, 다수 singleton, normal-incompatible cut 비율)를 해소하는지 판단하지 않는다. 실측(§9)은 **혼재된 결과**를 보여준다 — 최대 subset 비율과 cut edge 비율은 소폭 개선됐지만 singleton/fallback 비율은 오히려 소폭 악화됐고, 여전히 하나의 거대 subset이 서로 다른 orientation의 영역(바닥+산울타리)을 가로질러 이어진다(§10.E). 이 결과의 해석과 다음 단계(subset-local Trustable surfel 추정으로 진행할지, 파티션 자체를 더 다룰지)는 사용자가 §11의 시각 검토 후 결정한다.
