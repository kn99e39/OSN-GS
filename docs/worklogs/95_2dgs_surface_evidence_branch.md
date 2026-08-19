# Worklog 95: 2DGS surface-evidence 병렬 아키텍처 실험 브랜치

> 상태: **구현 완료 / 실측 진행 중** — 이 문서는 실제 학습 결과가 나오는 대로 §7~§9를 채운다.
> 이 브랜치는 `voxel-surface-regions`에 병합하지 않는다.

## 0. 브랜치와 기준 커밋

| 항목 | 값 |
|---|---|
| 브랜치 | `exp/2dgs-nurbs-surface-evidence` |
| base commit | `97c5c54` (Worklog 94: bounded surface-evidence representation architecture gate -- Decision 3) |
| base 브랜치 | `voxel-surface-regions` (수정·병합 없음) |

Worklog 94 Decision 3은 "constructor-level 재설계를 중단하고, 다음 architecture target은 training 중 visible geometric evidence 생성 자체(upstream)로 옮긴다"였다. 이 브랜치는 그 upstream 후보 중 하나인 2DGS를 **원 논문/원 구현에 최대한 충실하게** 이식해, OSN-GS curve-network/NURBS 구성에 더 적합한 evidence를 만드는지 같은 downstream 계약으로 판정하기 위한 것이다.

## 1. 조사한 논문·공식 구현 리비전

| 항목 | 값 |
|---|---|
| 논문 | Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao. *2D Gaussian Splatting for Geometrically Accurate Radiance Fields.* SIGGRAPH 2024 / ACM TOG |
| 논문 버전 | **arXiv:2403.17888v3** (2025-02-22 개정, 현재 최신) |
| 공식 구현 | `hbb1/2d-gaussian-splatting` @ **`335ad612f2e783a4e57b9cbc4d1e167bd599fc98`** (2025-11-24) |
| 공식 rasterizer | `hbb1/diff-surfel-rasterization` @ **`e0ed0207b3e0669960cfad70852200a4a5847f61`** (2024-11-10, "fix near cull") |
| glm submodule | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` (glm 0.9.9.9) |

읽은 범위: 논문 §4.1 Modeling(eq. 4-6), §4.2 Splatting(eq. 7-12, degenerate solutions/low-pass filter), §5 Training(eq. 13-16), §6.1 Implementation. 공식 코드: `train.py`, `gaussian_renderer/__init__.py`, `scene/gaussian_model.py`, `utils/point_utils.py`, `arguments/__init__.py`, `scripts/{dtu,tnt,m360,nerf}_eval.py`, `cuda_rasterizer/{forward.cu,backward.cu,auxiliary.h,config.h}`.

## 2. Primitive parameterization (논문 §4.1)

`osn_gs/gaussian/torch_surfel_model.py`의 `TorchGaussianSurfelModel`:

| 논문 기호 | 텐서 | 형상 |
|---|---|---|
| `p_k` | `_xyz` | (N, 3) |
| `R = [t_u, t_v, t_w]` | `_rotation` (wxyz quaternion) | (N, 4) |
| `t_w = t_u x t_v` | **파생** (`get_normal`, R의 3번째 열) | (N, 3) |
| `(s_u, s_v)` | `_scaling` (log domain) | **(N, 2)** |
| opacity | `_opacity` (logit) | (N, 1) |
| SH appearance | `_features_dc` / `_features_rest` | 3DGS와 동일 |

**세 번째 scale은 존재하지 않는다.** `scale_dim = 2`이므로 gradient step·densification·checkpoint load·opacity reset 중 어떤 것도 normal 방향 두께를 만들 수 있는 텐서 원소 자체가 없다. 이것이 "작은 세 번째 scale을 가진 3D Gaussian"과의 구조적 차이다.

논문 eq. 5의 `H`는 `splat_to_world_uv1()`(rows `[0, 1, 3]`, `(u, v, 1) -> world`)이다. 공식 코드의 `get_covariance`가 반환하는 4x4에는 unit normal `t_w` 행이 자리표시자로 들어있지만(그 자리 scale은 `scale_to_mat`가 1로 두는 상수이지 학습 파라미터가 아니다), CUDA `compute_transmat`는 3열 `mat3x4`만 만들고 `compute_cov3D_python` 경로는 `[0,1,3]`으로 그 행을 명시적으로 버린다 — 즉 논문의 "세 번째 열 = 0"과 동치다.

Base class `TorchGaussianModel`은 `scale_dim = 3`/`ply_scale_properties`만 클래스 속성으로 뽑아내는 최소 변경을 했고, 기본 동작은 완전히 동일하다(vanilla arm 회귀 없음).

## 3. Renderer (논문 §4.2)

**공식 CUDA rasterizer를 그대로 vendoring했다.** `osn_gs/render/vendor/diff_surfel_rasterization`의 컴파일 대상 소스는 upstream `e0ed020`과 **byte-identical**이다(`OSN_GS_PROVENANCE.md`). 3DGS affine covariance projection은 이 경로에서 전혀 쓰이지 않으며, **torch fallback을 의도적으로 두지 않았다** — ray-splat intersection을 파이썬으로 근사하면 2DGS라는 주장 자체가 무의미해지므로 실패는 조용한 대체가 아니라 예외로 드러난다.

| 논문 요소 | 위치 |
|---|---|
| x-plane/y-plane -> local uv (eq. 8) | `forward.cu::renderCUDA`, `k = pix.x*Tw - Tu`, `l = pix.y*Tw - Tv` |
| 두 평면의 교선 (eq. 9) | `p = cross(k, l)` |
| perspective division (eq. 10) | `s = {p.x/p.z, p.y/p.z}` |
| Gaussian kernel (eq. 6) | `rho3d = s.x^2 + s.y^2`, `alpha = opa * exp(-0.5*rho)` |
| object-space low-pass filter (eq. 11) | `rho = min(rho3d, rho2d)`, `FilterInvSquare = 2.0` (= `sigma = sqrt(2)/2`) |
| intersection depth (eq. 7) | `depth = s.x*Tw.x + s.y*Tw.y + Tw.z` |
| front-to-back alpha compositing (eq. 12) | `T = T * (1 - alpha)` 누적 |
| camera-facing normal | `DUAL_VISIABLE` |

패키징 관련 변경 2건만 있고 컴파일 소스에는 없다: (1) glm 헤더를 기존 3DGS vendored 트리와 공유(두 `glm/` 트리는 byte-identical), (2) provenance 문서 추가.

`osn_gs/render/surfel_rasterizer.py`는 공식 `gaussian_renderer/__init__.py`의 이식이며, `osn_gs/render/surfel_geometry.py`는 `utils/point_utils.py`(eq. 15의 depth-derived normal)의 이식이다.

### 노출되는 기하 출력 (지시 §8)

`render`, `rend_alpha`, `rend_normal`(world space), `rend_dist`, `surf_depth`, `surf_normal`, `depth_expected`, `depth_median`, `radii`, `visibility_filter`/`visibility_mask`, `viewspace_points`.

## 4. Losses (논문 §5, eq. 16) 및 paper vs official code

`L = L_c + alpha * L_d + beta * L_n`. `L_c`는 OSN-GS 기존 photometric term(`(1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM)`)을 그대로 쓴다 — 공식 2DGS `train.py`와 같은 식이다.

### Depth distortion

- **PAPER_FORMULATION** (eq. 13): `L_d = sum_{i,j} w_i w_j |z_i - z_j|`, `z_i`는 perspective-correct ray-splat intersection depth.
- **OFFICIAL_CODE_FORMULATION**: 동일한 pairwise 양이지만 **normalized inverse depth(disparity)** `m = far_n/(far_n-near_n) * (1 - near_n/depth)` (`near_n=0.2`, `far_n=100.0`, `auxiliary.h` 상수) 위에서 계산하고, Mip-NeRF360식 O(N) 전방 재귀로 CUDA 안에서 누적한다. 축약은 픽셀 `mean`(논문은 sum).
- **이 브랜치가 구현한 것: OFFICIAL_CODE_FORMULATION.**
- **이유**: 누적이 공식 CUDA 커널 내부에 있고, 이 브랜치는 perspective-correct rasterization을 "증명 가능하게 공식 코드"로 유지하기 위해 그 커널을 byte-identical로 vendoring했다. 논문의 raw-`z` 좌표로 바꾸려면 `forward.cu`/`backward.cu`를 고쳐야 하고, 그러면 renderer의 OFFICIAL_CODE_FAITHFUL 주장이 깨진다. 논문 자신도 "we implement this regularization term efficiently with CUDA in a manner similar to [Sun et al. 2022b]"로 이 효율적 구현을 지목한다. 두 형식을 섞는 것(예: CUDA `m`-space 위에 파이썬 raw-`z` pairwise를 덧붙이는 것)은 지시가 금지한 바다.

### Normal consistency

- **PAPER_FORMULATION** (eq. 14): ray당 `L_n = sum_i w_i (1 - n_i^T N) = A - R . N` (`A = sum_i w_i`, `R = sum_i w_i n_i`).
- **OFFICIAL_CODE_FORMULATION**: 픽셀당 `1 - A * (R . N)`. 즉 `rend_normal`(= `R`)과 `surf_normal`(= `N * A_detached`)의 내적을 1에서 뺀 뒤 픽셀 평균. `A = 1`(포화 ray)에서 두 식은 일치하고, 부분 투명 ray의 가중 방식만 다르다. 축약도 mean(논문 sum), `p_s`도 median이 아니라 `depth_ratio` 혼합(`depth_ratio=1`이 논문의 median).
- **이 브랜치가 구현한 것: OFFICIAL_CODE_FORMULATION** (`normal_consistency_loss`).
- **이유**: 공개된 2DGS 결과가 나온 형식이고, vendored rasterizer의 un-normalized alpha-weighted `rend_normal` 출력과 end-to-end로 정합적이며, 지시가 "하나의 정합적인 원본 형식"을 요구하기 때문이다. 논문 형식은 `normal_consistency_loss_paper_form`으로 **진단 전용**으로만 구현해 두어(학습 목적함수에 절대 들어가지 않는다) 두 형식의 차이를 가정이 아니라 측정으로 다룰 수 있게 했다. 실측 차이는 `tests/test_surfel_losses.py`가 고정한다.

## 5. Activation staging (지시 §5)

공식 `train.py`의 스케줄을 **재조정 없이 그대로** 쓴다.

| 항목 | 값 | 출처 |
|---|---|---|
| depth distortion 활성화 | `iteration > 3000` | 공식 `train.py` |
| normal consistency 활성화 | `iteration > 7000` | 공식 `train.py` |
| 총 iteration | **30,000** | 공식 기본값 |
| densification 창 | 500..15000, interval 100 | 공식 기본값 |
| opacity reset | 3000마다 | 공식 기본값 |

OSN-GS의 기존 reference 학습은 3,100 iteration이었으므로 그대로 쓰면 normal consistency가 한 번도 켜지지 않는다. 지시가 준 두 선택지 중 **"충분히 긴 run에서 원 스케줄을 재현한다"**를 택했다(마일스톤을 downstream NURBS 성공 기준으로 튜닝하지 않았다). `SurfelRegularizationSchedule.matches_official_staging()`이 이 사실을 런타임에 로그로 남긴다.

`lambda_normal = 0.05`(공식 기본값). `lambda_dist`는 아래 §10의 명시적 판단 사항이다.

## 6. Initialization (지시 §6)

`TorchOSNGSPipeline._surfel_compatible_scale_rotation`은 공식 `create_from_pcd`의 텐서 단위 이식이다:

```
dist2  = clamp_min(distCUDA2(points), 1e-7)
scales = log(sqrt(dist2))[..., None].repeat(1, 2)     # 두 tangent scale만
rots   = torch.rand((N, 4))                            # 3DGS의 identity와 다른 RANDOM
```

- 세 번째 scale은 채우지 않는다 — OSN-GS의 covariance 유래 normal thickness를 2DGS primitive에 넣지 않는다는 지시를 구조적으로 만족한다(넣을 슬롯이 없다).
- rotation이 identity가 아니라 **random**인 것은 upstream의 선택이며 그대로 재현했다. identity면 모든 surfel의 tangent plane이 world-axis-aligned로 시작해 normal-consistency term에 전역 상관된 퇴화 초기 방향을 준다.
- `gaussian_initialization_mode`(baseline_compatible / covariance_knn)는 `surfel_2d`에 적용되지 않는다(채울 세 번째 scale이 없다). 명시적으로 분기해 두었다.
- surface reconstruction용 local-PCA covariance 경로는 **미변경**이다. 두 arm 모두 같은 canonical construction covariance를 쓴다.
- OSN-GS stable Gaussian ID·ownership·checkpoint traceability는 그대로 유지된다.

## 7. Density control (지시 §7)

공식 `densify_and_prune` 항목별 감사는 `osn_gs/gaussian/torch_surfel_density_control.py` 모듈 docstring에 8개 항목으로 기록했다. 요약:

| 감사 항목 | 결론 |
|---|---|
| densification gradient 통계 | 공식은 `means2D.grad` norm인데, 2DGS의 `means2D`는 forward에 참여하지 않고 backward가 **3D center gradient의 screen 투영**을 써 넣는다(논문 §6.1). OSN-GS의 `grad[:, :2]`는 z 성분이 항상 0이므로 동일. **변경 없음** |
| tangent scale 상속 | clone은 그대로, split은 `/(0.8*N)` — 열 개수만 다르고 OSN-GS 로직이 이미 column-agnostic. **변경 없음** |
| planar primitive의 child 위치 | 공식은 세 번째 std를 **0**으로 둬 child가 부모 tangent plane 안에만 놓인다. **여기만 변경**: `_shape_transaction_candidates`가 `scale_dim == 2`일 때 0 std 열을 덧댄다(3열 volumetric 동작은 불변) |
| rotation/tangent frame 전파 | 부모 quaternion 그대로 복사. **변경 없음** |
| opacity pruning | 공식 `opacity_cull = 0.05` (논문 §6.1). 3DGS/OSN-GS 기본은 0.005. `surfel_density_control_config`가 0.05로 고정 |
| screen-space size pruning | `max_radii2D > 20`, `scale.max > 0.1 * extent`, gate는 `iteration > opacity_reset_interval`. OSN-GS 표현으로 동치 매핑 |
| opacity reset | `min(opacity, 0.01)`, 3000마다. opacity만 건드리므로 normal 두께를 되살릴 수 없다. **변경 없음** |
| low-pass filter 상호작용 | `radius = ceil(max(extent, 3.0 * 0.707106))` 이므로 렌더된 splat은 항상 반경 >= 3px. (a) prune 임계 20보다 훨씬 낮아 오탐 없음, (b) edge-on splat도 gradient/통계를 계속 받는다(eq. 11의 안정화). CUDA 테스트로 확인 |

NURBS를 아는 새 ADC 규칙은 도입하지 않았다. stable-ID 의미론은 clone/split/prune을 통과해 유지된다(공식 코드에는 stable ID 개념 자체가 없다).

## 8. Checkpoint / model 호환성

- checkpoint payload에 `primitive_class`, `scale_dim`을 기록한다.
- surfel checkpoint를 volumetric 모델로(또는 그 반대로) load하면 **fail-closed**로 `ValueError`다. 없는 normal 두께를 만들어내거나 학습된 축을 버리는 복구는 하지 않는다.
- 필드가 없는 기존 checkpoint는 volumetric으로 해석한다(항상 참이다).
- PLY는 surfel일 때 `scale_0`, `scale_1`만 선언한다.

## 9. Legacy 분석 코드용 adapter (지시 §2 후단, §13.9)

- `osn_gs/gaussian/torch_surfel_analysis_adapter.py`: 읽기 전용·detach된 covariance 형태 view. 기본 `exact_rank2`는 `Sigma = R diag(s_u^2, s_v^2, 0) R^T`로 **최소 고유값이 정확히 0**이다. 학습 경로는 이 모듈을 절대 읽지 않는다.
- `osn_gs/gaussian/torch_primitive_evidence_adapter.py`: 두 arm이 통과하는 **단일 진입점**. checkpoint가 스스로 기록한 primitive로 분기해 동일한 `(positions, covariance, opacity, normals, stable_ids)` 묶음을 준다. volumetric 쪽은 worklog 89/92/94 replay가 이미 쓰던 `covariance_from_scale_rotation`과 tensor 단위로 동일하므로 기존 baseline 수치와 직접 비교 가능하다.
- **명시된 한계**: `torch_gaussian_manifold_affinity.py`의 `normal_direction_separation_over_thickness = gap / average_thickness` 같은 per-primitive 두께 나눗셈 지표는 진짜 surfel에서 분모가 0이라 `extract_covariance_frame`의 floor(`1e-6`)에 걸려 포화된다. 이것은 adapter나 surfel model의 버그가 아니라 **"surface element에는 per-primitive band thickness가 정의되지 않는다"**는 사실이며, 따라서 그 위에 세워진 OSN-GS 기준은 2DGS evidence에 대해 ill-posed다. 감추지 않고 결과로 보고한다. 필요 시 `epsilon_regularized` 모드가 있으나 사용처마다 반드시 공개해야 한다.

## 10. Fair comparison contract 구현 (지시 §12)

두 arm은 `scripts/experiments/run_2dgs_vs_vanilla_30k.sh` 하나에서 나온다. 고정된 변수: dataset, calibrated camera, train/test split(`--eval --llffhold 8`, train 161 / test 24), scene normalization, resolution, SH degree와 증가 주기, background, position-LR extent mode, densification 창/간격/threshold/percent_dense/split_samples, screen/world prune 임계, opacity reset 주기, evaluation view, 저장 주기, downstream NURBS 평가 계약.

2DGS 방법론이 요구해서 유지되는 차이: planar surfel primitive, perspective-correct surfel rasterization, depth distortion, normal consistency, 2DGS density control(`opacity_cull = 0.05`). 어느 것도 vanilla에 가깝게 만들려고 제거하지 않았다. monocular depth 감독·external normal prior·NURBS loss·chart loss 등 후속 표면복원 기법은 **추가하지 않았다**.

### 명시적 판단이 필요했던 두 지점

1. **`lambda_dist` 값.** 논문 §6.1은 bounded `alpha = 1000`, unbounded `alpha = 100`을 명시한다. 그러나 공식 `scripts/m360_eval.py`(unbounded)는 `--lambda_dist`를 아예 넘기지 않아 `OptimizationParams` 기본값 **0.0**이 쓰인다. 대상 장면(COLMAP 185장, LLFF 형식, 실외 large capture)은 unbounded 계열이다. 여기서 공식 스크립트를 따르면 지시 §12가 "필수 구성요소"로 못박은 depth distortion이 사실상 제거되므로, **논문 값 `alpha = 100`**을 쓰고 공식 unbounded 스크립트가 0을 쓴다는 사실을 이렇게 공개한다. `depth_ratio = 0`(unbounded, 공식/논문 동일).
2. **primitive 개수 상한.** `--adc_max_gaussians 3000000`을 **두 arm에 동일하게** 적용했다. 공식 3DGS/2DGS 어느 쪽도 상한이 없고 2DGS 논문은 RTX 3090(24GB)을 쓴다. 이 호스트는 RTX 3080 Ti 12GB(가용 ~10.8GB)이고 이 해상도에서 백만 primitive당 ~2.1GB가 측정되어, 상한 없이는 iteration 9~10k 부근에서 OOM이다 — 즉 normal consistency(7000부터)가 작용할 시간을 갖기 전에 죽는다. 상한은 **공식 스케줄을 끝까지 돌리기 위한 것이지 어느 arm의 기하를 다듬기 위한 것이 아니다.** 공개하는 부작용 2가지: (1) 상한이 걸리면 두 arm이 같은 예산에서 densification을 멈추므로 비교가 "동일 primitive 예산" 비교가 된다, (2) OSN-GS `_limited_indices`가 gradient 순위가 아니라 row index로 자르므로 상한이 걸린 step에서 채택되는 부분집합은 결정론적이되 최고 gradient 집합은 아니다. 두 부작용 모두 양쪽 arm에 동일하게 적용된다.

## 11. 검증 (지시 §9)

CUDA 검증은 실제 RTX 3080 Ti에서 수행했다(`tests/test_surfel_rasterization_cuda.py`, `tests/test_surfel_regularization_cuda.py`, 21 passed).

| 지시 §9 요구 | 검증 방법 | 결과 |
|---|---|---|
| primitive가 실제로 평면인가 | `scale_dim == 2`, `_scaling.shape[1] == 2`, ADC/opacity reset/checkpoint 왕복 후에도 유지 | 통과 |
| 세 번째 scale이 두께를 되찾을 수 없는가 | 세 번째 열 자체가 없음. `get_splat2world`의 normal 행 길이는 scaling_modifier 0.5/1.0/2.0에서 모두 정확히 1 | 통과 |
| ray-splat intersection이 perspective-correct인가 | 55도 기울인 단일 surfel의 렌더 depth를 **closed-form ray-plane 해**와 비교. 전체 footprint에서 최대 오차 / depth span < **1e-3**. 대조군으로 center-only affine 근사는 같은 fixture에서 **100배 이상** 나쁨 | 통과 |
| normal이 tangent plane 내재값인가 | `t_w == t_u x t_v` (max err 3.6e-7), R orthonormal·det=+1, `rend_normal`이 camera-facing t_w와 일치 | 통과 |
| depth distortion이 depth 집중을 만드는가 | depth-only 최적화 80 step에서 per-ray expected-vs-median depth gap이 **절반 이하**로 감소(측면 이탈 경로는 gradient 마스크로 차단) | 통과 |
| normal consistency가 orientation을 바꾸는가 | 무작위로 어긋난 sheet의 rotation만 최적화 → loss 감소, 평면 normal 정렬 증가 | 통과 |
| 새 renderer로 gradient가 흐르는가 | `_xyz`/`_scaling`/`_rotation`/`_opacity`/`_features_dc` 전부 finite·nonzero. `viewspace_points.grad`의 z는 정확히 0(2DGS가 x/y만 기록) | 통과 |
| densification이 유효한 planar surfel을 유지하는가 | 실제 render+backward+ADC 4회 반복에서 2열 scaling 유지, stable ID 유일, normal 단위길이 유지, 이후에도 렌더 가능 | 통과 |
| checkpoint save/load 재현 | 모든 텐서 bitwise 일치, primitive mismatch는 fail-closed | 통과 |
| low-pass filter | 89.9도 edge-on splat도 radius >= 3px로 rasterize되고 gradient를 받음 | 통과 |

## 12. 실측 결과

### 12.1 학습 설정 (양 arm 동일)

DATASET(COLMAP 185장) `--eval --llffhold 8` → train 161 / test 24, **648x420**, SH degree 3, 30,000 iteration, densification 500..15000(interval 100), opacity reset 3000마다, `position_lr_extent_mode=scene`, `--surface_update_interval 0`, primitive 예산 2,000,000 동일.

2DGS만: `lambda_dist=100`(논문 unbounded alpha), `lambda_normal=0.05`, `depth_ratio=0`, dist>3000/normal>7000(공식 스케줄, 재조정 없음), `opacity_cull=0.05`.

**세 번의 실패한 시도가 이 설정을 결정했다** — 전부 `scripts/experiments/run_2dgs_vs_vanilla_30k.sh`에 기록:

1. 1600x1036 / cap 3.0M: vanilla이 iteration 5000에서 cap에 걸렸고, 당시 cap 집행이 남은 예산을 clone에 먼저 다 써서 이후 **모든** step에서 `split_parents=0`이 됐다. vanilla의 30k train PSNR(21.01)이 자기 자신의 2.9k 기준값(23.29)보다 낮아졌다. 2DGS는 cap에 닿지 않았으므로 그때 관측된 4.9dB 격차는 **방법이 아니라 cap을 측정한 것**이다. 폐기.
2. 1297x840(공식 2DGS m360 `-i images_4`) / cap 4.0M: vanilla이 iteration 6300에 3.84M으로 **CUDA OOM**. 2DGS는 같은 설정을 완주(peak 2.55M, final 2.13M, held-out PSNR 26.39/SSIM 0.830).
3. 648x420 / cap 4.0M: vanilla이 iteration 6900에 2.99M이고 100 iteration당 +18k로 계속 상승 — 해상도를 낮춰도 멈추지 않음.

**결론**: 이 12GB 카드에서 vanilla 3DGS는 어떤 해상도에서도 uncapped 공식 30k 스케줄을 완주하지 못한다. 그래서 **동일 primitive 예산(2M)** 설계로 확정했고, 그 전에 cap 집행 자체를 고쳤다(demand 비례 분배 + 최고 gradient 우선). 최종 run은 `scripts/devtools/check_adc_cap.sh`로 **양 arm 모두 cap 미도달**을 확인했다(vanilla peak 1,996,726 / 2DGS peak 1,407,281).

### 12.2 학습 결과와 렌더 품질

| | vanilla 3DGS | 2DGS surfel |
|---|---:|---:|
| 최종 primitive | 1,996,479 | **1,193,268** |
| peak primitive | 1,996,726 | 1,407,281 |
| cumulative clone / split / prune | 2,421,416 / 689,090 / 908,248 | 3,118,787 / **1,704,264** / **2,916,417** |
| train PSNR | 31.22 | 30.28 |
| **held-out PSNR** (24 cam) | **29.11** | 28.24 |
| **held-out SSIM** | **0.914** | 0.899 |
| 학습 wall clock | 25:58 | **19:34** |

2DGS는 appearance를 약 0.9dB 양보한다 — 논문이 보고하는 방향과 일치하며, geometry를 위한 정상적인 trade-off다. 동시에 primitive를 40% 적게 쓰고 학습도 25% 빠르다. **radiance field 동작이 파괴되지 않았다.**

### 12.3 Primitive 퇴화 (같은 iteration, 같은 예산, 같은 해상도)

ADC clone-parent anisotropy:

| iteration | vanilla | 2DGS |
|---:|---:|---:|
| 3,000 | 12.7 | **2.45** |
| 6,000 | 87.3 | **2.94** |
| 9,000 | 4,935 | **4.11** |
| 12,000 | 23,705 | **5.81** |
| 14,900 | 38,979 | **22.5** |

densification 종료 시점에 **1,730배** 차이다. vanilla의 volumetric primitive는 needle로 붕괴하고(OSN-GS covariance frame이 `needle_like`로 분류하는, 신뢰할 normal이 없는 상태), 2DGS surfel은 tangent aspect ratio를 유지한다. `needle_like` 비율도 16.9% → **1.75%**로 줄었다.

### 12.4 구조 evidence 비교 (동일 evaluator, 동일 threshold)

`scripts/devtools/primitive_structural_evidence_comparison.py`. 양 arm 모두 `torch_primitive_evidence_adapter`를 통해 **같은 unmodified downstream chain**(Worklog 82 → 83 → 89 → 79 coverage → PCA-UV → 6x6 NURBS → held-out)으로 들어간다. 2DGS는 `exact_rank2`(진짜 rank-2 기하, 최소 고유값 정확히 0)로 측정했다.

| 지표 | vanilla | 2DGS | 판정 |
|---|---:|---:|---|
| normal coherence (kNN, 부호무시) | 0.748 | **0.966** | 크게 개선 |
| single-sheet 비율 | 0.950 | **0.979** | 개선 |
| persistent multilayer | 3.41% | **1.47%** | 절반 이하 |
| `needle_like` | 16.9% | **1.75%** | 거의 제거 |
| affinity `same_surface` 비율 | 2.63% | **12.56%** | 4.8배 |
| affinity `crease` 비율 | 47.6% | **35.3%** | 개선 |
| local band thickness / spacing | **1.65** | 2.09 | 악화 |
| tangent-plane residual / spacing | **0.402** | 0.442 | 소폭 악화 |
| region 수 | 4 | **74** | 18.5배 |
| usable curve network region 수 | 3 | **43** | 14.3배 |
| structural curve segment 수 | 9 | **153** | 17배 |
| curve 총 길이 / representative spacing | 5.72 | **159.3** | 27.8배 |

Downstream NURBS 계약 — **비율과 절대량을 함께 봐야 한다**(수용된 evidence 총량이 37.4배 다르므로):

| 지표 | vanilla 비율 | 2DGS 비율 | vanilla 절대 | 2DGS 절대 | 절대 배율 |
|---|---:|---:|---:|---:|---:|
| region 수용 evidence | — | — | 6,366 | 238,383 | **37.4x** |
| coherent chart-unit | 67.7% | 89.2% | 4,307 | 212,656 | **49.4x** |
| cut-boundary recoverable | 0.298% | 0.077% | 19 | 184 | **9.7x** |
| **`valid_supported`** | **0.000%** | 0.015% | **0** | **35** | 0 → 35 |
| `unsafe_geometry` | 0.298% | 0.058% | 19 | 138 | 7.3x |
| `unresolved` | 67.4% | 89.2% | 4,288 | 212,580 | 49.6x |
| held-out p95 | **3.24** | 5.57 | — | — | 악화 |

### 12.5 판정

**2DGS는 OSN-GS가 볼 수 있는 표면 evidence를 실질적으로 개선한다. 그러나 NURBS 구성 문제를 해결하지는 않는다.**

개선 근거(전부 같은 evaluator·같은 threshold):
- primitive 방향 품질이 근본적으로 다르다 — normal coherence 0.75→0.97, needle 16.9%→1.75%, anisotropy 38,979→22.5.
- constructor가 수용하는 evidence가 37.4배, coherent chart-unit evidence가 49.4배 늘었다.
- curve network를 만들 수 있는 region이 3개→43개, structural curve가 9→153 segment로 늘었다.
- **지금까지 시험한 어떤 조건에서도 나오지 않던 `valid_supported` NURBS evidence가 처음으로 0이 아닌 값(35)으로 나왔다.** Worklog 94의 네 representation은 전부 0.2% 미만이었고, 이번 vanilla arm은 정확히 0이다.

미해결 근거:
- `unresolved`가 89.2%로 오히려 올라갔고, coherent evidence 대비 recoverable 비율은 0.44%→0.087%로 **떨어졌다**. 즉 evidence는 훨씬 많아졌지만 그중 boundary로 회수되는 *비율*은 나빠졌다.
- NURBS materialization rate는 0.077%로 여전히 1% 미만이다.
- held-out p95가 3.24→5.57로 악화했다(단, 두 값은 서로 다른 evidence 모집단 위에서 계산된 것이므로 직접 비교에 한계가 있다).
- center 기준 band thickness/spacing이 1.65→2.09로 악화했다. depth-distortion은 **ray 교차점**을 모으는 항이지 center 위치를 모으는 항이 아니므로, 이 지표가 개선되지 않는 것은 방법의 실패가 아니라 지표가 측정하는 대상이 다르기 때문이다.

따라서 Worklog 94 Decision 3이 지목한 upstream 방향은 **부분적으로 옳았다**: training-time evidence 생성을 바꾸면 구조 evidence가 실제로 크게 좋아진다. 그러나 2DGS 하나만으로 downstream NURBS 계약이 충족되지는 않는다.

### 12.6 명시적으로 기록하는 한계

1. **Legacy 지표의 ill-posedness.** `torch_gaussian_manifold_affinity`의 `normal_direction_separation_over_thickness = gap / average_thickness`는 진짜 surfel에서 분모가 0이라 `extract_covariance_frame`의 `sqrt(1e-12)` floor에 걸려 포화된다(`normal_thickness_is_at_the_degenerate_floor: true`로 보고됨). `same_surface` 분기가 먼저 평가되고 이 값을 읽지 않으므로 영향은 `ambiguous` → `parallel_but_separate` 방향뿐이지만, **surface element에 per-primitive band thickness라는 개념이 정의되지 않는다**는 사실 자체를 결과로 보고한다. 공개된 `epsilon_regularized` 보조 측정이 별도로 있다.
2. **Shape class 분류기 artifact.** 2DGS의 `planar_surfel` 비율(14.0%)이 vanilla(14.8%)보다 높지 않은데, 이는 평면성이 나빠서가 아니라(planarity median 1.65e7) `elongation = (s_u/s_v)^2`가 median 23.5로 `elongation_threshold=3`을 넘어 대부분 `ambiguous_shape`로 떨어지기 때문이다. rank-2 evidence에 대한 분류기 한계이지 기하 진술이 아니다.
3. **Region evidence 예산.** Worklog 82의 `build_same_surface_adjacency`가 region-owned evidence에 대해 dense `cdist`(O(n^2) 메모리)를 계산한다. 2DGS는 단일 region에 evidence를 훨씬 많이 모으므로(한 region이 57,274점 → 12.2GiB 요구) 양 arm에 **동일한 결정론적 20,000점 예산**을 적용했다. region 0(57,274)과 63(51,536)만 잘렸다. Worklog 94의 full-evidence replay와 절대값을 직접 비교할 수 없다.
4. **비율 대 절대량.** 수용 evidence 총량이 37.4배 다르므로 downstream *비율* 비교는 분모가 다르다. 위 표는 두 가지를 모두 제시한다.

## 13. Fidelity 분류

| 구성요소 | 분류 | 근거 / OSN_GS_ADAPTATION 사유 |
|---|---|---|
| 2D surfel primitive (§4.1) | `PAPER_FAITHFUL` + `OFFICIAL_CODE_FAITHFUL` | `scale_dim=2`, normal 파생, eq. 4-6 그대로 |
| Rasterizer (§4.2) | `OFFICIAL_CODE_FAITHFUL` | 컴파일 소스 byte-identical, eq. 8-11 + DUAL_VISIABLE + CUDA distortion 누적 미변경 |
| Depth distortion | `OFFICIAL_CODE_FORMULATION` | 논문은 raw `z`, 공식 CUDA는 normalized disparity `m(z)`. 커널을 고치면 renderer의 faithful 주장이 깨지고, 논문 자신이 이 효율적 구현을 지목한다 |
| Normal consistency | `OFFICIAL_CODE_FORMULATION` | 논문 `A - R.N` vs 코드 `1 - A(R.N)`. 공개 결과가 나온 형식이며 vendored rasterizer 출력과 정합적. 논문 형식은 진단 전용으로만 구현 |
| Activation staging (§5) | `OFFICIAL_CODE_FAITHFUL` | 3000/7000 재조정 없음, 30k 완주 |
| Initialization (§6) | `OFFICIAL_CODE_FAITHFUL` | `create_from_pcd` 그대로(random quaternion 포함) |
| Density control (§7) | `OFFICIAL_CODE_FAITHFUL` | 8항목 감사, 실질 변경은 planar child sampling(세 번째 std = 0) 하나 |
| Rendering outputs (§8) | `OFFICIAL_CODE_FAITHFUL` | 공식 binding이 제공하는 전부. per-(surfel,pixel) omega_i는 커널 내부에만 존재하므로 노출 불가 — 커널 수정 없이는 기술적으로 불가능하며 그 사실을 코드에 기록 |
| glm 공유 | `OSN_GS_ADAPTATION` (packaging) | 헤더 트리가 byte-identical이라 3MB 중복을 피함. 컴파일 소스 무변경 |
| Photometric clamp | `OSN_GS_ADAPTATION` | 공식 2DGS는 clamp하지 않지만 OSN-GS vanilla arm은 `[0,1]` clamp한다. 공정 비교 계약상 두 arm의 photometric 항을 동일하게 유지해야 해서 clamp를 맞췄다(`clamp_render=False`로 공식 경로 복원 가능) |
| Analysis / evidence adapter | `OSN_GS_ADAPTATION` | 원본 2DGS는 downstream이 TSDF meshing이라 이런 adapter가 필요 없다. OSN-GS는 covariance-frame 구조 분석이 필요하므로 read-only detached view가 필요하다. 학습 경로는 이 모듈을 읽지 않는다 |
| Primitive 예산 cap | `OSN_GS_ADAPTATION` | 공식 3DGS/2DGS 어느 쪽도 cap이 없고 2DGS 논문은 RTX 3090(24GB)을 쓴다. 이 호스트는 12GB이고 vanilla은 어떤 해상도에서도 uncapped 30k를 완주하지 못한다(§12.1). 양 arm 동일 적용, 최종 run에서는 **어느 arm도 cap에 닿지 않았다** |
| 해상도 648x420 | `OSN_GS_ADAPTATION` | 위와 같은 이유. 공식 평가 프로토콜도 장면/GPU에 따라 해상도를 바꾼다. 양 arm 동일 |
| Region evidence 예산 20k | `OSN_GS_ADAPTATION` | Worklog 82 adjacency가 O(n^2) 메모리. 양 arm 동일 결정론적 적용 |

## 14. Downstream 준비 상태

**준비됨.** 학습된 surfel 집합은 `osn_gs.gaussian.torch_primitive_evidence_adapter.load_primitive_evidence()` 하나로 기존 OSN-GS curve-network/NURBS 평가 체인에 들어가며, 이번 비교가 그 경로를 실제로 통과시켜 측정했다. 2DGS 전용 constructor나 완화된 기준은 만들지 않았다.

다만 §12.6의 두 계약 문제를 downstream이 인지해야 한다: (1) per-primitive normal thickness에 의존하는 기준은 rank-2 evidence에 대해 ill-posed이고, (2) region-owned evidence가 훨씬 커져서 O(n^2) 단계가 예산을 필요로 한다.

## 15. 남은 위험과 다음 후보

- `unresolved` 89.2%와 materialization rate 0.077%는 여전히 미해결이다. 2DGS는 evidence의 **방향 품질**을 고쳤지 boundary 회수 자체를 고치지 않았다.
- coherent evidence 대비 recoverable 비율이 오히려 떨어진 것(0.44%→0.087%)이 다음 조사 대상이다. evidence가 49배 늘었는데 회수율이 5배 떨어졌다면, 병목은 evidence 생성이 아니라 **cut-boundary 회수 단계의 scaling**일 수 있다.
- 이 브랜치는 병합하지 않는다. 2DGS를 NURBS 성공 방향으로 튜닝하지 않았고, 이번 비교의 목적상 튜닝해서도 안 된다.
