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

(실행 결과는 §12에 기록)

## 12. 실측 결과

*(학습 완료 후 채운다)*

## 13. Fidelity 분류

*(최종 보고와 함께 §12 이후 확정)*
