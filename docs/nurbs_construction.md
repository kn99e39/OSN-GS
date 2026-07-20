# Gaussian → NURBS 생성 파이프라인 (기술 상세)

이 문서는 OSN-GS가 관측된 3D Gaussian center 집합으로부터 실제로 어떻게 rational NURBS 표면(들)을 구성하는지, 코드 레벨에서 정확히 설명한다. 상위 개념/모티베이션은 `docs/architecture.md`를 참고하고, 이 문서는 **알고리즘과 수식 자체**에 집중한다.

대상 코드:

- `osn_gs/surface/torch_voxel_regions.py` — 밀도 적응형 voxel bootstrap, patch topology
- `osn_gs/surface/torch_nurbs.py` — NURBS 표현, Cox-de Boor basis, IDW/LSQ fitting, foot-point projection
- `osn_gs/core/torch_pipeline.py` — 전체 orchestration (`TorchOSNGSPipeline.initialize`, `maintain_surface_from_certain`)
- `osn_gs/core/torch_trainer.py` — 학습 중 NURBS control point의 지속적 최적화

검증 도구: `nurbs_constructor_benchmark/`가 실제 이 경로를 그대로 호출해 analytic ground-truth와 비교한다 (`docs`가 아니라 코드가 항상 최신 진실이라는 점을 이 벤치마크로 보장한다).

## 0. 파이프라인 개요

```text
관측 Gaussian center (N, 3) + color (N, 3)
  -> [A] Voxel Bootstrap: 밀도 적응형 공간 분할 + normal 기반 patch topology
  -> [B] Patch별 curve placement point (voxel region centroid)
  -> [C] Patch별 NURBS control-point 예산 배분 (밀도/경계 가중)
  -> [D] Patch별 NURBS fitting: IDW seed -> 정규화 LSQ -> foot-point parameter correction (반복)
  -> [E] 모든 개별 Gaussian을 해당 patch에 foot-point 투영해 (patch_id, u, v) binding
  -> [F] 학습 루프: control_grid/weights를 trainable parameter로 backprop 최적화
          + 주기적 품질 점검 -> 지속 실패 patch만 국소 재분할 (전역 rebuild 없음)
```

이 중 [A]~[E]가 `TorchOSNGSPipeline.initialize()` 한 번 호출로 수행되는 "construction" 단계이고, [F]가 `TorchOSNGSTrainer.train()` 루프 안에서 일어나는 "지속적 진화" 단계다.

## 1. 입력

`initialize(points, colors)`가 받는 것은 단순히:

- `points`: `(N, 3)` float tensor — 관측 Gaussian center (COLMAP point3D 또는 synthetic scene)
- `colors`: `(N, 3)` float tensor — RGB, SH DC 계수로 변환되어 `TorchGaussianModel`에 저장됨 (NURBS 생성과는 무관)

카메라나 이미지는 이 단계에 전혀 관여하지 않는다 — geometry만으로 표면을 만든다.

## 2. Stage A — Voxel Bootstrap (`build_torch_voxel_surface_regions`)

목적: (1) 수만~수십만 개의 개별 Gaussian을 다루기 쉬운 수의 "curve placement point"로 집약하고, (2) normal 불연속을 기준으로 몇 개의 NURBS patch가 필요한지 topology를 결정한다.

### 2.1 Coarse voxelization

`points`를 axis-aligned bounding box로 정규화한 뒤 `grid_resolution`(기본 16) 큐브 grid에 floor로 떨어뜨린다:

```text
normalized = (points - min_corner) / span   # [0, 1)^3
coarse_index = floor(normalized * base_resolution)   # (N, 3) integer
```

각 coarse voxel의 "밀도"는 `density_weights`(기본: 전부 1, 즉 point 개수)의 voxel 내 합이다.

### 2.2 밀도 적응형 subdivision (2단계, 재귀적 octree 아님)

`adaptive_voxel_density=True`이고 `voxel_max_subdivision_depth=depth>0`일 때:

```text
threshold = quantile(coarse_voxel_densities, voxel_density_quantile)   # 기본 0.75
level(voxel) = depth   if voxel_density >= threshold
level(voxel) = 0       otherwise
```

주의: 이는 **딱 두 단계**(coarse 그대로 vs. `depth`만큼 한 번에 더 세분화된 fine grid)이지 depth를 1씩 올려가며 재귀적으로 쪼개는 진짜 octree가 아니다. `depth=1`이면 밀집 영역은 `base_resolution * 2`, `depth=2`면 `base_resolution * 4` grid로 바로 재배치된다. 각 point는 자신이 속한 coarse voxel의 `level`에 따라 `point_resolution = base_resolution * 2^level`짜리 fine grid 위에서 다시 voxel index를 계산한다.

### 2.3 Mixed-resolution cell을 공통 좌표계로 표현

서로 다른 level의 cell이 공존하므로, 모든 cell을 "가장 미세한 grid(`fine_resolution = base_resolution * 2^depth`) 기준 AABB"로 통일해서 표현한다:

```text
cell_scale = 2^(depth - level)
bounds_min = cell_index * cell_scale
bounds_max = bounds_min + cell_scale
```

이렇게 하면 level이 다른 두 cell도 정수 좌표 AABB로 겹침/인접 검사를 할 수 있다.

### 2.4 Region center와 normal

각 region의 center는 그 안에 속한 point들의 **밀도 가중 평균 위치**다 (density_weights로 weighted). Normal은 각 region center 기준 `voxel_normal_knn`(기본 16)개의 최근접 point를 뽑아 PCA(SVD)의 최소 분산 축으로 추정한다 — 이 부분은 region 전체에 대해 배치 SVD 한 번으로 벡터화되어 있다 (`_estimate_region_normals`). 이후 평균 normal 방향에 맞춰 부호를 통일한다.

### 2.5 Face adjacency graph

`_adaptive_face_edges`가 서로 다른 해상도의 AABB들 사이 face-adjacency를 찾는다: 각 axis에서 "이 cell의 +면 좌표"와 "저 cell의 −면 좌표"가 일치하는 쌍을 해시맵으로 그룹핑한 뒤, 나머지 두 axis에서 실제로 겹치는지(overlap) 확인한다. Fine cell 하나가 인접한 coarse cell의 넓은 면 일부와만 접할 수 있으므로 이 overlap 체크가 필요하다.

### 2.6 Boundary 판정 + Patch 라벨링 (connected components)

인접한 두 region의 normal 내적 `|n_i · n_j|`이 `cos(voxel_boundary_angle_degrees)`(기본 35°) 미만이면 그 edge는 "boundary"로 표시되고(둘 다 `boundary_mask=True`), **patch adjacency 그래프에서 제외**된다. 반대로 threshold 이상이면 두 region은 "같은 patch"로 연결된다. 이후 남은 adjacency 그래프에서 BFS로 connected component를 구해 `region_patch_ids`를 할당한다 (`_patch_ids_from_edges`). 즉:

> **하나의 NURBS patch = normal이 부드럽게 이어지는(각도 threshold 이내) region들의 연결 성분.**

`crease` 벤치마크 scene(두 평면이 각지게 만나는 형태)에서 patch가 4개로 쪼개지는 것이 이 메커니즘 때문이다 — 능선을 가로지르는 edge의 normal 각도차가 threshold를 넘어 끊기고, 각 평면 조각이 독립된 connected component가 된다.

`point_patch_ids`는 각 개별 point가 속한 region의 patch id를 그대로 물려받는다.

## 3. Stage B — Curve Placement Point

`_curve_placement_points`: voxel region이 있으면 (region 개수 ≥ 2) **개별 Gaussian이 아니라 region center들**을 이후 curve/surface fitting의 입력으로 쓴다. 이는 (a) 수만 개의 raw point를 수십~수백 개의 안정된 centroid로 downsampling하는 효과와 (b) voxel 평균화에 따른 smoothing 효과를 동시에 낸다. Voxel 비활성화(`use_voxel_surface_regions=False`) 시에는 raw point를 그대로 쓴다.

`fit_torch_base_curves`는 patch id별로 그룹을 나눠(있다면), 각 그룹에서 PCA 주축을 구하고 그 축으로 정렬한 뒤 `base_curve_count`(기본 8)개 chunk로 쪼개 `[시작점, 평균, 끝점]` 3-control-point curve를 만든다. **이 base curve는 NURBS control grid 계산에 직접 쓰이지 않는다** — `nurbs_surface.json`의 `base_curves` 필드로 저장/스트리밍되는 시각화·진단용 부산물이다. 실제 표면 fitting은 아래 Stage D가 `curve_points`(voxel centroid)를 직접 입력으로 쓴다.

## 4. Stage C — Patch별 Control-Point 예산 배분 (`_fit_surface_patches`)

여러 patch가 있을 때, 전체 control point 개수 상한(`max_surface_control_points`, 기본 65536)을 patch들에 어떻게 나눌지 정한다.

```text
score[patch] = sqrt(sum(region_density in patch)) * (1 + mean(boundary_mask in patch))
target_total = min(max(4 * num_patches, max_surface_control_points),
                    (resolution_u * resolution_v) * num_patches)
raw_target[patch] = score[patch] / sum(score) * target_total
target[patch] = clamp(round(raw_target[patch]), 4, resolution_u * resolution_v)
```

`sqrt`를 쓰는 이유는 밀도가 매우 큰 patch 하나가 예산을 독식하지 않도록 압축하기 위해서다. `(1 + boundary_fraction)` 보너스는 경계(능선 등)에 인접한 patch에 조금 더 해상도를 준다. 반올림 후 합이 `target_total`을 넘으면, 가장 큰 target을 가진 patch부터 1씩 깎는 루프로 정확히 맞춘다.

각 patch의 목표 control 개수 `target`은 다시 `resolution_u × resolution_v` 직사각형으로 환원된다 (`base_u/base_v` 비율을 유지하며 `resolution_v`를 줄여 target 이하로 맞춤).

## 5. Stage D — Patch별 NURBS Fitting

### 5.1 Rational NURBS 표현 (`TorchNURBSSurface`)

Control grid `P ∈ R^{U×V×3}`, weight `w ∈ R^{U×V}`(전부 1이면 non-rational B-spline), degree `(p, q)`(기본 2, 2). u/v 각 축에 대해 **clamped/open uniform knot vector**를 만든다 (`_clamped_knot_vector`):

```text
knot 길이 = U + p + 1
앞뒤 (p+1)개 knot = 0, 1로 고정 (표면이 첫/마지막 control point 행을 보간하도록)
내부 knot = 균등 간격
```

Basis function은 Cox-de Boor recursion으로 **완전히 벡터화**되어 계산된다 (`_bspline_basis_pair`, `osn_gs/surface/torch_nurbs.py:50`):

```text
N_{i,0}(u) = 1  if  knot_i <= u < knot_{i+1}  else 0
N_{i,p}(u) = (u - knot_i) / (knot_{i+p} - knot_i) * N_{i,p-1}(u)
           + (knot_{i+p+1} - u) / (knot_{i+p+1} - knot_{i+1}) * N_{i+1,p-1}(u)
```

Control point 수가 degree보다 작으면(`_effective_degree`) 그 축의 degree를 자동으로 낮춰 안전하게 평가한다.

표면 평가(rational tensor-product NURBS):

```text
S(u, v) = [ Σ_i Σ_j N_i(u) N_j(v) w_ij P_ij ] / [ Σ_i Σ_j N_i(u) N_j(v) w_ij ]
```

미분은 quotient rule로: `A(u,v) = Σ N_i N_j w_ij P_ij`, `W(u,v) = Σ N_i N_j w_ij`라 하면

```text
S = A / W
∂S/∂u = (∂A/∂u - ∂W/∂u · S) / W      (∂v도 동일)
```

`∂N/∂u`는 표준 B-spline 미분 공식 `N'_{i,p} = p·(N_{i,p-1}/(t_{i+p}-t_i) - N_{i+1,p-1}/(t_{i+p+1}-t_{i+1}))`로 구한다 (`_bspline_basis_derivative`). 표면 normal은 `normalize(S_u × S_v)`.

### 5.2 IDW seed fit (`fit_torch_visible_surface`)

정식 fitting 이전의 **초기값**이자, `--fit-mode idw`일 때는 최종 결과이기도 하다.

1. `pca_parameterize_points`: point cloud를 중심화 → SVD → 첫 두 주성분 축에 투영 → `[0,1]^2`로 min-max 정규화. 이것이 초기 UV 파라미터화다.
2. `resolution_u × resolution_v` 정규 격자를 UV 공간에 만든다.
3. 각 격자점마다 UV 공간에서 최근접 `k=min(16, N)`개 데이터 point를 찾고, **역거리 가중치**(`1/dist`, 정규화)로 그 point들의 3D 위치를 평균 내 control point로 삼는다.

이건 오차를 최소화하는 회귀가 아니라 "근처에 데이터가 있는 곳에 control point를 놓는" 휴리스틱이라는 점이 중요하다.

### 5.3 정규화 최소제곱(LSQ) fitting (`fit_torch_visible_surface_lsq`, 기본 모드)

핵심 통찰: **fitting 시점에는 `weights ≡ 1`이므로 표면이 control point에 대해 정확히 선형**이다 — `S_k = B_k · P`, 여기서 `B_k = vec(N_i(u_k) N_j(v_k))`는 point `k`의 basis row. 따라서 다음 정규화 최소제곱은 닫힌 형태로 풀린다 (`_solve_control_grid_lsq`):

```text
minimize_P  Σ_k weight_k ||B_k P - X_k||^2  +  λ_s P^T L P  +  λ_t ||P - P_seed||^2

정규방정식:  (BᵀWB / scale + λ_s L + λ_t I) P = BᵀWX / scale + λ_t P_seed
```

- `weight_k`: patch에 속한 point의 voxel region 밀도 (선택적, `point_weights`) — 밀도가 높은 영역의 관측이 fitting에 더 큰 영향을 준다.
- `L`: thin-plate 스타일 2차 차분 페널티. Control grid를 U축/V축 각각에 대해 2차 차분(`[1,-2,1]`) 행렬 `D`로 만들고 `L = kron(DᵤᵀDᵤ, I_v) + kron(I_u, DᵥᵀDᵥ)`로 조립 — smoothness()가 학습 중 쓰는 것과 동일한 2차 미분 개념을 fitting 단계 정규화 항으로도 재사용한 것.
- `λ_t` (Tikhonov) 항은 **0이 아니라 seed(IDW) grid에 anchor**된다. 그래서 관측이 거의 없는 control point는 발산하지 않고 부드러운 seed를 따라간다.
- `torch.linalg.solve` 실패 시(특이 행렬 등) `torch.linalg.lstsq`로 조용히 fallback한다.

**반복(parameter correction) 구조**: `correction_rounds`(기본 2)번, 다음을 번갈아 수행한다:

1. 현재 UV로 위 선형계를 풀어 control grid 갱신
2. 갱신된 표면에 대해 모든 point를 **foot-point projection**으로 재투영해 UV 갱신 (5.4절)

이것이 표면 fitting 문헌의 표준 "parameter correction" 절차다 — geometry(고정 UV에서 최적)와 parameterization(고정 geometry에서 최근접점)을 번갈아 최적화한다.

### 5.4 Foot-point projection (`project_torch_points_to_nurbs`) — Gauss-Newton

임의의 3D point에 대해 표면 위 최근접 `(u,v)`를 찾는다.

1. **초기화**: `min(max(2·U,8),64) × min(max(2·V,8),64)` 크기의 dense UV grid를 평가해두고, 각 query point에 대해 (3D 거리 기준) 가장 가까운 grid sample의 UV를 초기값으로 쓴다.
2. **Damped Gauss-Newton** 반복 (`surface_projection_iterations`, 기본 4회):

```text
residual r = S(u,v) - X
Jacobian J = [S_u, S_v]                       (3×2)
(JᵀJ + damping·I) Δuv = -Jᵀr                  (damping = 1e-6 * mean(diag(JᵀJ)))
Δuv = clamp(Δuv, -0.25, 0.25)                 # grid cell 하나 이상 한 번에 못 넘어가게 제한
uv ← clamp(uv + Δuv, 0, 1)
```

3. 매 반복마다 residual이 실제로 **줄었을 때만** `best_uv`를 갱신한다 — 즉 결과는 grid 초기화보다 절대 나빠지지 않는다 (monotonic improvement 보장).

`torch.no_grad()` 안에서 실행되며, UV binding은 학습 그래프의 일부가 아니라 데이터로 취급된다.

### 5.5 정리: 하나의 patch가 만들어지는 전체 흐름

```text
patch에 속한 curve_points (voxel centroid)
  -> PCA parameterize -> UV 초기값
  -> IDW seed control grid
  -> [반복 2회] 정규화 LSQ 선형계 풀이 -> foot-point 재투영
  -> 최종 TorchNURBSSurface (control_grid, weights, degree_u/v)
```

## 6. Stage E — 개별 Gaussian을 Patch/UV에 결합 (`project_points_to_patches`)

Stage D의 fitting 입력은 (voxel로 뭉친) `curve_points`였지만, 최종적으로 **모든 개별 Gaussian**이 자기 patch와 UV를 가져야 한다 (`TorchGaussianModel.cluster_ids`, `surface_uv`). `initialize()`에서:

1. 각 Gaussian의 `cluster_ids`는 자신이 속한 voxel region의 `point_patch_ids`를 그대로 물려받는다(Stage 2.6).
2. 각 Gaussian의 `surface_uv`는 자기 patch에 대해 **다시 한 번 foot-point projection**(5.4절과 동일한 Gauss-Newton)을 수행해 얻는다 — patch가 여러 개면 patch별로 마스킹해서 각각 투영한다 (`project_points_to_patches`). Patch id가 유효 범위를 벗어나면 patch 0으로 fallback.

즉 **fitting 해상도(voxel-집약된 소수의 centroid)와 binding 해상도(전체 Gaussian)가 분리**되어 있다 — 이 덕분에 Gaussian이 수십만 개여도 LSQ 선형계 크기(`(U·V)×(U·V)`)는 patch당 control point 수에만 비례해 작게 유지된다. Binding은 point 개수에 비례하지만 embarrassingly parallel한 chunked projection이라 확장이 쉽다.

## 7. Stage F — 학습 중 NURBS의 지속적 진화

Construction은 `initialize()`에서 한 번 끝나지만, NURBS는 거기서 얼려지지 않는다.

### 7.1 Control grid는 trainable parameter

`TorchOSNGSTrainer._setup_surface_optimizer`가 모든 patch의 `control_grid`, `weights`를 `requires_grad_(True)` leaf tensor로 만들고, Gaussian 파라미터와 **별도의** Adam optimizer(`surface_optimizer`, lr=`surface_lr` 기본 1e-4)로 최적화한다. 매 iteration:

```text
total_loss = image_loss + nurbs_surface_loss(smoothness) + uncertain_anchor_loss + uncertain_confidence_loss
total_loss.backward()
surface_optimizer.step()
weights.clamp_(1e-3, 1e3)   # rational weight 발산 방지
```

즉 렌더링 loss의 gradient가 (image → Gaussian position/anchor → `surface_uv`로 평가된 patch 위 anchor point → control_grid) 경로를 타고 **NURBS control point 자체를 직접 갱신**한다. Construction 단계의 LSQ는 "좋은 출발점"을 주는 것이고, 진짜 최종 형태는 학습이 결정한다.

### 7.2 주기적 품질 점검과 국소 재분할 (전역 rebuild 없음)

`surface_rebuild_interval`(기본 1000)마다 `maintain_surface_from_certain`이 실행된다. **voxel topology 자체는 다시 만들지 않는다** — 대신:

1. (옵션) 각 certain Gaussian의 `surface_uv`를 현재 patch에 대해 foot-point로 재투영해 최신화.
2. 각 patch마다 `mean(||Gaussian_xyz - patch.evaluate(uv)||) / scene_extent` 비율을 계산.
3. 이 비율이 `surface_residual_ratio_threshold`(기본 0.03)를 `surface_residual_patience`(기본 3)회 연속 초과하면 그 patch를 "실패 후보"로 표시.
4. 실패한 patch만 `_split_failed_patch`로 처리: 그 patch에 속한 Gaussian만 (opacity·covariance volume 기반 밀도 가중으로) 다시 voxel화해 connected component를 구하고, 지배적 component를 제외한 나머지 충분히 큰 component들을 **새 독립 patch**로 떼어내 IDW+LSQ로 새로 fitting한다.
5. 기존에 문제없던 patch들과 그 optimizer 상태는 전혀 건드리지 않는다. 새로 생긴 patch의 파라미터만 `surface_optimizer`에 추가 등록된다(`_sync_surface_optimizer`, 기존 Adam moment는 보존).

이 설계는 "voxel/curve 재구성은 초기화 시 1회, 이후에는 학습 가능한 control point가 진짜 geometry, topology 수정은 지속적으로 실패하는 부분에만 국소적으로"라는 원칙을 따른다.

## 8. 설정 파라미터 요약 (`TorchPipelineConfig`)

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `voxel_grid_resolution` | 16 | Coarse voxel 해상도 |
| `adaptive_voxel_density` | True | 밀도 적응형 subdivision on/off |
| `voxel_max_subdivision_depth` | 1 | 밀집 영역의 세분화 배율(2^depth) |
| `voxel_density_quantile` | 0.75 | subdivision 여부를 가르는 밀도 분위수 |
| `voxel_normal_knn` | 16 | Region normal 추정용 이웃 수 |
| `voxel_boundary_angle_degrees` | 35.0 | patch 경계/분리 판정 각도 threshold |
| `base_curve_count` | 8 | patch당 base curve 개수 (시각화용) |
| `visible_surface_resolution_u/v` | 8 / 4 | patch당 최대 control grid 해상도 |
| `surface_fit_mode` | "lsq" | "lsq" 또는 "idw" |
| `surface_degree_u/v` | 2 / 2 | NURBS degree |
| `surface_fit_smoothness` (λ_s) | 1e-4 | thin-plate 정규화 강도 |
| `surface_fit_tikhonov` (λ_t) | 1e-4 | seed-anchoring 정규화 강도 |
| `surface_fit_rounds` | 2 | LSQ ↔ foot-point 반복 횟수 |
| `surface_projection_iterations` | 4 | foot-point Gauss-Newton 반복 횟수 |
| `max_surface_control_points` | 65536 | 전 patch 합산 control point 상한 |
| `surface_rebuild_interval`(trainer) | 1000 | 품질 점검 주기 |
| `surface_residual_ratio_threshold`(trainer) | 0.03 | patch "실패" 판정 기준 (scene extent 대비 비율) |
| `surface_residual_patience`(trainer) | 3 | 실패로 확정하기 전 연속 초과 횟수 |
| `surface_lr`(trainer) | 1e-4 | control_grid/weights Adam 학습률 |

## 9. 코드 참조 지도

| 단계 | 함수 | 위치 |
|---|---|---|
| Voxel bootstrap 전체 | `build_torch_voxel_surface_regions` | `osn_gs/surface/torch_voxel_regions.py:31` |
| Region normal 추정 | `_estimate_region_normals` | `osn_gs/surface/torch_voxel_regions.py:321` |
| Patch 경계 판정 | `_boundary_mask_from_edges` | `osn_gs/surface/torch_voxel_regions.py:258` |
| Patch 라벨링(connected components) | `_patch_ids_from_edges` | `osn_gs/surface/torch_voxel_regions.py:285` |
| Base curve 추출 | `fit_torch_base_curves` | `osn_gs/surface/torch_nurbs.py:258` |
| PCA UV 파라미터화 | `pca_parameterize_points` | `osn_gs/surface/torch_nurbs.py:306` |
| IDW seed fit | `fit_torch_visible_surface` | `osn_gs/surface/torch_nurbs.py:329` |
| LSQ 정규방정식 풀이 | `_solve_control_grid_lsq` | `osn_gs/surface/torch_nurbs.py:416` |
| LSQ + parameter correction | `fit_torch_visible_surface_lsq` | `osn_gs/surface/torch_nurbs.py:474` |
| Foot-point projection | `project_torch_points_to_nurbs` | `osn_gs/surface/torch_nurbs.py:528` |
| Cox-de Boor basis | `_bspline_basis_pair` | `osn_gs/surface/torch_nurbs.py:50` |
| Rational 평가/미분/normal | `TorchNURBSSurface.evaluate*`/`normals` | `osn_gs/surface/torch_nurbs.py:197` |
| Patch별 control 예산 배분 | `_fit_surface_patches` | `osn_gs/core/torch_pipeline.py:341` |
| Gaussian-patch UV binding | `project_points_to_patches` | `osn_gs/core/torch_pipeline.py:447` |
| 전체 construction entrypoint | `TorchOSNGSPipeline.initialize` | `osn_gs/core/torch_pipeline.py:100` |
| 품질 점검 + 국소 재분할 | `maintain_surface_from_certain` | `osn_gs/core/torch_pipeline.py:151` |
| 실패 patch 재분할 | `_split_failed_patch` | `osn_gs/core/torch_pipeline.py:265` |
| 학습 중 surface optimizer | `_setup_surface_optimizer` | `osn_gs/core/torch_trainer.py` |

## 10. 알려진 한계 (코드/worklog 기준)

- Voxel subdivision은 진짜 재귀 octree가 아니라 2단계(coarse/fully-subdivided)뿐이다.
- LSQ의 선형성 가정은 `weights ≡ 1`일 때만 성립한다. 학습 중 rational weight가 1에서 멀어진 patch를 나중에 다시 LSQ로 재fit하는 경로는 아직 없다 (`docs/worklogs/11_least_squares_nurbs_fit.md` 남은 위험).
- 데이터가 전혀 없는 UV 영역(구멍)에 대한 명시적 trimming/마스킹은 없다 — 그런 control point는 seed-anchoring으로 안정화될 뿐, "관측된 표면"이라는 의미는 없다.
- Patch topology는 quantile 경계 근처에서 rebuild마다 흔들릴 수 있다 (`docs/worklogs/07_density_adaptive_voxel_nurbs.md` 남은 위험).
