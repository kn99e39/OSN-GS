# Intruction: 각 TODO 항목에 대해 목표 달성이 확인된 경우, 해당 문서에서 그 항목에 대한 내용은 삭제하도록 한다.

# TODO: baseline 3DGS 대비 Scene 품질 격차 — 남은 동해상도 A/B 검증

정적 원인 후보였던 image loss 차이는 원본과 같은 L1+D-SSIM으로 정정했고, NURBS가 visible Gaussian을 끌던 역방향 anchor도 제거했으며, 학습 view는 이제 seed 재현 가능한 epoch별 무작위 순열로 sample한다. 기존 5k 결과는 OSN-GS가 `--low_vram` 반해상도이고 baseline은 full resolution이라 품질 격차의 공정한 증거로 사용할 수 없다.

- [x] **A/B 인프라 구축 완료 (2026-07-22, `docs/worklogs/59`, `60`)**: baseline의 held-out test-camera 분리(`llffhold`)와 해상도 자동축소 로직을 `osn_gs/data/vendor/graphdeco_scene_split.py`로 그대로 vendoring(license 보존, upstream 실제 함수와 실측 대조해 held-out 24/185장·해상도 1600x1036·scale 3.241875 완전 일치 확인). `load_colmap_scene_with_eval_split` + `osn_gs/eval/held_out_metrics.py`를 `train.py`/`scripts/train_osn_gs_torch.py`(`--eval --llffhold --resolution --resolution_scale`)에 연결하고, 실제 DATASET에서 CUDA smoke test(5 iteration)로 held-out PSNR/SSIM 리포트(`held_out_eval.json`)가 정상 생성되는 것까지 확인했다. 171 tests passing(1 skipped).
- [ ] 동일 dataset·동일 해상도·동일 iteration/save/eval 조건으로 OSN-GS와 Graphdeco 3DGS 10k A/B를 실제로 실행한다(인프라는 준비됨, 실행은 사용자 지시 대기).
- [ ] train-view PSNR뿐 아니라 동일 holdout camera의 PSNR/SSIM과 Gaussian 수, 평균 iteration 시간, ADC spike를 함께 기록한다.
- [ ] 변경 전 checkpoint가 동일 조건으로 남아 있지 않으므로 수치 동일성 대신 deterministic replay와 구조 회귀를 보장하고, 품질 acceptance는 위 A/B로 판정한다.

---

# NURBS 표면 생성 품질: 세 안건 평가 도구 + 개선 타깃

`nurbs_constructor_benchmark`가 이제 GT 대비 세 안건을 분리 측정한다(`docs/worklogs/20_ground_truth_nurbs_metrics.md`, `nurbs_constructor_benchmark/README.md`). 개선 작업은 이 지표로 before/after를 재는 것을 전제로 한다.

- **Surface Support metric -- density_gradient calibration**: 레거시 기준 남은 0.66 extrapolation 문제(global median NN spacing이 조밀 클러스터에 지배됨)는 아직 해소되지 않았다. 현재 production 기본 constructor(`boundary_first`)에서 재확인(2026-07-22)해도 `density_gradient` 씬의 `extrapolation_global=0.558`로 유사하게 높다 — Phase 2가 KDE 기반 support 추정으로 방식 자체는 바뀌었지만, 희소/비균일 관측 밀도에서 extrapolation을 과대 추정하는 근본 문제는 남아 있다.
- Accuracy(chamfer_rms)는 여전히 무난한 수준 — 주 문제는 support/topology 쪽에 남아 있다.

---

# OSN-GS NURBS Construction — 안정화 로드맵

> **Planning status:** this section is retained as a backlog and evidence index, not as the authoritative implementation sequence. For all new NURBS construction work, follow the phase gates and constraints in [OSN_GS_Final_Boundary_First_NURBS_Direction.md](docs/Urgent_Work/OSN_GS_Final_Boundary_First_NURBS_Direction.md). The historical voxel-driven migration plan is not a substitute for that governing plan.

## 범위와 문제 분리

현재 단계의 목표는 **clean synthetic Gaussian을 oracle input으로 사용해 NURBS constructor 자체의 정확성·support·topology를 검증하는 것**이다. 실전 COLMAP/3DGS에서 발생하는 floating/invalid Gaussian 문제는 constructor 안정화 이후 별도의 input-eligibility 계층에서 처리한다.

다음 네 문제를 한 실험에서 혼합하지 않는다.

1. **Geometry fitting** — 올바른 Gaussian이 주어졌을 때 NURBS geometry가 표면을 정확히 복원하는가.
2. **Support-domain estimation** — rectangular NURBS domain 중 실제 관측 support가 어디인가.
3. **Chartability/topology** — 하나의 Gaussian component를 하나의 rectangular chart로 표현할 수 있는가, 언제 split해야 하는가.
4. **Input eligibility** — 실전 데이터에서 어떤 Gaussian이 construction에 들어갈 자격이 있는가.

Synthetic benchmark에서는 `Input Eligibility Filter`를 identity로 둔다.

## 목표 patch 표현

각 patch는 geometry와 유효 domain을 분리해 관리한다.

```text
Patch_k = (
    S_k(u, v),       # tensor-product NURBS geometry
    M_obs,k(u, v),   # observed-support mask
    C_ext,k(u, v)    # controlled extrapolation confidence
)
```

- `S(u,v)`: 전체 parametric geometry.
- `M_obs(u,v)`: Gaussian support로 관측이 확인된 UV 영역.
- `C_ext(u,v)`: 관측 경계 밖에서 occluded-surface extension을 허용할 신뢰도.
- `M_obs`와 `C_ext`가 모두 낮은 영역은 invalid domain으로 간주한다.
- support mask 때문에 control grid나 knot structure를 즉시 변경하지 않는다. 초기에는 geometry와 독립된 diagnostic/runtime artifact로 유지한다.

## 권장 모듈 경계

```text
Synthetic / Raw Gaussian Source
        ↓
[Input Eligibility Filter]
        ↓
[Patch Topology Builder]
        ↓
[UV Parameterizer]
        ↓
[NURBS Geometry Fitter]
        ↓
[Support Domain Estimator]
        ↓
[Boundary / Chart Validator]
        ↓
[Occluded Extension Model]
```

각 단계는 입력·출력과 metric ownership을 분리한다. 특히 `NURBS Geometry Fitter`가 support threshold나 real-data outlier 판정을 소유하지 않도록 한다.

## 현재 construction 기준선

- 입력: observed Gaussian center `(N,3)`와 color. 초기 construction에는 camera/image를 사용하지 않는다.
- adaptive voxel bootstrap은 recursive octree가 아니라 coarse/fine 2단계 구조다.
- voxel centroid와 normal을 추정한 뒤, voxel face adjacency와 normal-angle threshold로 connected component patch를 구성한다.
- patch별 control-point budget을 할당하고 PCA 기반 UV initialization을 수행한다.
- IDW control-grid seed 후 regularized LSQ를 수행한다.
- LSQ에는 second-difference smoothness와 IDW seed Tikhonov anchoring이 있다.
- foot-point projection으로 UV를 갱신하며 LSQ와 반복한다.
- fitting 입력은 raw Gaussian이 아니라 voxel-region centroid다.
- construction 시 rational weight는 모두 1이므로 현재 fitting은 사실상 tensor-product B-spline LSQ다.
- `base_curves`는 PCA 주축 기반 quadratic Bézier 진단선이며 실제 surface fitting에는 사용되지 않는다.
- initialize 후 모든 Gaussian을 patch ID와 UV에 binding한다.
- 학습 중 control grid와 rational weight는 trainable이지만, 현재 constructor benchmark는 initialize 직후·학습 0회 결과다.

## Priority 1 — Rectangular baseline 안정화

대상 scene:

- [ ] elongated plane
- [ ] mild curved sheet

필수 metric:

- [ ] Jacobian condition number 분포
- [ ] Gaussian seed, voxel density, LSQ regularization 변화에 대한 stability

### Rectangular baseline의 알려진 한계

- `visible_surface_resolution_u/v`의 기본 상한이 사실상 `8/4` 수준이면 5:1 이상의 긴 patch를 충분히 표현하지 못할 수 있다.
- patch PCA extent에 따른 U/V 비율 배분은 일부 반영됐지만 global `base_u`/budget cap이 극단적 anisotropy를 제한한다.
- regularization 기본값은 synthetic 기준이며 실제 COLMAP 분포에서 검증되지 않았다.
- CI/regression threshold가 아직 강제되지 않는다(게이트는 있으나 기본 비활성).

## Priority 2 — 극단적 aspect ratio와 anisotropic control budget

- [ ] 총 control-point budget을 유지한다.
- [ ] 각 축에서 `degree + 1` 이상의 최소 control-point 조건을 보장한다.
- [ ] U/V 축 swap에 대해 결과가 동치인지 검증한다.
- [ ] elongated rectangular patch와 실제 curved ribbon을 별도 scene으로 분리한다.
- [ ] 단순히 긴 patch를 topology split으로 회피하지 않도록 patch-count penalty/diagnostic을 둔다.
- [ ] control-point 증가가 overfitting과 ill-conditioning을 유발하는지 condition number로 확인한다.

## Priority 3 — Surface support와 non-rectangular domain

### 방법론

raw 3D Gaussian density에서 직접 boundary curve를 찾지 않는다. 3D density threshold는 자연스럽게 volumetric isosurface/shell을 만들며 Gaussian scale·opacity·sampling density에 민감하다. 먼저 coarse NURBS geometry를 구성하고 Gaussian을 patch UV에 binding한 뒤, **UV domain에서 support density와 boundary를 추정**한다.

```text
Gaussian + coarse NURBS
→ patch별 Gaussian UV binding
→ UV support density / occupancy
→ support mask
→ outer/hole contour extraction
→ valid observed domain
```

### 기존 구현과 다음 단계

현재 UV trimming 자체는 구현되어 있고 plane/sine/crease에서 unsupported ratio를 줄인 결과가 있다. 따라서 다음 작업은 단순한 "mask 최초 구현"이 아니라 아래의 **일반화·진단·lifecycle 안정화**다.

- [ ] occupancy grid resolution과 threshold에 대한 sensitivity sweep을 추가한다.
- [ ] global median NN spacing 기반 support tau를 local-density-adaptive tau로 교체한다.
- [ ] patch-relative threshold와 quantile threshold를 비교한다.
- [ ] high/low hysteresis threshold를 추가해 경계 깜빡임과 topology 변화를 줄인다.
- [ ] smoothing, closing/opening, small-island removal을 각각 독립 ablation 가능하게 한다.
- [ ] hole preservation을 configurable policy로 둔다. 기본 cleanup이 실제 hole을 메우지 않게 한다.
- [ ] mask version과 UV-binding version을 함께 기록해 lifecycle mismatch를 탐지한다.

### Annulus O-grid의 outer-boundary conformance가 구조적으로 나쁨, 아직 한 번도 타겟된 적 없음 (2026-07-22)

- [ ] `build_annulus_chart`의 `phase2_boundary_conformance` 지표 기준, outer(바깥) loop conformance가 inner(hole) loop보다 항상 훨씬 나쁘다. 2026-07-22 재실측(coupled boundary fit 적용된 현재 production 기본값 기준, `planar_hole`/`planar_hole_offcenter`/`planar_hole_elliptical`/`planar_hole_density_gradient`, seed=0, count=600):

  | scene | inner chamfer/coverage | outer chamfer/coverage |
  |---|---|---|
  | planar_hole | 0.021 / 0.985 | 0.099 / 0.610 |
  | planar_hole_offcenter | 0.020 / 1.000 | 0.069 / 0.693 |
  | planar_hole_elliptical | 0.021 / 0.928 | 0.073 / 0.646 |
  | planar_hole_density_gradient | 0.026 / 0.964 | 0.098 / 0.518 |

- Phase 4 하드닝 Step 3(`docs/worklogs/39_phase4_hardening_step1_3.md`)에서 처음 발견된 이후 Step 6/6-B(`docs/worklogs/50`, `51`)와 Step 4-D 재평가(`docs/worklogs/52`, `53`)까지 매번 재확인됐지만, 시도된 모든 수정(arc-length seed, seam-phase offset, Hermite seed, worst_wedge_optimized, profile_constrained, coupled boundary fit)은 전부 inner-corner/seam 메커니즘만 겨냥했고 outer conformance를 직접 타겟한 적은 한 번도 없다.
- Phase 1의 per-leaf plane-AABB polygon union이 outer 방향으로 과대 확장되는 것이 근본 원인 후보로 지목됐다(`docs/worklogs/45`, `46`) — convex-hull clipping 프로토타입이 부분적으로 도움이 됐으나 sparse-boundary scene에서 새 under-coverage를 만들어 채택되지 않았고, 그 대신 채택된 boundary-leaf eligibility classifier(`docs/worklogs/47`-`49`)는 전체 false_fill/coverage는 개선했지만 이 outer-conformance 지표 자체를 별도로 개선하지는 못했다(위 표가 그 증거).
- **Phase 5 연관성**: Phase 5의 extension chart(`docs/Urgent_Work/OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md` §5.1-5.4)는 바로 이 boundary 추정치를 boundary tangent/local frame의 시드로 사용하므로, outer conformance가 나쁜 채로 Phase 5로 넘어가면 그 오차가 extension chart 품질에 직접 전파될 위험이 있다. Phase 5 본편 착수 전에 검토 필요.

### Synthetic boundary benchmark

대상 scene:

- [ ] trapezoid
- [ ] wedge
- [ ] L-shape

필수 output/metric:

- [ ] outer contour와 hole contour

### 단계적 support estimator

1. **Stage 1 — Center occupancy**: Gaussian UV sample을 2D grid에 누적하고 threshold/cleanup한다.
2. **Stage 2 — UV kernel density**: `D(u,v)=Σ K_i(u,v)` 형태의 patch-local KDE를 사용한다.
3. **Stage 3 — Covariance-aware footprint**: NURBS tangent `S_u`, `S_v`에 Gaussian covariance를 투영해 UV ellipse/covariance를 만들고 opacity/confidence로 가중한다.

각 Stage는 동일 benchmark와 metric으로 비교하며, 복잡한 estimator가 Stage 1보다 실제 개선을 보이지 않으면 채택하지 않는다.

## Priority 4 — Curved chartability와 UV validity

대상 scene:

- [ ] curved ribbon
- [ ] cylinder strip
- [ ] sphere cap
- [ ] saddle
- [ ] strongly bent sheet

검증 항목:

- [ ] PCA projection distortion
- [ ] UV overlap / self-overlap
- [ ] UV orientation sign consistency
- [ ] Jacobian condition number와 near-zero area ratio
- [ ] surface fold-over
- [ ] foot-point projection convergence/failure rate
- [ ] 하나의 chart로 유지할지 split할지 판정 기준

PCA UV가 실패하는 경우를 단순 LSQ 실패로 분류하지 않는다. `UV Parameterizer` 실패와 `NURBS Geometry Fitter` 실패를 별도 상태 코드와 metric으로 기록한다.

## Priority 5 — Multi-patch topology

대상 scene:

- [x] crease -- 해결 확인 (2026-07-22): production 기본 constructor(`boundary_first`)에서 `patches=2(gt 2)`, `topology_label_ari=1.000` 실측. 아래 레거시 회귀 타깃 문단은 이 확인으로 종료.
- [ ] T-junction
- [ ] disconnected surfaces
- [ ] close parallel sheets
- [ ] crossing sheets
- [ ] thin-shell front/back

검증 항목:

- [ ] expected/actual patch count
- [ ] topology label ARI
- [ ] Gaussian assignment accuracy
- [ ] inter-patch overlap/gap
- [ ] spurious split
- [ ] failed split
- [ ] failed merge
- [ ] close parallel sheet 간 잘못된 adjacency
- [ ] front/back normal ambiguity

### 곡률 있는 컴포넌트의 topology 오분류 (실데이터 영향 우려, 2026-07-22)

- [ ] `curved_annulus`(sine 곡률 + hole 1개)가 Phase 1(`build_surface_components`)에서 `annulus`로 인식되지 않고 `disk_like`/`complex` 컴포넌트 2개로 쪼개짐 -- `--bf-normal-threshold-degrees`/`--bf-offset-threshold-ratio` 조정과 무관하게 재현됨(`docs/worklogs/39_phase4_hardening_step1_3.md`, `docs/worklogs/43_phase4_hardening_step4d_worst_wedge_optimizer.md`, `docs/Urgent_Work/OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md`). Phase 4/5(chart generator, coupled boundary fit)의 책임 범위가 아니라 Phase 1 component builder 자체의 구조적 한계로 판단됨.
- [ ] 반대 방향 오분류도 확인됨(2026-07-22, `osn-gs benchmark --scenes mild_curved_sheet` 직접 실행): hole이 전혀 없는 단일 곡면(`mild_curved_sheet`, GT=1)이 오히려 `annulus`로 오분류되어 O-grid 8-wedge로 쪼개짐(`patches=8(gt 1)`, `topology_label_ari=0.000`) -- Phase 2의 hole-loop 검출이 곡률이 있는 컴포넌트에서 존재하지 않는 hole을 spurious하게 만들어내는 것으로 보임. `curved_annulus`와 반대 방향(hole 없음→있음으로 오판)이지만 같은 근본 원인(곡률이 있을 때 Phase 1/2의 loop/topology 판정이 불안정)을 공유하는 것으로 추정.
- **영향 평가(에이전트 판단, 사용자 질의에 대한 답)**: 실데이터에서 "곡률 + occlusion으로 인한 hole"은 평면보다 오히려 흔한 조합이며, 이 실패 모드는 크래시 없이 조용히 품질을 깎아먹는 유형(시각적으로 렌더를 봐야 발견됨, chamfer_rms 등 집계 지표만으로는 안 잡힘)이라 실데이터 적용 시 영향이 작지 않을 것으로 예상됨. 다만 이는 Phase 1(Surface-Cell Component Builder)의 책임 범위이며, Phase 4/5 쪽에서 국소적으로 패치하기보다 Phase 1 자체를 다시 열어 근본 원인을 찾는 것이 맞는 방향으로 판단됨(현재 미착수).

### Gaussian covariance 기반 topology boundary

3DGS Gaussian의 covariance 방향성을 patch topology 관측값으로 활용한다. Gaussian rotation 전체를 surface frame으로 간주하지 않고, covariance의 최소 고유값에 대응하는 축을 **surface-normal candidate**로 사용한다.

```text
Sigma_i = R_i diag(s_i,1^2, s_i,2^2, s_i,3^2) R_i^T
normal candidate n_i = eigenvector of min eigenvalue
```

단, 3DGS Gaussian은 image reconstruction을 위해 최적화되므로 covariance normal이 항상 실제 geometry normal이라는 보장은 없다. 거의 spherical한 Gaussian, floaters, densification 직후 Gaussian, 비정상적으로 elongation된 Gaussian은 topology 판단에서 낮은 신뢰도를 가져야 한다.

- [ ] Gaussian별 minimum-axis normal candidate를 계산한다.
- [ ] normal sign ambiguity 때문에 초기 비교는 `abs(dot(n_i,n_j))`를 사용한다.
- [ ] patch 생성 후 graph propagation으로 normal orientation sign을 일관되게 정렬한다.
- [ ] axis separation 기반 normal/planarity confidence를 정의한다. 예: `1 - s_min / (s_mid + eps)`.
- [ ] near-spherical Gaussian은 covariance normal의 topology weight를 낮춘다.
- [ ] Gaussian covariance normal과 neighborhood PCA/voxel normal을 함께 export해 상호 불일치를 진단한다.
- [ ] coarse NURBS fitting 이후 `normalize(S_u × S_v)`와 Gaussian normal의 mismatch를 계산한다.

### Topology adjacency graph와 boundary score

Patch boundary는 mesh UV chart segmentation의 dihedral-angle seam과 유사하게, 인접 surface observations 사이의 orientation-field discontinuity로 정의한다. 다만 normal angle 하나만으로 edge를 끊지 않는다.

인접 Gaussian 또는 voxel-region pair `(i,j)`의 boundary score 후보:

```text
B_ij =
    w_normal   * normal_discontinuity
  + w_distance * normalized_spatial_gap
  + w_offset   * normal_direction_offset
  + w_scale    * covariance_scale_mismatch
  + w_density  * local_density_discontinuity
  + w_conf     * normal_confidence_penalty
```

- `normal_discontinuity = 1 - abs(dot(n_i, n_j))`
- `normal_direction_offset = abs(dot(x_j - x_i, n_i)) / local_spacing_i`
- spatial gap과 density는 patch 내부의 adaptive local scale로 정규화한다.

- [ ] raw Gaussian 또는 voxel-region adjacency graph 중 어느 수준에서 boundary score를 평가할지 benchmark한다.
- [ ] voxel graph로 coarse component를 만든 뒤 Gaussian graph로 boundary를 refine하는 hierarchical path를 우선 검토한다.
- [ ] `tau_high` 이상은 boundary 확정, `tau_low` 이하는 adjacency 유지, 중간 영역은 neighborhood connectivity로 결정하는 hysteresis를 적용한다.
- [ ] boundary confidence와 split reason을 edge별 diagnostic artifact로 export한다.
- [ ] threshold 변화에 따른 patch-count stability와 topology ARI를 측정한다.

### Smooth curvature와 true crease 구분

이웃 normal angle이 크다는 사실만으로 patch seam을 만들면 cylinder, sphere cap, strongly bent sheet가 과분할된다. 구분해야 하는 것은 normal variation 자체가 아니라 **normal field가 locally smooth하게 설명되는지 여부**다.

- [ ] `angle / local_distance` 기반 curvature proxy를 추가하되 이것만으로 hard split하지 않는다.
- [ ] neighborhood normal field를 local plane/quadratic model로 fitting하고 residual discontinuity를 측정한다.
- [ ] 점진적 normal rotation은 동일 patch로 유지하고 짧은 거리의 비연속 jump만 crease 후보로 분류한다.
- [ ] curved ribbon, cylinder strip, sphere cap에서 false split rate를 측정한다.
- [ ] crease에서 missed split rate와 함께 ROC/threshold sweep을 기록한다.

### Close parallel sheets와 thin shell 분리

공간적으로 가깝고 normal도 평행한 두 surface는 normal-angle 기반 graph에서 잘못 연결될 수 있다. 상대 위치가 tangent plane 내부인지 normal 방향으로 분리되어 있는지를 함께 평가한다.

- [ ] `(x_j-x_i)`의 normal-direction component를 adjacency score에 포함한다.
- [ ] mutual kNN, tangent-plane distance, normal offset을 결합해 front/back 또는 parallel sheet edge를 차단한다.
- [ ] close parallel sheets와 thin-shell front/back에서 cross-sheet adjacency rate를 측정한다.

### Topology boundary와 support boundary의 역할 분리

두 boundary는 시각적으로 겹칠 수 있지만 architecture상 다른 개념이다.

```text
Topology boundary:
    하나의 NURBS chart가 담당할 수 있는 범위와 chart seam을 정의

Support boundary:
    해당 chart의 UV domain 중 실제 observed Gaussian support가 존재하는 범위를 정의
```

- topology boundary는 `Patch Topology Builder`가 소유한다.
- support boundary는 coarse NURBS와 UV binding 이후 `Support Domain Estimator`가 소유한다.
- topology boundary를 support mask threshold로 암묵적으로 생성하지 않는다.
- support boundary 때문에 topology patch를 무조건 split하지 않는다. hole과 non-rectangular support는 우선 trimming/mask로 표현한다.

### Boundary-constrained NURBS refitting

추출된 topology boundary와 UV support contour는 초기 단계에서 curve-first surface construction으로 직접 전환하지 않는다. 기존 tensor-product NURBS LSQ를 유지하고 boundary를 추가 constraint 또는 regularization으로 사용하는 방식을 먼저 구현한다.

```text
coarse patch topology
→ initial UV parameterization
→ coarse NURBS fitting
→ UV support contour extraction
→ boundary correspondence / confidence validation
→ boundary-constrained refitting
```

목표 함수 후보:

```text
E = E_data
  + lambda_smooth * E_smooth
  + lambda_seed   * E_seed
  + lambda_boundary * E_boundary
```

- [ ] topology boundary sample과 support contour sample을 구분해 저장한다.
- [ ] topology boundary는 inter-patch seam consistency 또는 edge control-point constraint로 사용한다.
- [ ] support contour는 NURBS rectangular domain을 강제로 변형하기보다 mask boundary fitting/diagnostic으로 먼저 사용한다.
- [ ] hard positional constraint와 soft weighted constraint를 ablation한다.
- [ ] boundary confidence가 낮으면 refitting weight를 자동 감소시킨다.
- [ ] boundary constraint가 interior point-to-surface accuracy를 악화시키는지 측정한다.
- [ ] adjacent patch 사이에 필요한 경우 positional continuity(C0)와 tangent continuity(G1/C1)를 별도 옵션으로 검증한다.

### Curve-network surface construction은 후속 대안으로 격리

Boundary curve와 interior U/V curve network를 먼저 만들고 Coons/Gordon surface 또는 multi-patch spline complex를 구성하는 방식은 장기 대안으로 남긴다. arbitrary Gaussian cloud에서는 curve ordering, correspondence, T-junction, hole, branching 처리가 새로운 주요 failure source가 될 수 있으므로 현재 baseline을 대체하지 않는다.

- [ ] four-sided structured patch에서만 Coons/Gordon prototype 적용 가능성을 평가한다.
- [ ] 기존 LSQ + boundary constraint보다 명확한 개선이 있을 때에만 main path로 승격한다.
- [ ] curve extraction 실패가 geometry fitting 실패로 오인되지 않도록 별도 benchmark와 상태 코드를 사용한다.

### Gaussian topology 방법론의 완료 조건

- close parallel sheet와 thin shell의 cross-surface adjacency가 지정 threshold 이하가 된다.
- near-spherical/low-confidence Gaussian을 포함해도 patch topology가 seed와 density에 대해 안정적이다.
- Gaussian covariance normal, PCA/voxel normal, fitted NURBS normal 간 mismatch가 diagnostic으로 재현 가능하다.
- topology boundary와 support boundary의 산출물·metric·lifecycle이 독립적으로 추적된다.

## Priority 6 — Real-data Input Eligibility

Synthetic constructor가 안정화되기 전에는 구현 우선순위에서 제외한다. 이후 별도 계층으로 추가한다.

```text
Raw COLMAP / trained 3DGS Gaussian
→ validity / eligibility filter
→ validated NURBS constructor
```

후보 신호:

- reprojection/image residual
- opacity/confidence
- multi-view visibility consistency
- local density/isolation
- scale/covariance anomaly
- normal consistency
- training age 또는 ADC provenance

이 계층은 geometry fitter 내부의 ad-hoc outlier rejection으로 숨기지 않는다. filter 전후 Gaussian count, rejected reason, patch 영향도를 기록한다.

## Priority 7 — Persistent lifecycle

- [ ] image residual을 patch basis weight로 backtracking한다.
- [ ] patch quality가 지속적으로 악화될 때 split/merge 후보를 생성한다.
- [ ] patch merge 정책과 merge validation을 구현한다.
- [ ] orphan patch cleanup을 구현한다.
- [ ] UV refresh, control-grid update, weight update, support-mask refresh의 실행 순서와 version contract를 정의한다.
- [ ] rational weight가 학습으로 변한 뒤 재-fitting/maintenance가 필요한 경우 nonlinear path를 분리 설계한다. 현재 LSQ는 weight=1일 때만 선형이다.

## Priority 8 — Training performance optimization 및 scene 품질 격차

성능 구현 항목(ADC 단일 shape transaction, bounded pinned-memory snapshot, final 중복 제거, maintenance patch budget)과 결정론적 순환 view sampling 교체는 완료했다. 완료 근거는 `docs/worklogs/57_priority8_training_performance.md`에 기록했으며, 완료된 세부 항목은 이 TODO에서 삭제했다.

남은 acceptance 검증:

- [ ] 동일 해상도 10k OSN-GS/Graphdeco A/B로 scene 품질 격차를 다시 측정한다.
- [ ] 같은 seed의 OSN-GS deterministic replay와 A/B 결과를 이용해 품질·학습시간·ADC spike regression 기준을 확정한다.

## 실행 순서

1. Renderer multi-patch/parity/provenance correctness.
2. Rectangular·elongated baseline과 anisotropic control resolution.
3. Non-rectangular boundary benchmark 및 local-density-adaptive support calibration.
4. UV support-mask lifecycle 안정화.
5. Curved chartability와 chart split 판정.
6. Gaussian covariance 기반 topology boundary와 multi-patch topology.
7. Boundary-constrained refitting 검증.
8. Real COLMAP/3DGS Gaussian eligibility.
9. Persistent lifecycle.
10. Training performance optimization.

## 당장 수행할 구현 단위

기존 fitting 알고리즘은 변경하지 않고 아래를 하나의 benchmark-focused change set으로 묶는다.

- [ ] Gaussian minimum-axis normal, planarity confidence, PCA/voxel normal 비교 artifact 추가.
- [ ] crease·curved ribbon·close parallel sheets용 topology boundary score diagnostic 추가.
- [ ] 기존 rectangular benchmark 결과가 변하지 않는지 regression 확인.

이 change set의 목적은 새 support 알고리즘을 즉시 확정하는 것이 아니라, **geometry fitting 실패와 support-domain estimation 실패를 독립적으로 측정할 수 있는 검증 기반을 만드는 것**이다.
