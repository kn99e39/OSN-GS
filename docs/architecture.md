# OSN-GS Architecture

OSN-GS는 3D Gaussian Splatting을 표면 중심(surface-centric) 구조로 확장하는 프레임워크이다. 기존 3DGS가 Gaussian primitive 자체를 장면 표현의 중심으로 최적화한다면, OSN-GS는 NURBS 기반 parametric surface를 장면의 canonical geometry로 두고 Gaussian을 그 표면에서 파생된 렌더링 샘플로 취급한다.

핵심 목표는 단순히 누락된 Gaussian을 추가하는 것이 아니라, 관측 표면에서 추출한 구조적 prior를 이용해 비관측 영역까지 이어지는 연속적인 표면 가설을 만들고, 그 표면 위에서 Gaussian 분포를 생성, 검증, 갱신하는 것이다.

```text
Scene
  -> NURBS Surface (Canonical Geometry)
  -> Certain / Uncertain Gaussian Distribution
  -> Differentiable Rendering
  -> Residual Analysis
  -> Surface Update
```

## Core Principles

### Surface-Centric Representation

NURBS surface는 OSN-GS의 단일한 geometric source of truth이다. Gaussian은 독립적인 geometry가 아니라 표면에서 평가된 위치와 방향을 가진 렌더링 instance로 해석한다.

이 관점에서 다음 원칙을 따른다.

- surface 수정은 Gaussian 위치와 normal을 갱신한다.
- ADC는 geometry 자체를 바꾸는 과정이 아니라 surface 위 sampling density를 조절하는 과정이다.
- rendering residual은 개별 Gaussian 오류에 머무르지 않고, 해당 Gaussian이 속한 surface patch와 control point를 검증하는 신호로 사용한다.

### Persistent Gaussian-Surface Binding

모든 Gaussian은 자신이 어떤 surface patch와 parameter 위치에서 파생되었는지 저장한다.

각 Gaussian이 보관해야 하는 surface 관련 정보:

- patch id
- `(u, v)` surface parameter
- surface normal
- observed / occluded flag
- confidence
- optional basis-function weights
- ADC history

따라서 렌더링 결과의 오류는 다음 경로로 역추적할 수 있다.

```text
Pixel
  -> Gaussian
  -> Patch
  -> (u, v)
  -> Surface Control Points
```

이 binding은 OSN-GS가 단순 Gaussian cloud가 아니라 surface hypothesis를 검증하는 구조가 되기 위한 핵심이다.

## Motivation

### Why 3DGS

비관측 표면을 예측하려면 관측 표면으로부터 명시적인 geometry sample을 얻을 수 있어야 한다. NeRF나 NeuS 계열은 scene을 implicit field로 표현하므로 관측된 표면 위의 구조적 패턴을 직접 추출해 새로운 Gaussian 배치로 연결하기 어렵다.

3DGS는 Gaussian의 위치, covariance, opacity, color가 명시적으로 존재하므로 다음 작업에 적합하다.

- observed surface point cloud 추출
- local geometry와 visibility 기반 filtering
- surface curve fitting
- surface parameter domain 위 Gaussian sampling
- certain / uncertain Gaussian 분리 학습
- ADC 패턴과 rendering residual 분석

### Why NURBS

비관측 표면에 Gaussian을 배치하려면 전체 표면 구조를 표현하는 중간 representation이 필요하다. Mesh는 vertex 배치에 대한 별도 prior가 필요하고, 관측된 vertex 간의 구조적 관계를 안정적으로 외삽하기 어렵다. SDF는 관측 이미지에서 표면을 찾는 데 강점이 있으나, 관측되지 않은 표면을 구조적으로 연장하는 직접 제약은 부족하다.

NURBS는 control point, knot vector, degree, weight를 통해 연속적인 parametric surface를 표현할 수 있으므로 다음 장점이 있다.

- 관측 curve에서 비관측 curve로 구조를 확장하기 쉽다.
- smoothness, curvature, continuity를 명시적으로 제어할 수 있다.
- surface parameter domain 위에서 Gaussian을 안정적으로 샘플링할 수 있다.
- Gaussian residual을 surface patch와 control point 갱신으로 되돌리기 쉽다.

## High-Level Pipeline

```text
Scene Loader
  -> Initial 3DGS Gaussians
  -> Observed Gaussian Filtering
  -> Observed Surface Point Cloud
  -> Base Curve Extraction
  -> Structural Prior Analysis
  -> Occlusion Curve Prediction
  -> NURBS Surface Construction
  -> Uncertain Gaussian Sampling
  -> Joint Rendering
  -> Residual Analysis
  -> Surface Update
  -> Surface-aware ADC
```

현재 구현의 우선순위는 visible surface reconstruction과 Gaussian-surface binding을 안정화하는 것이다. Occluded surface generation, uncertain-to-certain promotion, full surface update policy는 별도 단계로 분리한다.

## Surface Reconstruction

### Observed Surface Point Cloud

초기 Gaussian center를 observed surface point cloud로 사용한다. 필요하다면 opacity, scale, visibility score, normal confidence를 기준으로 신뢰 가능한 Gaussian만 필터링한다.

### Base Curve Extraction

관측 point cloud에서 base curve를 추출한다. curve extraction은 다음 신호를 함께 고려한다.

- local geometry grouping
- principal direction estimation
- normal consistency
- visibility confidence
- color or appearance consistency
- structural continuity

### Structural Prior Analysis

관측 curve를 비관측 영역으로 연장하기 위해 다음 prior를 분석한다.

- tangent continuity
- curvature continuity
- directional consistency
- local repetition
- shape regularity
- neighboring patch relation

이 prior는 정답 표면을 직접 보장하지 않고, occluded curve와 surface patch 후보를 만드는 structural hypothesis로 취급한다.

### Algebraic Extension

관측 curve는 비관측 영역으로 algebraic하게 확장된다. exact extension operator는 OSN-GS의 주요 연구 문제로 남긴다. 초기 구현에서는 tangent, curvature, repetition prior를 이용한 보수적 curve continuation을 우선한다.

### NURBS Surface Construction

관측 curve와 예측 curve는 함께 NURBS surface patch를 정의한다. 각 patch는 다음 정보를 가진다.

- control points
- knot vectors
- degree
- weights
- parameter domain
- observed / occluded region mask
- neighbor relations
- structural prior metadata
- associated Gaussian references

## Gaussian Representation

### Certain Gaussian

Certain Gaussian은 충분한 관측 근거가 있는 Gaussian이다. 기존 3DGS 학습 과정에서 생성되거나 COLMAP/초기 3DGS로부터 안정적으로 추출된 관측 표면 sample로 해석한다.

주요 학습 신호:

- image similarity loss
- opacity and scale regularization
- standard 3DGS adaptive density control
- color and spherical harmonics optimization

### Uncertain Gaussian

Uncertain Gaussian은 NURBS parameter domain에서 직접 생성되는 Gaussian이다.

```text
(u, v)
  -> Surface Evaluation
  -> 3D Position / Normal
  -> Gaussian Attributes
```

Uncertain Gaussian의 역할은 단순히 빈 공간의 density를 채우는 것이 아니라, supporting surface hypothesis가 rendering 관점에서 일관적인지 검증하는 것이다.

Uncertain Gaussian이 지속적인 image residual을 발생시키면, 이는 다음 중 하나를 의미할 수 있다.

- Gaussian appearance 초기화 오류
- Gaussian covariance 또는 opacity 오류
- surface parameter 위치 오류
- supporting NURBS patch 또는 control point 오류
- occluded curve hypothesis 오류

따라서 uncertain residual은 Gaussian parameter만 직접 수정하는 신호가 아니라 surface backtracking과 surface update 후보로 전달되어야 한다.

## Visibility-Driven Validation

OSN-GS는 uncertain Gaussian을 geometric probe로 사용한다. 렌더링 residual은 Gaussian-surface binding을 통해 surface hypothesis로 되돌아간다.

```text
Rendered Pixel
  -> Residual
  -> Uncertain Gaussian
  -> Patch ID
  -> (u, v)
  -> Basis Weights
  -> Surface Control Points
```

이 구조 덕분에 image-space residual을 직접 surface geometry에 연결할 수 있다. surface correction은 개별 pixel에서 바로 geometry를 추정하는 방식이 아니라, Gaussian이 가진 patch association과 parameter coordinate를 통해 수행한다.

## Surface-Aware Adaptive Density Control

기존 3DGS의 ADC는 Gaussian primitive의 clone, split, prune으로 density를 조절한다. OSN-GS에서는 ADC를 surface 위 adaptive sampling으로 재해석한다.

```text
Surface
  -> Sampling
  -> Gaussian
  -> ADC Signal
  -> Higher or Lower Surface Sampling Density
```

원칙:

- Certain Gaussian은 기존 3DGS ADC 정책을 따른다.
- Surface-bound Gaussian의 child는 parent의 patch id, `(u, v)` neighborhood, normal, confidence metadata를 상속한다.
- Uncertain Gaussian은 독립적으로 geometry를 바꾸기보다 surface parameter domain의 sampling density를 조절한다.
- 지속적인 high residual은 단순 densification이 아니라 surface update 후보로 전달한다.
- uncertain-to-certain promotion은 현재 구현에서 금지한다. promotion 정책은 추후 별도 stage에서 정의한다.

## Internal Data Model

### Gaussian Record

각 Gaussian은 다음 정보를 포함한다.

- xyz position
- covariance scale and rotation
- opacity
- color or spherical harmonics attributes
- patch id
- surface parameter `(u, v)`
- surface normal
- observed / occluded flag
- confidence
- ADC history
- optional basis-function weights

### Surface Patch Record

각 NURBS surface patch는 다음 정보를 포함한다.

- control points
- knot vectors
- degree
- weights
- parameter domain
- observed / occluded mask
- neighbor patch relations
- structural prior
- Gaussian references
- residual statistics

## Training Loop

```text
for iteration in training_iterations:
    batch = scene_loader.sample_views()

    rendered = rasterizer.render(
        certain_gaussians,
        uncertain_gaussians,
        cameras=batch.cameras,
    )

    image_loss = image_similarity(rendered, batch.images)
    surface_loss = nurbs_regularization(nurbs_surface)
    uncertainty_loss = residual_to_surface_loss(rendered, batch.images, bindings)

    total_loss = image_loss + surface_loss + uncertainty_loss
    total_loss.backward()

    update_gaussian_attributes()
    update_surface_bound_gaussian_positions()

    if should_analyze_residuals(iteration):
        backtrack_residuals_to_surface()
        mark_surface_update_candidates()

    if should_update_surface(iteration):
        update_base_curves()
        update_occlusion_curves()
        rebuild_nurbs_surface()
        refresh_surface_bound_gaussians()

    if should_run_density_control(iteration):
        run_certain_adc()
        run_surface_aware_adc()
```

## Module Responsibilities

현재 코드베이스의 구현은 Torch path를 중심으로 진행한다. 아래 모듈 경계는 목표 구조이며, 실제 파일명은 구현 단계에 따라 `torch_*` 계열로 존재할 수 있다.

### `osn_gs/core`

전체 실행 흐름, pipeline, trainer, state를 관리한다.

- training loop
- Gaussian/NURBS lifecycle
- streaming and output coordination
- surface update scheduling

### `osn_gs/gaussian`

Gaussian model, covariance, opacity, color, ADC, surface binding metadata를 관리한다.

- certain / uncertain Gaussian distinction
- Graphdeco-style primitive export
- covariance initialization and optimization
- surface-aware density control

### `osn_gs/surface`

Observed point cloud, curve extraction, NURBS construction, sampling, structural prior를 관리한다.

- visible surface reconstruction
- NURBS intermediate representation
- surface parameter sampling
- residual-to-surface metadata

### `osn_gs/render`

3DGS rasterizer를 OSN-GS 학습 루프에 연결한다.

- CUDA rasterizer bridge
- Torch fallback renderer
- packed Gaussian streaming for external visualization

### `osn_gs/data`

COLMAP/Graphdeco-style scene, camera, image batch를 로드한다.

- per-view image staging
- camera transforms
- training view sampling

## Current Implementation Boundary

현재 구현은 다음 범위를 우선한다.

- COLMAP/초기 Gaussian에서 visible surface NURBS intermediate 생성
- NURBS를 최종 출력이 아니라 메모리 및 출력 manifest의 중간 산출물로 유지
- Gaussian primitive와 NURBS visualization payload를 외부 렌더러로 streaming
- covariance scale을 original 3DGS 방식에 가깝게 nearest-neighbor spacing에서 초기화
- ADC를 basic 3DGS style로 연결하되, uncertain-to-certain promotion은 금지
- output iteration folder는 숫자 이름을 사용

아직 별도 stage로 남겨둔 범위:

- full occluded surface generation
- algebraic extension operator 확정
- residual 기반 NURBS control point update
- promotion policy
- surface-aware ADC의 완전한 sampling-density formulation

## Key Research Questions

1. 관측 Gaussian에서 안정적인 base curve를 어떻게 추출할 것인가?
2. Algebraic extension operator를 어떤 형태로 정의할 것인가?
3. 어떤 structural prior가 occluded curve prediction에 가장 효과적인가?
4. Uncertain residual을 appearance 오류, covariance 오류, surface 오류 중 어떻게 분해할 것인가?
5. Visibility-driven validation을 surface control point update로 어떻게 안정적으로 연결할 것인가?
6. Surface update가 너무 잦을 때 학습 안정성이 깨지지 않도록 어떤 scheduler를 둘 것인가?
7. Surface-aware ADC를 clone/split/prune이 아니라 sampling density 조절로 어떻게 정식화할 것인가?
8. Uncertain Gaussian의 color, covariance, opacity를 surface와 neighboring certain Gaussian에서 어떻게 초기화할 것인가?
9. Uncertain-to-certain promotion을 허용한다면 어떤 조건과 검증 stage가 필요한가?

## Implementation Roadmap

### Phase 1: 3DGS Baseline Bridge

- Torch-based Gaussian model 구성
- CUDA rasterizer 또는 Torch fallback renderer 연결
- COLMAP/Graphdeco scene loader 연결
- Graphdeco-style PLY export와 external renderer streaming 지원

### Phase 2: Visible NURBS Reconstruction

- Gaussian center 기반 observed point cloud 생성
- visible surface filtering
- base curve extraction prototype
- NURBS-like visible surface intermediate 생성
- NURBS payload 저장 및 streaming

### Phase 3: Persistent Surface Binding

- Gaussian에 patch id, `(u, v)`, normal, confidence metadata 저장
- NURBS surface patch가 Gaussian reference를 추적
- renderer output residual을 Gaussian-surface binding으로 backtracking

### Phase 4: Surface-Derived Uncertain Gaussian

- NURBS parameter domain sampling
- uncertain Gaussian 초기 위치, normal, covariance, color prior 생성
- visibility-driven validation loss 추가
- surface consistency regularization 추가

### Phase 5: Surface Update and Surface-Aware ADC

- residual 기반 curve/surface update scheduler 구현
- control point update 또는 patch rebuild policy 구현
- ADC를 surface sampling density 조절로 재정의
- promotion policy는 별도 실험 stage에서만 검토

## Expected Contributions

- Surface-centric 3D Gaussian representation
- Explicit NURBS reconstruction for visible and eventually occluded geometry
- Persistent Gaussian-surface association
- Visibility-driven surface validation
- Surface-aware adaptive density control
- NURBS 기반 parametric surface와 3DGS 학습 루프의 결합 방식
