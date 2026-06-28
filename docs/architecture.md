# OSN-GS Architecture

OSN-GS는 관측된 표면 위에 투사된 Gaussian들을 이용해 NURBS 기반의 parametric surface를 구성하고, 그 구조적 특성을 바탕으로 가려진 표면에 Gaussian을 배치하는 3D Gaussian Splatting 프레임워크이다.

핵심 목표는 기존 3DGS가 약한 비관측 영역의 표면 구조를 단순 visibility, MVS, pseudo-view 보강이 아니라 관측 표면에서 추출한 구조적 prior로 예측하는 것이다.

## Core Idea

1. 기존 3DGS와 같이 초기 Gaussian을 생성하고 학습을 시작한다.
2. 초기 Gaussian들을 point cloud로 간주해 관측된 표면 위의 base curve를 추정한다.
3. base curve들의 구조적 연속성, 곡률, 방향성, 반복 패턴을 이용해 occluded space에 대응되는 curve를 생성한다.
4. 관측 curve와 추정 curve를 함께 사용해 NURBS surface를 구성한다.
5. NURBS surface 위의 비관측 영역에 uncertain Gaussian을 샘플링한다.
6. 이미지 기반 학습을 반복하면서 certain Gaussian은 일반 3DGS 방식으로 최적화하고, uncertain Gaussian은 렌더링 loss와 surface consistency를 이용해 위치와 surface basis를 갱신한다.

## Motivation

### Why 3DGS

비관측 표면을 예측하려면 관측 표면으로부터 명시적인 구조 representation을 얻을 수 있어야 한다. NeRF나 NeuS 계열은 scene을 implicit field로 표현하므로, 관측된 표면 위의 구조적 패턴을 직접 추출해 새로운 Gaussian 배치로 연결하기 어렵다.

3DGS는 Gaussian의 위치, covariance, opacity, color가 명시적으로 존재하므로 다음 작업에 적합하다.

- 관측 표면 point cloud 추출
- surface curve fitting
- 비관측 영역 후보 위치 샘플링
- certain/uncertain Gaussian 분리 학습
- density control 패턴 분석

### Why NURBS

비관측 표면에 Gaussian을 배치하려면 전체 표면 구조를 표현하는 중간 representation이 필요하다.

Mesh는 vertex 배치에 대한 별도 prior가 필요하고, 관측된 vertex 간의 구조적 관계를 안정적으로 추출하기 어렵다. SDF는 관측 이미지에서 표면을 찾는 데 강점이 있으나, 관측되지 않은 표면을 구조적으로 외삽하는 데에는 직접적인 제약이 부족하다.

NURBS는 control point, knot, degree, weight를 통해 연속적인 parametric surface를 표현할 수 있으므로 다음 장점이 있다.

- 관측 curve에서 비관측 curve로 구조를 확장하기 쉽다.
- smoothness, curvature, continuity를 명시적으로 제어할 수 있다.
- surface parameter domain 위에서 Gaussian을 안정적으로 샘플링할 수 있다.
- base curve 재계산을 통해 uncertain Gaussian의 위치 보정과 연결하기 쉽다.

## High-Level Pipeline

```text
Scene Loader
    -> Initial 3DGS Gaussians
    -> Observed Surface Point Cloud
    -> Base Curve Fitting
    -> Occlusion Curve Prediction
    -> NURBS Surface Construction
    -> Uncertain Gaussian Sampling
    -> Joint Optimization
    -> Curve and Surface Update
```

## Gaussian Types

### Certain Gaussian

Certain Gaussian은 기존 3DGS 학습 과정에서 생성되거나, 충분한 관측 근거가 있는 Gaussian이다.

주요 학습 신호:

- image similarity loss
- opacity and scale regularization
- standard 3DGS adaptive density control
- color and spherical harmonics optimization

담당 모듈:

- `osn_gs/gaussian/certain_gaussians.py`
- `osn_gs/gaussian/projection.py`
- `osn_gs/losses/image_similarity.py`

### Uncertain Gaussian

Uncertain Gaussian은 NURBS surface의 비관측 영역 위에 배치되는 Gaussian이다. 이 Gaussian은 직접 관측된 표면에서 온 것이 아니므로, 초기에는 구조적 추정에 기반한 후보로 취급한다.

중요한 해석:

Uncertain Gaussian이 image similarity loss를 크게 발생시킨다면, 이는 해당 Gaussian이 잘못된 위치 또는 잘못된 surface 위에 배치되었을 가능성을 의미한다. 따라서 단순히 Gaussian parameter만 최적화하는 것이 아니라, NURBS base curve와 surface 자체를 재계산하는 신호로 사용한다.

주요 학습 신호:

- image similarity residual
- NURBS surface consistency
- neighboring certain Gaussian cluster prior
- curve smoothness and continuity prior

담당 모듈:

- `osn_gs/gaussian/uncertain_gaussians.py`
- `osn_gs/surface/sampling.py`
- `osn_gs/optim/curve_update.py`
- `osn_gs/losses/uncertainty.py`

## Surface Reconstruction

### Observed Point Cloud

초기 Gaussian의 center를 point cloud로 사용한다. 필요하다면 opacity, scale, normal confidence, visibility score를 기준으로 관측 표면에 해당하는 Gaussian만 필터링한다.

담당 모듈:

- `osn_gs/surface/point_cloud.py`

### Base Curve Fitting

관측 표면 point cloud에서 base curve를 생성한다. 초기 구현에서는 다음 기준을 고려한다.

- local geometry grouping
- normal consistency
- principal direction estimation
- color cluster consistency
- camera visibility confidence

담당 모듈:

- `osn_gs/surface/base_curves.py`
- `osn_gs/surface/structural_prior.py`

### Occlusion Curve Prediction

base curve의 방향성, 곡률, 간격, 반복성을 이용해 occluded space 안의 curve를 추정한다. 이 단계는 OSN-GS의 핵심 차별점이다.

추정된 curve는 직접 정답이 아니라 NURBS surface를 생성하기 위한 structural hypothesis로 취급한다.

담당 모듈:

- `osn_gs/surface/occlusion_curves.py`
- `osn_gs/surface/structural_prior.py`

### NURBS Surface Construction

관측 base curve와 추정 occlusion curve를 함께 사용해 NURBS surface를 만든다.

NURBS surface는 다음 정보를 가진다.

- control points
- degree
- knot vectors
- weights
- parameter domain
- observed/occluded region mask

담당 모듈:

- `osn_gs/surface/nurbs_surface.py`

## Color Assignment

Uncertain Gaussian의 색상은 직접 관측된 색상이 없으므로 certain Gaussian의 색상 분포를 기반으로 초기화한다.

초기 전략:

1. Certain Gaussian을 색상 또는 spherical harmonics coefficient 기준으로 clustering한다.
2. 각 uncertain Gaussian을 가장 가까운 surface region 또는 curve neighborhood의 cluster에 할당한다.
3. 할당된 cluster의 color prior를 uncertain Gaussian에 공유한다.
4. 학습 중 image residual이 낮아지는 방향으로 color parameter를 제한적으로 업데이트한다.

담당 모듈:

- `osn_gs/gaussian/color_clusters.py`

## Adaptive Density Control

Certain Gaussian은 기존 3DGS의 adaptive density control을 따른다.

Uncertain Gaussian은 독립적으로 densify/prune하기보다, 같은 color/geometry cluster에 속한 certain Gaussian의 ADC 패턴을 모방한다.

초기 전략:

- 같은 cluster의 certain Gaussian split/clone/prune 통계를 기록한다.
- surface parameter domain 위에서 uncertain Gaussian의 density를 보정한다.
- image residual이 지속적으로 큰 uncertain Gaussian은 위치 이동 또는 curve update 후보로 넘긴다.
- confidence가 충분히 높아진 uncertain Gaussian은 certain Gaussian으로 승격할 수 있다.

담당 모듈:

- `osn_gs/gaussian/density_control.py`

## Training Loop

```text
for iteration in training_iterations:
    batch = scene_loader.sample_views()

    rendered = rasterizer.render(
        certain_gaussians,
        uncertain_gaussians,
        cameras=batch.cameras,
    )

    certain_loss = image_similarity(rendered, batch.images)
    uncertain_loss = uncertainty_loss(rendered, batch.images, nurbs_surface)
    surface_loss = nurbs_regularization(nurbs_surface)

    total_loss = certain_loss + uncertain_loss + surface_loss
    total_loss.backward()

    update_certain_gaussians()
    update_uncertain_gaussians()

    if should_update_curves(iteration):
        update_base_curves()
        update_occlusion_curves()
        rebuild_nurbs_surface()
        resample_uncertain_gaussians()

    if should_run_density_control(iteration):
        run_certain_adc()
        run_uncertain_adc_from_cluster_patterns()
```

담당 모듈:

- `osn_gs/core/trainer.py`
- `osn_gs/core/pipeline.py`
- `osn_gs/core/state.py`
- `osn_gs/render/rasterizer_adapter.py`

## Module Responsibilities

### `osn_gs/core`

전체 프레임워크의 실행 흐름을 관리한다.

- `framework.py`: OSN-GS 상위 API
- `pipeline.py`: surface construction과 Gaussian update 단계 연결
- `trainer.py`: 학습 루프
- `state.py`: Gaussian, NURBS, optimizer, iteration state 보관

### `osn_gs/gaussian`

Certain/uncertain Gaussian의 생성, 업데이트, 색상, density control을 담당한다.

- `certain_gaussians.py`: 관측 기반 Gaussian container와 update
- `uncertain_gaussians.py`: NURBS 기반 Gaussian container와 confidence 관리
- `projection.py`: Gaussian projection 및 observed surface point 추출
- `color_clusters.py`: 색상 기반 cluster prior
- `density_control.py`: certain ADC와 uncertain ADC 모방 정책

### `osn_gs/surface`

Point cloud에서 curve, NURBS surface, Gaussian sampling 위치를 생성한다.

- `point_cloud.py`: Gaussian center 기반 point cloud 변환과 필터링
- `base_curves.py`: 관측 표면 base curve fitting
- `occlusion_curves.py`: 비관측 영역 curve prediction
- `nurbs_surface.py`: NURBS surface representation
- `sampling.py`: NURBS surface 위 Gaussian sampling
- `structural_prior.py`: curve continuity, curvature, repetition prior

### `osn_gs/losses`

이미지 기반 loss와 surface 관련 regularization을 정의한다.

- `image_similarity.py`: L1, SSIM, perceptual 형태의 image loss
- `nurbs_regularization.py`: smoothness, curvature, continuity regularization
- `uncertainty.py`: uncertain Gaussian confidence와 position correction loss

### `osn_gs/render`

기존 3DGS rasterizer를 OSN-GS 학습 루프에 맞게 감싼다.

- `rasterizer_adapter.py`: certain/uncertain Gaussian을 함께 렌더링하는 adapter

### `osn_gs/optim`

Curve와 surface 갱신 정책, scheduler를 담당한다.

- `curve_update.py`: uncertain residual을 기반으로 base curve와 occlusion curve 재계산
- `schedulers.py`: curve update, NURBS rebuild, ADC 실행 주기 관리

### `osn_gs/data`

Scene, camera, image batch를 로드한다.

- `scene_loader.py`: dataset entry point
- `cameras.py`: camera parameter wrapper

## Key Research Questions

1. 관측 Gaussian에서 안정적인 base curve를 어떻게 추출할 것인가?
2. Occluded curve prediction에서 어떤 structural prior가 가장 효과적인가?
3. Uncertain Gaussian의 image loss를 위치 오류, 색상 오류, 표면 오류 중 무엇으로 해석할 것인가?
4. 언제 uncertain Gaussian을 certain Gaussian으로 승격할 것인가?
5. Certain Gaussian의 ADC 패턴을 uncertain Gaussian에 어느 정도까지 모방시킬 것인가?
6. NURBS surface update가 너무 잦을 때 학습 안정성이 깨지지 않도록 어떤 scheduler를 둘 것인가?

## Implementation Roadmap

### Phase 1: Skeleton and Baseline Bridge

- 기존 3DGS 학습 코드와 연결할 rasterizer adapter 작성
- Gaussian container 인터페이스 정의
- Scene loader와 config 구조 정의
- 기본 train script 작성

현재 구현 상태:

- `TorchGaussianModel`은 3DGS의 `GaussianModel` 속성 계약과 유사하게 `get_xyz`, `get_features`, `get_opacity`, `get_scaling`, `get_rotation`을 제공한다.
- `TorchRasterizerAdapter`는 `diff_gaussian_rasterization`이 설치되어 있으면 CUDA rasterizer를 사용하고, 없으면 Torch fallback renderer를 사용한다.
- `TorchOSNGSTrainer`는 학습 결과로 PLY, PPM preview, Torch checkpoint, metrics file을 저장한다.

### Phase 2: Observed Surface Curves

- Gaussian center를 point cloud로 변환
- 관측 confidence 기반 filtering
- base curve fitting prototype 작성
- curve visualization 도구 작성

### Phase 3: NURBS Surface and Uncertain Gaussian

- NURBS surface representation 구현
- surface parameter domain sampling 구현
- uncertain Gaussian container 구현
- color cluster 기반 초기 색상 할당

### Phase 4: Joint Optimization

- certain/uncertain Gaussian joint rendering
- uncertainty loss 추가
- curve update scheduler 추가
- NURBS rebuild와 uncertain resampling 연결

### Phase 5: ADC and Promotion Policy

- certain cluster별 ADC 패턴 기록
- uncertain Gaussian density control 구현
- uncertain to certain promotion rule 구현
- ablation 실험 구성

## Expected Contributions

- 관측 표면의 구조적 특성을 이용해 비관측 표면을 추정하는 3DGS 확장 프레임워크
- NURBS 기반 parametric surface를 3DGS 학습 루프에 결합하는 방법
- Certain/uncertain Gaussian 분리와 confidence 기반 갱신 전략
- Color cluster와 ADC pattern transfer를 이용한 비관측 Gaussian 초기화 및 density control
