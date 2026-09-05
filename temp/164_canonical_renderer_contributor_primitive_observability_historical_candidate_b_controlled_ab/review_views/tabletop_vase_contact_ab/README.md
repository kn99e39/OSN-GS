# Tabletop/Vase Contact A/B

W162에 이미 존재하는 fixed tabletop 및 vase/curved-neighbor review record의 union만 사용해 contact population을 고정한다. candidate state를 표시하고 baseline GLOBAL OCCLUDED → candidate GLOBAL OBSERVED transition은 yellow로 overlay한다. 새 ROI, 새 Region, 거리 threshold는 만들지 않는다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Legend: green/red/gray=candidate contact-population global state; yellow=A/B changed row; gray=context.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
