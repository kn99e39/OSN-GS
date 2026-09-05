# Baseline Occluded to Candidate Observed

Historical GLOBAL OCCLUDED였지만 contributor-aware candidate에서 GLOBAL OBSERVED가 된 Gaussian만 yellow로 표시한다. 다른 row는 gray context이다. 이 view는 A/B transition의 changed population만 공간적으로 보여준다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Legend: yellow=historical GLOBAL OCCLUDED → candidate GLOBAL OBSERVED, gray=other/context. yellow는 canonical primitive observation evidence가 추가된 transition이지 physical hidden-surface 판정이 아니다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
