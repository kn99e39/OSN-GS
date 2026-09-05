# Historical Global State

각 PNG는 Gaussian center에 대해 frozen Candidate-B의 POINT_QUERY_STATE를 161개 relevant-camera aggregation으로 계산한 historical global state를 표시한다. green=GLOBAL OBSERVED, red=GLOBAL OCCLUDED, gray=GLOBAL UNRESOLVED이다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

state palette: green=OBSERVED (0.10, 0.85, 0.35), red=OCCLUDED (0.92, 0.18, 0.18), gray=UNRESOLVED (0.60, 0.60, 0.62). 이 view는 W160/W162 historical baseline을 재현하며 PRIMITIVE_OBSERVATION_STATE나 contributor positive override를 포함하지 않는다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
