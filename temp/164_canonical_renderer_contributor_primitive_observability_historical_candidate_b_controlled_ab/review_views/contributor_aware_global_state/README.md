# Contributor-Aware Global State

Historical POINT_QUERY_STATE에 대해 per-camera `CONTRIBUTED_IN_CAMERA(g,v)`가 true인 canonical primitive를 OBSERVED로 positive override한 뒤 frozen all-relevant aggregation을 적용한 Candidate B state이다. 색은 candidate PRIMITIVE_OBSERVATION_STATE의 global 결과를 나타낸다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

state palette: green=OBSERVED (0.10, 0.85, 0.35), red=OCCLUDED (0.92, 0.18, 0.18), gray=UNRESOLVED (0.60, 0.60, 0.62). Non-contributor의 historical OCCLUDED ordering은 그대로 유지된다. contributor count voting이나 confidence weighting은 없다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
