# Common World Views

`perspective.png`, `top.png`, `side.png`는 전체 canonical Gaussian world XYZ를 X-Z, X-Y, Y-Z로 투영한 common-world diagnostic이다. historical GLOBAL OCCLUDED 전체는 dark red, candidate GLOBAL OCCLUDED는 light red, historical OCCLUDED → candidate OBSERVED는 yellow로 구분한다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Legend: dark red=historical GLOBAL OCCLUDED, light red=candidate GLOBAL OCCLUDED, yellow=baseline OCCLUDED → candidate OBSERVED, gray=display-only context. camera perspective가 아니며 W161 spatial domain을 만들거나 Gate O2를 닫지 않는다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
