# Contributor-Aware Global Occluded Only

Contributor-aware candidate에서 GLOBAL OCCLUDED로 남은 Gaussian만 red로 표시하고 나머지는 gray context로 표시한다. red population의 감소는 오직 exact per-primitive contributor positive evidence에 의한 것이다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Legend: red=candidate GLOBAL OCCLUDED, gray=other/context. Candidate의 red는 contributor가 없는 경우에도 historical OCCLUDED가 유지된 결과일 수 있으며, non-contribution 자체를 physical proof로 읽지 않는다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
