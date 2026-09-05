# W164 Canonical Primitive Observation A/B

이 output은 W163에서 확인한 median-ordering conflict를 대상으로 historical Candidate-B의 POINT_QUERY_STATE와 contributor-aware PRIMITIVE_OBSERVATION_STATE를 matched A/B로 비교한다. canonical Gaussian primitive g가 frozen canonical renderer의 기존 acceptance path에서 한 개 이상의 pixel에 accepted contributor로 기록되면, camera별 CONTRIBUTED_IN_CAMERA(g,v)=true이고 해당 primitive state만 OBSERVED로 override한다. arbitrary XYZ에는 이 규칙을 적용하지 않는다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

W163의 K=16 pixel prefix는 이 batch의 participation 판정에 사용하지 않는다. isolated diagnostic sibling의 per-primitive forward_accepted bit가 exact sparse predicate이며, contribution magnitude threshold, percentage, area, confidence, new T/alpha threshold는 없다. `POINT_QUERY_STATE`와 `PRIMITIVE_OBSERVATION_STATE`는 raw NPZ와 report에서 별도로 inspectable하다.

state palette: green=OBSERVED (0.10, 0.85, 0.35), red=OCCLUDED (0.92, 0.18, 0.18), gray=UNRESOLVED (0.60, 0.60, 0.62).

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
