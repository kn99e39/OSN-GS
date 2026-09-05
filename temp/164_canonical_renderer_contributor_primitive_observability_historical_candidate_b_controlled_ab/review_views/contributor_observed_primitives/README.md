# Contributor-Observed Primitives

161개 training camera 중 하나 이상에서 exact `forward_accepted` bit가 true인 canonical Gaussian primitive를 blue로 표시한다. contributor가 0개 camera인 primitive와 나머지 context는 gray이다. 이 view는 global state가 아니라 positive primitive evidence coverage를 나타낸다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Legend: blue=CONTRIBUTED_IN_CAMERA in at least one camera, gray=no positive contributor evidence/context. zero contributor는 OCCLUDED proof가 아니다.

Review limitation: PRIMITIVE_OBSERVATION_STATE는 canonical primitive에 renderer-native positive observation evidence가 있다는 뜻일 뿐 physical first-hit truth가 아니다. POINT_QUERY_STATE와 arbitrary XYZ occlusion은 별도 의미이며, contributor가 없는 Gaussian은 OCCLUDED의 증거로 해석하지 않는다.
