# W164 Review Views

각 하위 directory는 동일한 canonical Gaussian population에 대한 한 가지 visualization semantics를 담는다. camera PNG는 `<camera_name_stem>.png` 형식으로 directory 바로 아래에 있고, 각 directory에 이 visualization의 의미, input/state semantics, palette/legend, 공통 rendering 조건, review limitation을 개별 기록한다.

공통 rendering 조건: frozen checkpoint/iteration, 1,190,469 Gaussian rows, 161 training cameras, 648x420 calibration, OSNSurfelRasterizer, black background. 모든 view는 동일한 camera, renderer, resolution, background, Gaussian row count를 사용하며 position, scale/covariance, rotation, opacity, geometry는 바꾸지 않고 display color만 바꾼다. 이 batch는 diagnostic-only이며 Candidate-B, W160 state/cache, W161, W162/W163, production renderer, Region, t_w, TSDF, topology, Boundary First, NURBS, continuation을 변경하지 않는다.

Historical A는 Gaussian center의 immutable Candidate-B POINT_QUERY_STATE를 사용한다. Candidate B는 그 local state에 exact per-primitive contributor positive evidence만 적용한다. common_world는 world XYZ diagnostic projection이며 W161 spatial field나 physical first-hit reconstruction이 아니다.
