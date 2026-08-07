# Worklog 72: Dense Region-Owned Boundary Support 및 Topology

## 구현

`osn_gs/surface/torch_region_owned_dense_boundary_support.py`를 추가했다. Worklog 71의 sparse typed half-edge loop를 입력으로 쓰지 않고, ownership gate를 통과한 region-owned full-cloud evidence에서 kNN tangent-plane angular gap을 측정해 local `observed_support_termination` 후보를 만든다. 후보 reason은 nearby sparse seed에서 상속하지 않는다.

연결은 raw radius graph가 아니다. full-evidence median nearest-neighbor scale을 별도 계산하고, 각 후보의 `+/- tangent` half-line에서 distance/local-scale, reason, tangent, normal을 모두 통과한 가장 가까운 후보만 선택한 뒤 상호 선택된 edge만 남긴다. 동등 경쟁은 임의 prune하지 않고 `ambiguity`로 기록한다. `distance_local_scale`, `reason_incompatibility`, `tangent_mismatch`, `normal_mismatch`, `ambiguity` rejection attribution을 반환한다. representative `mean_spacing`은 report-only로 분리 보존했다.

Worklog 71 replay script에는 additive `dense_boundary_support` report를 연결했다. 기존 sparse topology/fitting을 변경하지 않으며 dense closed boundary가 없으면 NURBS fitting 또는 forced containment 검사는 수행하지 않는다. crossing/planarity는 loop가 있을 때만 별도 보고하고 surface self-intersection은 계속 `not_checked`다.

## 실제 baseline-compatible@2900 측정

7개 region에서 full-evidence scale은 0.0234--0.0431, representative mean spacing은 0.1085--0.2542로 명확히 분리됐다. local boundary-support 후보는 785개로 Worklog 71 sparse seed보다 훨씬 넓게 포착됐지만, mutual directional connectivity 결과는 `open_or_ambiguous` 621개, closed loop 0개였다. 따라서 `interior_outside_boundary`를 비교할 evidence-containing recovered loop가 없고 fitting은 0회다.

rejection attribution은 distance/local-scale 244,224, tangent mismatch 400, normal mismatch 104, ambiguity 10이다. 결론은 boundary support 자체의 부재가 아니라, full evidence에서 추출된 local support가 현재 orientation/directional evidence로는 닫힌 ordered topology를 증명하지 못한다는 것이다. 기하 fallback, gap bridging, scale 확장, seed-loop densification은 적용하지 않았다.

참조 baseline@2900도 후보 5,605개와 open/ambiguous 4,601개, closed loop 0개로 같은 fail-closed 양상을 보였다.

## 검증

`tests/test_region_owned_dense_boundary_support.py`와 `tests/test_region_owned_full_evidence_boundary_topology.py`: **16 passed**. Full pytest는 지시대로 실행하지 않았다.

## 다음 위험

현 predicate는 local observed support만 사용한다. 닫힌 topology를 주장하려면 open support가 실제 boundary gap인지, orientation/normal disagreement인지, 혹은 region evidence가 genuinely non-single-chart인지 더 강한 관측 근거가 필요하다. 이 결과를 이유로 hull, PCA rectangle, forced closure, 타-region merge를 도입하면 안 된다.
