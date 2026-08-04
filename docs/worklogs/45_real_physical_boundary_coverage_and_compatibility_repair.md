# Worklog 45: Real Physical Boundary Coverage 및 Compatibility 복구

## 수행 내용

- Real 3k/5k/10k의 production physical ordering 입력이 각각 153/181/121인데도 closed loop와 materialized boundary가 0인 원인을 directed compatibility 단계에서 추적했다.
- Box fixture의 6번째 face(`face_pz`)가 닫히지 않는 직접 원인을 perimeter-adjacent missing edge 두 개로 분리했다. 두 pair 모두 topology support, 거리, lateral, normal gate는 통과했지만 target `boundary_direction` 부호가 source와 반대로 저장되어 `tan_align < -0.15`에서 거부됐다.
- `_compatible_directed_edges`에서 source tangent는 directed traversal의 forward/lateral gate로 유지하고, target tangent는 local boundary line orientation으로 보아 `abs(dot(source.boundary_direction, target.boundary_direction))`로 평가하도록 수정했다. threshold 완화, 강제 폐쇄, gap 보간, shape별 예외, NURBS fitting 변경은 하지 않았다.
- target tangent sign flip이 지원된 successor를 거부하지 않는 회귀 테스트를 추가했다.
- orientation repair 이후 기존 rigid-transform robustness fixture는 baseline 쪽만 정상 loop를 복구하고 transformed 쪽은 candidate-scarce(13 vs 4)로 남는 차이를 보였다. 테스트를 동일 성공 여부가 아니라 candidate evidence에 근거한 허용 가능한 divergence로 조정했다.

## 결과

### Before/After 핵심 수치

| 대상 | Before closed/materialized | After closed/materialized | 비고 |
| --- | ---: | ---: | --- |
| Real 3k | 0 / 0 | 0 / 0 | physical 153, directed edges 29 -> 53 |
| Real 5k | 0 / 0 | 2 / 2 | physical 181, directed edges 59 -> 98 |
| Real 10k | 0 / 0 | 0 / 0 | physical 121, directed edges 23 -> 38 |
| Box | 5 / 5 | 6 / 6 | 6번째 face 복구 |
| Cylinder | 3 / 3 | 3 / 3 | 유지 |
| Sphere | 0 / 0 | 0 / 0 | physical candidate 0 유지 |
| Thin slab | 2 / 2 | 2 / 2 | 양면 분리 유지 |
| Floater | 1 / 1 | 1 / 1 | floater closed-loop 유입 없음 |

Real composition after:

| checkpoint | sector-only physical sources | continuation-only physical sources | both | ordering physical | closed | materialized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3k | 279 | 112 | 47 | 153 | 0 | 0 |
| 5k | 293 | 88 | 96 | 181 | 2 | 2 |
| 10k | 272 | 70 | 54 | 121 | 0 | 0 |

## 평가

- 입증된 production 결함은 compatibility gate의 target tangent orientation 처리였다. target tangent 부호는 corner 또는 local frame transport에서 뒤집힐 수 있으므로 oriented vector 동등성으로 거부하면 실제 physical successor edge가 누락된다.
- 수정 후 Box 6번째 face가 닫히고, real 5k에서도 closed/materialized boundary가 0에서 2로 회복됐다. 3k/10k는 compatibility edge 수는 증가했지만 closed loop는 여전히 0이다.
- Negative controls는 유지됐다: cylinder closed=3, sphere physical=0, thin slab 분리, floater closed-loop 비유입, close/hole/topology safety focused tests 통과.

## 검증

- `python C:\tmp\osn_gs_boundary_old_gate_summary.py`: old-gate replay before 수치 확인.
- `python C:\tmp\osn_gs_boundary_after_summary.py`: 최종 after 수치 확인.
- `.venv\Scripts\python.exe -m pytest -q tests\test_directed_boundary_ordering.py tests\test_boundary_adjacency_semantics.py tests\test_boundary_topology_safety.py tests\test_full_cloud_continuation_shell.py tests\test_termination_neighborhood_scale_replay.py`: `64 passed in 19.30s`.
- `.venv\Scripts\python.exe -m pytest -q`: `715 passed, 1 skipped, 1 warning, 8 subtests passed in 183.22s`.

## 남은 단일 병목

- Real 3k/10k의 남은 병목은 directed ordering solver 자체가 아니라 physical candidate coverage/fragmentation이다. 주요 region에서 perimeter 후보가 sparse하게 생성되어 closed boundary를 만들 충분한 candidate chain이 아직 없다.