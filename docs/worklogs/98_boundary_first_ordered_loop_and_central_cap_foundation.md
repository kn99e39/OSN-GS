# Worklog 98 — Ordered Loop 및 Central-cap 기반

## 수행

- raw boundary-cell 집합과 ordered correspondence polyline을 분리했다.
- label mask의 marching-squares segment를 연결하는 `ordered_closed_boundary_world_loops`를 추가했다.
- open/branched contour는 임의로 닫지 않고 ordered loop를 제공하지 않는다.
- support-network는 raw boundary-cell fallback을 제거했고 ordered contour가 없으면 `ordered_boundary_required`로 보류한다.
- outer-only disk-like 입력을 위해 실제 component 관측점에서 선택한 medoid anchor와 explicit pole-aware central-cap patch foundation을 추가했다.
- 격리 visible builder/pipeline은 component 관측점과 원래 Gaussian index를 central-cap 경로에 전달한다.

## 검증

- ordered closed square 결정성 및 open contour 보류 테스트 통과.
- ordered builder/pipeline 관련 8개 테스트 통과.
- 새 모듈·연결 코드 `py_compile` 통과.
- 이후 runtime import는 공유 작업 트리의 append adapter가 누락된 `project_and_register_occluded_chart_owner_id`를 import해 실패했다. 이 Worklog 범위의 코드 변경과 무관한 별도 Agent 범위이며, full runtime suite는 해당 결함 해소 뒤 재실행이 필요하다.

## 제한

- central cap은 pole singularity를 provenance에 명시한다. 일반 Jacobian-positive 규칙으로 성공을 주장하지 않으며 pole-aware regularity gate가 필요하다.
- triangle/U-shape처럼 현재 extractor가 open contour를 내는 구조는 ordered correspondence가 확보될 때까지 계속 unsupported다.
- 기본 dispatcher·production training 경로는 변경하지 않았다.