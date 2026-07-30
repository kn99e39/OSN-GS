# Worklog 128 — WebRenderer Gaussian 진단 시각화

## 작업

- OSN-GS PLY에 `uncertain`, `confidence`, `surface_u/v`, `cluster_id`, `surface_owner_kind/id`, `stable_gaussian_id`를 기록했다.
- trainer WebSocket packed snapshot에 같은 행 정렬의 `ids`, `uncertain`, `confidences`, `surfaceUvs`, `clusterIds`, `surfaceOwnerKinds`, `surfaceOwnerIds`를 추가했다.
- `WebRenderer`에 Certain/Reliability, confidence heatmap, surface ownership, NURBS patch ID 모드를 추가했다.

## 색상 규약

- Reliability: certain/reliable 녹색, uncertain 빨강, 필드 없는 기존 PLY는 회색
- Confidence: 낮음 빨강에서 높음 녹색
- Ownership: unassigned 회색, visible patch 청록, occluded chart 주황
- Patch ID: `cluster_id`별 안정적인 categorical 색

## 검증

- Python PLY header 및 packed stream payload 회귀: `2 passed`.
- `git diff --check` 통과.
- 이 환경에는 Node.js가 없어 WebRenderer JavaScript smoke test는 실행하지 못했다. 테스트 자체는 새 PLY/stream schema와 reliability·ownership GPU color packing을 검증하도록 갱신했다.
