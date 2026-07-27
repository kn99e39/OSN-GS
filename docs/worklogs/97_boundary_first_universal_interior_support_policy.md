# Worklog 97 — Universal Boundary-first interior support 정책

## 방향 보정

- `curved_annulus`는 split-component recovery의 positive control일 뿐, Boundary-first의 대표 topology 또는 성공 판정 기준이 아니다.
- 모든 구조는 관측 outer boundary와, topology에 맞는 observed inner boundary 또는 provenance 있는 interior support를 바탕으로 구성해야 한다.
- review artifact의 15개 scene 상태는 현재 coverage를 보여 주는 근거이며, annulus 계열만의 품질 주장에 사용하지 않는다.

## 구현 전제

- `ComponentBoundaryResult`의 loop descriptor는 현재 boundary cell 집합을 제공하지만, universal constructor가 사용할 ordered closed polyline 계약은 아직 없다.
- outer-boundary-only disk/triangle/crease/density-gradient 계열에 단순 축소 outer loop를 가짜 inner loop로 넣으면 중앙 영역이 비거나 central pole Jacobian이 퇴화한다.
- 따라서 다음 구현은 임의 중심점/축소 loop를 자동 생성하지 않는다.

## 다음 구현 단위

1. component UV frame에서 연결된 contour segment를 deterministic ordered closed boundary polyline으로 복원한다.
2. loop의 observed/derived 상태, source component, correspondence, confidence를 가진 `InteriorSupportCurve` 계약을 추가한다.
3. disk-like 입력은 annulus multi-patch를 재사용하지 않고, explicit central-cap topology와 pole/regularity 계약을 별도로 materialize한다.
4. concave/multi-loop 입력은 correspondence가 확인되지 않으면 unsupported로 유지한다.
5. 모든 topology에 같은 report/export 형식으로 boundary preservation, source-point fidelity, normal/curvature, regularity를 기록한다.

## 금지 조건

- rectangle/PCA fallback으로 전환하지 않는다.
- `v=0/1` 또는 outer boundary라는 이유만으로 source/provenance 검증을 생략하지 않는다.
- annulus positive control의 통과를 disk, concave, multi-loop의 통과로 일반화하지 않는다.
- 기본 dispatcher와 production training 경로는 위 generic contracts와 quality gate 이전에 변경하지 않는다.