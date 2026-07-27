# Worklog 89: 통일 Boundary-first Support-Curve Foundation

## 상태

모든 topology가 공유하는 Boundary-first support-curve foundation 구현·검증 완료. 기존 constructor와의 integration은 미착수다.

## 배경과 결정

`curved_annulus`는 true annulus 구조가 component/boundary 단계에서 소실된 뒤 `trimmed_rect_fallback`으로 routing되어, inner/outer boundary와 그 사이 support curve를 source of truth로 유지하지 못했다. 사용자는 annulus만의 O-grid 방법론이 아니라 모든 visible-surface topology가 명시적 boundary pair/loop와 ordered support-curve family에서 출발해야 한다고 지시했다.

## 구현

- 새 `osn_gs/surface/torch_boundary_support_network.py`를 추가했다.
- `PatchBoundarySegment` 두 개와 명시적 reversal/phase correspondence를 받아 world-arclength 기준 paired samples 및 transverse support-curve family를 생성한다.
- closed loop(annulus 등)와 open boundary pair(strip/bridge 등)를 같은 계약으로 지원한다.
- provenance, boundary IDs, patch IDs, source kind, correspondence를 payload로 보존한다.
- 서로 다른 open/closed topology, 동일 boundary 재사용, zero-length/non-finite segment는 명시적으로 거부한다. topology를 box/rectangle fallback으로 조용히 바꾸지 않는다.

## 충돌 회피

다른 Agent가 작업 중인 `nurbs_constructor_benchmark/boundary_first.py`, `torch_surface_components.py`, `torch_component_boundary.py`, `torch_annulus_chart.py`, trainer/renderer에는 수정하지 않았다. 새 foundation과 새 전용 테스트만 추가했다. 기존 dispatcher 연결, topology recovery, support-curve constrained fitting은 별도 integration 단계다.

## 검증

- `.venv\Scripts\python.exe -B -m unittest -v tests.test_boundary_support_network tests.test_patch_boundary tests.test_annulus_chart`
  - 56 tests passed
- `.venv\Scripts\python.exe -B -m pytest`
  - 공유 작업트리 기준 398 passed, 1 skipped, 1 warning
  - warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion이다.

## 남은 범위

- 기존 boundary-first dispatcher가 fallback 대신 이 support-curve contract를 소비하도록 연결
- curved annulus의 component/loop topology recovery와 correspondence 선택
- support curves를 실제 boundary-constrained NURBS solver constraint로 연결
- disk/non-convex/multi-loop의 support network topology 정책 및 benchmark gate
- production pipeline/trainer/renderer integration