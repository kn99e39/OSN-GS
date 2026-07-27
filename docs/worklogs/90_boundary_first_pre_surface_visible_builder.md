# Worklog 90: Pre-Surface Boundary-first Visible Builder

## 상태

국소 Boundary-first 재구축 약 70% 진행. pre-surface outer/hole loop에서 support-curve network와 explicit seam multi-patch NURBS를 생성하는 foundation을 구현·검증했다. 기존 dispatcher integration 및 curved annulus topology recovery는 미착수다.

## 문제 교정

Worklog 89의 첫 foundation은 `PatchBoundarySegment`도 입력으로 받았으나, 이것만 사용하면 surface fit 뒤의 boundary를 다시 소비하는 순환이 된다. 이번 보완으로 `ComponentBoundaryResult`의 observed outer/hole loop를 chart fit 이전에 직접 입력으로 받도록 했다.

## 구현

- `ObservedBoundaryCurve`: pre-surface observed loop와 provenance를 표현한다.
- `observed_boundary_curves_from_annulus_component`: 하나의 outer loop와 하나의 hole loop를 실제 Boundary-first 입력으로 변환한다.
- `build_boundary_support_curve_network`: world-arclength correspondence, reversal, closed-loop phase를 보존해 transverse support-curve family를 만든다.
- `build_boundary_constrained_surface`: open pair는 exact clamped linear single patch, closed pair는 인접 support curve마다 one patch인 cyclic seam multi-patch를 만든다.
- `build_boundary_first_visible_surface`: explicit outer/hole pair가 있으면 pre-surface annulus를 구성하며, pair가 없거나 multi-loop correspondence가 불명확하면 rectangle fallback 대신 `unsupported`를 반환한다.

## 보존한 경계

- 기존 `boundary_first.py`, component builder, annulus fitter, trainer/renderer는 수정하지 않았다.
- 새 builder는 legacy dispatcher에 아직 연결되지 않았다.
- curved annulus의 split component를 자동 merge하지 않았다. 현재처럼 outer/hole pair가 소실된 경우에는 잘못된 box surface 대신 unsupported가 되는 것이 우선 계약이다.

## 검증

- 전용 + boundary/annulus 회귀 unittest: 61 tests passed
- `.venv\Scripts\python.exe -B -m pytest`: 공유 작업트리 기준 403 passed, 1 skipped, 1 warning
- warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion이다.

## 다음 구현 범위

- split component에서 annulus 구조를 복구하기 위한 boundary/loop correspondence evidence
- disk/non-convex/multi-loop의 interior support-network 정책
- support curve를 data/fairness constrained solver로 정교화
- 기존 dispatcher에 feature-gated integration 및 curved_annulus benchmark gate