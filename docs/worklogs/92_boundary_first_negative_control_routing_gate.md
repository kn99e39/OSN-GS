# Worklog 92: Boundary-first Negative-Control Routing Gate

## 상태

기존 dispatcher integration 준비 약 35% 진행. isolated proof path의 annulus positive control과 false-hole negative control을 보강했다. dispatcher/default는 미변경이다.

## 관찰

초기 isolated sweep에서 `plane`과 `close_parallel_sheets`도 작은 noise hole 때문에 annulus로 오인되어 support multi-patch를 만들었다. 이는 curved annulus recovery를 연결하기 전에 막아야 하는 false-positive였다.

## 구현

- `build_boundary_first_visible_surface`에 `minimum_hole_area_ratio=0.02`의 conservative plausibility gate를 추가했다.
- outer/hole pair가 하나씩 있어도 hole 면적 비율이 2% 미만이면 `hole_area_ratio_too_small`으로 `unsupported`를 반환한다.
- recovered `curved_annulus`의 hole/outer 면적 비율은 약 15%로 통과한다.
- plane의 noise hole은 약 0.2%, close parallel sheets의 noise hole은 약 0.4–1.2%로 차단된다.

## 검증

- positive: `curved_annulus` → recovery → annulus → 8 seam multi-patch, fallback 없음
- negative: `plane`, `close_parallel_sheets` → `unsupported`, annulus multi-patch 미생성
- `.venv\Scripts\python.exe -B -m pytest`: 공유 작업트리 기준 410 passed, 1 skipped, 1 warning

## 남은 integration

- disk/non-convex/multi-loop의 evidence-supported interior support-network
- recovery threshold의 scene/seed/density/rotation sweep
- quality metric과 resource bound
- feature-gated legacy dispatcher 연결 및 production adoption Gate