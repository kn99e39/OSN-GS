# Worklog 86: Phase F.1 Chart Hardening 보완 완료 및 Gate F.1 승인 요청

날짜: 2026-07-24

상태: **구현·검증 완료, Gate F.1 사용자 승인 요청.** 자동 승인은 처리하지 않았다. Phase G, Gaussian proposal/append, global ranking·selection, conflict resolution, production integration은 미착수다.

## 기준 상태와 결함 분류

- Worklog 85의 `13 tests, 2 failures`는 당시의 fixture 계약 결함 기록이다. 보완 시작 전 현재 트리에서 `tests.test_occluded_chart_hardening`은 18 tests OK였으므로, 과거 상태를 재현하려고 코드를 되돌리지 않았다.
- partial evidence 실패는 candidate/domain/boundary coverage provenance 없이 `review_required`를 기대한 fixture 결함이었다. 실제 candidate와 두 domain registry를 포함한 fixture로 교체했다.
- separated-chart 실패는 geometry만 분리하고 source provenance를 공유한 fixture 결함이었다. domain, boundary, patch ID를 모두 분리했다.

## 구현과 fixture

- `OccludedChartConflictEdge` 생성은 keyword argument를 사용하며 `unresolved` bool을 검증한다. payload에 AABB, distance, intersection count, normal agreement, area ratio, coverage ratio, reason, provenance를 raw로 보존한다.
- sampled triangle에서 exact coplanarity와 tolerance-near를 분리했다. 분리된 근접면은 false hard crossing이 아닌 near-contact로 기록된다.
- actual candidate/domain/boundary provenance, central bridge scope, allowed/non-source/`t>0` contact, self-pair accounting, exact coplanar/near contact, multi-patch ordering, conflict types, evidence/uncertainty, deterministic payload, read-only invariant fixture를 검증했다.
- curvature, area, continuation extent uncertainty는 coverage가 유효한 경우 독립적으로 `review_required`로 보존된다.

## 검증

- `python -B -m unittest tests.test_occluded_chart_hardening` → **29 tests, OK**
- `python -B -m unittest tests.test_occluded_chart_hardening tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain tests.test_annulus_chart tests.test_surface_candidate_graph` → **161 tests, OK**
- `python -B -m pytest` → **341 passed, 1 skipped, 1 warning**
- Gate F 기준 `311 passed, 1 skipped, 8 subtests passed` 대비 pytest passed는 **+30**, skipped는 동일이다. pytest는 subtests를 별도 집계하지 않아 증감은 확정하지 않았다. warning은 기존 `torch_voxel_hierarchy.py` tensor-to-scalar warning이며 본 변경과 무관하다.

## 잔여 리스크

- 검사는 sampled piecewise-linear triangle surface crossing에 한정된다. continuous NURBS, analytic surface, CAD-level validity guarantee는 제공하지 않는다.
- global ranking/selection, conflict resolution, Gaussian proposal/append, production integration은 미구현이며 본 변경에 포함되지 않다.

> Phase F.1 보완 구현과 회귀 검증 결과를 제출하고, Gate F.1 승인 여부에 대한 사용자 검토를 요청합니다.
