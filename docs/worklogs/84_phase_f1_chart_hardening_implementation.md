# Worklog 84-B: Phase F.1 Occluded Chart Hardening 구현

날짜: 2026-07-24

상태: **보완 구현 진행 중, Gate F.1 승인 대기. Phase G 및 production integration은 미착수.**

## 변경 파일

- `torch_sampled_surface_geometry.py`: NURBS sampled triangle mesh, 기존 AABB broad phase 재사용, deterministic tolerance-aware triangle intersection.
- `torch_occluded_chart_hardening.py`: immutable chart state와 분리된 `OccludedChartSafetyResult`, self-intersection·visible penetration·coverage·eligibility.
- `torch_chart_conflict.py`: `OccludedChartConflictEdge`, unresolved conflict를 safety result에 연결하는 helper.
- `tests/test_occluded_chart_hardening.py`: crossing, deterministic mesh, registry 부재, conflict 차단 fixture.

## 결과

각 sampled mesh payload는 resolution, triangle count, diagonal convention, tolerance, dtype/device, `method=sampled_piecewise_linear_triangle_intersection`, `continuous_surface_guarantee=false`를 보존한다. self 검사에서는 동일/같은 cell/parameter-grid 인접 cell을 제외한다. visible penetration은 명시적으로 증명된 contact만 허용할 수 있도록 기본 구현에서 무조건 boundary exclusion을 하지 않으며, sampled triangle intersection만 hard evidence로 취급한다.

coverage는 `central_bridge_only`와 transition strip 미모델링을 기록한다. 현재 coverage provenance가 없거나 visible registry가 없으면 Phase G eligibility는 `ineligible`이다. near contact와 central-bridge 제한은 raw uncertainty로 보존하며, conflict edge는 해결하지 않고 자동 proposal을 차단한다.

## 검증

- `python -B -m unittest tests.test_occluded_chart_hardening` → `4 tests, OK`
- 기존 Phase F chart 상태/control grid를 수정하거나 trainer/pipeline을 import하지 않았다.

## 남은 위험

- 검사는 sampled piecewise-linear mesh에 한정되며 analytic NURBS surface 무교차 증명이 아니다.
- 추가 fixture 및 scene/seed calibration distribution은 다음 gate의 검토 대상으로 남긴다.


## 2026-07-24 보완 구현 기록

상태: **보완 구현 진행 중, Gate F.1 승인 대기.** Phase G, Gaussian proposal/append, global ranking·selection, conflict resolution, production integration은 미착수다.

- 실제 `OccludedRegionCandidate`, 두 `ContinuationDomain`, `PatchBoundarySegment` fixture로 chart/candidate/domain/boundary/patch source chain을 검증했다. source registry 누락·ID 불일치는 `support_coverage_failed`로 `ineligible`이며, 완전한 source chain은 `central_bridge_only` provenance 때문에 차단되지 않는다.
- coverage payload는 selected `t`의 min/max/median, `selected_t_over_extent_median`, visible-boundary-to-chart-support minimum distance를 보존한다. partial/conflicting evidence는 hard contradiction이 없을 때 `review_required`다.
- visible contact는 chart mesh의 `v=0/1` 또는 외곽이라는 이유로 허용하지 않는다. 같은 source patch/boundary, selected `t==0` 또는 명시적 `source_boundary_correspondence`, source boundary polyline 구간 안, `allowed_contact_tolerance` 이내의 intersection point를 모두 만족할 때만 허용한다. `t>0` 교차는 hard crossing으로 기록한다.
- chart conflict payload는 AABB, minimum/normalized distance, intersection count, normal agreement, area ratio, coverage ratio, reasons, `unresolved: bool`을 raw로 보존한다.

## 검증 갱신

- `python -B -m unittest tests.test_occluded_chart_hardening tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain tests.test_annulus_chart tests.test_surface_candidate_graph` → **150 tests, OK**.
- `python -B -m pytest` → **330 passed, 1 skipped**. Gate F 기준 `311 passed, 1 skipped, 8 subtests passed` 대비 pytest passed는 **+19**, skipped는 동일이다. 현 pytest 출력에는 subtests 별도 집계가 없어 `8 subtests passed`의 증감은 검증 불가이며, 이를 0으로 환산하지 않는다.

## 남은 보완 범위

사용자 계약의 나머지 adversarial fixture 확장(비인접 coplanar overlap, near-contact/AABB-only, curvature·area·extent uncertainty와 모든 chart-conflict 조합)은 계속 진행한다. 따라서 canonical plan과 이 worklog의 Gate F.1 상태는 승인 대기로 유지한다.
