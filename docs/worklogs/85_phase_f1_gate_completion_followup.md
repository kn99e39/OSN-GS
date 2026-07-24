# Worklog 85: Phase F.1 Gate 보완 진행 상태 및 fixture 계약 결함 기록

날짜: 2026-07-24

상태: **진행 중. Gate F.1 승인 대기. Phase G/production integration 미착수.**

## 배경

Worklog 84-B의 최초 구현은 sampled chart safety 계층의 골격과 기본 테스트 4개만 제공했다. 이후 사용자 재검토에서 central-bridge eligibility deadlock, sampled visible-surface 검사의 의미 과장, fixture 부족이 지적됐다. 이 기록은 새 설계가 아니라, 보완 작업 도중 실제로 확인한 코드 상태와 다음 수정 단위를 보존한다.

## 이번에 반영한 보완

- `coverage_scope="central_bridge_only"`와 `transition_surface_modeled=false`를 eligibility 자동 차단 사유에서 제거했다. 둘은 Phase F의 정상 canonical scope/provenance이며, proposal이 시작될 경우에도 그대로 전달되어야 한다.
- `visible_surface_penetration` payload가 signed solid inside/outside 또는 volumetric penetration 검사가 아님을 명시하도록 보완했다. 현재 검사는 sampled triangle surface crossing만 검출하며 continuous NURBS guarantee는 제공하지 않는다.
- self-intersection 결과에 `excluded_same_cell_pair_count`, `excluded_adjacent_pair_count`, `coplanar_overlap_count`를 추가해 fixture가 raw pair accounting을 검증할 수 있게 했다.
- F.1 설계 문서, canonical plan, Worklog 84-B의 상태를 `보완 구현 진행 중, Gate F.1 승인 대기`로 되돌렸다.

## 테스트 확장 시도와 발견된 결함

기존 `tests/test_occluded_chart_hardening.py`를 4개에서 13개 테스트로 확장하는 중 2개 실패가 발생했다.

1. partial evidence fixture는 candidate/domain coverage provenance 없이 `evaluate_occluded_chart_safety()`를 호출했다. 현재 계약에서 coverage provenance 누락은 `ineligible`이므로 `review_required` 기대값이 잘못됐다. 올바른 fixture는 실제 `OccludedRegionCandidate`와 두 `ContinuationDomain` registry를 전달한 뒤 partial/conflicting evidence만 단독 uncertainty로 검증해야 한다.
2. separated-chart fixture가 기본 helper의 동일 `supporting_domain_ids`/`supporting_boundary_ids`를 그대로 사용했다. conflict registry는 같은 source를 competing chart로 올바르게 감지했으므로, fixture가 실제로 분리된 source provenance를 주도록 고쳐야 한다.
3. `OccludedChartConflictEdge` 생성에서 positional dataclass 인자가 `unresolved` 대신 `provenance` 위치에 들어가는 결함을 발견했다. 이 때문에 payload의 `unresolved`가 bool이 아니라 dict가 되는 위험이 있다. keyword 인자로 바꾸어 수정해야 한다.

따라서 확장 테스트 실행은 이 시점에 `13 tests`, 실패 `2`로 끝났고, Gate F.1 회귀 결과로 사용하면 안 된다.

## 남은 필수 보완

- 실제 candidate/domain/boundary provenance를 가진 central-bridge coverage fixture
- explicit allowed source-boundary contact와 `t_selected > 0` unallowed contact fixture
- nonadjacent self-crossing/coplanar overlap/near-contact 및 adjacency exclusion fixture
- multi-visible patch, AABB-only overlap, deterministic payload fixture
- near-duplicate/crossing/same-source/conflict-free/reversed-order chart conflict fixture
- full/partial/conflicting evidence와 curvature/area/extent uncertainty fixture
- hardening 전후 chart state, control grid, weights, chart ID, payload 불변 fixture
- 지정된 Phase D/E/F 관련 regression 및 전체 pytest

## 범위 확인

이번 작업은 기존 chart control grid/state를 바꾸지 않는 read-only sampled safety gate 보완에 한정된다. analytic/CAD-level surface validity, global ranking/selection, conflict resolution, Gaussian proposal/append, production integration은 수행하지 않았다.
