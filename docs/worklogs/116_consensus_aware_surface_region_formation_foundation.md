# Worklog 116 — Consensus-aware Surface-Region Formation Foundation

## 상태

## 전달용 요약

### 수행 작업

- Worklog 113/115의 covariance-guided pairwise affinity를 입력으로 받아, 단일 same_surface edge가 아닌 local consensus·bridge veto·path consistency를 이용하는 격리된 SurfaceRegionCandidate formation을 추가했다.
- stable core seed, consensus attachment, ambiguous/conflict/rejected node 상태, region-level accepted/conflict edge와 reliability/frame/scale 통계를 구현했다.
- gap=0.02 close-parallel false pair가 두 sheet를 병합하던 pending weak-bridge/region-growth 경로를 차단했다.
- sparse-but-continuous fixture에서 triangle 부재만으로 core가 생기지 않던 문제를, 충분한 규모·무모순 network에 한정한 fallback core 정책으로 보완했다. isolated pair는 계속 core가 될 수 없다.

### 결과와 검증

- clean plane, smooth curved sheet, density-gradient 및 sparse-but-continuous control은 region을 유지했다.
- perpendicular surfaces, oversized planar bridge, gap=0.02 close-parallel sheets는 false merge 없이 분리되었다.
- rejected isotropic node는 core/primary region evidence로 사용되지 않았고, stable ID input-order determinism을 확인했다.
- 신규 테스트 7 passed, Boundary-first 회귀 포함 69 passed, Worklog 113/115 회귀 포함 55 passed.
- 전체 pytest: **568 passed, 1 skipped, 1 warning, 8 subtests passed**.

### 명시적 경계

이는 ordered boundary/half-edge graph, builder adapter, NURBS patch, default dispatcher, trainer 및 production integration을 포함하지 않는 격리 foundation이다. production-ready region 또는 Boundary-first Gate 완료를 주장하지 않는다.

완료(이번 독립 foundation 범위). Worklog 113/115의 covariance-guided reliability/manifold-affinity 입력 계약을 유지하면서, pairwise `same_surface` edge의 단순 connected component를 대체하는 consensus-aware region candidate 형성을 추가했다. 이는 object segmentation, ordered boundary extraction, NURBS patch 생성 또는 production 통합이 아니다.

## 계약과 상태

- covariance eigenframe, sign-independent normal, intrinsic/contextual reliability, tangent-major/minor scale와 normal thickness, candidate/endpoint/relation/confidence의 직교 상태 및 stable ID 계약을 그대로 소비한다.
- `SurfaceRegionCandidate`는 stable region ID, core/attached/rejected member, accepted/ambiguous/conflict edge, reliability/frame/scale 통계, confidence, formation/unresolved reason, policy version을 가진다.
- region 상태는 `core_region`, `growing_region`, `stable_region`, `review_required`, `rejected_region`, `small_review_region`만 사용한다. `eligible`/`production_ready`는 추가하지 않았다.
- node 상태는 `core_member`, `consensus_attached`, `ambiguous_unassigned`, `conflict_boundary_candidate`, `rejected_structural_node`으로 분리한다. 모호한 중복 소속을 강제 tie-break하지 않는다.

## 구현 및 결함 교정

- high-confidence same-surface core를 shared-neighbor consensus, local tangent-frame/path diagnostic, bridge veto와 함께 seed한다. isolated pair는 core가 될 수 없다.
- sparse-but-continuous surface처럼 triangle이 없는 입력은, strict core가 전혀 없고 충분한 규모의 contradiction-free network일 때만 reviewable fallback core로 승격한다. 작은 pair에는 적용되지 않는다.
- local consensus payload에는 shared support, mutual agreement, contradiction, rejected contamination, density, independent path, triangle, tangent transport residual, bridge likeness를 기록한다.
- pending weak bridge가 여러 개라는 이유만으로 merge하던 경로를 제거했다. 최종 merge는 well-supported consensus와 well-supported bridge veto를 동시에 만족하는 independent cross-edge만 허용한다.
- gap=0.02의 close-parallel false pair는 scale-normalized normal-direction separation과 borderline tangent residual을 함께 가진 shortcut으로 conflict 처리하고, region growth도 conflict edge를 support로 사용하지 않는다. 이는 smooth curved sheet의 큰 normal-thickness 정규화 값만으로 veto하지 않도록 tangent residual을 함께 요구한다.

## synthetic 결과

- clean plane: 81개 Gaussian이 하나의 stable/core region으로 유지.
- smooth curved sheet, gradual-density, sparse-but-continuous: 불필요한 fragmentation 없이 region 유지.
- perpendicular surfaces와 oversized planar bridge: floor/wall false merge 없음.
- gap=0.02 close-parallel sheets: false `same_surface` pair가 있어도 mixed-label region merge 없음.
- rejected isotropic node는 region member/core evidence가 될 수 없음.
- stable ID 입력 순서 shuffle에 대해 membership set이 동일함을 회귀 테스트로 고정.

## 검증

- 신규 `tests/test_gaussian_surface_region_formation.py`: 7 passed.
- Worklog 110/112 Boundary-first 회귀 + 신규 테스트: 69 passed.
- Worklog 113/115 covariance/reliability/affinity 회귀 + 신규 테스트: 55 passed.
- 저장소 전체 pytest: **568 passed, 1 skipped, 1 warning, 8 subtests passed**. warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 경고이며 이번 모듈과 무관하다.

## 명시적 비범위 및 다음 Gate

- ordered world-space boundary chain/loop, half-edge graph, builder adapter, NURBS patch/control-grid, raster/KDE 제거, dispatcher/trainer/renderer/checkpoint/ownership 통합은 구현하지 않았다.
- real trained snapshot의 region-level diagnostic과 phase-alias 전용 synthetic fixture는 다음 Gate 범위다. 이번 foundation은 이들을 위한 consensus/path/bridge payload를 제공하지만 production 품질을 주장하지 않는다.
- default dispatcher 및 production path는 수정하지 않았다. Boundary-first 전체 Gate 완료나 production integration 가능 상태를 주장하지 않는다.
