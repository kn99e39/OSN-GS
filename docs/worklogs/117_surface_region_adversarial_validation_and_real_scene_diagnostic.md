# Worklog 117 — Surface-Region Adversarial Validation and Real-Scene Diagnostic

## 상태

진행 중. Worklog 116의 isolated consensus-aware region formation을 유지하며, ordered boundary graph 이전의 adversarial validation/readiness evidence를 추가한다. Worklog 번호 117은 `docs/worklogs/`에서 사용 가능함을 확인했다.

## 현재 완료 영역

- `osn_gs/surface/torch_surface_region_validation.py`에 read-only boundary-input readiness diagnostic을 추가했다. 이 모듈은 boundary/half-edge/NURBS를 생성하지 않고, region의 core 존재·ambiguity·internal contradiction·conflict/rejected adjacency·connectivity·crease/parallel evidence를 `ready_for_boundary_graph_experiment` 등 review-only 상태로 투영한다.
- 실제 trained snapshot `output/osn_gs_ab_3k/final/point_cloud.ply`를 read-only로 두 번 순회하여 global centroid에 가장 가까운 4,000 Gaussian을 deterministic하게 선택했다. covariance는 저장된 log scale 및 quaternion에서 복원했으며 threshold는 변경하지 않았다.
- real diagnostic에서 candidate cross-edge가 core consensus payload를 갖지 않을 수 있는데 merge가 이를 무조건 조회하던 `KeyError`를 발견해 guard를 추가했다. 기존 region/readiness tests는 수정 후 8 passed다.

## 실제 snapshot 중간 결과

| 항목 | 결과 |
| --- | --- |
| 입력 crop | global centroid nearest 4,000 Gaussian |
| pairwise connected-component | 413 regions, 최대 크기 132/100/94/80/62 |
| consensus-aware candidate | 80 regions, 최대 크기 29/17/17/14/13 |
| ambiguous-unassigned | 3,409 |
| conflict-boundary candidate | 43 |
| rejected structural node | 5 |

이는 region 수 감소를 성능 향상으로 주장하는 결과가 아니다. 특히 ambiguity를 대량으로 남기며 giant region으로 붕괴하지 않았다는 read-only 관찰이다.

## 남은 범위

phase-alias 전용 fixture, fallback-core adversarial matrix, genuine narrow connection, competing membership, forced internal-contradiction fixture, region-level invariance/density/gap-sweep 확장 및 전체 regression은 계속 수행한다. ordered boundary graph, builder adapter, dispatcher/production integration은 범위 밖이다.

## 종료 결과 — Adaptive nonlocal shortcut policy

- canonical policy를 `off|auto|force` mode로 전환했고 기본값은 `auto`다. auto는 candidate-degree 분포로 broad candidate graph를 감지해 formation-level nonlocal filtering을 활성화한다.
- broad curved-sheet에서 pairwise shortcut 350개는 유지되지만 accepted topology shortcut은 0개이며, local curved sheet는 1 region으로 유지된다.
- deterministic real crop(4,000): pairwise 413 components, consensus 80 regions, 최대 29, ambiguous-unassigned 3,409, conflict 43, rejected 5. giant component는 없다. 이는 full segmentation이 아니라 reliable core extraction이다.
- targeted suite 84 passed, 전체 pytest 574 passed, 1 skipped.
- readiness의 `ready_for_boundary_graph_experiment`는 reliable core seed의 실험적 사용 가능성만 뜻하며 coverage는 `reliable_core_only`, full coverage claim은 false다.

Part A Gate 통과 후 Worklog 118에서 experimental world-space boundary half-edge candidate foundation을 시작했다. ordered loop/builder/production integration은 계속 미착수다.
