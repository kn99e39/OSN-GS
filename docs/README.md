# OSN-GS 문서 안내

## 현재 기준 문서

- [Urgent Work Master](Urgent_Work/OSN_GS_Urgent_Work_Master.md): 현재 방향, 활성 작업, 승인 경계의 단일 기준
- [작업로그 보존 정책](worklogs/README.md): 유지 중인 최소 검증 기록 목록
- [Architecture](architecture.md): 프레임워크 수준의 설계 결정
- [NURBS Construction](nurbs_construction.md): NURBS 중간 표현과 구현 계약

## 현재 상태

현재는 두 범위를 병행한다.

- 모든 topology에 공통인 isolated Boundary-first visible-surface hardening. 기본 dispatcher나 production path에는 아직 연결하지 않는다.
- Uncertain Gaussian의 proposal/append/ownership model foundation. optimizer, trainer, renderer, checkpoint 및 global selection 통합은 범위 밖이다.
- NURBS Construction benchmark는 depth-bearing 3D shell과 baseline-like flattened covariance를 기본 입력으로 사용한다. [Worklog 111](worklogs/111_nurbs_construction_synthetic_3d_gaussian_dataset.md)을 따른다.
- covariance-guided pairwise affinity 위에 consensus-aware surface-region candidate foundation을 추가했다. 이는 ordered boundary/builder/production path와 연결되지 않은 격리 진단 단계다. [Worklog 116](worklogs/116_consensus_aware_surface_region_formation_foundation.md)를 따른다.
- trimmed-component Jacobian test-health 부채를 해소했고, 최신 전체 pytest는 `568 passed, 1 skipped, 1 warning, 8 subtests passed`다. [Worklog 114](worklogs/114_trimmed_component_jacobian_test_health.md)와 [Worklog 116](worklogs/116_consensus_aware_surface_region_formation_foundation.md)을 따른다.

이전 실험·폐기 방향의 상세 작업로그는 작업 트리에서 제거했다. 필요한 경우 Git 이력으로 조회하며, 현재 결정을 위해 과거 로그를 canonical source로 사용하지 않는다.
## 2026-07-29 Boundary-first NURBS materialization 상태

- consensus-aware region formation의 real trained Gaussian 결과는 full sheet segmentation이 아니라 reliable-core-only core-island 추출로 해석한다. Worklog 117을 따른다.
- world-space half-edge와 ordered graph의 admissible closed outer loop는 canonical evaluable visible NURBS materialization adapter로 전달한다. dispatcher, builder, production path에는 연결하지 않는다. Worklog 120을 따른다.
- 최신 전체 pytest: 577 passed, 1 skipped, 1 warning, 8 subtests passed.
- Worklog 123: [Canonical Tangent Frame Invariance Repair](worklogs/123_canonical_tangent_frame_invariance_repair.md) — Gaussian-only smooth_curved_sheet의 rotation/scale/shuffle/sign-equivalence 안정성 수정 및 검증.
