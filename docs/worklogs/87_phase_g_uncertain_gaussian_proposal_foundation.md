# Worklog 87: Phase G Uncertain Gaussian Proposal Foundation 구현 및 Gate 검증 보강

날짜: 2026-07-26

상태: **Proposal foundation 구현·검증 완료, Gate G 사용자 승인 완료.** Gaussian append/integration과 production integration은 미착수다.

## 작업

- immutable eligibility decision과 batched proposal을 구현했다. `eligible`만 sample을 생성하며 `review_required`와 unresolved conflict는 review decision으로 보존한다.
- known-free contradiction, `ineligible`, `unsupported`, 필수 provenance 누락은 hard rejection으로 검증한다.
- cell-centered deterministic UV, derivative frame, quaternion, linear scale, duplicate validation을 추가했다.
- `TorchGaussianModel` mutation, append, optimizer, trainer, renderer, checkpoint는 수행하지 않았다.

## 검증

- `python -B -m unittest -v tests.test_uncertain_gaussian_proposal` → **8 tests, OK**
- Phase D/E/F/F.1 지정 회귀 군 → **161 tests, OK**
- `python -B -m pytest` → **349 passed, 1 skipped, 2 warnings**

## Warning 분석

- `torch_voxel_hierarchy.py`의 tensor-to-scalar `UserWarning`은 기존 warning이며 Phase G 변경과 무관하다.
- pytest cache write `PytestCacheWarning`은 workspace permission 제한이며 test correctness에 영향이 없다.

## 불변식과 잔여 범위

- proposal 생성 전후 chart state, chart ID, control grid, weights를 비교하는 테스트로 read-only를 검증했다. ID, UV ordering, reason ordering은 반복 실행에서 결정적이다.
- adaptive sampling, review workflow, conflict resolution, appearance prior, Gaussian append/integration은 후속 승인 범위다.

> Phase G proposal foundation의 계약 검증 결과를 제출하며, Gaussian append 또는 production integration은 수행하지 않았다. Gate G 승인 여부에 대한 사용자 검토를 요청한다.
