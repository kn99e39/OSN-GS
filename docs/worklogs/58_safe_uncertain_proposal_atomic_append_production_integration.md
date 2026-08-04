# Worklog 58: Safe Uncertain Proposal → Atomic Append Production Integration

## 실제 연결 지점

`osn_gs/gaussian/torch_safe_uncertain_append_production.py`를 추가했다.

- `run_safe_uncertain_proposals_and_append_from_gaussians(...)`는 raw Gaussian evidence를 Worklog 57의 `run_safe_uncertain_proposals_from_gaussians(...)`에 전달한 뒤, `proposed` 상태 batch만 기존 `UncertainGaussianAppendAdapter.append(...)`에 전달한다.
- `append_safe_uncertain_proposals(...)`는 이미 생성된 Worklog 57 result를 동일한 atomic append 경로로 전달한다.
- `initialization_provider`가 `UncertainAppendInitialization`을 명시적으로 제공해야 한다. provider가 없거나 `None`을 반환하면 `appearance_initialization_required` typed rejection이며 appearance/opacity 값을 합성하지 않는다.
- candidate/chart/proposal ID와 supporting domain/boundary/patch provenance는 adapter receipt와 adapter-owned sidecar에 보존되며, model-owned batch-ID ledger는 같은 `proposal_batch_id` 재append를 차단한다.

## Transaction 결과

- planar candidate-ready fixture: proposal valid sample 수만큼 model tensor가 0개에서 receipt의 `appended_sample_count`로 증가했고, receipt/sidecar/owner registry/ledger가 모두 commit됐다.
- 동일 safe proposal의 두 번째 실행: model tensor 수는 증가하지 않고 `duplicate_proposal_batch` receipt를 `duplicate`로 accounting했다.
- `_commit_ledger` failure injection: attempt는 `rolled_back`으로 기록됐다. model tensor, owner registry, sidecar, model-owned ledger가 모두 pre-transaction snapshot과 동일함을 확인했다.
- Worklog 57에서 이미 rejected인 candidate, Box/Thin-slab의 degenerate/rejected 경로, Sphere raw no-candidate 경로는 모두 append 0이며 model을 변경하지 않는다.

## 검증

```text
python -m pytest -q tests/test_safe_uncertain_append_production.py
7 passed, 2 subtests passed

python -m pytest -q
751 passed, 1 skipped, 1 warning, 14 subtests passed in 249.23s
```