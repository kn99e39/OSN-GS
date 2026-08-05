# Worklog 60: Atomic Trainer Activation Contract Closure

## 실제 trainer 연결 지점

`osn_gs/core/torch_trainer.py`의 `TorchOSNGSTrainer.activate_and_train_uncertain_step(state, positions, *, initialization_provider, camera, target, ...)`가 실제 production 연결 지점이다. 독립 하네스(`run_one_training_step`)로 완료 처리하지 않고, `_train_loop`가 매 iteration마다 쓰는 것과 동일한 `self.rasterizer.render(camera, state.model, background)` → `image_reconstruction_loss(...)` 경로로 forward/backward를 수행한 뒤, `masked_optimizer_step()`으로 이번 호출에서 새로 activate된 row만 골라 step한다.

내부적으로는 `run_safe_uncertain_proposals_append_and_activate(positions, model=state.model, ...)`(raw evidence → Worklog 57 proposal → composite append+activate, worklog 56-58 미변경)를 호출한다.

## Composite transaction (append+activate)

`osn_gs/gaussian/torch_uncertain_trainer_activation.py`의 `append_and_activate()`를 candidate 단위 단일 트랜잭션으로 재작성했다. 각 candidate마다: activation 직전 `model.snapshot_state()` + adapter의 `_sidecar` dict 복사본 + `model.appended_uncertain_batch_ids` 복사본 + `model.occluded_chart_owner_registry` 복사본 + (활성 optimizer가 있다면) group-name 기준 optimizer state 복사본을 스냅샷한 뒤, 기존(미변경) `append_safe_uncertain_proposals()`를 단일-attempt sub-batch로 호출한다. activation(`activate_appended_rows`, worklog59의 grow-in-place 로직 그대로 재사용)이 실패하면 스냅샷 전체를 복원한다 — model tensor(`model.restore_state()`), sidecar, ledger, owner registry, optimizer 전부가 append 이전 상태로 되돌아가며, 이 candidate는 `rolled_back`으로 기록된다. Lower-level `activate_appended_receipts()`(append는 이미 끝난 배치를 받아 activation만 재시도, append 자체는 되돌리지 않음)는 그대로 남겨뒀고, 그 함수만 여전히 `appended_inactive`를 반환할 수 있다 — production 경로(`append_and_activate`)는 이 상태를 절대 노출하지 않는다.

## Row-level isolation (Adam momentum 포함)

`masked_optimizer_step(model, row_mask)`을 새로 구현했다. `torch.optim.Adam.step()`을 그대로 쓰지 않고 Adam 갱신식을 직접 row-mask로 게이팅해 재구현한다 — `row_mask`가 선택하지 않은 row는 `param.grad`가 0이어도 `exp_avg`/`exp_avg_sq`를 아예 읽지도 쓰지도 않는다(기존 residual momentum에 의한 미세 drift까지 완전 차단). step/bias-correction counter는 파라미터당 공유 scalar를 유지한다(`_preserve_optimizer_state`가 이미 scalar step을 그로우 대상에서 제외하는 기존 관례와 일치, `torch.optim.SparseAdam`의 shared-step 관례와도 동일).

## Row/state 전후 비교

사전 학습(실제 `exp_avg`/`exp_avg_sq` 확보)된 3-visible-row 모델에 candidate-ready planar fixture(worklog 57-59와 동일)를 append+activate 후:

- **uncertain-only step**: visible 3행의 `_xyz` 값과 `exp_avg`/`exp_avg_sq` 전부 bit-for-bit 불변, uncertain 신규행은 값 변경.
- **visible-only step**: uncertain 신규행의 값과 `exp_avg`/`exp_avg_sq` 전부 bit-for-bit 불변.
- **실제 `TorchOSNGSTrainer` 경로**(`activate_and_train_uncertain_step`, 실제 rasterizer render+backward 사용): 기존 visible row의 `_xyz` 값과 전체 Adam state(`exp_avg`/`exp_avg_sq`) bit-for-bit 불변, `step_result["loss"]` 유한값 확인.

## Optimizer parameter identity

성공 경로: `_preserve_optimizer_state`가 `model._optimizer_named_params()`(현재 model attribute)를 직접 읽어 `group["params"] = [new_param]`으로 설정하므로 자동으로 identity 일치.

롤백 경로: `model.restore_state()`는 항상 새 `nn.Parameter` 객체를 생성하므로, 스냅샷 시점의 param 객체 참조는 무의미해진다. `_restore_transaction_state()`가 `model._optimizer_named_params()`로 복원 직후의 CURRENT 객체를 다시 조회해 `group["params"]`를 재대입하고 optimizer state를 group name 기준으로 재-keying한다 — 롤백 후 매 param_group이 `model.<param>`과 identity 기준으로 정확히 일치함을 테스트로 직접 확인했다.

## Rollback 결과

`_preserve_optimizer_state`에 activation 단계에서만 실패를 주입(append 자체의 내부 no-op 호출은 무영향): candidate의 append_attempt 자체는 `appended`(worklog58 트랜잭션 성공)이지만 최종 attempt 상태는 `rolled_back`. model 행 수, `_xyz` 값, `appended_uncertain_batch_ids`, `occluded_chart_owner_registry` 전부 append 이전 스냅샷과 완전히 동일하게 복원됨을 확인. duplicate/rejected/no-candidate(Sphere) 경로는 attempt가 전부 `not_activated`이며 model/optimizer 변화 없음.

## 테스트 결과

```text
python -m pytest -q tests/test_uncertain_trainer_activation.py tests/test_safe_uncertain_append_production.py \
    tests/test_safe_uncertain_proposal_production.py tests/test_eligible_boundary_continuation_bridge.py \
    tests/test_occluded_chart.py tests/test_occluded_chart_hardening.py tests/test_uncertain_gaussian_proposal.py \
    tests/test_uncertain_gaussian_append_adapter.py tests/test_torch_pipeline_smoke.py
144 passed, 6 subtests passed in 59.78s

python -m pytest -q
760 passed, 1 skipped, 1 warning, 14 subtests passed in 238.78s
```

(worklog 59의 759 passed에서 `tests/test_uncertain_trainer_activation.py`를 8개→9개 테스트로 재작성해 순증 1)
