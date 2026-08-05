# Worklog 59: Appended Uncertain Gaussian Trainer Activation

## 활성화 구조

새 모듈 `osn_gs/gaussian/torch_uncertain_trainer_activation.py`.

`UncertainGaussianAppendAdapter.append()`는 `model.optimizer is None`을 자체 전제조건으로 요구한다(worklog 58, 미변경). 이 모듈은 그 전제조건을 만족시키기 위한 optimizer의 임시 분리/재부착과, append 이후 실제 optimizer 확장을 담당한다.

- `append_and_activate(safe_proposals, model=..., initialization_provider=...)` / `run_safe_uncertain_proposals_append_and_activate(positions, ...)`: `model.optimizer`를 보관한 뒤 `None`으로 설정 → worklog 58의 `append_safe_uncertain_proposals`(미변경, 재감사 없음) 호출 → 각 attempt를 분류·활성화.
- `appended` receipt만 `activate_appended_rows()`를 호출하고, `rejected`/`duplicate`/`rolled_back`은 보관해둔 이전 optimizer를 그대로 복귀시키는 no-op으로 처리한다.
- `activate_appended_rows()`는 `TorchGaussianModel`의 기존 `_preserve_optimizer_state()`(ADC clone/split가 이미 쓰는 grow-in-place 로직, 미변경)를 재사용한다. `old_params`(append 이전 optimizer가 참조하던 원래 param 객체)와 `old_count`(receipt의 `model_count_before`)를 넘기면, 기존 Adam state(`exp_avg`/`exp_avg_sq`)는 앞 `old_count`행에 그대로 복사되고 새로 append된 행은 0으로 초기화된 상태로 optimizer가 in-place 확장된다 — optimizer 객체 자체(identity)는 재생성되지 않는다.
- `run_one_training_step(model, loss_fn)`: 최소한의 forward/backward/optimizer.step 하네스. 실제 `TorchTrainer`의 렌더러 기반 학습 루프는 재구현하지 않았다 — 이번 라운드의 범위는 activation 자체이며, `run_safe_uncertain_proposals_append_activate_and_train_step(...)`가 raw evidence → proposal → append → activation → 1 step을 하나로 잇는 단일 production 진입점이다.
- Visible/uncertain 분리는 새로 구현하지 않았다. append adapter가 이미 모든 append 행에 `uncertain_mask=True`(`is_uncertain`)를 부여하며(worklog 58, 미변경), 기존 `torch_density_control.py`(gradient 기반 clone/split 후보 mask, `certain = ~model.is_uncertain`)와 `torch_pipeline.py`(canonical visible-NURBS construction 호출부 3곳, `~model.is_uncertain`)가 이미 이 플래그로 visible-only 경로를 걸러낸다 — 이 모듈은 그 플래그를 건드리지 않을 뿐이다.
- 실패 처리: `activate_appended_rows()` 실행 전 optimizer의 `param_groups`/`state`를 얕은 복사로 스냅샷한다(`_preserve_optimizer_state`는 각 group의 `"params"`를 재할당만 하고 `optimizer.state`의 기존 inner dict는 절대 in-place 변경하지 않으므로 얕은 복사로 충분). 예외 발생 시 스냅샷으로 정확히 복원하고 `model.optimizer`는 원래 객체 그대로 유지한다. 이때 model tensor는 이미 worklog 58의 append 트랜잭션이 커밋한 상태 그대로 두고(재감사·롤백하지 않음), attempt에는 `appended_inactive` 상태를 명시한다 — half-registered optimizer 없이, "행은 존재하지만 아직 학습에 참여하지 않음"을 정확히 기록한다.

## Tensor/optimizer state 전후

- Planar candidate-ready fixture(worklog 57/58과 동일 fixture)로 사전 학습된 optimizer(첫 step으로 실제 `exp_avg`/`exp_avg_sq`를 확보)가 있는 모델에 append+activate를 실행: `model.optimizer`는 동일 객체(identity 불변), 기존 3행의 `exp_avg`/`exp_avg_sq`는 bit-for-bit 보존, 새로 append된 행은 `exp_avg`/`exp_avg_sq` 전부 0, 기존 3행의 `_xyz` 값 자체도 append 시점엔 불변.
- 1 training step(uncertain-only loss) 실행: `model._xyz.grad`의 visible 3행은 정확히 0, uncertain 신규행은 0이 아님(gradient 격리 직접 확인) → step 이후 uncertain 행 값 변경, visible 행은 residual Adam momentum으로 인한 통상적 미세 drift 외 이 loss에서 유입된 gradient가 전혀 없음을 확인.
- 반대 방향(visible-only loss)도 대칭 확인: 신규 uncertain 행은 정확히 불변.

## 학습-step 결과와 rollback 검증

- 동일 proposal batch 재실행: 두 번째 실행은 `duplicate`(worklog 58 own ledger) → activation은 `not_activated`, `activated_row_count=0`, 모델 행 수 불변.
- `_preserve_optimizer_state`에 실패를 주입(append 트랜잭션 자체는 건드리지 않고 activation 호출에만 국한): attempt 상태 `appended_inactive`, model 행 수는 append가 이미 커밋한 대로 증가한 채 유지, `model.optimizer`는 원래 객체·`param_groups`·`state` 전부 스냅샷과 bit-for-bit 동일하게 복구됨을 확인.
- `rejected`(초기화 미제공)와 raw Gaussian Sphere(candidate 0) 경로 모두 activation 0건, model/optimizer 불변.
- 전체 단일 진입점(`run_safe_uncertain_proposals_append_activate_and_train_step`)을 Sphere raw evidence로 실행: activation 0건, `step_result=None`(아무것도 학습 가능해지지 않음), 에러 없이 정상 종료.

## 테스트 결과

```text
python -m pytest -q tests/test_uncertain_trainer_activation.py tests/test_safe_uncertain_append_production.py \
    tests/test_safe_uncertain_proposal_production.py tests/test_eligible_boundary_continuation_bridge.py \
    tests/test_occluded_chart.py tests/test_occluded_chart_hardening.py tests/test_uncertain_gaussian_proposal.py \
    tests/test_uncertain_gaussian_append_adapter.py
141 passed, 6 subtests passed in 58.36s

python -m pytest -q
759 passed, 1 skipped, 1 warning, 14 subtests passed in 236.50s
```

(worklog 58의 751 passed에서 `tests/test_uncertain_trainer_activation.py` 신규 8개 순증)
