# Worklog 88: Uncertain Gaussian Append Adapter Foundation

## 상태

Append adapter foundation 구현·검증 완료. Gate 검증 보강(transaction/contract) 완료. 최종 계약 보완(receipt strong exception guarantee, shared model-owned ledger, cluster ID downstream audit) 완료. **Append Adapter Gate 사용자 승인 완료.** Occluded Chart Ownership Foundation은 `docs/worklogs/96_occluded_chart_ownership_foundation.md`에서 별도로 다룬다. optimizer, trainer, renderer 및 checkpoint production integration은 여전히 수행하지 않았다.

## 수행 내용

- 승인된 `UncertainGaussianProposalBatch`를 위한 read-only preflight, tensor conversion, model-only atomic append, immutable receipt를 구현했다.
- `TorchGaussianModel.append_gaussians_model_only`는 optimizer가 없는 model만 허용하며, 모든 incoming tensor를 먼저 변환·shape 검증한 뒤 `replace_tensors` 한 번으로 commit한다.
- linear scale은 기존 model 계약에 따라 log-scale로 변환했고, canonical quaternion은 raw rotation tensor로 보존한다. valid mask 순서와 dtype/device 정렬을 유지한다.
- appearance와 opacity는 proposal 계약상 unset이다. adapter는 값을 임의 생성하지 않으며, 명시적 `UncertainAppendInitialization`이 없으면 `appearance_initialization_required`로 차단한다.
- 동일 process 내 batch ID ledger로 duplicate append를 차단했다. batch/sample ID와 chart/candidate/patch/domain/boundary provenance는 batched sidecar에 보존한다. checkpoint 영속 ledger는 이번 범위 밖의 deferred gap이다.

## 기존 model 계약 감사

- 필수 per-Gaussian tensor는 xyz, DC/remaining SH features, opacity logit, log scaling, raw quaternion, confidence logit, uncertain mask, UV, cluster ID다.
- 기존 `append_uncertain`은 RGB와 opacity/scale 상수를 요구하고 `initialize()`를 호출하므로 unset appearance/opacity 계약 및 model-only 경계에 부적합하여 사용하지 않았다.
- 기존 `append_gaussians_raw`는 optimizer state 보존 경로를 포함한다. 새 model-only API는 active optimizer를 명시적으로 차단한다.

## 원자성 및 provenance

- preflight 거부, appearance blocker, conversion shape 실패의 경우 model tensor snapshot이 호출 전과 동일함을 테스트했다.
- 성공 시 모든 per-Gaussian tensor의 길이가 함께 증가하고 uncertain mask, UV, cluster ID가 연결됨을 확인했다.
- sidecar는 proposal batch ID, sample IDs, source chart/candidate, patch/domain/boundary IDs, append origin을 보존한다. source patch ID 하나만 model `cluster_ids`에 투영되며, 다중 patch 전체 정보는 sidecar에 남긴다.

## 검증 결과

- `.venv\Scripts\python.exe -B -m unittest -v tests.test_uncertain_gaussian_append_adapter tests.test_uncertain_gaussian_proposal tests.test_occluded_chart_hardening tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain tests.test_annulus_chart tests.test_surface_candidate_graph`
  - 173 tests passed
- `.venv\Scripts\python.exe -B -m pytest`
  - 353 passed, 1 skipped, 2 warnings
  - warning 1건은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion이며, 나머지 1건은 Windows `.pytest_cache` 권한 경고다.

## 잔여 범위

- appearance/opacity initialization policy, optimizer registration/state expansion, trainer, renderer, checkpoint, review workflow 및 production integration은 미착수다.

> Uncertain Gaussian append adapter foundation의 구현 및 검증 결과를 제출하며, optimizer, trainer, renderer 및 checkpoint production integration은 수행하지 않았다. 사용자 Gate 검토를 요청한다.

---

## Gate 검증 보강 (2026-07-26)

상태: **transaction/contract 검증 보강 완료. 사용자 Gate 검토 대기(변경 없음). optimizer, trainer, renderer, checkpoint integration 미착수.**

새로운 production integration을 추가하지 않고, 기존 §1의 결정(§1은 그대로 유지)을 지키면서 transactional atomicity, sidecar consistency, duplicate ledger, append contract 검증을 보강했다.

### 1. `replace_tensors` 실제 atomicity 감사 결과

`TorchGaussianModel.replace_tensors`(osn_gs/gaussian/torch_model.py)를 직접 읽고 확인했다.

- 10개의 per-Gaussian 필드(`_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`, `_confidence`, `is_uncertain`, `surface_uv`, `cluster_ids`)를 **순차적인 Python attribute assignment**로 대입한다. 각 대입은 자체 `.reshape(count, ...)`를 포함하며, 이는 shape 불일치 시 `RuntimeError`를 던질 수 있다.
- 검증(존재한다면)은 각 대입 시점에 개별적으로 일어나며, **전체 대입 전에 한 번에 끝나지 않는다.** 예를 들어 `features_rest`의 reshape가 3번째 대입에서 실패하면, 이미 대입된 `_xyz`/`_features_dc`는 새 count로 바뀐 채 남고 나머지는 이전 count로 남아 **혼합된 상태**가 된다.
- 마지막에 호출되는 `_reset_density_stats(count)`도 10개 필드 대입 이후 실행되므로, 이것이 실패하면 per-Gaussian 필드는 새 count인데 `xyz_gradient_accum`/`denom`/`max_radii2D`는 이전 count로 남는 추가 불일치가 생길 수 있다.
- **rollback/undo 메커니즘은 `replace_tensors`/`append_gaussians_model_only` 어디에도 없다.** 즉 이 메서드들은 실제로 atomic하지 않다 — 이전 버전의 `append_gaussians_model_only` docstring이 "Atomically append"라고 주장한 것은 부정확했다.

**조치**: `append_gaussians_model_only`의 docstring을 감사 결과대로 정정하고("NOT internally atomic"), model 클래스에 최소 추가로 `snapshot_state()`/`restore_state()`를 신설했다(10개 per-Gaussian 필드 + 3개 density-stat 필드, 총 13개를 clone/restore). 이 두 메서드는 순수 추가이며 기존 `replace_tensors`/`append_gaussians_model_only`/기타 호출부의 동작을 변경하지 않는다.

### 2. Commit-stage failure injection 결과

`tests/test_uncertain_gaussian_append_adapter.py`의 `AppendAdapterTransactionTest`에서 강제 실패를 주입했다.

- **model commit 내부 실패**: `model.append_gaussians_model_only`를 몽키패치해 `_xyz`/`cluster_ids`를 새 count로 일부 변경한 뒤 예외를 던지도록 했다(실제 `replace_tensors`의 부분-변경 실패를 그대로 모사). `adapter.append()` 호출 후 모델의 10개 tensor 전부가 호출 전과 **완전히 동일**함을 확인했다(`test_model_commit_failure_rolls_back_fully`).
- **sidecar commit 실패**: `_commit_sidecar`를 몽키패치해 예외를 던지게 하고, model이 호출 전과 동일하며 sidecar/ledger 모두 비어 있음을 확인했다(`test_sidecar_commit_failure_rolls_back_model`).
- **ledger commit 실패**: `_commit_ledger`를 몽키패치해 예외를 던지게 하고, sidecar entry가 롤백되고(sidecar에서 제거) model도 호출 전과 동일함을 확인했다(`test_ledger_commit_failure_rolls_back_sidecar_and_model`).
- **conversion 실패**: 잘못된 shape의 initialization을 넣어 model 접촉 이전에 예외가 나고 model/sidecar/ledger가 전부 무변경임을 확인했다(`test_conversion_failure_before_model_commit_leaves_everything_untouched`).
- **receipt 생성 실패**: `_build_receipt`를 몽키패치해 예외를 던지게 했다. 이 경우 이미 ledger commit까지 끝난 뒤이므로 **model count 증가, sidecar entry, ledger entry가 모두 그대로 유지**됨을 확인했다(`test_receipt_failure_after_commit_does_not_roll_back`) — receipt 실패는 이미 성공한 append를 되돌리지 않는다.

### 3. Model/sidecar/ledger transaction 결과

구현한 commit 순서:

```text
preflight
→ 순수 conversion (모델 미접촉, 실패 시 롤백 대상 없음)
→ model.snapshot_state()
→ model.append_gaussians_model_only(...)   실패 시 model.restore_state(snapshot) 후 raise
→ self._commit_sidecar(...)                실패 시 model.restore_state(snapshot) 후 raise
→ self._commit_ledger(...)                 실패 시 sidecar 롤백 + model.restore_state(snapshot) 후 raise
→ receipt 생성                              실패해도 이미 끝난 transaction은 롤백하지 않음
```

5개 독립 실패 케이스(conversion 이전 / model commit 내부 / model 성공 후 sidecar 실패 / sidecar 성공 후 ledger 실패 / duplicate ledger 거부) 모두 개별 테스트로 검증했다. Duplicate ledger 거부는 `preflight()`가 `_appended_batch_ids`를 조회해 처리하며, append 시도 자체가 model/sidecar/ledger를 건드리지 않는다(`test_duplicate_batch_id_second_append_is_blocked`).

**Receipt vs. rejection/예외 계약을 명시했다**(모듈 docstring): 정상 거부(eligibility, provenance, optimizer, duplicate, initialization 누락, 파라미터 검증)는 항상 `append_state="not_appended"` receipt를 반환하며 예외를 던지지 않는다. `append()`에서 예외가 나오는 경우는 preflight를 통과한 뒤 transaction 자체가 깨진 경우(결함)이며, 이 경우 model/sidecar/ledger를 롤백한 뒤 예외를 전파한다.

### 4. Duplicate ledger ownership과 lifecycle

**소유권을 adapter-owned로 유지하고 명시했다**(`__init__`의 `_appended_batch_ids`/`_sidecar`는 인스턴스 전용, in-process, in-memory). 모듈 docstring에 다음을 명시:

- 새 `UncertainGaussianAppendAdapter()` 인스턴스는 빈 ledger로 시작하며, **다른 인스턴스가 이미 append한 batch ID를 인식하지 못한다** — 같은 model을 대상으로 해도 마찬가지다.
- model reset/재생성은 ledger를 건드리지 않고, adapter 재생성은 model을 건드리지 않는다(서로 독립).
- `test_ledger_is_adapter_owned_not_shared_across_instances`로 이 위험을 실제로 재현했다: adapter 1이 append한 batch ID를 adapter 2가 같은 model에 대해 다시 append하면 **차단되지 않고 성공**하여 동일 내용의 Gaussian이 두 번 들어간다. 이는 "소유권이 불명확하면 duplicate protection이 adapter instance 교체로 우회될 수 있다"는 지적을 그대로 증명하는 테스트다.
- checkpoint 영속 ledger는 이번 범위 밖으로 그대로 유지한다(§1 결정 불변).

### 5. Cluster ID 투영 규칙

기존 구현은 `cluster_ids = torch.full((count,), int(batch.metadata["source_patch_ids"][0]), ...)`로 **Python list의 첫 원소를 임의로** 사용하고 있었다 — 실제 결함으로 확인됐다.

기존 코드베이스 전체(Phase E/F candidate/chart provenance)를 감사했으나 "canonical owner patch"라는 기존 계약은 존재하지 않았다(두 patch id는 domain-id 문자열 정렬 순서로만 저장되며 patch identity 기준 순서가 아니다). 따라서 새 명시적 규칙을 정의했다.

```text
CLUSTER_ID_PROJECTION_RULE = "min_source_patch_id"
cluster_id = min(int(p) for p in source_patch_ids)
```

- deterministic, input list 순서 무관(`test_cluster_id_projection_independent_of_list_order`로 `(2,9)`와 `(9,2)` 양쪽 모두 `cluster_id=2`임을 확인).
- `source_patch_ids[0]`이 아니라 실제 min임을 `(7,3)` fixture로 확인(`test_cluster_id_projects_deterministic_min_not_first_element`, `cluster_id=3` 확인, `7`이 아님).
- sidecar의 `cluster_id`/`cluster_id_projection_rule` 필드와 receipt의 동일 필드에 기록해 추적 가능하다(`test_sidecar_preserves_full_provenance`).

### 6. Test coverage mapping

| Contract | Implementation | Test method/subtest | Result |
|---|---|---|---|
| Eligibility rejection (review_required/ineligible/unsupported) | `preflight()` | `AppendAdapterEligibilityTest.test_review_required_proposal_is_blocked`, `.test_ineligible_proposal_is_blocked`, `.test_unsupported_proposal_is_blocked` | PASS |
| Active optimizer rejection | `preflight()` | `AppendAdapterEligibilityTest.test_active_optimizer_model_is_blocked` | PASS |
| Initialization blocker | `preflight()` | `AppendAdapterEligibilityTest.test_missing_initialization_is_blocked` | PASS |
| Schema/provenance/duplicate/append-state/zero-valid vetoes | `preflight()` | `test_unsupported_schema_is_blocked`, `test_missing_provenance_is_blocked`, `test_already_appended_state_is_blocked`, `test_zero_valid_samples_is_explicit_rejection`, `test_known_free_contradiction_is_blocked` | PASS |
| Scale conversion (zero/negative/extremely small/dtype) | `_convert()`/`preflight()` | `AppendAdapterConversionTest.test_scale_converts_to_log_scale_numerically`, `.test_zero_scale_is_rejected`, `.test_negative_scale_is_rejected`, `.test_extremely_small_scale_is_rejected`, `.test_scale_dtype_device_and_ordering_preserved` | PASS |
| Quaternion conversion | `_convert()`/`preflight()` | `.test_rotation_component_order_and_values_pass_through`, `.test_nonnormalized_quaternion_is_rejected` | PASS |
| Valid-mask ordering | `_convert()` | `.test_valid_mask_filtering_preserves_sample_id_alignment` | PASS |
| Appearance/opacity pass-through, no hidden default | `_convert()` | `AppendAdapterAppearanceOpacityTest.test_initialization_values_pass_through_unmodified`, `.test_no_hidden_default_appearance_or_opacity` | PASS |
| Initialization digest/provenance | `_initialization_digest()` | `.test_initialization_digest_recorded_and_stable` | PASS |
| Successful atomic append | `append()` | `AppendAdapterTransactionTest.test_successful_append_is_atomic_across_model_sidecar_ledger` | PASS |
| Commit-stage rollback (model) | `snapshot_state`/`restore_state` | `.test_model_commit_failure_rolls_back_fully` | PASS |
| Sidecar rollback | `_commit_sidecar` + rollback | `.test_sidecar_commit_failure_rolls_back_model` | PASS |
| Ledger rollback | `_commit_ledger` + `_rollback_sidecar` | `.test_ledger_commit_failure_rolls_back_sidecar_and_model` | PASS |
| Receipt failure does not roll back | `_build_receipt` | `.test_receipt_failure_after_commit_does_not_roll_back` | PASS |
| Failed batch ID retryable | ledger commit ordering | `.test_failed_transaction_batch_id_can_be_retried` | PASS |
| Duplicate append (ID-keyed, payload-independent) | `preflight()` | `AppendAdapterDuplicateLedgerTest.test_duplicate_batch_id_second_append_is_blocked`, `.test_duplicate_block_ignores_payload_changes` | PASS |
| Different batch IDs, same geometry, both append (policy) | `preflight()` | `.test_different_batch_ids_same_geometry_both_append` | PASS |
| Ledger ownership/lifecycle (adapter-owned, not shared) | ledger design | `.test_ledger_is_adapter_owned_not_shared_across_instances` | PASS |
| Provenance preservation | sidecar entry | `AppendAdapterProvenanceAndClusterTest.test_sidecar_preserves_full_provenance` | PASS |
| Cluster mapping (min, order-independent) | `_project_cluster_id()` | `.test_cluster_id_projects_deterministic_min_not_first_element`, `.test_cluster_id_projection_independent_of_list_order` | PASS |
| Receipt correctness (success/rejection/determinism) | `UncertainAppendReceipt` | `AppendAdapterReceiptTest.test_receipt_fields_on_success`, `.test_receipt_fields_on_rejection`, `.test_receipt_json_is_deterministic` | PASS |
| Model snapshot/restore round-trip | `TorchGaussianModel.snapshot_state/restore_state` | `ModelStateSnapshotTest.test_snapshot_restore_round_trip_is_exact` | PASS |

### 7. Targeted regression 및 전체 pytest 결과

```text
python -B -m unittest -v tests.test_uncertain_gaussian_append_adapter
  Ran 42 tests in 0.421s — OK

python -m pytest tests/test_uncertain_gaussian_append_adapter.py tests/test_uncertain_gaussian_proposal.py \
    tests/test_occluded_chart_hardening.py tests/test_occluded_chart.py tests/test_occluded_region_candidate.py \
    tests/test_candidate_evidence.py tests/test_continuation_domain.py
  152 passed

python -B -m pytest
  391 passed, 1 skipped, 1 warning, 8 subtests passed  (이전 353 passed에서 +38 = 신규 append-adapter 테스트 42개 - 기존 4개)
```

warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 1건이다(무관, 회귀 아님).

### 8. 범위 확인

optimizer state expansion, trainer/renderer/checkpoint integration, appearance/opacity 추정 정책, review workflow, conflict resolution, global ranking, uncertain-to-certain promotion, source chart/observed Gaussian 수정은 이번 보강에서도 수행하지 않았다. 실제 발견된 defect(atomicity 부재, cluster ID 투영 오류) 2건은 adapter/model-only 범위 안에서 최소 수정했다(`snapshot_state`/`restore_state` 신설, `_project_cluster_id` 신설 — 둘 다 순수 추가, 기존 `replace_tensors` 호출부 동작 불변).

## 최종 보고

### A. 결론

- append adapter Gate 검토 가능: **예** — transaction/contract 검증 보강을 완료했다.
- transaction defect 수정 여부: **예** — (1) `replace_tensors`/`append_gaussians_model_only`가 실제로는 atomic하지 않음을 확인하고 `snapshot_state`/`restore_state` 기반 adapter-level rollback으로 보완, (2) `cluster_ids`가 `source_patch_ids[0]`을 임의로 쓰던 결함을 `min(source_patch_ids)` 결정론적 규칙(`_project_cluster_id`)으로 수정.
- production integration 미착수 확인: **예** — optimizer/trainer/renderer/checkpoint 어디에도 연결하지 않았다.

### B. Atomic transaction

- model: `snapshot_state()`/`restore_state()` 신설, model commit 실패 시 전체 복원.
- sidecar: `_commit_sidecar`/`_rollback_sidecar`로 별도 commit 단계, 실패 시 model도 함께 복원.
- ledger: `_commit_ledger`로 최종 단계, 실패 시 sidecar+model 복원.
- failure injection: model/sidecar/ledger/conversion/receipt 5개 지점 각각 독립 테스트로 검증(§2, §6 표).
- rollback 방식: **사전 snapshot 후 rollback**(옵션 1) 채택 — `replace_tensors` 자체를 건드리지 않고 adapter가 스스로 model 상태를 관리.

### C. Conversion contract

- scale: linear→log 수치 검증, zero/negative/극소값(0 underflow) 각각 별도 거부 사유로 검증.
- rotation: pass-through 순서·unit-norm 계약 검증, 비정규화 quaternion 거부 검증.
- initialization: 값 그대로 통과, 숨겨진 default 없음, content-digest로 추적 가능.
- dtype/device/order: float32/cpu 정렬 및 valid-mask 필터링 후 sample ID 정렬 일치 검증.

### D. Provenance

- sidecar: batch ID, sample IDs, chart/candidate/patch/domain/boundary ID, append origin, initialization digest, cluster_id/rule, appended index range 전부 보존 확인.
- cluster mapping: `min(source_patch_ids)`로 결정론적·순서 무관 투영, receipt/sidecar 양쪽에 기록.
- ledger ownership: adapter-owned, in-process, 인스턴스 간 공유 안 됨을 문서화하고 실제로 우회 가능함을 테스트로 증명.

### E. Test coverage

§6 표 참고(19개 계약 항목, 전부 실제 test method에 연결, PASS).

### F. Test results

```text
python -B -m unittest -v tests.test_uncertain_gaussian_append_adapter  →  Ran 42 tests — OK
python -m pytest tests/test_uncertain_gaussian_append_adapter.py tests/test_uncertain_gaussian_proposal.py tests/test_occluded_chart_hardening.py tests/test_occluded_chart.py tests/test_occluded_region_candidate.py tests/test_candidate_evidence.py tests/test_continuation_domain.py  →  152 passed
python -B -m pytest  →  391 passed, 1 skipped, 1 warning, 8 subtests passed
```

### G. Deferred scope

optimizer state expansion, trainer integration, renderer integration, checkpoint save/load(및 checkpoint-persistent ledger), appearance/opacity 추정 정책, review workflow, conflict resolution, global ranking, uncertain-to-certain promotion — 전부 미착수, 이번 보강에서도 시작하지 않았다.

Uncertain Gaussian append adapter의 transaction 및 contract 검증 결과를 제출하며, optimizer, trainer, renderer와 checkpoint integration은 수행하지 않았다. 사용자 Gate 승인 여부에 대한 검토를 요청한다.

---

## 최종 계약 보완 (2026-07-26 후속)

상태: **API 계약 결함 2건 수정 완료. 사용자 Gate 검토 대기(변경 없음). optimizer, trainer, renderer, checkpoint integration은 여전히 미착수.**

새 production integration은 추가하지 않고, 다음 두 계약 결함만 수정했다.

### 1. Receipt strong exception guarantee

**문제**: 이전 구현은 model/sidecar/ledger commit이 전부 성공한 뒤 `_build_receipt()`를 호출했다. commit은 이미 끝났는데 receipt 생성 자체가 실패하면 호출자 입장에서 "append가 됐는지 안 됐는지" 판단할 수 없는 ambiguous 상태였다.

**수정**: receipt(그리고 sidecar entry)를 **model을 건드리기 전에** 전부 미리 구성하도록 순서를 바꿨다.

```text
preflight
→ conversion(순수, model 미접촉)
→ model_count_before/after 계산(count는 conversion 결과에서 이미 확정)
→ 완전한 success receipt 후보 구성            ← model 접촉 전
→ 완전한 sidecar entry 구성                    ← model 접촉 전
→ model.snapshot_state()
→ model commit                                 실패 시 model.restore_state() 후 raise
→ sidecar commit                               실패 시 model.restore_state() 후 raise
→ ledger commit(model-owned, §2)               실패 시 sidecar 롤백 + model.restore_state() 후 raise
→ 사전 구성된 receipt 반환                      (이 지점 이후 실패 가능한 코드 없음)
```

Receipt의 모든 필드(`requested_sample_count`, `valid_sample_count`, `appended_sample_count`, `rejected_sample_count`, `model_count_before/after`, `appended_index_range`, `appended_sample_ids`, `conversion_summary`, `cluster_id`, `cluster_id_projection_rule`, `initialization_digest`)는 `batch`/`preflight`/`converted`와 `before`/`after = before + count`만으로 전부 계산 가능했다 — commit 결과에 의존하는 필드는 없었다(`model.append_gaussians_model_only`가 성공하면 정확히 `count`개를 추가한다는 concatenation 계약을 그대로 신뢰). 따라서 receipt를 뒤로 미룰 이유가 없었다.

이 재구성으로 **"commit 성공 후 receipt 생성 실패"라는 경우 자체가 구조적으로 사라졌다** — receipt 생성이 실패할 수 있는 경우는 이제 전부 "아직 model을 건드리기 전" 단계에 있으므로, conversion 실패와 동일한 카테고리(아무것도 커밋되지 않음)가 됐다.

**테스트**(`AppendAdapterTransactionTest`):
- `test_receipt_candidate_failure_leaves_everything_untouched`: `_build_receipt`를 몽키패치해 실패시키고 model/sidecar/ledger 전부 무변경 확인.
- `test_sidecar_entry_build_failure_leaves_everything_untouched`: `_build_sidecar_entry` 실패도 동일하게 무변경 확인.
- `test_model_commit_failure_rolls_back_fully`, `test_sidecar_commit_failure_rolls_back_model`, `test_ledger_commit_failure_rolls_back_sidecar_and_model`: 기존 유지, 각각 model/sidecar/ledger 무변경.
- `test_successful_commit_always_returns_success_receipt`: 성공 시 반드시 success receipt 반환(신설, 강한 보장 자체를 명시적으로 검증).
- 기존 `test_receipt_failure_after_commit_does_not_roll_back`(commit 후 receipt 실패는 롤백 안 함)는 **새 계약과 모순되므로 삭제**했다 — 새 설계에서는 그 시나리오 자체가 존재하지 않는다(receipt가 이미 commit 전에 완성돼 있다).
- receipt serialization 결정성(`test_receipt_json_is_deterministic`)은 그대로 유지·통과.

### 2. 동일 model 내 duplicate ledger 공유

**문제**: ledger(`_appended_batch_ids`)가 adapter 인스턴스 소유였다. 새 `UncertainGaussianAppendAdapter()`를 만들면 같은 model에 동일 batch ID를 다시 append할 수 있어 duplicate protection 계약을 실질적으로 만족하지 못했다.

**수정**: ledger 소유권을 `TorchGaussianModel`로 이동했다.

```python
# osn_gs/gaussian/torch_model.py, __init__
self.appended_uncertain_batch_ids: set[str] = set()
```

- `preflight()`는 `self._appended_batch_ids` 대신 `model.appended_uncertain_batch_ids`를 조회한다.
- `_commit_ledger(model, batch_id)`는 `model.appended_uncertain_batch_ids.add(batch_id)`로 커밋한다(adapter는 더 이상 자체 ledger 상태를 갖지 않는다).
- provenance sidecar(`self._sidecar`)는 그대로 **adapter-instance 소유**로 유지했다 — sidecar는 "이 adapter 인스턴스가 무엇을 했는지"의 참고 기록일 뿐 duplicate 판정 메커니즘이 아니므로, ledger처럼 공유해야 할 이유가 없다(작업 지시 §2도 ledger만 이동을 요구했다).
- `replace_tensors`/`snapshot_state`/`restore_state`는 이 집합을 건드리지 않는다 — ledger 등록은 model/sidecar commit이 이미 성공한 뒤 마지막 단계에서만 일어나므로 rollback 대상이 될 필요가 없다(등록 실패 시 애초에 아무것도 추가되지 않는다).
- **model reset 정책**: 명시적 reset API는 없다. batch ID의 존재 여부는 이 **model Python 객체의 생존 기간**에 묶인다 — 새 `TorchGaussianModel()`은 빈 ledger로 시작한다. checkpoint save/load를 통한 영속화는 여전히 deferred gap이다(module docstring에 명시).

**조건별 검증**(`AppendAdapterDuplicateLedgerTest`):
- 같은 model + 다른 adapter → 차단: `test_duplicate_blocked_across_adapter_instances_same_model`(adapter 1이 append한 뒤 adapter 2가 같은 model에 같은 batch ID로 시도 → `duplicate_proposal_batch`로 거부, model tensor 무변경).
- 다른 model → 독립: `test_same_batch_id_different_models_are_independent`(동일 batch ID가 model A/B 양쪽에 독립적으로 append 성공, 세 번째 신규 model에는 없음).
- failed transaction → batch ID 미등록: `test_failed_transaction_does_not_consume_batch_id`(기존 유지).
- successful transaction → batch ID 등록: `test_successful_transaction_registers_batch_id`(기존 유지).
- **다른 adapter로 재시도 성공**: `test_failed_transaction_retry_with_different_adapter_succeeds`(adapter A에서 ledger commit을 강제 실패시킨 뒤, 완전히 다른 adapter B 인스턴스로 같은 batch/model에 재시도 → 성공. 이전 vulnerability-reproduction 테스트였던 `test_ledger_is_adapter_owned_not_shared_across_instances`는 삭제하고 이 fixed-contract 테스트로 교체했다).
- ledger/sidecar/model rollback 일관성: 기존 `test_ledger_commit_failure_rolls_back_sidecar_and_model` 그대로 유지(model이 이제 ledger를 갖고 있어도 rollback 대상은 여전히 model tensor + adapter sidecar뿐이라는 점을 재확인).
- duplicate rejection 시 model tensor 불변: `test_duplicate_batch_id_second_append_is_blocked`(기존 유지).

### 3. Cluster ID downstream 감사 결과

`cluster_ids`/`surface_patch_ids`(별칭 property)의 모든 read site를 `osn_gs/` 전체에서 검색했다(cluster schema 자체는 확장하지 않음).

| 위치 | 용도 분류 | 실제 동작 |
|---|---|---|
| `osn_gs/utils/torch_checkpoint.py:37,84` | **serialization/checkpoint** | 다른 per-Gaussian tensor와 함께 그대로 save/load됨(기존부터 이미 checkpoint되고 있었다 — deferred인 것은 append **ledger**뿐, `cluster_ids` 텐서 자체는 아니다). |
| `osn_gs/gaussian/torch_density_control.py:260,284,318` | **densification/pruning(ADC)** | clone/split 후보에 parent의 `cluster_ids`를 `repeat_interleave`/인덱싱으로 그대로 전달, `replace_tensors`로 prune 후 유지. 값 기반 의사결정은 없고 patch 소속을 끊기지 않게 전달만 한다. |
| `osn_gs/core/torch_pipeline.py:206-230`(초기화) | **초기화/diagnostics** | voxel-region 결과로 초기 `cluster_ids` 결정 → `project_points_to_patches`로 그 patch에 대한 초기 `surface_uv` 계산. diagnostics(`final_gaussian_indices/uv`)에도 사용. |
| `osn_gs/core/torch_pipeline.py:677-688`(`maintain_surface_from_certain`) | **training-time 동작** | `model.cluster_ids == patch_id`로 그 patch 소속 certain Gaussian을 골라 `surface_uv`를 **그 patch 기준으로** 재투영한다 — cluster_ids 값이 어떤 patch의 fitting 대상이 되는지를 직접 결정한다. |
| `osn_gs/core/torch_pipeline.py:762-818`(`_split_failed_patch`) | **grouping 재구성(동작)** | 실패한 patch에 속한 Gaussian을 `cluster_ids`로 모아 로컬 재군집한 뒤 `model.cluster_ids[component_indices] = new_patch_id`로 **재대입**한다 — 학습 도중 patch 소속이 바뀌는 실제 동작. |
| `osn_gs/core/torch_pipeline.py:1020-1029`(UV support mask) | **rendering에 영향** | `cluster_ids`로 patch별 UV 집합을 모아 `patch.uv_support_mask`를 만든다 — `TorchNURBSSurface.support()`가 이 마스크로 렌더링/측정 범위를 제한하므로 렌더링 결과에 직접 영향을 준다. |
| `osn_gs/core/torch_pipeline.py:1278`(`_assign_uncertain_colors`) | **미사용(dead code)** | 자체 docstring이 "Stage 2 legacy helper for future uncertain Gaussian initialization"라고 명시하며, `model.cluster_ids`가 아니라 완전히 다른 로컬 변수(`color_cluster_count` 나머지 연산)다. 코드베이스 전체에서 이 함수를 호출하는 곳이 없음을 grep으로 확인 — 현재 아무 영향 없음. |
| `osn_gs/losses/torch_losses.py:130` | **training/loss 계산** | `state.model.surface_patch_ids[indices]`로 `active_mask`를 만들어 patch별 smoothness/anchor loss 항의 대상을 결정한다 — cluster_ids 값이 어떤 Gaussian이 어떤 patch의 loss 그래프에 들어가는지를 직접 결정한다. |

**결론**: `cluster_ids`는 metadata/diagnostics 전용이 아니다. **training-time 동작(uv 재투영 대상, patch 재군집, loss membership)과 rendering(UV support mask)에 직접 영향을 주는 behavioral 필드**다. 이 adapter가 쓰는 `min(source_patch_ids)` 규칙은 두 patch 중 하나를 "주인"으로 임의 지정하는 것과 마찬가지이므로, occluded chart가 실제로는 두 patch **사이**를 잇는 bridge라는 사실을 무시하게 된다. 이번 작업에서는 cluster schema를 확장하지 않고 **현재 규칙(`min(source_patch_ids)`)을 유지**하되, 다음을 deferred risk로 명시적으로 기록한다.

**Deferred risk (production integration 전 필수 해결)**: model-only append로 만들어진 uncertain Gaussian이 실제로 optimizer/trainer 경로에 연결되는 시점에는, `torch_pipeline.py`의 patch-maintenance 로직(`maintain_surface_from_certain`, `_split_failed_patch`, UV support mask 생성)과 `torch_losses.py`의 patch-loss membership이 이 Gaussian을 `min(source_patch_ids)` patch에게만 속한 것처럼 취급하게 된다 — 두 patch 모두에 걸친 bridge라는 provenance(`source_patch_ids` 전체, sidecar에 보존됨)는 `cluster_ids` 하나로는 표현되지 않는다. Production integration 전에 다음 중 하나를 별도로 설계·승인해야 한다: (a) bridge Gaussian을 위한 **primary-owner patch** 공식 계약(현재 규칙을 그 계약으로 승격), 또는 (b) 두 patch에 걸친 membership을 표현하는 **synthetic/dual cluster** 계약(schema 확장 필요, 이번 범위 밖). 지금은 어느 쪽도 결정하지 않았고, `min(source_patch_ids)`는 어디까지나 "결정론적이고 순서-무관한 임시 값"이라는 지위로만 유지한다.

### 4. 회귀 결과

```text
python -B -m unittest -v tests.test_uncertain_gaussian_append_adapter
  Ran 46 tests — OK

python -m pytest tests/test_uncertain_gaussian_append_adapter.py tests/test_uncertain_gaussian_proposal.py \
    tests/test_occluded_chart_hardening.py tests/test_occluded_chart.py tests/test_occluded_region_candidate.py \
    tests/test_candidate_evidence.py tests/test_continuation_domain.py tests/test_training_regressions.py
  180 passed

python -B -m pytest
  398 passed, 1 skipped, 1 warning, 8 subtests passed
```

`test_training_regressions.py`를 targeted suite에 추가했다 — `torch_model.py`(production 파일, ADC/checkpoint/pipeline이 공유)를 수정했으므로 기존 학습 경로 회귀도 별도로 재확인했다. warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 1건으로 무관.

### 완료 조건 확인

- `append()` 예외 발생 시 side effect 없음: **충족**(§1, 5개 실패 지점 전부 테스트).
- commit 성공 시 반드시 success receipt 반환: **충족**(§1, 구조적으로 실패 불가능한 경로로 재구성).
- 동일 model의 다른 adapter에서도 duplicate batch 차단: **충족**(§2).
- failed batch ID는 재시도 가능(다른 adapter 포함): **충족**(§2).
- 다른 model의 ledger는 독립: **충족**(§2).
- checkpoint persistence는 미구현 상태로 명시: **충족**(ledger만 deferred, `cluster_ids` 텐서 자체는 기존처럼 checkpoint됨 — §3 감사로 확인).
- optimizer/trainer/renderer/checkpoint integration 미착수: **충족**.
- 전체 회귀 통과: **충족**(§4).

---

Append Adapter의 receipt 및 process-local duplicate protection 계약을 보완했으며, optimizer, trainer, renderer와 checkpoint integration은 수행하지 않았다. 사용자 Gate 승인 검토를 요청한다.