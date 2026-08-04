# Worklog 3: Occluded Chart Ownership Foundation

날짜: 2026-07-27

상태: **Ownership Foundation 사용자 Gate 승인 완료(2026-07-27).** O-grid/planar/annulus/topology 관련 별도 작업(다른 에이전트 담당)은 진행 중이며 이번에도 조사·수정하지 않았다. Optimizer, trainer, renderer 및 checkpoint production integration은 미착수.

Append Adapter Gate: APPROVED. Occluded Chart Ownership Foundation Gate: APPROVED(§1-14 전체, owner registry transaction consistency·append strong exception guarantee·`cluster_id>=0 → VISIBLE_PATCH`/`cluster_id<0 → UNASSIGNED`·visible patch reassignment 시 `cluster_ids`/`surface_owner_id` 동기화·uncertain Gaussian owner=occluded chart·source patch IDs=provenance only 전부 승인 근거로 유지). 저장소 전체 `pytest` 결과는 `471 passed, 1 skipped, 2 failed`이며, 실패 2건은 이 Gate와 무관한 진행 중인 boundary-first/annulus/cyclic-topology 병렬 작업(worklog 89-98) 소관이다 — Ownership Foundation Gate 승인 자체는 이 실패들과 별개이며, repository-wide integration state는 아직 green이 아니다. Optimizer/trainer/renderer/checkpoint integration은 이 승인 이후에도 별도 승인 없이는 착수하지 않는다.

## 1. Canonical ownership 결정

```
Observed/certain Gaussian    -> visible NURBS patch ownership
Occluded/uncertain Gaussian  -> occluded NURBS chart ownership
Visible source patch/domain/boundary/candidate -> provenance only
```

`source_patch_ids`(및 append adapter의 `cluster_id = min(source_patch_ids)`)는 uncertain Gaussian의 생성 근거·provenance일 뿐 behavioral owner가 아니라는 계약을 그대로 유지했다 — primary visible-patch owner, production training membership, visible patch loss/maintenance/support-mask membership 어느 것으로도 승격하지 않았다. Dual/multi-membership schema 확장도 하지 않았다(별도 설계 대상으로 남김).

## 2. Ownership representation — Option A vs B 비교와 채택 결과

| 기준 | Option A(reserved namespace, `cluster_ids` 재사용) | Option B(explicit `surface_owner_kind`+`surface_owner_id`) |
|---|---|---|
| 기존 `cluster_ids` read site 변경 범위 | 작음(값만 다르게) | 각 read site에 별도 gate 추가 필요 |
| accidental visible-patch lookup 위험 | 큼 — namespace 해석을 잊으면 그대로 오인 가능(실제로 `_assign_uv_support_masks`의 `patch_id==0` catch-all이 `cluster_ids>=n_patches`로 큰 값을 전부 흡수하는 기존 코드를 발견 — namespace 값을 `cluster_ids`에 넣으면 이 catch-all이 모든 occluded Gaussian을 patch 0에 흡수해버림) | 작음 — kind 태그가 항상 먼저 확인됨 |
| one-way dependency 명시성 | 약함(암묵적 정수 해석) | 강함(타입처럼 명시적) |
| ADC transport | 가능(정수 하나만 전달) | 가능(정수 두 개 전달, 동일 난이도) |
| checkpoint migration | 쉬움(필드 하나) | 쉬움(필드 두 개, 이번엔 저장 안 함) |
| loss/maintenance 필터링 용이성 | 매번 namespace 비교 필요 | `is_visible_patch_owned()` 한 줄 |
| 테스트 가능성 | 낮음(정수 하나로 여러 의미 검증) | 높음(kind/id 독립 검증) |
| 장기 schema 명확성 | 약함(타입 안정성 없음) | 강함 |

**Option B(explicit ownership representation) 채택.** 근거는 감사 중 실제로 발견한 `_assign_uv_support_masks`의 `patch_id==0` catch-all(`cluster_ids < 0 | cluster_ids >= n_patches`)이 Option A의 "큰 정수 namespace"를 `cluster_ids`에 직접 넣었을 경우 정확히 이 catch-all에 흡수되어 모든 occluded Gaussian을 patch 0 소유로 잘못 분류했을 것이라는 점 — 이는 이론적 위험이 아니라 기존 코드에서 실제로 확인된 결함 조건이다. 다만 defense-in-depth로 Option A의 namespace-offset 아이디어는 `surface_owner_id`에도 그대로 적용했다(`OCCLUDED_CHART_NAMESPACE_BASE = 10^12`) — kind 태그를 실수로 무시하고 owner_id만 읽어도 실제 patch_id와 충돌하지 않도록.

## 3. Owner identity contract

새 모듈 `osn_gs/gaussian/torch_surface_ownership.py`:

```python
SURFACE_OWNER_UNASSIGNED = 0
SURFACE_OWNER_VISIBLE_PATCH = 1
SURFACE_OWNER_OCCLUDED_CHART = 2
OWNERSHIP_SCHEMA_VERSION = 1
OCCLUDED_CHART_NAMESPACE_BASE = 1_000_000_000_000

def project_occluded_chart_owner_id(source_chart_id: str, *, schema_version=OWNERSHIP_SCHEMA_VERSION) -> int:
    digest = hashlib.sha256(f"{schema_version}|{source_chart_id}".encode()).hexdigest()
    return OCCLUDED_CHART_NAMESPACE_BASE + int(digest[:15], 16)
```

- 같은 `source_chart_id` → 같은 id(동일 프로세스/재실행 무관, `hashlib.sha256` 사용 — Python 내장 salted `hash()` 미사용).
- 다른 `source_chart_id` → 다른 id.
- 입력이 chart_id 문자열 하나뿐이라 순서 문제 자체가 없음.
- `PYTHONHASHSEED`를 0/1/12345로 바꿔 별도 subprocess 3회 실행 후 동일 값 확인(`test_identity_independent_of_python_hash_seed`) — Python hash-seed 무관을 실제로 증명.
- `>= 10^12`라 어떤 현실적 visible `patch_id`와도 충돌하지 않음.
- floating-point control point/UV byte를 전혀 쓰지 않고 이미 안정적인 `chart_id` 문자열(Phase F `_chart_id`)만 해시.

## 4. Compatibility projection 처리

append adapter의 `cluster_id = min(source_patch_ids)`는 그대로 유지했다(수치·규칙 변경 없음). 다만:
- `_ConvertedAppend`/sidecar/receipt에 `surface_owner_kind`/`surface_owner_id`를 **별도 필드로 추가**했다 — `cluster_id`를 대체하지 않았다.
- `model.append_gaussians_model_only(...)`에 새 필수 인자 `surface_owner_kind`(항상 `SURFACE_OWNER_OCCLUDED_CHART`), `surface_owner_id`(`project_occluded_chart_owner_id(source_chart_id)`)를 추가했다.
- behavioral read site(§5)는 전부 `surface_owner_kind`/`is_visible_patch_owned()`만 신뢰하며 `cluster_ids`를 owner 판정에 단독으로 쓰지 않는다.

## 5. 수정한 behavioral read site

### `_assign_uv_support_masks` (osn_gs/core/torch_pipeline.py) — 실제 결함 수정

기존 코드는 `cluster_ids == patch_id`만으로 UV occupancy를 계산했고, `is_uncertain`/ownership 게이트가 전혀 없었다 — occluded-chart-owned Gaussian의 `cluster_ids`(compatibility projection)가 patch_id와 우연히 같으면 그 patch의 support mask를 실제로 팽창시켰다. `is_visible_patch_owned(model.surface_owner_kind)`를 AND 조건으로 추가해 고쳤다(`patch_id==0` catch-all에도 동일 게이트 적용).

### `maintain_surface_from_certain`, `_split_failed_patch` — 이미 올바름을 확인, 코드 변경 없음

두 함수 모두 이미 `certain = ~model.is_uncertain`으로 게이트하고 있어(`torch_pipeline.py:644`, `:762`) occluded-chart-owned Gaussian은 코드 변경 전부터 정상적으로 제외되고 있었다. 새 테스트로 이 기존 정확한 동작을 회귀-잠금했다(코드는 손대지 않음).

### `nurbs_surface_loss` (osn_gs/losses/torch_losses.py) — 이미 올바름을 확인, 코드 변경 없음

`certain = ~state.model.is_uncertain`으로 이미 게이트되어(`torch_losses.py:112`) patch별 smoothness/anchor loss membership에 uncertain Gaussian이 애초에 들어가지 않는다. 새 테스트로 회귀-잠금만 했다.

**중요 정정**: 이전 worklog 2의 deferred-risk 서술은 "cluster_ids 값이 어떤 Gaussian이 어떤 patch의 loss 그래프에 들어가는지 직접 결정한다"고 다소 과장돼 있었다 — 실제로는 `is_uncertain` 게이트가 이미 loss/maintenance/split 세 곳 모두를 보호하고 있었고, 진짜 gap은 `_assign_uv_support_masks` 단 하나였다. 이번 감사로 이 부분을 바로잡는다.

### ADC(osn_gs/gaussian/torch_density_control.py) — ownership-preserving transport만 추가, 정책 변경 없음

`apply_adaptive_density_control`의 clone/split 대상(`clone_mask`/`split_mask`)은 이미 `certain`으로만 게이트되므로 occluded-chart-owned Gaussian은 애초에 clone/split되지 않는다(패스스루만 됨). `_shape_transaction_candidates`의 `raw`/`split_values` dict와 `_commit_shape_transaction`의 `model.replace_tensors(...)` 호출에 `surface_owner_kind`/`surface_owner_id`를 `cluster_ids`와 동일한 패턴으로 추가해, clone/split/prune 전 구간에서 ownership이 그대로 실려가도록 했다.

## 6. One-way dependency invariant

```
observed Gaussian -> visible patch -> occluded chart -> uncertain proposal -> uncertain appended Gaussian
```

구조적으로 보장: `initialize()`가 만드는 모든 Gaussian은 `surface_owner_kind=VISIBLE_PATCH`(마이그레이션 기본값)이고, `append_gaussians_model_only`만 `SURFACE_OWNER_OCCLUDED_CHART`를 부여할 수 있으며, ADC/prune은 값을 전달만 할 뿐 kind를 변환하지 않는다. `test_uncertain_owner_never_converted_to_visible_owner`, `test_uncertain_gaussian_not_in_certain_maintenance_indices`로 검증.

## 7. Ownership data contract

```python
# TorchGaussianModel (osn_gs/gaussian/torch_model.py)
self.surface_owner_kind: Any  # long tensor (N,)
self.surface_owner_id: Any    # long tensor (N,)
```

Sidecar provenance(변경 없음, owner와 혼동하지 않음): `source_chart_id`, `source_candidate_id`, `source_patch_ids`, `supporting_domain_ids`, `supporting_boundary_ids`, `proposal_batch_id`, `proposal_sample_ids`, `append_origin` — 이제 `surface_owner_kind`/`surface_owner_id`/`cluster_id`/`cluster_id_projection_rule`도 함께 기록해 추적 가능(§10 "receipt와 sidecar의 owner identity 일치" 테스트로 확인).

## 8. 기존 model invariant 확인

- `TorchGaussianModel.__init__`에 `surface_owner_kind`/`surface_owner_id`를 빈 텐서로 추가(순수 추가, 기존 필드 무변경).
- `initialize()`: 새 optional 인자, `None`이면 `owner_kind=VISIBLE_PATCH`, `owner_id=cluster_ids`(§9 마이그레이션 규칙 그대로).
- `replace_tensors()`: 새 optional 인자(같은 fallback) — **checkpoint 로드(`torch_checkpoint.py`, 무변경)가 이 새 인자 없이 호출해도 깨지지 않도록 optional로 유지**. 단 실제 ownership을 추적하는 내부 호출부(`prune`, ADC `_commit_shape_transaction`, `append_gaussians_model_only`)는 전부 명시적으로 전달하도록 수정해 fallback이 실사용 경로에서 절대 실제 값을 덮어쓰지 않게 했다.
- `append_gaussians_model_only()`: 새 인자 **필수**(fallback 없음) — 이 경로가 바로 실제 ownership을 부여해야 하는 지점이므로 묵시적 기본값을 허용하지 않는다.
- `prune()`: `surface_owner_kind[keep_mask]`/`surface_owner_id[keep_mask]`를 명시적으로 슬라이스해 전달하도록 수정.
- `snapshot_state`/`restore_state`(`_STATE_TENSOR_NAMES`)에 두 필드 추가 — append adapter의 rollback 계약이 ownership도 그대로 커버.
- 기존 checkpoint tensor, existing ADC certain path, existing renderer certain path, append transaction/duplicate ledger 계약은 전부 무변경(회귀 테스트로 확인).

## 9. 테스트

**신규 파일** `tests/test_surface_ownership.py`(20개): identity(6), 헬퍼(1), behavioral isolation(6, support mask 실제 leak 재현+수정 확인 포함), ADC transport(4), one-way dependency(2).

**`tests/test_uncertain_gaussian_append_adapter.py`에 추가**(3개 신규 + 기존 `model_snapshot` 헬퍼에 owner_kind/owner_id 편입으로 기존 rollback/duplicate 테스트 전체가 ownership도 자동 검증):
- `test_appended_rows_are_occluded_chart_owned_not_visible_patch_owned`
- `test_receipt_and_sidecar_owner_identity_match`
- `test_ownership_row_alignment_survives_valid_mask_filtering`

기존 `model_snapshot()`에 `surface_owner_kind`/`surface_owner_id`를 추가했으므로, 이미 있던 model-commit/sidecar-commit/ledger-commit 실패 롤백 테스트, duplicate-append 테스트 전부가 **추가 코드 없이** ownership tensor 불변까지 검증하게 됐다.

## 10. Checkpoint 경계

`torch_checkpoint.py`는 수정하지 않았다. Deferred gap(향후 저장 필요):
- ownership schema version
- `surface_owner_kind`, `surface_owner_id` 텐서
- occluded chart registry/ID mapping(현재 없음 — owner_id는 `chart_id`에서 그때그때 재계산 가능하므로 별도 registry 불필요할 수 있음, 별도 검토 필요)
- appended batch ledger(`model.appended_uncertain_batch_ids`, worklog 2에서 이미 지적)
- proposal provenance sidecar

현재 프로세스에서만 유효한 ownership: `replace_tensors`의 optional fallback 때문에 **old 체크포인트를 로드하면 모든 Gaussian이 `VISIBLE_PATCH`/`owner_id=cluster_id`로 마이그레이션되고, 저장 시점에 실제로 `OCCLUDED_CHART`였던 Gaussian이 있었다 해도 그 사실은 checkpoint reload 후 복구할 수 없다**(애초에 이 기능이 아직 어떤 저장 경로에도 연결되지 않았으므로 현재는 해당 사례가 없지만, 향후 checkpoint에 ownership을 실제로 저장하기 전까지는 이 한계가 유효하다).

## 11. 별도 에이전트 작업 경계 확인

O-grid, planar construction, annulus construction, cyclic topology, topology-specific fitting branch, construction fallback, topology branch inventory 및 관련 policy/테스트/문서 — grep으로 확인한 결과 이번 변경분(`torch_surface_ownership.py`, `torch_model.py`, `torch_pipeline.py`의 `_assign_uv_support_masks`, `torch_density_control.py`, `torch_uncertain_append_adapter.py`, 신규 테스트 2개 파일) 중 어느 것도 `osn_gs/surface/torch_boundary_*.py` 계열 파일을 import하거나 참조하지 않는다. 작업 시작 전 `git status`/mtime으로 다른 에이전트(Codex)가 이 시점에 `torch_boundary_*` 계열 파일(worklog 89-95)을 활발히 수정 중임을 확인했고, 파일 목록이 완전히 분리돼 있어 충돌 없이 병행 진행했다. 해당 파일들은 조사·수정하지 않았다.

## 12. 회귀 결과

```text
python -B -m unittest -v tests.test_uncertain_gaussian_append_adapter tests.test_surface_ownership \
    tests.test_training_regressions tests.test_uncertain_gaussian_proposal tests.test_occluded_chart_hardening \
    tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain
  Ran 203 tests — OK

python -B -m pytest
  437 passed, 1 skipped, 1 warning, 8 subtests passed
```

warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 1건으로 무관(회귀 아님).

## 최종 보고

### A. 결론

- ownership foundation 구현 여부: **완료**.
- Gate 검토 가능 여부: **가능**.
- 별도 topology 작업 비변경 확인: **확인**(§11).

### B. Ownership contract

- representation: Option B(explicit `surface_owner_kind`+`surface_owner_id`), Option A의 namespace-offset을 defense-in-depth로 병용.
- owner namespace: `SURFACE_OWNER_{UNASSIGNED,VISIBLE_PATCH,OCCLUDED_CHART}` = `{0,1,2}`; occluded owner_id `>= 10^12`.
- deterministic identity: `sha256(schema_version|source_chart_id)` 기반, hash-seed 무관 확인(subprocess 3회).
- provenance: sidecar에 `source_patch_ids` 등 전체 원본 그대로 보존, owner와 분리.
- compatibility projection: `cluster_id = min(source_patch_ids)` 유지, behavioral read site에서 미사용.

### C. Behavioral isolation

- maintenance: 기존 `is_uncertain` 게이트가 이미 정확 — 회귀 테스트만 추가.
- split/reassignment: 기존 게이트 정확 — 회귀 테스트만 추가.
- support mask: **실제 결함 발견·수정**(`is_visible_patch_owned` 게이트 추가).
- patch loss: 기존 게이트 정확 — 회귀 테스트만 추가.
- ADC transport: `surface_owner_kind`/`surface_owner_id`를 `cluster_ids`와 동일하게 clone/split/prune 전 구간에 전달하도록 추가.

### D. 수정 파일

- production: `osn_gs/gaussian/torch_surface_ownership.py`(신규), `osn_gs/gaussian/torch_model.py`, `osn_gs/gaussian/torch_uncertain_append_adapter.py`, `osn_gs/gaussian/torch_density_control.py`, `osn_gs/core/torch_pipeline.py`.
- tests: `tests/test_surface_ownership.py`(신규, 20개), `tests/test_uncertain_gaussian_append_adapter.py`(+3, 기존 헬퍼 확장).
- worklog: 이 문서(96), Worklog 2 상태선 갱신(Gate 승인 반영).

### E. Test results

```text
python -B -m unittest -v tests.test_uncertain_gaussian_append_adapter tests.test_surface_ownership \
    tests.test_training_regressions tests.test_uncertain_gaussian_proposal tests.test_occluded_chart_hardening \
    tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain
  Ran 203 tests — OK

python -B -m pytest
  437 passed, 1 skipped, 1 warning, 8 subtests passed
```

### F. Deferred scope

checkpoint(ownership tensor/schema version/registry 저장 미구현), optimizer, trainer, renderer, occluded chart 전용 loss, dual/multi-membership schema, uncertain-to-certain promotion workflow — 전부 미착수.

---

## 13. Lifecycle Invariant 최종 보완 (2026-07-27 후속)

상태: **Ownership lifecycle invariant 보완 완료. 사용자 Gate 검토 대기(변경 없음). topology 관련 별도 작업, optimizer/trainer/renderer/checkpoint integration 미착수.**

새 production integration을 추가하지 않고, §1-12의 결정을 그대로 유지하면서 ownership lifecycle invariant를 보완했다. O-grid/planar/annulus/cyclic topology 관련 파일은 이번에도 조사·수정하지 않았다.

### 13.1 Cluster membership write-site 감사 결과

Repository 전체에서 `cluster_ids`/`surface_patch_ids`의 모든 write site를 검색했다.

| Write site | 위치 | 분류 |
|---|---|---|
| `self.cluster_ids = torch.empty((0,), ...)` | `torch_model.py.__init__` | visible patch membership initialization(빈 부트스트랩) |
| `cluster_ids` 계산 후 `self.cluster_ids = cluster_ids` | `torch_model.py.initialize()` | visible patch membership initialization |
| `self.cluster_ids = torch.as_tensor(cluster_ids, ...)` | `torch_model.py.replace_tensors()` | infra passthrough(호출부별로 분류, 아래 항목 참고) |
| `cluster_ids` concat 후 `self.replace_tensors(...)` | `torch_model.py.append_gaussians_raw()` | dead code — 소스 전체에 live caller 없음(grep 확인) |
| `cluster_ids` concat 후 `self.initialize(...)` | `torch_model.py.append_uncertain()` | dead code — live caller 없음 |
| `cluster_ids=self.cluster_ids[keep_mask]` | `torch_model.py.prune()` | prune/filter transport |
| `cluster_ids=selected["cluster_ids"]` → `model.replace_tensors(...)` | `torch_density_control.py._commit_shape_transaction()` | ADC clone/split/prune transport |
| `cluster_ids = torch.full((count,), cluster_id, ...)` (`min(source_patch_ids)`) | `torch_uncertain_append_adapter.py._convert()` | occluded append compatibility projection |
| `cluster_ids=raw["cluster_ids"]` → `state.model.replace_tensors(...)` | `torch_checkpoint.py.load_torch_checkpoint()` | checkpoint load |
| `cluster_ids = self._point_region_ids(...)` → `model.initialize(...)` | `torch_pipeline.py.initialize()` | visible patch membership initialization |
| `cluster_ids = torch.full(...,-1,...)` → `cluster_ids[indices] = patch_id` → `model.initialize(...)` | `torch_pipeline.py._initialize_stage1()` | visible patch membership initialization |
| `model.cluster_ids[component_indices] = new_patch_id` | `torch_pipeline.py._split_failed_patch()` | **visible patch reassignment**(이번에 `surface_owner_id` 동기화 추가) |
| `cluster_ids = nearest % color_cluster_count` | `torch_pipeline.py._assign_uncertain_colors()` | diagnostics/test-only — `model.cluster_ids`와 무관한 로컬 변수, 호출부 없음(dead, 이전 세션에 이미 확인) |

`surface_patch_ids`는 읽기 전용 `@property`(`return self.cluster_ids`)이며, 이번 grep에서 `model.surface_patch_ids[...] =` 형태의 실제 write site는 발견되지 않았다(`torch_losses.py:130`은 read-only 사용).

### 13.2 Visible ownership synchronization invariant

```text
surface_owner_kind == VISIBLE_PATCH  ⇒  surface_owner_id == cluster_id
```

`_split_failed_patch`(유일한 실제 visible reassignment write site)에 동기화를 추가했다.

```python
model.cluster_ids[component_indices] = new_patch_id
model.surface_owner_id[component_indices] = new_patch_id
```

`component_indices`는 이미 `~model.is_uncertain`으로 제한된 집합의 부분집합이므로(§C "split/reassignment" 기존 게이트) occluded-chart-owned Gaussian은 이 대입에서 원천적으로 제외된다 — `surface_owner_kind`는 건드리지 않아 `VISIBLE_PATCH`로 유지된다. `test_reassignment_updates_cluster_id_and_owner_id_together`, `test_uncertain_owner_unaffected_by_reassignment`로 검증.

### 13.3 Ownership consistency validator

`osn_gs/gaussian/torch_surface_ownership.py`에 순수 read-only 함수 `validate_surface_ownership_consistency(model) -> tuple[str, ...]`을 추가했다(빈 튜플 = 위반 없음). 검사 항목: row count 일치(3개 텐서), `torch.long` dtype, device 일치, 알려진 enum 값만 존재, VISIBLE_PATCH 행의 `owner_id`가 occluded namespace 밖에 있고 `cluster_id`와 같음, OCCLUDED_CHART 행의 `owner_id`가 namespace 안에 있음. `SURFACE_OWNER_UNASSIGNED`는 어떤 코드 경로도 생성하지 않으므로 이 kind에는 owner_id 범위 제약을 걸지 않는다(정책으로 명시). **어떤 training hot path에도 자동 연결하지 않았다** — preflight/diagnostics/tests에서 호출자가 원할 때만 사용하는 순수 함수다.

### 13.4 Owner ID namespace와 collision 방어

`project_occluded_chart_owner_id`의 docstring/계약을 "충돌 없음을 보장"이 아니라 "충돌 저항적(collision-resistant)"으로 명확히 하고, 실제 방어는 새 함수로 구현했다.

```python
def project_and_register_occluded_chart_owner_id(model, source_chart_id, *, schema_version=...) -> int:
    owner_id = project_occluded_chart_owner_id(source_chart_id, schema_version=schema_version)
    existing = model.occluded_chart_owner_registry.get(owner_id)
    if existing is not None and existing != source_chart_id:
        raise OccludedChartOwnerCollisionError(...)
    model.occluded_chart_owner_registry[owner_id] = source_chart_id
    return owner_id
```

- **Registry 소유권: model-owned** — `TorchGaussianModel.__init__`에 `self.occluded_chart_owner_registry: dict[int, str] = {}`를 추가했다(worklog 2의 `appended_uncertain_batch_ids`와 동일 이유: adapter-owned였다면 adapter 인스턴스를 교체해 collision 이력을 우회할 수 있다). **Module-level global registry는 사용하지 않았다**(model 독립성·테스트 격리 문제 회피).
- 같은 chart ID 반복 projection은 허용(`existing == source_chart_id`이면 통과).
- 실제 SHA-256 충돌은 구성이 불가능하므로, `project_occluded_chart_owner_id`를 몽키패치해 서로 다른 chart가 같은 owner_id를 반환하도록 강제한 뒤 `OccludedChartOwnerCollisionError`가 실제로 발생하고 registry가 오염되지 않음을 테스트로 증명했다(`test_forced_collision_raises_explicit_error`).
- 서로 다른 model은 독립된 registry를 가짐을 확인(`test_different_models_have_independent_registries`).
- `reject_visible_patch_id_in_occluded_namespace(patch_id)`: visible patch_id가 `OCCLUDED_CHART_NAMESPACE_BASE` 이상이면 `ValueError`. topology/patch 생성 코드에는 연결하지 않았다(그 코드는 별도 에이전트 범위) — validator 내부에서 동일 조건을 이미 검사하며, 이 함수 자체는 diagnostics/tests용 독립 진입점으로 제공한다.
- append adapter의 `_convert()`가 이제 `project_and_register_occluded_chart_owner_id(model, ...)`를 사용하도록 변경 — 실제 append 경로가 자동으로 collision-checked됨(`test_append_adapter_uses_registered_projection`).
- Checkpoint persistence는 이번에도 범위 밖으로 유지했다(registry는 `_STATE_TENSOR_NAMES`에도 포함하지 않음 — 이유: owner_id→chart_id 바인딩은 한 번 참이면 이후 append 성공 여부와 무관하게 계속 참이므로 rollback 대상이 아니라는 점을 모듈 docstring에 명시).

### 13.5 테스트

`tests/test_surface_ownership.py`에 3개 클래스 신규 추가(총 17개 테스트):

- `PatchReassignmentTest`(5): `cluster_id`/`surface_owner_id` 동시 갱신, `VISIBLE_PATCH` 유지, mixed fixture에서 uncertain owner 불변, validator 전후 통과, 반복 호출 결정성.
- `WriteSiteRegressionTest`(5): `initialize()` fallback 일치, ADC clone/split 후 일치, prune 후 정렬·일치, occluded row는 불일치 허용 확인, snapshot/restore 후 validator 유지.
- `NamespaceCollisionTest`(7): namespace 미만/이상 허용·거부, 같은/다른 chart 정상 동작, 강제 collision 실패, model 간 registry 독립, append adapter의 실제 registered-projection 사용 확인.

### 13.6 회귀 결과

```text
python -B -m unittest -v tests.test_surface_ownership tests.test_uncertain_gaussian_append_adapter
  Ran 86 tests — OK

python -B -m unittest -v tests.test_surface_ownership tests.test_uncertain_gaussian_append_adapter \
    tests.test_training_regressions tests.test_uncertain_gaussian_proposal tests.test_occluded_chart_hardening \
    tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence tests.test_continuation_domain
  Ran 220 tests — OK

python -B -m pytest
  454 passed, 1 skipped, 1 warning, 8 subtests passed
```

(이전 §12 시점 437 passed에서 정확히 +17 = 이번에 추가한 신규 테스트 수.) warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 1건으로 무관.

### 13.7 범위 확인

O-grid, planar, annulus, cyclic topology, topology branch 관련 파일은 이번 변경분 어디에서도 import/참조하지 않는다(§11과 동일 방식으로 재확인). optimizer/trainer/renderer/checkpoint save-load 변경, occluded chart loss, dual membership, primary visible owner 승격 — 전부 수행하지 않았다.

---

Visible patch membership과 explicit ownership의 lifecycle consistency를 보완했으며, topology 관련 별도 작업과 optimizer, trainer, renderer 및 checkpoint integration은 수행하지 않았다. 사용자 Gate 검토를 요청한다.

---

## 14. Ownership Foundation Gate 최종 계약 보완 (2026-07-27 후속)

상태: **Owner registry transactional consistency 및 UNASSIGNED ownership semantics 보완 완료. 사용자 Gate 승인 완료(2026-07-27, §15 참고). topology 관련 별도 작업, optimizer/trainer/renderer/checkpoint integration 미착수.**

§1-13의 결정을 그대로 유지하면서, 사용자가 지적한 두 계약 결함만 수정했다. O-grid/planar/annulus/cyclic topology 관련 파일은 이번에도 조사·수정하지 않았다(§14.6).

### 14.1 결함 1 — Owner registry가 "pure" 단계에서 mutate되던 문제

§13.4의 `project_and_register_occluded_chart_owner_id(model, source_chart_id)`는 append adapter의 `_convert()`(transaction stage 1, "model에 아무것도 손대지 않는다"고 문서화된 단계) 안에서 호출되면서 **registry는 실제로 그 자리에서 mutate**되고 있었다 — 이후 model/sidecar/ledger commit 중 하나라도 실패하면 registry만 이미 바뀐 채로 남는, `append()`의 strong exception guarantee("append()가 예외를 반환하면 model tensors, adapter sidecar, owner registry, appended batch ledger가 모두 불변")를 위반하는 상태였다.

**수정**: `osn_gs/gaussian/torch_surface_ownership.py`에서 단일 함수를 셋으로 분리했다.

```python
def validate_occluded_owner_binding_read_only(model, source_chart_id, *, schema_version=...) -> tuple[int, bool]:
    """순수 read-only. owner_id를 projection하고 DIFFERENT chart와의 collision만 검사한다.
    Registry를 mutate하지 않는다. 반환값 (owner_id, already_registered)."""
    owner_id = project_occluded_chart_owner_id(source_chart_id, schema_version=schema_version)
    existing = model.occluded_chart_owner_registry.get(owner_id)
    if existing is not None and existing != source_chart_id:
        raise OccludedChartOwnerCollisionError(...)
    return owner_id, existing is not None

def commit_occluded_owner_binding(model, owner_id, source_chart_id) -> None:
    model.occluded_chart_owner_registry[owner_id] = source_chart_id

def rollback_occluded_owner_binding(model, owner_id, *, was_preexisting: bool) -> None:
    if not was_preexisting:
        model.occluded_chart_owner_registry.pop(owner_id, None)
```

`project_and_register_occluded_chart_owner_id`는 완전히 제거했다(더 이상 어디서도 참조하지 않음 — 이 문서를 포함해 §13.4의 코드 스니펫은 이번 라운드로 대체됨).

`osn_gs/gaussian/torch_uncertain_append_adapter.py`의 transaction을 새 단계로 재구성했다:

```text
preflight
  -> _convert(): validate_occluded_owner_binding_read_only(...)만 호출(pure, registry 미변경)
  -> receipt/sidecar entry 사전 구성(변경 없음, §Append Adapter Gate 계약 유지)
  -> model snapshot
  -> model tensor commit           (실패 시: model 복원)
  -> sidecar commit                (실패 시: model 복원)
  -> owner registry commit(NEW)    (실패 시: sidecar 롤백 + model 복원)
  -> appended batch ledger commit  (실패 시: registry 롤백[신규 생성분만] + sidecar 롤백 + model 복원)
  -> 사전 구성된 receipt 반환
```

`_ConvertedAppend`에 `owner_binding_preexisted: bool` 필드를 추가해 `validate_occluded_owner_binding_read_only`의 두 번째 반환값을 ledger-commit 실패 시점까지 실어 나른다. 어댑터에 `_commit_owner_registry`/`_rollback_owner_registry` 메서드를 `_commit_sidecar`/`_commit_ledger`와 동일한 패턴(monkeypatch로 실패를 주입할 수 있는 얇은 래퍼)으로 추가했다.

**"이전부터 존재하던 binding은 제거하지 않는다" 규칙**: ledger commit이 실패하면 `_rollback_owner_registry(model, owner_id, was_preexisting=converted.owner_binding_preexisted)`를 호출한다 — 이번 transaction이 새로 만든 binding(`was_preexisting=False`)만 지우고, 이전에 이미 성공적으로 커밋된 transaction이 만든 binding(`was_preexisting=True`, 예: 같은 occluded chart에서 나온 두 번째 batch)은 그대로 둔다.

### 14.2 결함 2 — UNASSIGNED ownership semantics 확정

**감사 결과**: `torch_pipeline.py._initialize_stage1()`은 비활성/skip된 voxel leaf의 Gaussian에 `cluster_id = -1`을 남기며, 이는 그 함수 자신의 docstring/주석("Gaussians in inactive/skipped leaves stay unassigned (cluster_id -1)...")에 의해 **canonical한 상태이지 transient 버그가 아니다**. 반면 `TorchGaussianModel.initialize()`/`replace_tensors()`의 fallback은 `surface_owner_kind`가 `None`이면 **무조건** `SURFACE_OWNER_VISIBLE_PATCH`를 대입하고 있었다 — `cluster_id`가 음수인 행까지 VISIBLE_PATCH로 잘못 낙인찍는 실제 결함이었다. `SURFACE_OWNER_UNASSIGNED` enum 값 자체는 §3에서 이미 정의돼 있었지만, 어떤 코드 경로도 실제로 이 값을 생성하지 않고 있었다.

`_assign_uv_support_masks`의 `patch_id==0` catch-all(`cluster_ids < 0 | cluster_ids >= n_patches`)은 의도적으로 음수 `cluster_ids` 행을 patch 0에 흡수한다 — 그러나 이 흡수는 `is_visible_patch_owned(model.surface_owner_kind)`와 AND 조건이므로(§5), 결함 수정 후에는 음수 `cluster_ids` 행이 더 이상 `VISIBLE_PATCH`가 아니라 `UNASSIGNED`가 되어 이 catch-all에서도 구조적으로 제외된다 — **catch-all 코드 자체는 건드리지 않았다**(음수 분기가 실질적으로 도달 불가능해질 뿐이며, 이는 "음수 cluster_ids 행은 절대 VISIBLE_PATCH가 될 수 없다"는 invariant가 유지되는 한 안전한 죽은 코드이지 버그가 아니다 — 별도 리팩터링 대상으로 남긴다).

**확정한 canonical 계약** (`derive_default_ownership`, `osn_gs/gaussian/torch_surface_ownership.py` 신규 함수):

```python
cluster_id >= 0  ->  surface_owner_kind = VISIBLE_PATCH,  surface_owner_id = cluster_id
cluster_id <  0  ->  surface_owner_kind = UNASSIGNED,      surface_owner_id = UNASSIGNED_OWNER_ID  # -1
```

`UNASSIGNED_OWNER_ID = -1`은 `cluster_ids`의 기존 "no patch" sentinel과 동일한 값으로 의도적으로 맞췄다(마이그레이션 규칙이 놀랍지 않도록).

`TorchGaussianModel.initialize()`/`replace_tensors()`: `surface_owner_kind`/`surface_owner_id` 중 하나라도 `None`이면 `derive_default_ownership(cluster_ids)`로 기본값 쌍을 계산하고, 명시적으로 전달된 쪽만 그 값으로 덮어쓴다(양쪽 다 전달되면 기존처럼 그대로 사용). Occluded chart ownership은 이 경로에서 절대 파생되지 않는다 — 여전히 append adapter만 명시적으로 부여한다. `replace_tensors()`는 old checkpoint 호환 fallback이라는 기존 문서화도 유지하되, "VISIBLE_PATCH로 일괄 마이그레이션"이 아니라 "cluster_id 부호에 따라 VISIBLE_PATCH 또는 UNASSIGNED로 마이그레이션"으로 docstring을 정정했다.

**Validator 강화** (`validate_surface_ownership_consistency`):
- VISIBLE_PATCH 행의 `surface_owner_id`가 음수이면(예: 실수로 UNASSIGNED sentinel이 들어간 경우) 위반으로 신고 — 기존에는 상한(`< OCCLUDED_CHART_NAMESPACE_BASE`)만 검사하고 하한을 검사하지 않아 `VISIBLE_PATCH + owner_id=-1`이 조용히 통과했다.
- UNASSIGNED 행의 `surface_owner_id`가 정확히 `UNASSIGNED_OWNER_ID`(-1)가 아니면 위반으로 신고.

### 14.3 UNASSIGNED 행의 behavioral 제외 확인

| 사이트 | 제외 메커니즘 | 코드 변경 |
|---|---|---|
| `_assign_uv_support_masks`(support/trim mask) | `is_visible_patch_owned(kind)` 게이트 — UNASSIGNED는 kind가 다르므로 자동 제외 | 없음(§5에서 이미 추가된 게이트가 그대로 커버) |
| `maintain_surface_from_certain`(visible patch maintenance) | `cluster_ids == patch_id` 매칭 — `-1`은 어떤 실제 patch_id와도 같을 수 없음 | 없음 |
| `_split_failed_patch`(failed-patch split/reassignment) | 위와 동일(`certain & (cluster_ids == patch_id)`) | 없음 |
| `nurbs_surface_loss`(patch loss membership) | `certain = ~is_uncertain` 게이트 + `patch_ids >= 0` 명시적 체크(`torch_losses.py`) | 없음 |

네 사이트 모두 **이번 라운드에서 실제로 코드를 바꾼 곳은 없다** — `initialize()`/`replace_tensors()`의 기본값 수정만으로 네 곳 전부가 올바르게 UNASSIGNED를 제외하게 된다(세 곳은 애초에 `cluster_ids`/`is_uncertain` 값 자체로 이미 배제되고 있었고, support mask 한 곳만 kind 게이트에 의존한다). `test_negative_cluster_row_excluded_from_patch_zero_support_mask_catch_all`, `test_negative_cluster_row_never_matches_any_real_patch_id`로 회귀-확인했다.

### 14.4 테스트

**`tests/test_surface_ownership.py`**:
- `UnassignedOwnershipTest`(신규 클래스, 9개): `cluster_id` 음수 → UNASSIGNED 초기화, 명시적 patch-0 할당은 VISIBLE_PATCH owner 0 유지, `replace_tensors` fallback이 `initialize()`와 동일한 기본값 산출, `derive_default_ownership` 자체 검증, validator가 `VISIBLE_PATCH+owner_id<0`/`UNASSIGNED+owner_id≠sentinel` 각각 위반으로 신고, patch-0 catch-all에서 UNASSIGNED 행 제외, 세 behavioral 사이트에서 음수 `cluster_ids` 행이 어떤 patch_id와도 매칭되지 않음.
- `NamespaceCollisionTest`: 기존 5개 테스트를 새 3-함수 API(`validate_occluded_owner_binding_read_only`/`commit_occluded_owner_binding`)로 재작성(동작 자체는 동일하게 검증). `rollback_occluded_owner_binding`의 신규-생성분만 제거/기존분 보존 분기를 검증하는 테스트 2개 추가.

**`tests/test_uncertain_gaussian_append_adapter.py`**: `AppendAdapterOwnerRegistryTransactionTest`(신규 클래스, 6개):
- `_convert()`가 registry를 mutate하지 않음(pure 검증).
- 성공적인 append가 registry에 정확히 하나의 binding을 커밋함.
- owner-registry-commit 실패 시 model/sidecar/registry/ledger 전부 원상복구(무엇도 새로 남지 않음).
- ledger-commit 실패 시 **이번 transaction이 새로 만든** registry entry는 롤백됨.
- ledger-commit 실패 시 **이전 transaction이 이미 커밋한**(같은 chart, 다른 batch) registry entry는 보존됨 — 이번 라운드의 핵심 계약.
- 실패한 transaction을 다른 adapter 인스턴스로 재시도하면 성공하고, registry가 최종적으로 일관된 상태로 남음.

### 14.5 회귀 결과

```text
python -B -m unittest -v tests.test_surface_ownership tests.test_uncertain_gaussian_append_adapter
  Ran 102 tests — OK

python -B -m unittest -v tests.test_surface_ownership tests.test_uncertain_gaussian_append_adapter \
    tests.test_training_regressions tests.test_uncertain_gaussian_proposal tests.test_occluded_chart_hardening \
    tests.test_occluded_chart tests.test_occluded_region_candidate tests.test_candidate_evidence \
    tests.test_continuation_domain tests.test_stage1_pipeline tests.test_torch_pipeline_smoke
  Ran 247 tests — OK

python -B -m pytest
  471 passed, 1 skipped, 2 failed, 1 warning, 8 subtests passed
```

`pytest` 전체 실행에서 실패한 2건은 이번 변경과 무관하다: `tests/test_boundary_first_visible_builder.py::test_unpaired_topology_is_explicitly_unsupported_not_rectangle`와 `tests/test_boundary_surface_quality.py::test_curved_annulus_boundary_samples_and_cyclic_seams_are_exact` — 둘 다 `git status`상 미추적(`??`) 상태인 Codex의 진행 중인 `torch_boundary_*`/annulus/cyclic topology 작업(worklog 89-98) 소관이며, 이번 라운드가 손댄 파일(`torch_surface_ownership.py`, `torch_model.py`, `torch_uncertain_append_adapter.py`, 테스트 2개)과 import/참조 관계가 전혀 없다. 이전 §12/§13의 437/454 passed 기준선과 비교해도 이번에 신규로 깨진 테스트는 없다(102개 타겟 테스트 전부 통과, 광의 회귀 247개 전부 통과).

warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 1건으로 이번 변경과 무관.

### 14.6 범위 확인

O-grid, planar, annulus, cyclic topology, topology branch 관련 파일은 이번 변경분(§14.1-14.4의 production/test 파일) 어디에서도 import/참조하지 않는다. `git status --short`로 재확인한 결과 Codex가 여전히 `osn_gs/surface/torch_boundary_*.py`/`nurbs_constructor_benchmark/boundary_first_support*.py`/`docs/worklogs/89-98` 범위를 병행 작업 중이며, 이번 라운드는 그 파일들을 조사·수정하지 않았다. optimizer/trainer/renderer 통합, checkpoint schema 변경, occluded chart 전용 loss, dual membership, primary visible owner 승격 — 전부 수행하지 않았다.

---

Owner registry의 transactional consistency와 UNASSIGNED ownership semantics를 보완했으며, topology 관련 별도 작업과 optimizer, trainer, renderer 및 checkpoint integration은 수행하지 않았다. 사용자 Gate 검토를 요청한다.

---

## 15. Gate 승인 반영 (2026-07-27, 상태 갱신만)

**Occluded Chart Ownership Foundation Gate: APPROVED.** 사용자가 §1-14 전체 구현을 검토 후 승인했다. 이번 절은 상태 반영만 수행하며, ownership production code/테스트는 추가로 변경하지 않았다.

```text
Append Adapter Gate: APPROVED
Occluded Chart Ownership Foundation Gate: APPROVED

Optimizer integration: NOT STARTED
Trainer integration: NOT STARTED
Renderer integration: NOT STARTED
Checkpoint integration: NOT STARTED
Production integration: NOT STARTED
```

**승인 근거(§1-14에서 이미 확립, 이번에 재변경 없음)**:
- owner registry transaction consistency 완료(§14.1: `validate_occluded_owner_binding_read_only`/`commit_occluded_owner_binding`/`rollback_occluded_owner_binding` 분리, registry commit이 sidecar-ledger 사이 독립 transaction 단계).
- append strong exception guarantee 유지(receipt는 커밋 이전에 완성, 실패 시 model/sidecar/registry/ledger 전부 원상복구).
- `cluster_id >= 0 → VISIBLE_PATCH`, `cluster_id < 0 → UNASSIGNED`(§14.2 `derive_default_ownership`).
- visible patch reassignment 시 `cluster_ids`와 `surface_owner_id` 동기화(§13.2, `_split_failed_patch`).
- uncertain Gaussian의 owner는 occluded chart(`SURFACE_OWNER_OCCLUDED_CHART`), never 하나의 visible patch로 승격되지 않음(§1, §6).
- source patch IDs는 provenance이며 behavioral owner가 아님(§1, §4).

**Repository-wide integration state: NOT GREEN.** 저장소 전체 `pytest` 결과는 `471 passed, 1 skipped, 2 failed`이며, 실패 2건(`test_boundary_first_visible_builder.py`, `test_boundary_surface_quality.py`)은 진행 중인 boundary-first/annulus/cyclic-topology 병렬 작업(worklog 89-98, 다른 에이전트 담당) 소관이다 — Ownership Foundation Gate 승인은 이 실패들과 무관하게 유효하며, 그 실패들이 해결되고 전체 pytest가 green이 된 뒤 사용자의 별도 승인에 따라 다음 production integration 단계를 결정한다. 이번 절에서 O-grid/planar/annulus/cyclic-topology 파일, 다른 에이전트의 실패 테스트, ownership production code/테스트는 일체 수정하지 않았다.
