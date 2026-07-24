# Worklog 78: Boundary-Conditioned Phase C — Gate C 보완

날짜: 2026-07-23

상태: **Gate C 보완 완료. 사용자 재검토 대기. Phase D 미착수.**

## 1. 배경

`docs/worklogs/77a_observation_evidence_phase_c.md`(Phase C 1차 구현)를 사용자가 대체로 승인했지만, Gate C를 닫기 전에 두 가지 보완을 요구했다.

1. multi-view aggregate에서 `on_surface_in`이 있는 sample이 `known_free_space`가 되지 않도록 수정.
2. `ObservationEvidence`의 content fingerprint를 실제 evidence를 바꾸는 모든 필드로 확장.

Phase D는 이번 작업에서 시작하지 않았다.

## 2. 보완 1 — on-surface veto

### 문제

기존 집계 규칙은 `free_space_confirmed_by`와 `behind_surface_in`의 공존만 `conflicting_evidence`로 처리했다. 한 카메라가 `on_observed_surface`(표면 tie band)를 보고하고 다른 카메라가 `known_free_space`를 보고하는 경우는 별도로 걸러지지 않아 `known_free_space`로 집계됐다 — worklog 77-A §4의 surface-band 테스트 취지(표면 위 sample은 free space로 승격하지 않는다)가 multi-view 집계 단계에서는 지켜지지 않는 구멍이었다.

### 선택한 의미론

사용자가 제시한 두 옵션(신규 aggregate 상태 추가 vs `conflicting_evidence`로 처리) 중 후자를 선택했다. 이유: 이미 `free_space_confirmed_by`와 `behind_surface_in`의 공존을 `conflicting_evidence`로 처리하는 선례가 있고, on-surface 증거도 "다른 카메라의 free-space 주장에 반하는 증거"라는 점에서 성격이 같다. 새 aggregate 상태를 추가하면 `SAMPLE_STATUSES` 집합이 커지고 Phase D/E의 소비자 코드가 한 가지 상태를 더 처리해야 하므로, 기존 상태를 재사용하는 편이 더 단순하고 "no scene-specific tuning"/"hard gate 최소화" 기존 관례와 일치한다.

`osn_gs/surface/torch_observation_evidence.py`의 aggregation 우선순위를 다음과 같이 수정했다.

1. `free_space_confirmed_by`와 `behind_surface_in`이 둘 다 있으면 → `conflicting_evidence` (`reason="free_space_and_occluded_conflict"`, 기존과 동일)
2. **(신규)** `free_space_confirmed_by`와 `on_surface_in`이 둘 다 있으면(1이 아닌 경우) → `conflicting_evidence` (`reason="free_space_and_on_surface_conflict"`)
3. `behind_surface_in`만 있으면 → `occluded_candidate` (기존과 동일)
4. `free_space_confirmed_by`만 있으면 → `known_free_space` (2에서 이미 on-surface와의 공존 여부를 걸렀으므로 이 분기는 on_surface_in이 비어있을 때만 도달)
5. `on_surface_in` 또는 `unobserved_in`만 있으면 → `unobserved` (기존과 동일)
6. 전부 `outside_valid_view`면 → `outside_valid_view` (기존과 동일)

선택한 의미론은 module docstring(파일 상단, "On-surface veto (Gate C follow-up, docs/worklogs/78)" 절)과 `classify_world_samples()`의 docstring에 모두 명시했다.

### 검증

신규 `test_multi_view_on_surface_vetoes_known_free_space`를 추가했다. 기존 multi-view-conflict fixture(camera 1이 wall A를 정면으로 관측, camera 3이 wall A 모서리를 비껴 wall B까지 관측)를 재사용해 sample을 `(0,0,3)` 대신 `(0,0,2)`(wall A 표면 정확히 위)로 바꿨다. 사전조건으로 camera 1의 per-view 상태가 `on_observed_surface`, camera 3의 per-view 상태가 `known_free_space`임을 먼저 확인한 뒤, 집계 결과가 `known_free_space`가 아니라 `conflicting_evidence`임을 확인했다.

## 3. 보완 2 — content fingerprint 확장

### 문제

`_topology_version()`은 `xyz`만 해시했고, `_camera_set_version()`은 `world_view_transform`만 해시하고 `full_proj_transform`(FoV·projection에 의해 결정되며 화면 위치 계산에 직접 쓰인다)을 빠뜨렸다. `evidence_cache_key()`는 두 문자열만 결합했을 뿐 `near`/`far`/`depth_epsilon`이나 backend/depth convention(같은 model·camera라도 CUDA/fallback 중 무엇으로 렌더했는지, 혹은 렌더 도중 CUDA가 실패해 fallback으로 전환됐는지)을 전혀 반영하지 않았다 — 이 값들이 달라지면 evidence 내용 자체가 달라지는데도 동일한 키가 나올 수 있었다.

### 수정

- `_topology_version(model, hierarchy=None)`: `model.get_xyz`뿐 아니라 `get_scaling`, `get_rotation`, `get_opacity`(rasterizer가 실제로 소비하는 effective 값)를 모두 해시하도록 확장했다. Color/SH는 depth/coverage evidence에 영향을 주지 않으므로 의도적으로 제외했다.
- `_camera_set_version(cameras)`: `world_view_transform`에 더해 `full_proj_transform`도 해시하도록 확장했다.
- `evidence_cache_key()`의 signature를 `(topology_version: str, camera_set_version: str)`에서 `(evidence: ObservationEvidence)`로 바꿨다. `evidence.near`/`far`/`depth_epsilon`과 각 view의 `backend_source`/`depth_kind`/`coverage_kind`를 키에 포함한다 — 같은 model·camera라도 backend가 다르거나 렌더 도중 CUDA에서 fallback으로 전환되면 다른 키가 나온다. 이 함수는 여전히 순수 함수이며 global cache dict/singleton은 추가하지 않았다(caller가 자체 캐싱을 원하면 이 키를 사용하되, 캐싱 자체는 caller 책임).

이 함수는 아직 다른 코드에서 호출되지 않는 신규 prototype API이므로 signature 변경에 따른 하위 호환 문제는 없다(grep으로 확인).

### 검증

신규 `test_evidence_cache_key_reacts_to_appearance_camera_and_config_changes`를 추가했다. 동일 model/camera/config로 두 번 빌드하면 키가 같음(결정성)을 확인한 뒤, scale만 변경/rotation만 변경/opacity만 변경/camera FoV(=`full_proj_transform`)만 변경/`depth_epsilon`만 변경한 각 경우에 키가 달라짐을 확인했다.

## 4. 회귀 검증

- 신규 파일만: `10 passed` (기존 8개 + 신규 2개)
- 전체 pytest suite: `234 passed, 1 skipped, 8 subtests passed` (worklog 77-A 시점 `230 passed`에서 정확히 +4 — 신규 2개 테스트 + 동시 세션 변경 포함으로 추정, 실패/회귀 없음)
- Production(`torch_pipeline.py`/`torch_trainer.py`, renderer 두 파일의 기존 키) 미변경. 이번 보완은 신규 `torch_observation_evidence.py`와 그 테스트 파일에만 국한된다.

## 5. 남은 제한

- `evidence_cache_key()`는 여전히 완성된 `ObservationEvidence`를 만든 뒤에만 계산할 수 있다 — 렌더링 전에 "다시 렌더링할 필요가 있는가"를 판정하는 pre-render 캐시 키는 아니며, worklog 77-A와 마찬가지로 이 모듈은 실제 캐시를 구현하지 않는다.
- Color/SH는 fingerprint에서 의도적으로 제외했다 — depth/coverage evidence에 영향을 주지 않기 때문이나, 향후 evidence가 색상 기반 신호를 포함하게 되면 재검토가 필요하다.

## 6. 중단 및 다음 승인

계획대로 Gate C 보완만 수행하고 멈춘다. Phase D — Parametric Continuation Domain은 별도 사용자 승인 없이는 시작하지 않는다.
