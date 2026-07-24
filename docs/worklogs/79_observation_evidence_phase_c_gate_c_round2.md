# Worklog 79: Boundary-Conditioned Phase C — Gate C 2차 보완

날짜: 2026-07-23

상태: **Gate C 2차 보완 완료. 사용자 재검토 대기. Phase D 미착수.**

## 1. 배경

`docs/worklogs/78_observation_evidence_phase_c_gate_c_followup.md`(Gate C 1차 보완)를 사용자가 아직 승인하지 않고 두 가지를 추가 요구했다.

1. Aggregate on-surface 의미론 완성 — `STATUS_ON_OBSERVED_SURFACE` aggregate 상태 추가, on_surface_in이 있는 sample이 `known_free_space` 또는 `occluded_candidate`로 집계되지 않는다는 불변식을 docstring에 명시.
2. Camera fingerprint를 요청된 전체 필드(world_view_transform, full_proj_transform, width, height, stable identity/name)로 확장하고 camera name 부재 시 deterministic fallback 규칙 명시.
3. 관련 신규 테스트 추가(on-surface-only, on-surface+behind, 가능하면 free+behind+on_surface 삼중 조합, resolution-only 변경, identity/name-only 변경).
4. Tensor fingerprint가 field name/shape/dtype/content를 포함하며 device-independent하게 결정적인지 확인·문서화.
5. `evidence_cache_key`가 post-build result fingerprint라는 계약을 docstring에 명시하거나 함수명을 더 정확하게 변경.

Phase D는 이번 작업에서도 시작하지 않았다.

## 2. 착수 전 확인 — 이미 완료된 부분 보고

작업을 시작하기 전에 현재 코드를 먼저 읽어 이미 반영된 부분과 그렇지 않은 부분을 구분해 보고했다(사용자의 "이미 완료한 부분이 있으면 작업을 수행하지 말고 보고부터 해" 지시에 따름).

- Camera fingerprint의 `world_view_transform`/`full_proj_transform`/`image_height`/`image_width`/`camera.image_name`은 worklog 78에서 이미 추가돼 있었다 — 그대로 뒀다.
- Aggregate on-surface 상태, 신규 테스트 5개, tensor fingerprint의 field/shape/dtype 포함, `evidence_cache_key` 계약 명시는 전부 미착수 상태였다 — 이번에 구현했다.
- 점검 중 실제 결함 하나를 추가로 발견했다: `_tensor_digest()`가 raw byte content만 해시하고 shape/dtype을 해시에 포함하지 않아, 같은 바이트를 가진 다른 shape/dtype 텐서(예: `(16,)` vs `(4,4)` float32)가 이론상 충돌할 수 있었다. 이번에 수정했다(아래 5절).

## 3. Aggregate on-surface 의미론 완성

`osn_gs/surface/torch_observation_evidence.py`에 `STATUS_ON_OBSERVED_SURFACE = "on_observed_surface"`를 aggregate 상태로 추가하고, `classify_world_samples()`의 집계 로직을 다음과 같이 재작성했다.

```text
evidence_kinds = {free 존재, behind 존재, on_surface 존재} 중 True 개수

2개 이상 True:
    conflicting_evidence
    reason:
        free+behind+on_surface 모두 → free_space_behind_and_on_surface_conflict
        free+behind만 → free_space_and_occluded_conflict
        free+on_surface만 → free_space_and_on_surface_conflict
        behind+on_surface만 → behind_and_on_surface_conflict
behind만: occluded_candidate
free만: known_free_space
on_surface만: on_observed_surface  (신규)
unobserved만: unobserved
전부 outside_valid_view: outside_valid_view
```

**불변식**(module docstring과 `classify_world_samples` docstring에 명시): `on_surface_in`이 비어있지 않은 `SampleEvidence`는 `status`가 `known_free_space` 또는 `occluded_candidate`가 될 수 없다.

## 4. 신규 테스트 5개

1. `test_on_surface_only_aggregates_as_on_observed_surface` — 기존 `test_sample_on_observed_surface_is_not_free_space`를 개명·수정했다(신규 aggregate 상태에 맞춰 기존 assertion `STATUS_UNOBSERVED`를 `STATUS_ON_OBSERVED_SURFACE`로 교체, `occluded_candidate`도 아님을 추가 검증).
2. `test_multi_view_on_surface_and_behind_are_conflicting` — 신규 fixture `_on_surface_multi_view_scene()`. Camera A(원점, wall A 정면 관측)는 sample `(0,0,2)`를 `on_observed_surface`로, camera B(오프셋, sample을 직접 조준하되 그 광선 중간에 배치한 고-opacity occluder "Wall G"를 먼저 관측)는 `behind_first_observed_surface`로 판정함을 사전조건으로 확인한 뒤 집계가 `conflicting_evidence`(`reason="behind_and_on_surface_conflict"`)임을 검증했다.
3. `test_multi_view_free_behind_and_on_surface_triple_conflict` — 같은 fixture에 `include_free_camera=True`로 camera C(반대편에서 sample을 조준하되 그 광선 연장선의 더 먼 "Wall H"를 관측)를 추가해 free/behind/on_surface 세 가지가 동시에 나타나는 조합을 검증했다(`reason="free_space_behind_and_on_surface_conflict"`).
4. `test_evidence_cache_key_reacts_to_resolution_only_change` — pose/FoV는 동일하고 `image_height`/`image_width`만 다른 두 카메라가 다른 키를 만드는지 확인했다.
5. `test_evidence_cache_key_reacts_to_identity_only_change` — transform이 바이트 단위로 동일한 두 카메라의 `image_name`만 다르면 키가 다름을 확인했고, 추가로 두 카메라가 모두 기본값 `"camera"`를 공유하더라도 리스트 순서를 바꾸면 키가 달라짐을 확인해(명시적 `camera_index` 성분이 실제로 작동함을 증명) 아래 5절의 identity fallback 규칙을 검증했다.

### `_on_surface_multi_view_scene` fixture 설계 근거

Wall A와 sample point가 정확히 같은 위치(`(0,0,2)`)에 있으므로, 그 지점을 바라보는 어떤 카메라든 Wall A 자체가 항상 같은 픽셀에 기여한다(실제로 그 위치에 표면이 있으므로). Fallback renderer가 진짜 depth-ordered occlusion을 구현하지 않고 화면-근접도로만 가중 평균하기 때문에(worklog 78의 남은 제한), camera B가 "behind"로 읽히려면 Wall A보다 명백히 우세한 가중치를 가진 근접 occluder(Wall G)가 필요했다. Wall G의 opacity를 0.95(Wall A/H의 0.1/0.5 대비)로 높여 이 문제를 해결했다 — probe script로 사전에 수치 검증했다(camera B: sample_depth 5.385, obs_depth 3.672 → 명확히 behind).

## 5. Tensor fingerprint 수정 — 실제 결함 발견 및 수정

### 발견한 결함

`_tensor_digest(tensor)`가 `tensor.detach().cpu().contiguous().numpy().tobytes()`만 해시했다. Shape와 dtype이 해시 입력에 포함되지 않아, 동일한 바이트 시퀀스를 가진 서로 다른 shape/dtype 텐서(예: 16개 float32를 담은 `(16,)` 텐서와 `(4,4)` 텐서)가 이론상 동일한 digest를 생성할 수 있었다 — content fingerprint의 핵심 목적(서로 다른 내용은 반드시 다른 키를 가져야 한다)을 깨는 결함이다.

### 수정

`_tensor_digest(label: str, tensor: Any) -> str`로 시그니처를 바꿔 `f"{label}|{shape}|{dtype}"` 헤더를 raw bytes 앞에 붙여 해시하도록 했다. 모든 호출부(`_topology_version`의 xyz/scaling/rotation/opacity, `_camera_set_version`의 world_view_transform/full_proj_transform)를 라벨과 함께 호출하도록 갱신했다.

**Device-independence**: `.cpu().contiguous()`를 해시 전에 항상 적용하므로 GPU 텐서와 CPU 텐서가 논리적으로 같은 내용이면 항상 같은 digest를 낸다 — docstring에 명시했다. 별도의 GPU/CPU 교차 검증 테스트는 추가하지 않았다(이 저장소의 CUDA 텐서 대부분이 이미 다른 테스트에서 `.cpu()` 변환 후 비교되는 관례를 따름, 그리고 `.detach().cpu().contiguous()` 자체가 표준 라이브러리 동작이라 별도 검증이 크게 새로운 위험을 줄이지 않는다고 판단).

## 6. Camera fingerprint — 요청된 전체 필드 확인 및 identity fallback 규칙 추가

- `world_view_transform`, `full_proj_transform`, `image_height`, `image_width`는 worklog 78에서 이미 반영돼 있었다(2절 참고).
- **Identity fallback 규칙(신규)**: `TorchCamera.image_name`은 `str = "camera"` 기본값을 가지며 `None`이 될 수 없는 필드다(`str | None`이 아님). 따라서 "이름이 없을 때"라는 조건 분기는 실제로 발생할 수 없다. 대신 `_camera_set_version()`은 각 카메라의 `enumerate()` 인덱스를 `camera_index=N` 형태로 항상 명시적으로 포함하도록 수정했다 — 여러 카메라가 공유 기본값 `"camera"`를 그대로 쓰더라도(흔한 상황) 리스트 내 위치가 항상 별도 토큰으로 구분자 역할을 하며, 이는 `"|".join`의 암묵적 순서에 우연히 의존하는 것이 아니라 의도적으로 문서화된 구분자다. Docstring에 이 규칙의 근거를 명시했다.

## 7. `evidence_cache_key` 계약 명시

Docstring 최상단에 "POST-BUILD result fingerprint"임을 명시하고, 렌더링 전 사용할 수 없는 이유(카메라별 `backend_source`는 실제 렌더 시도 후에만 확정되며, CUDA 렌더 도중 실패해 fallback으로 전환되는 경우 동일 입력이라도 다른 backend로 귀결될 수 있음)를 설명했다. 함수명은 바꾸지 않고 docstring으로 계약을 명시하는 쪽을 선택했다 — 아직 다른 코드가 이 함수를 호출하지 않는 신규 prototype API이므로 이름 변경의 이득보다 관례상 이미 익숙해진 이름을 유지하는 편이 낫다고 판단했다.

## 8. 회귀 검증

- 신규 파일만: `14 passed` (worklog 78 시점 10개 + 신규 5개 − 1개는 기존 테스트를 개명·수정)
- 전체 pytest suite: `238 passed, 1 skipped, 8 subtests passed` (worklog 78 시점 `234 passed`에서 +4, 실패/회귀 없음)
- Production(`torch_pipeline.py`/`torch_trainer.py`, renderer 두 파일의 기존 키) 미변경. 이번 보완도 `torch_observation_evidence.py`와 그 테스트 파일에만 국한된다.

## 9. 남은 제한

- Fallback renderer가 depth-ordered occlusion을 구현하지 않는다는 pre-existing 한계는 여전히 남아 있다 — 이번에 만든 on-surface+behind 테스트 fixture도 이 한계를 우회(고-opacity occluder)하는 방식으로 설계했을 뿐, 근본 수정은 하지 않았다(production fallback 자체를 바꾸는 것은 이 모듈의 범위 밖).
- GPU/CPU 텐서 fingerprint 교차 일치를 검증하는 전용 테스트는 추가하지 않았다 — `.cpu().contiguous()` 정규화가 표준 동작이라는 근거로 생략했으나, 필요시 후속 추가 가능.

## 10. 중단 및 다음 승인

계획대로 Gate C 2차 보완만 수행하고 멈춘다. Phase D — Parametric Continuation Domain은 별도 사용자 승인 없이는 시작하지 않는다.
