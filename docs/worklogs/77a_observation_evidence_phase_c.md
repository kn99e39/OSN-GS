# Worklog 77-A: Boundary-Conditioned Phase C — Observation Evidence and Free-Space Query

날짜: 2026-07-23

상태: **Phase C 구현·검증 완료. Gate C 사용자 검토 대기. Phase D 이후 미착수.**

## 1. 승인 범위

사용자가 Gate B(Phase A–B) 결과를 검토한 뒤 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`의 **Phase C — Observation Evidence와 Free-Space Query**만 승인했다. Phase D(continuation domain) 이후는 이번 작업에서 수행하지 않았다.

계획 수립 과정에서 사용자가 초안(behind_observed_surface 단일 상태, tie를 behind로 처리, "한 카메라라도 behind면 behind" 집계 규칙, CUDA invalid depth를 clamp만으로 처리, coverage_threshold를 backend 공통 설정으로 둔 점)을 5가지 근거로 반려했다. 최종 구현은 이 5가지 교정을 모두 반영했다 — 자세한 내용은 아래 3절.

## 2. 구현

### `osn_gs/render/torch_fallback.py`, `osn_gs/render/gaussian_rasterizer.py`

기존 render 반환 dict에 키를 추가만 했다(기존 키의 값·의미는 변경 없음).

- fallback: `"alpha"`, `"valid_depth_mask"` 추가.
- CUDA: `"valid_depth_mask"` 추가(vendor 코드 미변경, `depth_image.abs() > eps`만 사용).

편입 전 caller-safety grep 결과 — `torch_pipeline.py`/`torch_trainer.py`/`tests/`의 `.render(` 호출부(`osn_gs/eval/held_out_metrics.py:64`, `osn_gs/core/torch_trainer.py:216,917`)는 전부 `render_pkg["render"]`/`render_pkg.get("visibility_filter"/"viewspace_points"/"radii")` 형태의 dict 키 접근이며, positional unpack이나 key-set 동등 비교는 없다. 새 키 추가는 기존 호출부에 안전하다.

### 신규 모듈 `osn_gs/surface/torch_observation_evidence.py`

Phase A/B와 동일한 스타일(비-frozen dataclass, `payload()`, 문자열 상태를 `set`으로 검증, entry point 함수 하나가 결과 dataclass 하나를 반환)로 작성했다.

**Per-view 상태(5개)**: `known_free_space`, `on_observed_surface`, `behind_first_observed_surface`, `unobserved`, `outside_valid_view`. `behind_first_observed_surface`는 "이 카메라의 첫 관측 표면보다 멀다"는 ray-기하학적 사실일 뿐, "실제로 occluded geometry 내부"라는 주장이 아니다 — 벽 뒤의 빈 공간도 동일하게 관측된다는 점을 이름에 명시했다.

**Aggregate 상태(5개, per-view와 다른 집합)**: `known_free_space`, `occluded_candidate`, `unobserved`, `outside_valid_view`, `conflicting_evidence`. 집계 우선순위:

1. `free_space_confirmed_by`와 `behind_surface_in`이 둘 다 비어있지 않으면 → `conflicting_evidence`. (한 카메라가 자유공간이라고 보고하면 그 자체가 occlusion에 대한 강한 반증이므로, "한 카메라라도 behind면 이긴다"는 초안 규칙을 버리고 raw per-camera 리스트를 항상 payload에 남겼다.)
2. 그 외 `behind_surface_in`만 있으면 → `occluded_candidate`.
3. 그 외 `free_space_confirmed_by`만 있으면 → `known_free_space`.
4. `on_surface_in` 또는 `unobserved_in`만 있으면 → `unobserved` (tie band는 free space로 승격하지 않는다).
5. 전부 `outside_valid_view`면 → `outside_valid_view`.

**CUDA depth 복원**: `1/depth_image`를 무조건 계산하지 않고, `valid_depth_mask`로 먼저 걸러낸 뒤에만 역수를 취한다(`view_depth = full_like(..., inf); view_depth[valid] = 1/depth[valid]`). invalid pixel이 큰 유한값으로 둔갑해 downstream에서 far-plane처럼 취급되는 것을 막는다. `depth_is_approximate=True`로 `1/E[1/z] != E[z]` 근사를 명시했다.

**Fallback depth**: 기존 `fallback_render`는 카메라 pose를 전혀 사용하지 않는다(world-frame `xyz[:,2]`를 그대로 depth로 반환하는 pre-existing 결함, 이번 변경과 무관). 이 결함이 있는 production 경로를 그대로 재사용하지 않고, evidence 전용 `_fallback_view_depth()`를 새로 만들어 camera-space로 변환한 xy/z에 대해 `fallback_render`와 동일한 accumulation 공식(`_auto_chunk_size` 포함)을 다시 실행한다. Production `fallback_render`의 동작은 변경하지 않았다.

**Backend capability 메타데이터**: `CameraViewEvidence`에 `coverage_kind`(`alpha_fraction` | `binary_contribution_mask`), `depth_kind`(`direct_linear` | `inverted_expected_reciprocal`), `depth_is_approximate`를 추가해 두 backend를 동일 신뢰도로 취급하지 않도록 명시했다. CUDA에는 실제 coverage 비율이 없으므로 `coverage_alpha=None`.

**Empty-voxel query**: `query_empty_voxel_support()`는 `STATE_EMPTY` leaf와의 AABB overlap만 계산하고, 결과 `support` 필드는 구조적으로 `"no_observed_support"` 외의 값을 가질 수 없다(분기 자체가 없음). `classify_world_samples`/`SampleEvidence`와 연결하는 타입이 없어 occlusion candidate 신호로 흘러갈 경로가 없다.

**Cache invalidation**: `TorchGaussianModel`/`TorchVoxelGaussianHierarchy`에 기존 버전 카운터가 없어(확인됨), production state를 건드리지 않는 content fingerprint(`sha256(xyz bytes)` + 개수, 카메라 이름/크기/`world_view_transform` 해시)로 구현했다.

## 3. 사용자 교정 5건 반영 확인

| 교정 사항 | 반영 |
|---|---|
| `behind_observed_surface` 명칭이 과도하게 강함 | `behind_first_observed_surface`로 개명, docstring에 "occluded 확정 아님" 명시 |
| depth-epsilon tie를 behind로 처리 | `on_observed_surface` 신규 상태 추가, 집계 시 free space로 승격하지 않음 |
| "한 카메라라도 behind" aggregation이 과도함 | per-camera 5개 리스트를 항상 payload에 보존, `conflicting_evidence` 신규 aggregate 상태 추가 |
| CUDA invalid depth가 큰 유한값으로 둔갑 | mask-먼저-then-invert로 수정 (`torch.full_like(..., inf)` 후 valid만 역수) |
| coverage_threshold가 backend마다 의미 다름 | 전역 `coverage_threshold` 제거, `CameraViewEvidence`에 `coverage_kind`/`depth_kind`/`depth_is_approximate` 추가 |

테스트 보완 요청 2건(surface-band 테스트, multi-view contradiction 테스트) 및 명명 불일치(`outside_every_camera_view` 테스트가 `unobserved`라고 적혔지만 기대값은 `outside_valid_view`였던 점)도 모두 반영했다.

## 4. Gate C 검증 결과

`tests/test_observation_evidence.py`, 8개 테스트, 전부 통과.

| 검증 항목 | 테스트 | 결과 |
|---|---|---|
| 카메라 앞 sample → known_free_space | `test_sample_in_front_of_camera_is_known_free_space` | pass |
| 관측 표면 뒤 sample → occluded_candidate | `test_sample_behind_observed_surface_is_flagged` | pass |
| 모든 카메라 밖 sample → outside_valid_view | `test_sample_outside_every_camera_view_is_outside_valid_view` | pass |
| 표면 위(tie band) sample → free space 아님 | `test_sample_on_observed_surface_is_not_free_space` | pass (aggregate=`unobserved`) |
| Multi-view 상충(한 카메라 behind, 다른 카메라 free) → 보존 | `test_multi_view_conflict_is_preserved_not_collapsed` | pass (aggregate=`conflicting_evidence`) |
| CUDA/fallback depth convention parity | `test_cuda_fallback_depth_parity` | pass (이 환경에 CUDA 백엔드 존재, skip 아님) |
| Empty voxel 단독으로 no_observed_support 초과 안 함 | `test_empty_voxel_query_never_exceeds_no_observed_support` | pass |
| **Gate C 핵심**: 관측 표면 뒤 30-포인트 sweep에서 free-space false acceptance | `test_free_space_never_asserted_for_occluded_candidate_point` | pass, false accept 0/30 |

Multi-view 상충 테스트는 정밀한 카메라 기하를 요구했다 — camera 1(원점, +z 정면)은 wall A(z=2)를 가장 먼저 관측해 sample(0,0,3)을 `behind_first_observed_surface`로 판정하고, camera 3((5,0,0)에서 sample을 조준)의 광선은 wall A의 [-0.5,0.5] 범위를 z=2에서 x=1.667 지점으로 비껴가지만 z=6의 wall B(그 광선 경로상에만 배치한 작은 패치)에는 닿아 sample을 `known_free_space`로 판정한다. 두 판정 모두 사전조건으로 직접 검증한 뒤 집계가 `conflicting_evidence`임을 확인했다.

## 5. 회귀 검증

- 신규 파일만: `8 passed`
- 전체 pytest suite: `230 passed, 1 skipped, 8 subtests passed` (worklog 75 시점 `220 passed`에서 증가 — 신규 8개 외 동시 세션의 변경도 포함된 수치로 보이며, 실패/회귀는 없음)
- `git diff --check`: 통과(줄바꿈 정규화 경고만 존재, 이번 변경과 무관)
- production 변경: `torch_pipeline.py`/`torch_trainer.py` 미변경. Renderer 두 파일은 dict 키 추가만(기존 키 값 불변).

## 6. 남은 제한과 위험

- `classify_world_samples`는 카메라 수 × 샘플 수에 대해 Python 루프를 사용한다(성능 최적화 없음) — isolated prototype 범위이므로 문제 아니지만, Phase D/E에서 대량 continuation 샘플을 다룰 때는 벡터화가 필요할 수 있다.
- Nearest-pixel(반올림) lookup만 사용하고 bilinear는 구현하지 않았다 — 결정성 우선.
- CUDA depth 근사(`1/E[1/z]`)는 depth variance가 낮은 픽셀에서만 신뢰할 수 있다는 한계를 명시했고, parity 테스트 fixture는 이 조건을 만족하도록 의도적으로 설계했다(단일 정면 벽, on-axis sample).
- `_fallback_view_depth`는 fallback 자체 depth-order 합성(진짜 occlusion)을 구현하지 않는다 — production `fallback_render`와 동일하게 화면 근접도 가중 평균이다. Multi-view conflict 테스트에서 이 한계 때문에 두 벽을 카메라 1의 같은 픽셀에 동시에 두면 안 된다는 것을 확인했고(초기 시도에서 실제로 발견), 그에 맞춰 wall B를 camera 1의 frustum 밖에 배치했다.
- Empty-voxel query는 여전히 "no_observed_support" 그 이상을 표현할 수 없다 — Phase E의 occlusion-candidate 판정과는 별도 유지.

## 7. 중단 및 다음 승인

계획대로 Gate C에서 멈춘다. 다음 단계는 **Phase D — Parametric Continuation Domain**이며 별도 사용자 승인 없이는 시작하지 않는다.
