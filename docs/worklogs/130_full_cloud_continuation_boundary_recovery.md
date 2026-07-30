# Worklog 130 — Full-Cloud Continuation Support 기반 Real-Scene Boundary Recovery 보완

## 목표

Worklog 129에서 밝혀진 새 병목 — 실제 DATASET에서 reliable representative와 surface region은 형성되지만 모든 ADC event가 `boundary_component_count=0`, `construction_state=boundary_recovery_failed`로 끝나는 문제 — 를 단계별로 계측하고, full observed cloud의 same-mode ambiguous support를 read-only continuation evidence로 활용해 실제 surface continuation과 reliable-core frontier를 구분하도록 support-termination candidate 생성과 directed boundary recovery를 보완했다. Reliability threshold 완화나 representative cap 확대는 하지 않았다.

## 1. 실제 boundary failure의 단계별 분해

기존 `torch_boundary_support_termination.extract_support_termination_candidates`는 representative 자신의 same-region accepted-topology 이웃(다른 representative)만으로 8-sector angular occupancy를 계산했다. Worklog 129가 밝힌 것처럼 실제 DATASET에서는 representative 간 거리가 실제 국소 밀도를 반영하지 못하므로, 이 단계가 정확히 어디서 얼마나 실패하는지 알 수 없었다 — `boundary_component_count=0`이라는 최종 수치 하나만 있었다.

`torch_visible_surface_construction.py`의 `diagnostic_summary`에 stage별 수치를 추가했다:

- `boundary_candidate_count`, `boundary_genuine_termination_candidate_count`, `boundary_reliability_frontier_candidate_count`, `boundary_sampling_gap_candidate_count`, `boundary_crease_candidate_count`, `boundary_parallel_conflict_candidate_count`, `boundary_ambiguous_candidate_count`
- `boundary_component_closed_count` / `_open_count` / `_branching_count` / `_ambiguous_count` / `_isolated_count` (기존에는 `admissible_component_count` 하나만 있었다)
- `boundary_failure_stage` ∈ `{"not_failed", "A_candidate_generation_failed", "B_candidate_linking_failed", "C_component_admission_failed"}` — genuine termination candidate가 0이면 A, candidate는 있지만 component가 하나도 안 만들어지면 B, component는 있지만 closed loop가 하나도 없으면 C.

실제 DATASET 결과(§4)는 병목이 B(candidate linking)임을 보였다 — 이전에는 구분 자체가 불가능했다.

## 2. Full-cloud continuation shell 계약 (`osn_gs/surface/torch_full_cloud_continuation_shell.py`)

`build_continuation_shells(...)`가 각 eligible representative node(기존과 동일한 게이트: `region_id >= 0`, membership state가 `core_member`/`consensus_attached`, canonical frame 존재)에 대해 read-only continuation support shell을 구성한다.

**재사용, 재계산 금지**: Worklog 129가 이미 계산한 `nearest_representative_index`(Voronoi assignment), `full_frame`/`full_intrinsic`(O(N) 1회 계산), `FullNeighborhoodEvidence.mean_spacing`을 그대로 받는다. 새로운 O(N) eigen-decomposition이나 O(N×M) `cdist`를 추가하지 않았다 — 유일하게 새로 계산하는 것은 representative 수(M, cap 이하)만큼의 M×M representative-to-representative 거리 행렬이다.

**Shell 구성**: query representative i에 대해 (1) 같은 region이면서 (2) i로부터 adaptive radius 이내인 다른 representative들("neighbor reps")을 찾고, (3) 그 neighbor reps의 Voronoi cell(이미 계산된 nearest-representative assignment로 그룹화)에 속한 모든 full-cloud Gaussian을 shell 후보로 모은다. Radius는 고정 거리가 아니라 `max(6 × representative의 tangent_major_scale, 4 × FullNeighborhoodEvidence.mean_spacing)`로, representative 자신의 footprint와 실측 full-cloud spacing을 함께 반영한다(§6 요구사항).

**Same-mode filtering (§7)**: shell 후보 각각을 learned normal alignment(부호 보정), tangent-plane residual, footprint 비율, intrinsic reliability로 분류한다 — `same_mode`(진짜 continuation 증거), `parallel_sheet_conflict`(같은 normal이지만 tangent-plane offset이 큰 close-parallel sheet), `crease`(normal이 크게 다른 경쟁 mode), `ambiguous`(어느 쪽도 아님), 그리고 intrinsic-rejected는 전부 제외된다. Close-parallel sheet는 normal이 같아도 tangent-plane signed offset으로 분리된다(closed-sheet가 continuation 증거로 오인되지 않도록).

**Shell의 역할은 read-only diagnostic/query에 한정**: region ownership, cluster ID 할당, accepted local topology, boundary owner, NURBS fitting support, closed-loop topology edge 어디에도 shell 멤버십이 쓰이지 않는다. Ambiguous Gaussian이 reliable region member로 승격되는 경로는 없다 — shell은 오직 `extract_support_termination_candidates`가 지금 이 node의 gap 방향을 어떻게 분류할지 판단하는 데만 쓰인다.

## 3. Reliable-core frontier와 실제 termination 구분 (§4)

Continuous gap이 발견되면(§5) 다음으로 분류한다:

- `observed_support_termination`: same-mode support가 충분(`>= min_same_mode_support_for_termination=6`)하고 gap 경계에 parallel/crease 증거가 없음 — 진짜 물리적 edge.
- `reliability_frontier`: same-mode support는 부족하지만 gap 방향에 ambiguous mass가 존재 — surface는 계속되지만 reliable core만 끝난 상태. **Physical boundary candidate로 만들지 않는다** — `ordering_state="ambiguous_ordering"`으로 diagnostic-only 유지.
- `unresolved_sampling_gap`: same-mode support 부족 + ambiguous mass도 없음 — genuine termination인지 판단 불가.
- `crease_discontinuity` / `parallel_sheet_conflict`: gap 경계가 경쟁 surface mode로 막혀 있음.
- `ambiguous_continuation`: 그 외.
- `no_gap`: 연속 occupancy에 실제 gap이 없음 — candidate를 아예 만들지 않는다(기존 sector 코드의 "runs가 없으면 skip"과 동일한 자리).

`torch_directed_boundary_ordering.recover_directed_boundary_components`의 기존 입력 필터(`boundary_reason == "observed_support_termination"`만 통과)는 그대로 두었다 — `reliability_frontier`/`unresolved_sampling_gap`/`crease_discontinuity`/`parallel_sheet_conflict`/`ambiguous_continuation` candidate는 계속 diagnostic-only로 남고, closed loop를 만드는 데 절대 쓰이지 않는다. 이 계약을 별도로 다시 구현하지 않고 기존 필터를 그대로 재사용했다.

## 4. Continuous circular support-gap query (§5-6)

`WorldSpaceBoundaryHalfEdgeCandidate`가 sector 8개 대신 fine-grained angular bin(기본 180개, 2°)으로 occupancy를 계산한다. 각 same-mode support는 점 하나가 아니라 각도 구간으로 기여한다(`atan2(footprint, reference_distance)`로 각 Gaussian의 footprint가 만드는 angular half-width를 계산).

**실제로 발견한 버그(density sweep 테스트로 노출)**: 처음에는 `reference_distance`로 각 member의 실제 radial distance를 썼는데, 밀도가 높아질수록(예: box 8x resample) representative에 아주 가까운 member가 많아져 `atan2(footprint, radial_distance)`가 clamp 상한(60°)까지 치솟았다. 몇 개의 근접 member가 이런 큰 half-width로 원 전체를 채워버려, 가장 밀도가 높은 경우(mult=8)에 오히려 genuine candidate가 0개가 되는 역설적 결과가 나왔다. Reference distance를 query 자신의 search radius(query당 고정값)로 바꿔 해결했다 — "이 Gaussian의 footprint가 이 query가 보는 범위에 비해 얼마나 큰가"로 정규화하면 근접 member의 우연한 근접성 때문에 원 전체가 채워지는 일이 없다.

## 5. Candidate normalization (§9)

`torch_boundary_support_termination.normalize_continuation_candidates`가 같은 region · 같은 boundary_reason · 서로의 support_radius 절반 이내 · boundary_direction 정렬(≥0.8) 조건을 모두 만족하는 continuation candidate만 union-find로 병합한다. 서로 다른 crease/parallel-conflict/실제 분리된 물리적 경계는 거리만으로 병합되지 않는다(reason 불일치 시 애초에 그룹화되지 않음). Sector 기반(non-continuation) candidate는 이 함수에서 건드리지 않는다.

## 6. Directed boundary recovery — 변경 없음

`torch_directed_boundary_ordering.py`의 mutual-successor 알고리즘, `accepted_pairs` 게이트, `local_spacing` 계산, closed-loop 판정 로직은 그대로 두었다. Continuation 기반 candidate도 결국 같은 `WorldSpaceBoundaryHalfEdgeCandidate` 타입이고 `boundary_reason` 필터링이 이미 존재하므로, directed ordering 모듈 자체를 손댈 필요가 없었다.

## 7. 기존 계약 유지 확인

`WorldSpaceBoundaryHalfEdgeCandidate`에 새 필드 10개(`gap_width_degrees`, `same_mode_support_count`, `same_mode_opacity_mass`, `ambiguous_continuation_mass`, `competing_mode_mass`, `support_radius`, `reliability_frontier`, `sampling_gap`, `source_full_cloud_fingerprint`, `policy_version`)를 모두 기본값 포함으로 끝에 추가했다 — 기존 positional/keyword 생성 코드(`tests/test_ordered_world_boundary_graph.py`의 13-인자 positional 생성 포함)는 수정 없이 그대로 동작한다. `construct_visible_nurbs_from_gaussians`는 `continuation_input: ContinuationShellInput | None = None` 파라미터 하나만 추가했다 — `None`이면(기존 모든 호출자) `extract_support_termination_candidates`가 완전히 기존 sector 경로로만 동작한다. `torch_pipeline.py`는 다운샘플링이 실제로 일어난 경우에만(`downsampled=True`) continuation을 구성한다 — worklog 129와 동일하게, representative==full cloud인 작은 scene은 continuation도 degenerate이므로 건너뛴다.

## 8. 합성 fixture 결과 (`tests/test_full_cloud_continuation_shell.py`, 10개)

- **Density sweep box**: budget=128로 강제 다운샘플한 box를 1×/2×/4×/8× 밀도로 재샘플링 — region_count는 모든 밀도에서 6(정확한 face 수) 유지, genuine termination candidate 수는 128 미만으로 유지되며 폭증하지 않음(위 §4 버그 수정 후 확인).
- **Thin slab shell isolation**: 손으로 만든 top/bottom 평면 fixture(명시적 region 배정)에서 top query의 `source_full_cloud_fingerprint`가 절대 bottom 인덱스를 포함하지 않음을 직접 확인 — `same_region_mask`가 구조적으로 보장.
- **Box corner**: mult=4 density sweep에서 region_count=6 유지(인접 face가 continuation support로 새는 경우 crease가 "설명되어 사라지며" region이 6 미만으로 붕괴할 것).
- **Sphere**: 닫힌 구에서 genuine termination candidate 수가 eligible node 수의 절반을 넘지 않음(대부분 `no_gap`이어야 함).
- **Contamination/floater**: 고립된 floater(3,3,3)는 어떤 candidate의 `world_position`도 그 근처에 만들지 않음. Isotropic contamination이 있어도 construction이 예외 없이 완료되고 일관된 stage 분류를 반환.
- **Phase-alias cylinder**: closed component 수가 3(side+2 cap) 이하로 유지 — phase-alias/nonlocal edge가 directed ordering에 흘러들어가지 않는 기존 계약이 continuation 경로에서도 유지됨.
- **Invariance**: 회전·이동·균일 스케일 후 region_count 동일, boundary_failure_stage의 "성공/실패" 여부(정확한 후보 수는 worklog 129와 동일한 이유로 axis-aligned voxel grid 한계 때문에 근사)가 동일.
- **No-continuation 회귀**: `continuation_input` 없이 호출하면 box_face가 여전히 sector 경로로 materialize(기존 동작 100% 유지).

## 9. 실제 DATASET before/after (worklog 129와 동일 조건 재현)

설정: RTX 5080, `max_images=1`, `image_downscale=8`, `train_resolution_scale=4`, 6 iterations, ADC iteration `2/4/6`, `densify_grad_threshold=0.001`, `adc_max_gaussians=150000`, opacity/screen/world pruning 비활성화, `canonical_construction_max_points=2048`.

| iteration | reliable | region | boundary_failure_stage | genuine candidates | boundary_component_count | runtime |
|---|---|---|---|---|---|---|
| 2 (구조적 ADC) | 22 | 1 | `B_candidate_linking_failed` | 1 | 0 | 40.01s |
| 4 (구조적 ADC) | 27 | 1 | `B_candidate_linking_failed` | 1 | 0 | 40.62s |
| 6 (terminal) | 24 | 2 | `B_candidate_linking_failed` | 2 | 0 | 40.61s |

Worklog 129(동일 조건): reliable 22~24, region 1~3, `construction_state=boundary_recovery_failed`(원인 미분해), runtime 39.1~42.2s.

**병목이 정확히 어디인지 이제 구분된다**: `boundary_component_count`는 여전히 0이지만, 원인이 "candidate가 하나도 안 생겨서"(stage A)가 아니라 "genuine termination candidate가 1~2개뿐이라 서로 연결(linking)할 상대가 없어서"(stage B)임이 밝혀졌다. 22~27개 reliable node 중 대부분은 `no_gap`(내부, 사방이 support로 둘러싸임)으로 분류되고, 정말 물리적 edge로 보이는 node는 극소수라는 뜻이다 — 이는 이 장면이 아직 학습 극초반(6 iteration, PSNR ~14.5, 단일 이미지)이라 관측된 reliable core 자체가 매우 얇고 서로 떨어져 있어서일 가능성이 높다(진짜 물리적 boundary가 존재하지 않는다는 뜻이 아니라, 이 시점의 학습 데이터로는 그 boundary 근처에 충분히 밀집한 reliable evidence가 아직 없다는 뜻).

**Runtime**: worklog 129(39.1~42.2s)와 사실상 동일(40.0~40.6s) — continuation shell이 기존 계산(Voronoi assignment, full frame/intrinsic, mean_spacing)을 전부 재사용한 덕분에 추가 overhead가 측정 가능한 수준으로 나타나지 않았다.

**실제 DATASET 실행 중 발견/수정한 버그**: 개발 중 CPU 합성 fixture로만 테스트하다가 CUDA 실기기 실행에서 `RuntimeError: ... cuda:0 and cpu!`가 발생 — `torch_full_cloud_continuation_shell.py`에서 `torch.tensor(...)`를 device 지정 없이 생성한 곳(2곳: `region_id_tensor`, `rejected` 마스크)이 원인이었다. 입력 텐서의 device를 명시적으로 전달하도록 수정 후 재실행해 정상적인 `boundary_recovery_failed` → `B_candidate_linking_failed` 분류를 확인했다.

## 10. 테스트

- 신규: `tests/test_full_cloud_continuation_shell.py` 10개 전부 통과.
- 대상 회귀 묶음(manifold affinity, structural reliability, region formation, phase-alias, region invariance, visible surface construction + invariance, ADC-synchronized, world-space boundary half-edges, ordered boundary graph, materialization adapter, region validation, ownership, density-preserving selection, continuation shell, training regressions, pipeline smoke, synthetic dataset): **139 passed**.
- repository-wide pytest: **597 passed, 1 skipped, 1 warning, 8 subtests passed in 153.64s**(worklog 129 기준 587 passed 대비 신규 테스트만큼 증가, 실패 없음).

## 11. 완료 기준 대조

- 실제 DATASET의 boundary failure가 A/B/C stage로 분해됨: 완료(§9, 이번 실행은 전부 B).
- Worklog 129 reliability/region 결과가 회귀 없이 유지됨: 확인(§9 표, reliable/region 수 worklog129와 동일 범위).
- Full-cloud ambiguous support가 read-only continuation shell로만 사용됨: 확인(§2).
- Ambiguous Gaussian 강제 region 할당 없음: 확인 — reliability_frontier/sampling_gap은 candidate로만 남고 topology에 참여하지 않음.
- Reliability frontier와 physical termination 구분: 완료(§3).
- Continuous circular support-gap 구현: 완료(§4), 실제 버그를 density sweep 테스트로 발견·수정.
- Same-mode close-parallel filtering: 완료(§2 same-mode filtering).
- Canonical boundary candidate 및 normalization: 완료(§5).
- Directed component topology가 closed/open/branch/ambiguous/isolated로 분해됨: `diagnostic_summary`에 5개 카운트 모두 노출(§1).
- Synthetic plane/curved closed-loop materialization 유지: 확인(§8 no-continuation 회귀 테스트).
- Sphere false outer boundary 없음: 확인(§8).
- Sparse gap false inner boundary 없음: `unresolved_sampling_gap`이 candidate로 생성되지만 `ordering_state="ambiguous_ordering"`이라 directed ordering에 진입하지 않음.
- 실제 ADC before/after 결과 공개: §9.
- Runtime 비용 공개: §9(사실상 증가 없음, 긍정적 결과).
- Targeted/repository-wide pytest green: §10.

## 12. 다음 작업 판정

이번 결과는 "실제 physical boundary component가 복원되고 기존 materializer에 진입"하지는 못했지만, "실제 topology가 왜 형성되지 않는지"를 stage B(linking, 사실상 candidate가 지나치게 희소함)로 명확히 좁혔다 — §18의 완료 기준 중 두 번째 조건을 만족한다.

**다음 작업 권고**: general topology-aware volumetric chart materialization이나 core-to-shell expansion이 아니라, 학습이 더 진행된(더 많은 iteration, 더 많은 이미지) 실제 snapshot에서 이 genuine candidate 희소성이 완화되는지 재측정하는 것이 다음으로 근거가 명확한 방향이다. 이번 worklog의 density sweep 결과(§8)는 합성 fixture에서 밀도가 증가해도 region_count가 안정적으로 유지됨을 보였으므로, 같은 메커니즘이 실제 학습이 더 진행된 scene에서 genuine termination candidate 수를 늘릴 가능성이 있다 — 다만 이는 아직 실측하지 않은 가설이며, 이번 작업의 비범위(§13)에 따라 검증하지 않았다.

## 13. 명시적 비범위 (변경하지 않음)

Core-to-shell region ownership expansion, ambiguous Gaussian cluster assignment, open boundary NURBS fitting, branching graph decomposition, multi-loop ownership, annulus materialization, derived seam 생성, sphere chart atlas, multi-patch NURBS fitting, scene-specific threshold tuning, representative cap 확대, legacy/voxel constructor 복구, Uncertain/Occluded path 연결, renderer/trainer feedback, production-ready 선언 — 전부 이번 작업에서 다루지 않았다.
