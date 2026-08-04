# Worklog 24 — Density-Preserving Canonical Representative Evidence 및 Reliability 보완

## 목표

Worklog 20이 실제 DATASET(약 138,766~139,737 Gaussian)에서 어떤 `canonical_construction_max_points` cap(512/1024/2048)으로도 `reliable_count=0`, `region_count=0`, `materialized_surface_count=0`, `construction_state=no_admissible_region`으로 fail-closed했던 문제를 조사하고 보완했다. Topology graph의 vertex 수는 계속 제한하되, 각 representative가 전체 observed Gaussian neighborhood의 density·learned covariance·opacity·local manifold evidence를 보존하도록 canonical sampling과 reliability input을 재설계했다. General topology materialization이나 NURBS chart atlas 구현은 이번 범위가 아니다.

## 1. 기존 representative path의 density-loss 분석

`TorchOSNGSPipeline._canonical_construction_indices`(voxel-nearest-to-cell-center)는 occupied cell당 정확히 1개의 Gaussian만 남기고, 그 대표점의 local density/support는 어디에도 기록되지 않는다. 이어서 `evaluate_contextual_consistency`는 이 representative-only 집합에서 8-nearest-neighbor를 다시 계산한다. 실제 DATASET처럼 138k+ Gaussian을 895개 representative로 압축하면, representative 간 거리는 실제 국소 밀도를 전혀 반영하지 못하는 sparse point cloud가 되어 거의 모든 representative가 `contextual_insufficient`/`contextual_mixed`로 붕괴한다 — worklog 20의 `reliable_count=0`은 cap 값의 우연이 아니라 이 구조적 결함의 필연적 결과였다.

추가로 이 sampler는 cell당 대표점을 1개만 남기므로, 한 voxel cell에 close-parallel sheet·box corner의 여러 face·cylinder cap/side 접합처럼 구조적으로 다른 surface mode가 여럿 있으면 나머지는 topology 전체에서 조용히 소실된다.

## 2. 신규 representative/evidence 계약

### 2-1. Full-neighborhood evidence (`osn_gs/surface/torch_full_neighborhood_evidence.py`)

`compute_full_neighborhood_evidence(full_positions, full_frame, full_opacity, full_intrinsic, representative_positions, representative_frame, representative_ids)`가 각 representative에 대해 **전체 observed cloud를 그 representative의 Voronoi cell로 파티션**하여(`assign_nearest_representative`, chunked `cdist` argmin — 기존 `_nearest_canonical_sample_indices`와 동일한 복잡도 클래스) 다음을 집계한다: `support_count`, `opacity_sum`, `opacity_weighted_centroid`, `mean_spacing`/`spacing_std`, `normal_consensus`(부호-보정 resultant vector 크기), `tangent_residual_mean`/`std`, `eigenvalue_ratio_mean`/`std`, `competing_mode_mass`(정렬 낮은 비율), `rejected_neighbor_mass`, `local_density`. Topology(affinity/region formation)는 여전히 bounded representative에서만 O(N²)로 실행되며, 이 모듈은 O(N) eigen-decomposition(전체 cloud의 `extract_covariance_frame`)과 chunked nearest-representative assignment만 추가한다.

### 2-2. Learned covariance와 local aggregate의 역할 분리

`evaluate_structural_reliability_from_full_evidence(frame, evidence)`(`torch_gaussian_structural_reliability.py`)를 추가했다. **Intrinsic reliability는 그대로 representative 자신의 learned covariance만 사용**(변경 없음 — `reconstruct_visible_after_adc`는 이미 `model.get_scaling`/`get_rotation`에서 나온 실제 learned covariance를 representative에 전달하고 있었으므로, 이 축은 이미 production primary evidence였다). **Contextual consistency만** representative-only kNN에서 full-neighborhood evidence 기반(`evaluate_contextual_consistency_from_full_evidence`)으로 교체했다. 두 경로 모두 동일한 `combine_reliability(intrinsic, contextual, config)`로 기존 3-tier(`reliable`/`ambiguous`/`rejected`) 계약에 투영되므로 downstream(affinity/region formation/boundary)은 입력 evidence의 출처를 모른다.

`construct_visible_nurbs_from_gaussians`에는 `reliability: StructuralReliabilityResult | None` override 파라미터 하나만 추가했다 — 제공되면 내부 representative-only reliability 계산을 건너뛴다. 이것이 유일한 injection point이며, affinity/region formation/boundary recovery/materialization 등 다른 모든 canonical stage는 변경하지 않았다. CLI에 노출되는 새 constructor selector는 없다.

**대표성이 없어지는 경우의 명시적 fallback**: representative 수가 전체 cloud와 같으면(다운샘플링이 실제로 일어나지 않은 작은 scene) full-neighborhood Voronoi 집계는 "자기 자신만의 이웃"으로 퇴화해 오히려 기존 kNN보다 나쁜 evidence가 된다. 이 경우 `_construct_canonical_with_full_evidence`는 기존 `evaluate_structural_reliability`(representative-only kNN)로 자동 폴백한다 — 이는 근사가 아니라, representative 집합과 full cloud가 literally 동일 집합일 때 두 경로가 수학적으로 같은 입력을 다르게(하나는 올바르게, 하나는 퇴화되게) 처리하기 때문에 필요한 정확성 수정이다.

### 2-3. Multiple surface mode 처리 (`osn_gs/surface/torch_density_preserving_representative_selection.py`)

`select_density_preserving_representatives(points, frame, opacity, stable_ids, max_points, config)`:

1. 기존과 동일한 voxel grid(occupied cell)로 candidate grouping.
2. Cell마다 `_split_cell_into_modes` — stable-ID 오름차순으로 순회하며, 기존 mode 중 `normal alignment >= mode_normal_alignment_min(0.6)` **AND** tangent-plane offset이 `mode_offset_max_thickness_ratio(3.0) * thickness` 이내인 것에 합류시키고, 아니면 새 mode를 만든다. `max_modes_per_cell(4)`로 명시적으로 bound하며, 초과분은 최고-정렬 mode로 강제 병합한다(scene별 튜닝이 아니라 문서화된 고정 상한).
3. Mode마다 opacity-가중 centroid에 가장 가까운 실제 Gaussian을 representative로 선택(동률은 stable-ID로 결정).
4. Candidate 수가 budget 이하면 전부 채택. 초과하면 **support/opacity 가중 farthest-point selection**으로 budget개를 결정적으로 선택(seed는 support·opacity 최대, 이후 매 스텝 "미선택 후보까지의 최소 거리 × support/opacity 가중치"가 최대인 후보를 채택, 동률은 stable-ID). Full C×C pairwise 행렬은 만들지 않고 반복마다 신규 대표점 1개에 대한 거리 벡터만 계산한다(대형 scene에서 메모리 폭증 방지).

Cell ID는 후보 grouping/acceleration에만 쓰이고 region ID로 재사용되지 않는다.

### 2-4. Full-Gaussian propagation (`TorchOSNGSPipeline._propagate_with_evidence_gating`)

기존 `_propagate_canonical_patch_ids`(nearest-representative만으로 강제 할당)와 별도로, `reconstruct_visible_after_adc`에 새 evidence-gated 버전을 추가했다: 각 full Gaussian을 가장 가까운 representative에 매핑한 뒤, **자신의 learned normal이 그 representative의 normal과 정렬되고(`alignment >= 0.5`) tangent-plane residual이 그 representative의 tangent scale 대비 허용 범위(`<= 4x`) 안일 때만** 해당 region으로 배정한다. 둘 중 하나라도 어긋나면 coverage를 억지로 늘리지 않고 `-1`(미배정)로 남긴다. Eigenvector 부호는 원래 모호하므로 모든 지표는 `abs()`만 사용해 orientation 계산을 별도로 두지 않았다.

## 3. Scope 결정: `_initialize_canonical`/`maintain_surface_from_certain`은 이번 변경 대상에서 제외

이번 수정은 `reconstruct_visible_after_adc`(worklog 20이 실패를 관측한 바로 그 경로)에만 새 representative selection + full-neighborhood evidence + evidence-gated propagation을 적용했다. `_initialize_canonical`은 원시 COLMAP point cloud에서 학습 전 최초 1회 실행되며, 이 시점에는 "production model이 제공하는 learned covariance" 자체가 존재하지 않는다(사용 가능한 유일한 covariance는 대표 sample에서 계산한 local PCA 근사) — 전체 cloud에 대해 O(N)로 확장 가능한 학습된 covariance가 없으므로 full-neighborhood evidence를 적용할 입력이 없다. `maintain_surface_from_certain` 역시 (기존부터) learned covariance가 아닌 local PCA를 사용하는 별도 호환 경로이며, worklog 20의 실패 재현 대상이 아니다. 두 경로 모두 변경하지 않았으므로 회귀 위험이 없다.

## 4. 실제 DATASET 검증 (worklog 20과 동일 조건 재현)

설정: RTX 5080, `max_images=1`, `image_downscale=8`, `train_resolution_scale=4`, 6 iterations, ADC iteration `2/4/6`, `densify_grad_threshold=0.001`, `adc_max_gaussians=150000`, opacity/screen/world pruning 비활성화, `visible_nurbs_update_schedule=adc_post_commit`.

### cap=2048 (worklog 20과 동일 cap)

| iteration | reliable | ambiguous | region | boundary_component | state | representative | occupied_cell | multi_mode_cell | full_evidence_support_mean | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 22 | 2026 | 1 | 0 | `boundary_recovery_failed` | 2048 | 894 | 731 | 67.9 | 39.1s |
| 4 | 23 | 2025 | 3 | 0 | `boundary_recovery_failed` | 2048 | 894 | 731 | 68.1 | 41.6s |
| 6(terminal) | 24 | 2024 | 1 | 0 | `boundary_recovery_failed` | 2048 | 894 | 730 | 68.1 | 42.2s |

Worklog 20(동일 cap): `reliable_count=0`, `region_count=0`, `materialized_surface_count=0`, state=`no_admissible_region`, runtime `0.881~0.902s`.

### cap=512 (worklog 20과 동일 cap)

| iteration | reliable | region | state | representative | occupied_cell | multi_mode_cell | full_evidence_support_mean | runtime |
|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 1 | `boundary_recovery_failed` | 512 | 265 | 225 | 271.8 | 36.9s |
| 4 | 0 | 1 | `boundary_recovery_failed` | 512 | 265 | 225 | 272.4 | 37.4s |
| 6(terminal) | 0 | 1 | `boundary_recovery_failed` | 512 | 265 | 225 | 272.4 | 39.7s |

cap=512에서는 3-tier `reliable_count`가 0이지만 `region_count=1`이다 — region formation의 core-seeding은 3-tier reliable이 아니라 **intrinsic axis만**(`INTRINSIC_RELIABLE` + `core_degree>=2`) 요구하므로, `intrinsic_reliable + contextual_mixed`(진짜 crease 형태 evidence)인 representative만으로도 region이 형성될 수 있다. cap이 낮을수록 representative당 Voronoi cell이 커져(support_mean 271.8 vs cap2048의 67.9) 하나의 representative가 여러 실제 surface mode를 섞어 집계하게 되고, 이는 "완전히 깨끗한" contextual-consistent 판정을 더 어렵게 만든다 — cap이 reliability 결과에 미치는 영향은 architectural로 설명 가능하며 우연이 아니다.

**공통 관찰**: `full_evidence_zero_support_count=0` — 모든 representative가 실제 full-cloud 지지를 갖는다(과거처럼 representative 자신 외에 아무것도 보지 못하는 경우가 없다). Occupied cell의 82%(cap2048: 731/894, cap512: 225/265)가 multi-mode였다 — "cell당 대표점 1개"가 실제 scene에서도 구조적 정보를 광범위하게 버리고 있었다는 증거다.

**병목의 재분류**: worklog 20의 병목은 reliability admission 전체 붕괴(모든 cap에서 `reliable_count=0`, `region_count=0`)였다. 이번 수정 이후 병목은 **boundary/topology recovery**로 이동했다(`region_count>0`이지만 `boundary_component_count=0` → `boundary_recovery_failed`) — reliability admission 자체도 cap=2048에서 여전히 얇다(2048개 중 22~24개, ~1.1%만 3-tier reliable). 두 층 모두 개선 여지가 있지만, 지금은 region이 형성돼도 boundary half-edge/loop 복원 단계에서 막힌다는 점이 명확히 드러났다 — 이는 이번 작업의 범위(§7 명시적 비범위: general topology materialization/multi-patch fitting 제외)를 벗어난 별도 스레드의 과제다.

**Runtime 비용 공개**: 이벤트당 runtime이 worklog 20의 ~0.9초에서 ~37~42초로 증가했다(약 40배). Full-neighborhood evidence의 O(N) eigen-decomposition과 chunked nearest-representative assignment, 그리고 mode-aware representative selection의 cell-순회 비용이 원인이다. Structural ADC는 학습 전체에서 드물게 발생하므로 연구용으로는 수용 가능하지만, `project_deferred_followups`(학습 속도는 NURBS representation 완성 후 함께 재검토)에 정직하게 편입되어야 할 실측 비용이다.

## 5. 합성 volumetric 결과 (box/cylinder/sphere/thin_slab)

Budget 이하 소규모 scene(다운샘플링이 일어나지 않는 경우)은 §3의 fallback으로 기존 경로와 동일하게 동작 — 회귀 없음을 확인했다(box: region=6/reliable=54, cylinder: region=3/reliable=152/materialized=1, sphere: region=4/reliable=199, thin_slab: region=2/materialized=2 — 모두 Worklog 19/23 기록과 일치, sphere의 4-region 분절은 여전히 미해결로 재확인만 함).

## 6. Density sweep (`make_gaussian_density_sweep_scene`, `nurbs_constructor_benchmark/gaussian_reliability_scenes.py`)

기존 legacy fixture는 건드리지 않고 신규 함수 하나만 추가했다: 동일한 box/cylinder 기하를 grid 해상도 배수(`resolution_multiplier`)로 재샘플링해 실제로 더 조밀한 Gaussian을 생성한다(중복+jitter가 아님). `budget=256`으로 강제 다운샘플링한 box에서:

| multiplier | N | representative | full_evidence support 평균 | reliable | region |
|---|---|---|---|---|---|
| 1 | 294 | 256 | 1.15 | 0 | 6 |
| 2 | 1014 | 256 | 3.96 | 0 | 6 |
| 4 | 3750 | 256 | 14.65 | 77 | 6 |
| 8 | 14406 | 256 | 56.27 | 194 | 6 |

Region count(6, box의 정확한 face 수)는 density 증가와 무관하게 안정적으로 유지되어 false merge가 없음을 확인했고, support/reliable count는 density에 단조 증가로 반응했다 — ADC가 실제로 하는 일(같은 물리적 위치에 더 많은 Gaussian이 누적)을 올바르게 반영한다.

## 7. Negative control / invariance (`tests/test_density_preserving_representative_selection.py`, 8 tests)

- **Multi-mode 보존**: 위치가 거의 같고(1e-5) normal이 반대인 두 Gaussian이 반드시 같은 cell에서 서로 다른 mode로 분리됨을 확인(`modes_per_cell_max>=2`).
- **Close-parallel sheet 보존**: thin slab의 top/bottom face가 coarse budget(8)에서도 둘 다 representative로 생존.
- **Order invariance**: stable-ID를 유지한 채 입력 순서를 shuffle해도 선택된 representative의 stable-ID 집합이 동일.
- **Rigid transform 안정성**: 회전+이동+균일 스케일 후 region_count는 정확히 동일, reliable_count는 근사(50% 이내) — axis-aligned voxel grid는 임의 회전 하에서 exact 불변은 아니라는 것을 문서화(기존 `_canonical_construction_indices`도 동일한 한계를 가진 axis-aligned 구조이며 이번 회귀는 아님).
- **Isolated floater**: representative로 선택되더라도 3-tier reliable로 승격되지 않음.
- **Isotropic contamination**: full-neighborhood support가 높아도(밀집 지역 인접) intrinsic axis에서 rejected로 남아 contextual 지지가 intrinsic rejection을 절대 덮어쓰지 않음을 확인.
- **Density sweep 회귀**(§6 표 포함) 2건.

## 8. 테스트

- 신규: `tests/test_density_preserving_representative_selection.py` 8개 전부 통과.
- 대상 회귀 묶음(manifold affinity, structural reliability, region formation, phase-alias, region invariance, visible surface construction + invariance, ADC-synchronized, world-space boundary half-edges, region validation, ownership, density-preserving selection, training regressions, pipeline smoke, synthetic dataset): **126 passed**.
- repository-wide pytest: **587 passed, 1 skipped, 1 warning, 8 subtests passed in 136.78s** (실패 없음, Worklog 22 기준 578 passed 대비 신규 테스트만큼 증가).

## 9. 완료 기준 대조

- learned Gaussian covariance가 production structural primary evidence로 사용됨: 유지(intrinsic axis 불변) + 재확인.
- contextual reliability가 full observed neighborhood를 반영함: 신규 구현, `full_evidence_zero_support_count=0`으로 실측 확인.
- representative cap과 full density evidence 분리: 신규(`FullNeighborhoodEvidence.support_count`가 cap과 독립적으로 실제 밀도를 반영).
- Cell 내부 multiple surface mode 보존: 신규, 실제 DATASET에서 occupied cell의 82%가 multi-mode였음을 실측.
- Voxel이 topology/patch 단위로 재사용되지 않음: 유지(cell ID는 grouping에만 사용).
- Deterministic representative selection: stable-ID tie-break, order-invariance 테스트로 확인.
- Full-Gaussian propagation에 ambiguity 유지: `_propagate_with_evidence_gating`이 evidence 불일치 시 `-1` 유지.
- Box/cylinder/sphere/thin-slab negative control 회귀 없음: §5, §7에서 확인.
- Density 증가가 aggregate evidence에 실제 반영됨: §6에서 실측.
- ADC snapshot의 before/after stage별 비교 완료: §4.
- 실제 DATASET 결과 및 failure reason 공개: §4(state=`boundary_recovery_failed`로 명시적 재분류, threshold 완화 없음).
- Targeted/repository-wide pytest green: §8.

## 10. 남은 병목과 다음 작업 판정

**병목은 이제 두 층으로 명확히 분리된다**: (1) reliability admission은 개선됐지만 cap=2048에서도 여전히 ~1%만 3-tier reliable(threshold를 낮추지 않았으므로 fail-closed 유지, 의도된 동작), (2) region이 형성돼도(`region_count>=1`) boundary half-edge/loop 복원이 실제 DATASET에서 매번 실패한다(`boundary_recovery_failed`, `boundary_component_count=0`). 두 번째가 지금 시점에 더 상위의 blocker다 — reliability를 더 개선해도 boundary recovery가 loop를 못 찾으면 materialize로 이어지지 않는다.

**다음 작업 권고**: general topology-aware volumetric chart materialization이나 core-to-shell expansion이 아니라, **실제 DATASET 규모(수백 개 reliable/region member)에서 `extract_support_termination_candidates`/`recover_directed_boundary_components`가 왜 closed loop를 하나도 복원하지 못하는지에 대한 독립적인 진단**이 다음으로 근거가 명확한 방향이다(이번 worklog의 §7 비범위 목록에 있는 open boundary NURBS fitting/multi-loop ownership/atlas 작업들의 전제 조건). Representative cap을 더 올리는 것은 이벤트당 런타임을 추가로 늘리므로(현재 cap=2048에서 ~40초/event), 성능 최적화 없이는 실용적이지 않다 — 이는 `project_deferred_followups`(학습 속도 이슈)와 함께 다뤄야 한다.

## 11. 명시적 비범위 (변경하지 않음)

Open boundary NURBS fitting, multi-loop ownership, annulus materialization, derived seam 생성, sphere chart atlas, branching graph decomposition, multi-patch NURBS fitting, core-to-shell label propagation, ambiguous Gaussian 강제 할당, scene-specific threshold tuning, fixture-specific dispatcher, legacy/voxel constructor 복구, Occluded/Uncertain path 연결, production-ready 선언 — 전부 이번 작업에서 다루지 않았다.
