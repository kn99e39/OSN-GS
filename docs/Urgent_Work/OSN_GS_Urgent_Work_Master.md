# OSN-GS Urgent Work Master

최종 갱신: 2026-08-04

이 문서는 현재 진행 방향과 승인 경계를 정의하는 canonical master다. 과거 실험의 상세 경과는 Git 이력에 보존하며, 현재 판단에 필요 없는 작업로그는 `docs/worklogs/`에서 제거했다. 작업로그 번호는 `docs/worklogs/README.md`의 최신 인덱스(1부터 재번호, 현재 62까지)를 기준으로 한다.

## 1. 목표 모델과 불변 조건

OSN-GS에서 NURBS는 관측 가능한 표면을 설명하고 가려진 영역의 불확실 Gaussian 생성에만 기하 정보를 제공하는 중간 표현이다. visible/certain Gaussian의 위치는 NURBS가 아니라 영상 손실로 최적화한다.

- visible surface는 topology별 별도 방법론으로 분기하지 않는다.
- 모든 topology는 observed boundary loop, boundary role, source provenance, interior support를 공통 입력 계약으로 사용한다.
- multi-hole은 outer loop와 모든 interior loop를 보존한다. 비중첩 planar partition 증거가 없으면 `review_required`이며 임의 central fill 또는 hole별 overlapping annulus 복제는 금지한다.
- artifact의 chart가 생성되었다는 사실은 품질·안전·사용 가능성을 뜻하지 않는다.
- **evidence가 없으면 강제로 닫지 않는다.** gap 보간, 임의 candidate 생성, closure만을 위한 낮은-evidence edge 선택은 어떤 단계에서도 금지한다 — 이 원칙은 아래 2절 전체에서 반복적으로 검증·재확인됐다.

## 2. 현재 활성 작업 A — Canonical visible-NURBS production pipeline (full-cloud continuation → representative selection → boundary ordering → materialization)

**2026-07-30(worklog 18) 기준으로 이 경로가 `train.py`/benchmark의 유일한 production 경로가 됐다.** `legacy`, `voxel_patch_stage1`, IDW/local split fallback과 해당 CLI는 전부 제거됐다. 이전 절(과거 버전)에서 "isolated, dispatcher 미연결"로 기술했던 world-space boundary half-edge → ordered graph → materialization adapter 계열 작업(worklog 12-15)은 이 병합 이후 더 이상 isolated가 아니다 — 지금은 `construct_visible_nurbs_from_gaussians` / `TorchOSNGSPipeline._construct_canonical_with_full_evidence`가 곧 production 경로다.

### 2.1 파이프라인 단계

1. **Representative selection** (`torch_density_preserving_representative_selection.py`): 대량 Gaussian을 `canonical_construction_max_points`(기본 2048) 예산으로 mode-aware farthest-point sampling. worklog 49에서 boundary-evidence swap-in(예산 경쟁으로 탈락한 진짜 orientation-divergent evidence를 안전하게 복구)을 추가했다.
2. **Reliability + region formation** (`torch_gaussian_structural_reliability.py`, `torch_gaussian_surface_region_formation.py`): full-cloud contextual evidence 기반 reliability, seed/merge DSU 기반 region consolidation.
3. **Boundary candidate 추출** (`torch_full_cloud_continuation_shell.py`, `torch_boundary_support_termination.py`): representative 주변 same-mode angular gap을 관측해 `observed_support_termination`/`no_gap`/`parallel_sheet_conflict`/`crease_discontinuity`/`ambiguous_continuation`/`reliability_frontier`/`unresolved_sampling_gap`으로 분류. worklog 47(cross-surface leakage), 48(fold/gap-crossing locality), 50(single-radius over-reach)에서 순차적으로 오분류를 좁게 수정했다.
4. **Directed ordering / materialization** (`torch_directed_boundary_ordering.py`, `torch_visible_boundary_region_status.py`): region별 compatible edge에 대한 exact one-in/one-out Hungarian matching으로 closed cycle과 open path를 결정론적으로 복원. worklog 53에서 downstream-invalid 2-cycle이 capacity를 낭비하는 결함과, 그로 인해 노출된 direct/reverse tangent quality 비교의 self-intersection 인식 결함을 수정했다. worklog 54는 region별 5-state 계약(`eligible_closed_boundary`/`open_observed_fragment`/`insufficient_observation`/`ambiguous_boundary`/`rejected_unsafe`)을 확정해 validated closed loop만 materialization에 전달하도록 제한했다.

### 2.2 현재 결론 (worklog 40~53에 걸친 6단계 소진 감사)

Real 3k/5k/10k checkpoint(cap 2048) replay에서 **5k만 2개 region을 닫고(closed/materialized 2/2), 3k와 10k는 0개**다. worklog 40부터 53까지 다음 단계를 각각 독립적으로 감사했고, 전부 실제 결함을 찾아 좁게 수정했지만 **어느 것도 closed-loop 개수를 바꾸지 못했다**:

| worklog | 감사 대상 | 결과 |
|---|---|---|
| 40 | region-pair 단위 sphere seam 오탐 | 수정, sphere 22 false candidate 제거 |
| 45 | directed compatibility gate의 target tangent 부호 | 수정, Box 6번째 face + real 5k 0→2 closed 회복 |
| 47 | no_gap의 cross-surface(mode) leakage | 수정, 효과는 재분류만 |
| 48 | no_gap의 candidate-local fold/gap-crossing | 수정, 3k/5k 각 1 node만 재분류 |
| 49 | representative selection의 FPS 예산 손실 | 수정(swap-in), real swap 22/51/76건이나 closed 불변 |
| 50 | 단일 4x 반경의 원거리 support 과다 신뢰 | 수정(scale-persistence), 76/36/39 node 재분류, closed 불변 |
| 51 | raw full-cloud에 representative 축약으로 잃은 chain이 있는지 | 기각(3곳 raw에도 chain 없음) + 2곳은 Hungarian 경쟁 문제로 재분류 |
| 52 | 그 Hungarian "경쟁" 주장 자체의 정확성 | 정정 — edge는 실제로 matching에 포함돼 있었음, 진짜 원인은 topology 불가능(region 52) 또는 evidence 열세(region 56) |
| 53 | region 56의 진짜 원인(2-cycle 낭비) | 수정, fragment 2개→4-node open path 1개로 개선(닫히진 않음, 진짜 최댓값이 open이므로) |
| 54 | region별 boundary 상태를 production 계약으로 확정 | `eligible_closed_boundary`/`open_observed_fragment`/`insufficient_observation`/`ambiguous_boundary`/`rejected_unsafe` 5-state 도입, validated closed loop만 materialize, 2-cycle budget exhaustion fail-closed 추가 |
| 55 | eligible region의 downstream 전달 완성 | fail-closed 일관성 검사 + provenance threading + 단일 진입점(`eligible_materialized_surfaces()`) 추가. 첫 시도(region당 최대 1개 closed loop만 유지)가 thin_slab materialization을 회귀시켜 즉시 되돌리고 "전부 유지 + reason으로 명시 disclosure"로 확정 |
| 56 | `eligible_materialized_surfaces()` → Phase D/E(continuation domain·occluded candidate) production bridge | `build_eligible_boundary_continuation_bridge()` + `run_eligible_boundary_continuation_bridge_from_gaussians()` 신설, Phase D/E 두 모듈은 미변경. Phase D의 최소 4-sample 계약이 기하적 안전 조건이 아니라 표현 밀도 관례임을 실측 확인하고 검증된 loop를 결정론적으로 재표본화 — real 5k 2개 region 모두 domain 생성(candidate는 AABB 비접촉으로 0), 남는 실패는 `eligible_visible_only_not_continuation_ready`로 typed 분리 |
| 57 | Occluded candidate → safe uncertain proposal production integration | 기존 Phase F constrained fit → Phase F.1 sampled safety → Phase G proposal을 `run_safe_uncertain_proposals_from_gaussians()`로 Worklog 56 bridge 뒤에 연결. candidate별 1:1 typed accounting과 provenance chain 유지, `degenerate` domain은 Phase E provenance로만 보존하고 Phase F/G 입력으로는 fail-closed rejection. model append/appearance/opacity는 미수행 |
| 58 | Safe uncertain proposal → atomic append production integration | Worklog 57의 `proposed` proposal만 기존 `UncertainGaussianAppendAdapter.append()`(model-only, 4-way transactional)로 전달하는 production 진입점 추가. `initialization_provider`가 명시적 appearance/opacity를 제공해야 하며 합성하지 않음. `_commit_ledger` 실패 주입 시 model/sidecar/registry/ledger 전부 pre-transaction 상태로 롤백 확인 |
| 59 | Appended uncertain Gaussian trainer activation | `appended` receipt만 소비해 기존 trainer optimizer에 안전하게 활성화. Append adapter의 `model.optimizer is None` 전제조건을 만족시키는 임시 분리/재부착을 이 모듈이 담당하고, 기존 `_preserve_optimizer_state()`(ADC clone/split 재사용, 미변경)로 기존 Adam state는 bit-for-bit 보존·신규 행은 0 초기화. Visible/uncertain 분리는 새로 구현하지 않고 기존 `is_uncertain` 마스킹 관행에 의존. 동기화 실패는 전부 롤백 후 `appended_inactive`로 typed 분리 |
| 60 | Atomic trainer activation contract closure | `TorchOSNGSTrainer.activate_and_train_uncertain_step()`으로 실제 rasterizer render+loss 경로에 연결. `append_and_activate()`를 candidate 단위 composite transaction으로 확장해 activation 실패 시 model/sidecar/registry/ledger/optimizer 전부 롤백(`rolled_back`). `masked_optimizer_step()`으로 Adam momentum까지 포함한 true row-level isolation 구현, 롤백 후 optimizer parameter identity를 group-name 재-keying으로 보장 |
| 61 | Real visible chart boundary materialization (physical 경로와 무관한 신규 확장, §2.2 6단계 감사 대상 아님) | Accepted region topology만으로 chart boundary를 구성하는 `eligible_parametric_chart_boundary` 경로 신설(§2.5). Physical `eligible_closed_boundary` 경로는 완전히 미변경. Real 5k/10k 합산 materialized 3→92/4→78 |

**결론: 남은 병목은 파이프라인 결함이 아니라 candidate evidence 자체의 밀도/위상이다.** 3k/10k의 큰 region은 perimeter 전체를 덮을 만큼 충분히 독립적인 observed-termination evidence가 없다 — representative를 더 정확한 위치로 옮기거나(49), 오분류를 고치거나(47/48/50), matching 낭비를 없애도(53) 이 밀도 자체는 바뀌지 않는다. 5k가 성공하는 이유는 이런 결함이 없어서가 아니라 해당 region이 작아서(candidate 4개)다. worklog 54는 이 결론을 region별 5-state production 계약으로 확정했다 — real 5k는 region 130/141만 `eligible_closed_boundary`, 3k/10k는 전부 나머지 4-state로 명시 기록되며 빈 결과/실패로 뭉뚱그려지지 않는다.

**Worklog 61 addendum**: 위 결론은 physical-termination 경로(`eligible_closed_boundary`)에만 적용된다 — 이 경로 자체는 worklog 61에서도 변경하지 않았다. 대신 accepted region topology만으로 chart boundary를 구성하는 완전히 별도인 `eligible_parametric_chart_boundary` 경로를 추가해, physical evidence 밀도와 무관하게 visible NURBS materialization coverage를 real 5k/10k 기준 수십 배 확장했다(§2.5 참고). physical 경로의 candidate evidence 밀도 문제 자체는 여전히 미해결이다.

### 2.3 검증된 negative control (모든 worklog가 반복 확인)

cap=64 기준: Box `physical=51 closed=6 materialized=6`, Cylinder `16/2/2`, Sphere `14/0/0`, Thin slab `37/3/3`. 이 수치는 worklog 47부터 61까지 매 라운드 재확인했다 — 새 작업은 이 표를 반드시 재현해야 하며, 벗어나면 즉시 원인을 규명한다(worklog 53의 2차 self-intersection 결함, worklog 55의 multi-loop exclusion 회귀가 각각 이렇게 발견됐다).

### 2.4 다음 착수 후보 (미승인, 방향 제안일 뿐)

- Candidate GENERATION 밀도 자체(왜 큰 region이 candidate 3~6개에 그치는지) — 지금까지 전부 candidate 전달/분류/정렬 단계만 감사했고 생성 자체의 근본 원인은 아직 안 봤다.
- ordering/quality 비교(`recover_directed_boundary_components`)가 direct/reverse 전체 scene 단위로 이뤄지는 구조 자체의 재검토 — worklog 53에서 이 결합이 예상치 못한 부작용을 만든 사례가 있었다.

### 2.5 Parametric chart boundary 경로 (worklog 61, physical 경로와 완전 분리)

`osn_gs/surface/torch_region_parametric_chart_boundary.py`가 physical termination evidence와 무관하게, region의 이미 승인된 accepted-edge topology만으로 chart boundary를 구성한다(convex hull/bounding box 미사용, leftmost-turn walk로 outer face 추적). Boundary edge는 `physical_termination`/`crease`/`observation_frontier`/`partition_seam` 중 하나로 분류되며 위장하지 않는다. `eligible_parametric_chart_boundary` region만 기존 `materialize_visible_boundary_component()`(미변경)로 materialize되고, `eligible_parametric_chart_surfaces()`로 physical 경로(`eligible_materialized_surfaces()`)와 완전히 분리 노출된다 — Worklog 56 continuation bridge 등 기존 소비자는 여전히 physical 경로만 읽는다(연결 안 됨, 별도 승인 필요).

Real 5k/10k(cap 2048)에서 physical+parametric 합산 materialized surface가 5k 3→92, 10k 4→78로 확장됐다(§2.2 표의 physical-only 숫자와는 다른 지표). §2.3의 cap=64 negative-control physical 수치(6/2/0/3)는 byte-identical로 유지되고, parametric 경로가 추가로 Box 4·Cylinder 1·Sphere 1(physical 0 유지, 별도 provenance만)·Thin-slab 3(partition_seam 1건 명시)를 materialize한다.

## 3. 현재 활성 작업 B — Isolated Boundary-first hardening (구 Section 2, 현재 사실상 비활성)

과거(2026-07 초) 이 절이 기술하던 exporter/cubic seam wedge/observed-anchor central cap 작업(worklog 4-6)은 그 자체로는 dispatcher/production에 연결된 적이 없다. 2026-07-30(worklog 18) 이후 production은 §2의 canonical 경로로 통합됐고, 이 isolated hardening 라인은 이후 세션에서 별도로 재개되지 않았다. 재개할 경우 다음을 여전히 유지한다: exporter의 outer/interior/support/seam/chart 명시적 분리, sampled crossing/fidelity gate, degree-1/fan 임시면의 최종 근거 사용 금지, false-hole evidence 부족 시 `review_required` 유지.

## 4. 현재 활성 작업 C — Uncertain Gaussian model foundations

Phase G proposal, model-only append adapter, occluded chart ownership foundation은 각각 구현·검증된 계약으로 유지한다. 이들은 visible-surface quality를 대신 증명하지 않으며, append 대상의 appearance/opacity와 downstream lifecycle은 여전히 명시적 차단 조건이다.

- raw Gaussian evidence → safe uncertain proposal → model-only atomic append(worklog 57/58) → trainer optimizer activation(worklog 59)까지 production orchestration이 구현됐다. `appended` receipt만 활성화 대상이며, appearance/opacity는 여전히 `UncertainAppendInitialization`이 명시 제공한 값만 사용한다(합성 없음).
- Worklog 59의 activation은 `TorchGaussianModel.optimizer`(Adam)를 기존 `_preserve_optimizer_state()` grow-in-place 로직으로 확장하는 데 그쳤다. Worklog 60은 `TorchOSNGSTrainer.activate_and_train_uncertain_step()`으로 실제 render+loss 경로에 연결하고, append+activate를 candidate 단위 composite transaction으로(activation 실패 시 append까지 전부 롤백), `masked_optimizer_step()`으로 Adam momentum까지 포함한 row-level 학습 격리를 구현했다 — 여전히 `train()`의 실제 iteration 루프/CLI/checkpoint 스케줄에는 연결하지 않았다(`activate_and_train_uncertain_step()`은 호출자가 명시적으로 불러야 하는 별도 메서드). Visible/uncertain 분리 자체는 새로 구현하지 않고 기존 `is_uncertain` 마스킹(`torch_density_control.py`/`torch_pipeline.py`)에 의존한다.
- global ranking/selection, conflict resolution, checkpoint schema 변경, renderer 통합, 실제 `TorchTrainer` 루프로의 배선은 여전히 시작하지 않는다.

근거: `docs/worklogs/1_phase_g_uncertain_gaussian_proposal_foundation.md`, `docs/worklogs/2_uncertain_gaussian_append_adapter_foundation.md`, `docs/worklogs/3_occluded_chart_ownership_foundation.md`, `docs/worklogs/57_occluded_candidate_safe_uncertain_proposal_production_integration.md`, `docs/worklogs/58_safe_uncertain_proposal_atomic_append_production_integration.md`, `docs/worklogs/59_appended_uncertain_gaussian_trainer_activation.md`, `docs/worklogs/60_atomic_trainer_activation_contract_closure.md`.

## 5. 명시적 비범위

다음은 현재 착수 금지다.

- §2 canonical 경로의 candidate/threshold를 scene-specific하게 튜닝하는 일 (box/cylinder/sphere/thin_slab처럼 shape별 예외를 만드는 것 포함)
- gap 보간, 임의 candidate 생성, 또는 closure만을 위한 낮은-evidence edge/cycle 강제 선택
- representative cap 증가로 evidence 부족을 우회하는 일
- optimizer/trainer/renderer/checkpoint 통합 범위 확대(worklog 59/60의 좁은 예외: 이미 append된 uncertain row를 `TorchGaussianModel.optimizer`에 활성화하고, `TorchOSNGSTrainer.activate_and_train_uncertain_step()`을 통해 실제 render+loss로 row-level 격리 학습하는 것까지만 승인됐다 -- `train()`의 실제 iteration 루프·CLI·checkpoint schema로의 배선은 여전히 금지)
- global chart ranking·selection 또는 conflict resolution
- 불완전한 false-hole evidence를 이용한 자동 topology 확정
- benchmark artifact만으로 visible surface 품질이 해결되었다고 선언하는 일

## 6. Training core parity (worklog 62, §2와 독립)

Baseline과의 render/scale/anisotropy 격차(§2와 무관, `output/osn_gs_scene`의 iteration 600 이후 극단적 anisotropy 성장 및 3k+ screen-size prune 폭주의 실제 원인)를 실제 baseline 코드와의 lockstep parity harness로 추적해 `osn_gs/data/colmap_scene.py`의 데이터 로더 결함 2건을 확정·수정했다:

1. `camera_fovs()`가 downscale된 렌더 해상도로 FoV를 재계산하고 있었다(baseline은 COLMAP 원본 해상도에서 1회만 계산하고 이후 절대 건드리지 않음) — 원본 해상도 고정으로 수정.
2. Ground-truth 이미지 resize filter가 `BILINEAR`였다(baseline은 Pillow 기본값 `BICUBIC`) — `BICUBIC`으로 수정.

Fix 후 lockstep 재검증: step 1은 float32 noise 수준까지 일치, step 600(ADC 비활성화, population 통계만) anisotropy median이 baseline과 거의 일치(1.58 vs 1.59) — fix 전 real training에서 관측된 극단적 격차가 사라짐. Surface reconstruction/reliability 코드는 미변경. 자세한 내용은 `docs/worklogs/62_graphdeco_lockstep_training_parity.md` 참고.

**Worklog 63 addendum**: 위 fix를 real 3k production 학습(2900/3000/3100)에 적용했으나, iteration 3100 anisotropy median이 fix 전후 사실상 동일(35.06→35.68, baseline 5.46)하고 screen-size prune도 줄지 않았다(오히려 소폭 증가) — **완료 기준 미충족.** loader 결함은 lockstep 조건에서의 확인된 최초 발산 원인 중 하나일 뿐, 실사용 규모 anisotropy 폭주의 지배적 원인은 아니었다. 자세한 내용은 `docs/worklogs/63_fixed_loader_3k_production_replay.md` 참고.

**Worklog 64 addendum(지배적 원인 확정 및 fix)**: OSN-GS의 유일한 Gaussian 초기화 경로(로컬 PCA planar-surfel)가 `normal_scale = tangent_scale * 0.04`로 모든 새 Gaussian의 iteration-0 anisotropy를 설계상 ~25로 고정하고 있었다 — 이것이 실제 지배적 원인이다. `TorchPipelineConfig.gaussian_initialization_mode`("baseline_compatible" 기본값 / "covariance_knn" experimental)를 신설해 surface construction용 covariance(미변경)와 모델 자신의 학습 가능한 scale/rotation init(신규, Graphdeco의 `create_from_pcd`와 텐서 단위로 동일한 등방 init)을 분리했다. 3-way parity harness로 baseline_compatible이 step 0(anisotropy 정확히 1.0)부터 step 600(ADC 후 population/split 후보 규모)까지 baseline에 근접함을 확인했다. `initialize_deferred` 스케줄은 surface reconstruction 불변 유지를 위해 의도적으로 이 플래그 영향 밖에 둔다. 자세한 내용은 `docs/worklogs/64_gaussian_initialization_parity.md` 참고.

**Worklog 65 addendum(실사용 3k production 검증)**: `baseline_compatible`을 실제 3k 학습(iteration 0/600/2900/3000/3100)으로 검증했다. anisotropy median 35.77(covariance_knn)→3.66(baseline_compatible, baseline 5.49)로 회복, min-scale collapse도 baseline보다 낮은 0.027%로 개선, chart materialization 수 감소(90→11 @2900)는 baseline PLY를 동일 파이프라인에 돌린 참조값(3~8 regions)과 비교해 진짜 over-segmentation 완화로 확인했다(covariance_knn은 그 참조값의 20~60배). 렌더 품질은 매 checkpoint에서 일관되게 baseline 방향 개선이나(PSNR +0.3~0.4) 격차를 완전히 닫지는 못한다. **다만 iteration 3100의 screen-size prune 건수는 224,164→233,178로 줄지 않았다** — anisotropy와는 별개로 opacity-reset 이벤트 자체에 내재한 현상으로 보이며, 아직 원인 미규명이다. 자세한 내용은 `docs/worklogs/65_baseline_compatible_3k_production_validation.md` 참고. **다음 자연스러운 후속은 opacity-reset 직후 screen-size prune storm 자체의 원인 추적이다(미착수).**

## 7. 현재 검증 상태와 알려진 위험

Repository-wide pytest 최신 기준(worklog 64, worklog 65는 코드 변경 없는 replay라 재실행 안 함): `783 passed, 1 skipped, 1 warning, 18 subtests passed`. §2.3의 negative-control 표와 함께 매 라운드 재확인한다.

알려진 위험:
- §2.2 표의 real 3k/10k closed-loop 부재는 미해결이며, §2.4 후보(candidate 생성 밀도) 전에는 착수 승인이 없다.
- §2 파이프라인의 direct/reverse tangent quality 비교는 scene 전체 단위 선택이라, 한 region의 수정이 다른 region의 결과를 바꿀 수 있다(worklog 53에서 실측). 이 구조를 변경하는 작업은 반드시 §2.3 negative-control 전체 재확인을 동반한다.
- §3(Isolated Boundary-first)은 재개 시 이 문서를 먼저 갱신하고 시작한다 — 현재는 활성 작업자가 없다.
- **`output/osn_gs_scene/*/checkpoint.pt`가 세션 사이 완료된 학습 run으로 교체됐다** — worklog 45-60이 참조하던 real 5k region 130/141 등 특정 region ID/개수는 더 이상 현재 checkpoint와 일치하지 않는다. 앞으로 real-checkpoint 수치를 보고할 때는 항상 그 라운드에서 실측한 값을 쓰고, 과거 worklog의 고정된 숫자와 다르다고 회귀로 오인하지 않는다.
- worklog 61에서 real 3k(cap 2048)가 `torch_gaussian_surface_region_formation.py`(2026-07-30, worklog 111-123)의 사전 존재 `KeyError`(`bridge_by_pair[(a, b)]`가 `consensus_by_pair`에는 있는 pair를 못 찾음)로 crash함을 발견했다 — 새 checkpoint 내용이 노출한 latent 결함이며 이번 작업 범위 밖이라 미수정. 3k(cap 1024)는 정상 동작한다. Region formation을 다루는 다음 작업자는 이 결함을 먼저 재현·수정 여부를 판단한다.
- **Known issue (worklog 64, 미수정)**: `gaussian_initialization_mode="baseline_compatible"`은 `_initialize_canonical`(production 기본 "initialize" 스케줄)에만 적용되고, `initialize_deferred`(`adc_post_commit`/`disabled` 스케줄)는 의도적으로 이 플래그를 무시하고 항상 `covariance_knn` 방식의 planar-surfel 초기값을 쓴다 — 그 경로의 첫 post-ADC surface 재구성이 모델 자신의 scale/rotation을 유일한 orientation evidence로 재사용하기 때문이다. 즉 `visible_nurbs_update_schedule=adc_post_commit`이나 `disabled`로 학습하면, `gaussian_initialization_mode=baseline_compatible`을 지정해도 실제로는 covariance-KNN 초기화가 적용되는 **의미 불일치**가 있다. Production 기본 스케줄("initialize")에는 영향 없음. deferred 스케줄을 실제로 쓰는 작업이 생기면 이 불일치를 먼저 해소해야 한다(현재는 미수정, 미착수).

다음 작업자는 먼저 이 문서와 `docs/worklogs/README.md`의 최신 인덱스, 그리고 가장 최근 작업로그(현재 `docs/worklogs/65_baseline_compatible_3k_production_validation.md`)를 읽고 이어서 작업한다. 과거 방향의 세부 기록은 필요할 때 Git history로만 조회한다.
