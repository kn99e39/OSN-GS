# OSN-GS 문서 안내

## 현재 기준 문서

- [Urgent Work Master](Urgent_Work/OSN_GS_Urgent_Work_Master.md): 현재 방향, 활성 작업, 승인 경계의 단일 기준
- [작업로그 보존 정책](worklogs/README.md): 유지 중인 최소 검증 기록 목록
- [Architecture](architecture.md): 프레임워크 수준의 설계 결정
- [현재 구현 파이프라인](current_framework.md): 논문·연구 보조용으로 현재 동작하는 Gaussian–canonical visible-NURBS–ADC lifecycle과 구현 수식을 설명
- [NURBS Construction](nurbs_construction.md): NURBS 중간 표현과 구현 계약

## 현재 상태

**2026-08-10 기준 최신 요약은 [Urgent Work Master](Urgent_Work/OSN_GS_Urgent_Work_Master.md)를 단일 기준으로 본다.** 아래 항목은 이후 남겨둔 이력이며, canonical visible-NURBS 경로(§2)는 worklog 18(2026-07-30) 이후 `train.py`/benchmark의 유일한 production 경로다 — "isolated, dispatcher 미연결"이라는 아래 서술은 그 이전 상태다.

- Uncertain Gaussian의 proposal/append/ownership model foundation. optimizer, trainer, renderer, checkpoint 및 global selection 통합은 범위 밖이다.
- NURBS Construction benchmark는 depth-bearing 3D shell과 baseline-like flattened covariance를 기본 입력으로 사용한다. [Worklog 5](worklogs/5_nurbs_construction_synthetic_3d_gaussian_dataset.md)을 따른다.
- covariance-guided pairwise affinity 위에 consensus-aware surface-region candidate foundation을 추가했다. 이는 ordered boundary/builder/production path와 연결되지 않은 격리 진단 단계다. [Worklog 10](worklogs/10_consensus_aware_surface_region_formation_foundation.md)를 따른다.
- trimmed-component Jacobian test-health 부채를 해소했고, 최신 전체 pytest는 `568 passed, 1 skipped, 1 warning, 8 subtests passed`다. [Worklog 8](worklogs/8_trimmed_component_jacobian_test_health.md)와 [Worklog 10](worklogs/10_consensus_aware_surface_region_formation_foundation.md)을 따른다.

이전 실험·폐기 방향의 상세 작업로그는 작업 트리에서 제거했다. 필요한 경우 Git 이력으로 조회하며, 현재 결정을 위해 과거 로그를 canonical source로 사용하지 않는다.
## 2026-07-29 Boundary-first NURBS materialization 상태

- consensus-aware region formation의 real trained Gaussian 결과는 full sheet segmentation이 아니라 reliable-core-only core-island 추출로 해석한다. Worklog 11을 따른다.
- world-space half-edge와 ordered graph의 admissible closed outer loop는 canonical evaluable visible NURBS materialization adapter로 전달한다. dispatcher, builder, production path에는 연결하지 않는다. Worklog 14을 따른다.
- 최신 전체 pytest: 577 passed, 1 skipped, 1 warning, 8 subtests passed.
- Worklog 17: [Canonical Tangent Frame Invariance Repair](worklogs/17_canonical_tangent_frame_invariance_repair.md) — Gaussian-only smooth_curved_sheet의 rotation/scale/shuffle/sign-equivalence 안정성 수정 및 검증.

## 2026-07-30 canonical visible NURBS 학습 통합

- `train.py`의 visible NURBS 초기화·주기적 재구축·file/stream payload는 이제 `construct_visible_nurbs_from_gaussians` 하나만 사용한다. `legacy`, `voxel_patch_stage1`, IDW/local split fallback과 해당 CLI는 제거했다.
- 대규모 점군은 deterministic voxel-center 표본(`canonical_construction_max_points`, 기본 2048)에서 canonical topology를 구축하고 ownership/UV/covariance frame을 전체 Gaussian으로 전파한다.
- 지원되는 curved sheet의 실제 trainer 1 iteration은 통과한다. 현재 canonical 범위 밖인 로컬 `DATASET` 복합 장면은 `review_required`로 fail-closed하며 fallback을 사용하지 않는다.
- 상세 근거와 남은 production blocker는 [Worklog 18](worklogs/18_canonical_visible_nurbs_training_integration.md)를 따른다.
- Worklog 18 최종 repository-wide pytest: `570 passed, 1 skipped, 2 warnings, 8 subtests passed in 153.38s`.
## 2026-07-30 ADC 동기화 canonical visible NURBS 실험

- 기본 `initialize` 스케줄은 그대로 유지한다. 실험 플래그 `--visible_nurbs_update_schedule adc_post_commit`은 초기 NURBS 없이 Gaussian 학습을 시작하고 구조적 ADC와 Gaussian optimizer commit 뒤에만 detached canonical 재구축한다.
- 실패/review/0 surface는 stale patch·optimizer·visible binding을 제거한다. clone/split/prune/checkpoint를 통과하는 stable Gaussian ID와 event별 sample/full/opacity coverage·fingerprint·runtime JSONL 진단을 추가했다.
- controlled multi-ADC 대조 실험은 Gaussian trainable tensor bitwise equality를 포함해 통과했다. 실제 `DATASET` 6-iteration/3-event CUDA 실험도 실행했다. 모든 bounded canonical sample은 `no_admissible_region`으로 fail-closed했고 stale NURBS 없이 Gaussian ADC는 계속됐다. cap sensitivity(512/1024/2048)도 같은 결론이었다.
- 최신 repository-wide pytest: `578 passed, 1 skipped, 1 warning, 8 subtests passed in 132.85s`. 실데이터 CUDA ADC는 run-to-run bitwise 재현적이지 않아 model equality는 controlled CPU 대조로만 보장한다. 최종 판정은 `PARTIALLY_SUPPORTED`다.
- 상세 구현·검증·남은 위험은 [Worklog 20](worklogs/20_adc_synchronized_canonical_visible_nurbs_experiment.md)을 따른다. 노트북 Train 셀은 [Worklog 21](worklogs/21_notebook_canonical_adc_nurbs_schedule.md)처럼 현재 canonical 경로와 `adc_post_commit` 스케줄을 기본으로 전달한다.

## 2026-07-30 WebRenderer Gaussian 진단

- `WebRenderer`는 PLY와 training stream 양쪽에서 certain/uncertain reliability, confidence, canonical surface ownership, NURBS patch ID를 시각화한다.
- field 없는 기존 Graphdeco PLY는 계속 로드되며 diagnostic mode에서는 중립 회색으로 나타난다. 상세 계약과 검증 한계는 [Worklog 22](worklogs/22_webrenderer_gaussian_diagnostics.md)을 따른다.

## 2026-07-31 Training stream keepalive

- trainer WebSocket client와 loopback stream server의 `ping_timeout`은 full Gaussian JSON snapshot 전송 중 기본 20초 deadline으로 연결이 끊기지 않도록 120초로 명시한다. ping 주기는 websockets 기본값을 유지한다.
- Train 셀의 5초 live monitor는 최신 일반 출력과 별도로 가장 최근의 detailed trainer 로그(`OSN-GS timing`, ADC, surface/NURBS 상태 또는 `detailed`/`diagnostic` 표기)를 한 줄 유지해 함께 표시한다. 마지막으로 관측한 `gaussians=` 값도 상태로 보존해, 이후 timing 로그가 출력돼도 Detailed log 끝에 계속 표시한다.

## 2026-07-30 Canonical reconstruction GPU synchronization optimization

- Full-cloud intrinsic reliability의 per-Gaussian CUDA scalar synchronization을 vectorized GPU mask와 bulk metadata transfer로 교체했다. 동일 real CUDA event는 38.146s에서 5.675s로 약 6.7x 단축됐다. stage profile, GPU power/temperature 해석, 검증 및 남은 greedy mode-selection 병목은 [Worklog 26](worklogs/26_canonical_reconstruction_gpu_synchronization_optimization.md)을 따른다.
- Mode-aware selection의 mode별 medoid 거리 138,766건을 하나의 bulk GPU→CPU transfer로 바꾸되 Torch aggregate와 CPU tie-break를 보존했다. v3 exact replay에서 candidate 및 FPS 선택 순서가 같고 median은 4.278s에서 2.324s로 단축됐다. 실패한 vectorized/NumPy 대안과 native splitter의 착수 기준은 [Worklog 27](worklogs/27_mode_aware_selection_phase2_exact_optimization.md)을 따른다.

## 2026-07-30 Native exact splitter gate

- C++ CPU native splitter prototype은 2.533ms까지 단축됐지만 v3 replay에서 첫 mode assignment가 달라 production에 채택하지 않았다. 현재 Python exact backend를 유지하며, 원인과 재시도 조건은 [Worklog 28](worklogs/28_native_exact_cell_splitter_gate.md)을 따른다.
## 2026-08-03 Candidate-local continuation 및 physical termination 감사

- Worklog 40의 region-pair global continuation verdict는 sphere seam suppression에는 유효하지만 candidate-local certificate가 아니며, mixed-relation pair에서의 false suppression 안전성은 아직 증명되지 않았다. 현재 typed provenance와 candidate-scale 전달 보정은 유지하되, global verdict의 production 대체는 보류한다.
- 실제 3k/5k/10k replay에서 evidence-based R1은 2~3 region, R2는 124~138 region으로 분리됐다. 그러나 두 candidate waterfall의 normalization/집계 수치가 아직 일치하지 않으므로, Box face 4 repair나 threshold 변경 전에 stable-ID 기준 trace를 통일해야 한다. 자세한 근거는 [Worklog 41](worklogs/41_candidate_local_continuation_certificate_and_physical_termination_audit.md)을 따른다.
## 2026-08-03 Representative-sector dual-scale lineage gate

- Representative-sector termination neighborhood를 read-only dual-scale replay로 검증했다. 동일 representatives/reliability/regions/accepted topology/downstream config에서 `equivalent_tangent_scale`와 `candidate_scale`만 바꿔 평가한다.
- Candidate branch는 모든 focused fixture에서 accepted-neighbor recall `1.0`, classified false support `0`, branch regression `0`을 기록했다. Box face/corner behavior는 124 raw/normalized, 110 physical, closed 5, open 2, branch 0, materialized 5로 명시 보고했다.
- 최신 targeted validation은 `11 passed`, 전체 suite는 `713 passed, 1 skipped, 1 warning, 8 subtests passed`다. 상세 lineage와 남은 continuation-path 위험은 [Worklog 43](worklogs/43_representative_sector_dual_scale_lineage_gate.md)을 따른다.
## 2026-08-03 Full-cloud candidate lineage

- Real 3k/5k/10k에서 full-cloud continuation 포함 production candidate lineage를 stable-ID와 candidate ID 양쪽으로 연결했다. 기존 136/153, 167/181, 106/121 차이는 production 결함이 아니라 `trace_physical_termination_gates.py`가 continuation-backed production candidate를 세기 전에 representative-sector local-neighbor gate로 먼저 탈락시키던 diagnostic trace 결함이었다.
- Corrected physical count는 3k/5k/10k 각각 153/181/121로 production waterfall과 exact match한다. Candidate-scale accepted-neighbor recall은 세 checkpoint 모두 1.0이며 footprint recall은 0.0523/0.0171/0.0184에 그쳤다.
- Frozen replay helper도 full-cloud continuation replay에서 `candidate_scale`을 전달하도록 맞췄다. Focused tests는 `28 passed`, 전체 suite는 `714 passed, 1 skipped, 1 warning, 8 subtests passed`다. 상세 근거는 [Worklog 44](worklogs/44_full_cloud_candidate_lineage_and_trace_count_alignment.md)을 따른다.


## 2026-08-03 Real physical boundary coverage

- Directed compatibility gate에서 target `boundary_direction` 부호를 oriented vector로 비교해 Box 6번째 face와 real 5k의 실제 successor edge를 거부하던 결함을 수정했다. Source tangent는 directed traversal에 계속 사용하고, target tangent는 방향 없는 boundary line alignment로 평가한다.
- Before/after: Box 5/5 -> 6/6, real 5k 0/0 -> 2/2 closed/materialized. Real 3k/10k는 edge 수는 늘었지만 closed는 0으로 남아, 남은 병목은 candidate coverage/fragmentation이다.
- 검증: focused `64 passed`, repository-wide `715 passed, 1 skipped, 1 warning, 8 subtests passed`. 상세 근거는 [Worklog 45](worklogs/45_real_physical_boundary_coverage_and_compatibility_repair.md)를 따른다.
## 2026-08-03 Real 3k/10k physical candidate chain 감사

- [Worklog 46: real 3k/10k physical candidate chain 원인 감사](worklogs/46_real_3k_10k_physical_candidate_chain_cause_audit.md)
- 3k/10k는 raw physical candidate 단계에서도 closed chain이 없고, 5k의 두 closed chain은 raw/normalized 모두 유지된다. candidate intermediary bridge를 허용하면 짧은 loop가 생기지만 Y-branch safety를 깨므로 production에 적용하지 않았다.
## 2026-08-03 Boundary-proximate no_gap local evidence

- [Worklog 47: boundary-proximate no_gap local evidence 복구](worklogs/47_boundary_proximate_no_gap_local_evidence_repair.md)
- `no_gap`은 모든 mode의 occupancy가 아니라 same-mode local coverage로만 승인한다. local parallel/crease/competing support가 이를 막으면 physical termination으로 승격하지 않고 typed nonphysical/ambiguous state로 보존한다.

## 2026-08-03~04 Full-cloud continuation 파이프라인 6단계 소진 감사 (worklog 48-53)

Boundary candidate 전달 경로(no_gap 분류 → representative selection → single-radius 신뢰 → raw-to-representative 축약 → directed matching → self-intersection 인식)를 순서대로 감사했다. 매 단계에서 실제 결함을 찾아 좁게 수정했지만, real 3k/10k의 closed-loop 개수는 끝까지 0으로 남았다 — 자세한 표는 [Urgent Work Master §2.2](Urgent_Work/OSN_GS_Urgent_Work_Master.md)를 따른다.

- [Worklog 48](worklogs/48_candidate_local_smooth_continuation_repair.md): no_gap의 candidate-local fold/gap-crossing leak 수정(accepted-topology 경로 vs 직선거리 비율).
- [Worklog 49](worklogs/49_representative_selection_boundary_evidence_recovery.md): representative selection의 FPS 예산 경쟁 손실을 안전한 swap-in으로 복구(5차 반복 끝에 box region_count 회귀 전부 해결).
- [Worklog 50](worklogs/50_multi_scale_local_termination_persistence.md): 단일 `4x candidate_scale` 반경의 원거리 support 과신을 scale-persistence 인증으로 수정.
- [Worklog 51](worklogs/51_raw_full_cloud_boundary_evidence_audit.md): raw full-cloud에 representative 축약으로 잃은 boundary chain이 있는지 감사, boundary-anchor sidecar 가설 기각.
- [Worklog 52](worklogs/52_directed_matching_objective_audit.md): worklog 51의 "Hungarian 경쟁 탈락" 주장을 정정 — edge는 실제로 matching에 포함돼 있었고, 진짜 원인은 topology 불가능 또는 evidence 열세.
- [Worklog 53](worklogs/53_downstream_valid_directed_matching_repair.md): region 56의 진짜 결함(downstream-invalid 2-cycle의 capacity 낭비)을 수정하고, 그 수정이 노출한 direct/reverse tangent quality 비교의 self-intersection 인식 결함도 함께 수정. 최신 전체 pytest: `720 passed, 1 skipped, 1 warning, 8 subtests passed`.

**결론**: 남은 병목은 candidate evidence의 밀도/위상 자체이며 파이프라인 결함이 아니다. 다음 단계 후보는 candidate 생성 밀도 자체의 근본 원인 조사이며, 아직 착수 승인은 없다.

## 2026-08-04 Eligible boundary downstream bridge

- Region별 visible-boundary status를 5-state fail-closed 계약으로 분리하고, `eligible_closed_boundary` component만 canonical visible NURBS materialization과 downstream continuation bridge에 전달한다.
- Real 5k의 region 130/141 두 surface만 continuation domain까지 연결되며 candidate는 0이다. 3k/10k는 eligible surface가 없어 downstream attempt가 없다.
- 상세 계약·검증·한계는 [Worklog 54](worklogs/54_visible_boundary_eligibility_and_unsupported_state_contract.md), [Worklog 55](worklogs/55_eligible_boundary_downstream_integration.md), [Worklog 56](worklogs/56_eligible_visible_surface_occluded_candidate_production_bridge.md) [Worklog 57](worklogs/57_occluded_candidate_safe_uncertain_proposal_production_integration.md)는 continuation-domain candidate를 safe uncertain proposal artifact까지 연결한 현재 model-only 경계를 기록한다.를 따른다.

## 2026-08-10 Full-region face membership-incidence 최종 판정

- Worklog 87의 candidate-anchored stable-ID daisy-chain은 최종 architecture 판정 근거로 기각했다.
- Worklog 88의 candidate-independent 방향은 유지하지만 global PCA rotation과 induced-subgraph largest outer-face 선택은 판정 근거에서 제거했다.
- [Worklog 89](worklogs/89_full_region_face_membership_incidence_final_go_no_go.md)은 기존 local normal/tangent frame으로 full-region observed face를 먼저 복원한 뒤 unit-supported face incidence에서 모든 boundary loop를 계산한다.
- 7-region 결과는 cut recoverable 0.170%(coherent 대비 0.193%), valid_supported 0%, mixed/seam-only 0%로 **실제 최종 NO-GO**다. Region→Charts canonical 통합은 하지 않았으며 visible-constructor boundary redesign은 종료한다.

## 2026-08-10 Surface-topology root-cause attribution

- [Worklog 90](worklogs/90_surface_topology_root_cause_attribution.md)은 Worklog 89가 full-region face topology를 만들지 못한 167 coherent unit을 center graph와 기존 covariance footprint로 읽기 전용 분류했다.
- primary evidence 91.44%가 `MULTILAYER_OR_VOLUMETRIC`, center undersampling은 4.59%, relation false negative는 0%였다. footprint representation으로 valid local complex를 추가 조사할 근거도 5.47%에 그쳤다.
- 따라서 production constructor나 Worklog 82 relation을 바꾸지 않는다. 다음 원인 조사는 training/ADC의 depth·visibility·covariance·layer distribution을 대상으로 하며 boundary heuristic을 재개하지 않는다.

## 2026-08-17 Surface-topology temporal + lineage attribution

- [Worklog 91](worklogs/91_surface_topology_temporal_lineage_attribution.md)은 Worklog 90의 `MULTILAYER_OR_VOLUMETRIC` evidence를 center-only PCA depth clustering(각 Gaussian 자신의 covariance orientation과 독립)과 covariance-only ambiguity로 분리하고, `baseline_compatible` checkpoint 5개(600/2900/3000/3100/final)에 걸친 ADC lineage(stable-ID 신규/소멸)와 실제 train camera 161개 기준 가시성/depth 분리를 측정했다.
- true-center multilayer 비율은 checkpoint 2900~final 4개 시점 전부에서 92.5%~95.9%로 안정적이었고, dominant competing-layer 구조 대부분은 초기 densification 파동에서 형성돼 이후 pruning에서도 유지됐다. 두 dominant layer는 161개 중 14개 카메라에서 동시에 보이며 평균 depth 분리 0.277이다.
- **Decision A**(당시 판정, Worklog 92가 재판정): true center-distribution multilayer가 압도적이고 안정적이므로, covariance frame이 부적합한 표현이라는 가설(B)은 기각한다. 다음 조사 대상은 boundary가 아니라 ADC/densification/pruning이 이 구조를 왜 만들고 유지하는지다.

## 2026-08-17 Global-SVD confound 제거 후 최종 center-geometry 귀속

- [Worklog 92](worklogs/92_local_center_geometry_attribution.md)는 Worklog 91이 chart unit 전체에 SVD 평면 하나를 적합해 곡률 있는 단일 표면을 여러 depth band로 오판할 수 있다는 confound를 제거했다. 각 node의 local kNN 이웃만으로 diagnostic-only 평면을 적합하고, gap이 양쪽 side 내부 spread보다 1.5배 이상 커야 하는 silhouette 검증과 공간적 persistence gate를 추가해 `LOCALLY_SINGLE_CURVED_SHEET`/`LOCALLY_THICK_UNIMODAL_SHEET`/`TRUE_PERSISTENT_TWO_LAYER`/`TRUE_PERSISTENT_MULTI_LAYER`/`SPARSE_SATELLITE_OR_OUTLIER`로 재분류했다.
- 실측 결과 `TRUE_PERSISTENT_TWO_LAYER`는 checkpoint 4개 전부에서 1.08%~2.06%, `TRUE_PERSISTENT_MULTI_LAYER`는 0%로 소수였고, `LOCALLY_THICK_UNIMODAL_SHEET`(62~70%)와 `LOCALLY_SINGLE_CURVED_SHEET`(25~32%)가 대부분을 차지했다. Persistent layer는 population 2~7개의 소규모 구조뿐, Worklog 91의 1378-vs-3 같은 대규모 split은 재현되지 않았다.
- **Decision E**: Worklog 91의 global-SVD 진단이 multilayer를 과다 귀속했으므로 Worklog 91 근거로 ADC를 architecture target으로 채택하지 않는다.
