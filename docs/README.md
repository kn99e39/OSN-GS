# OSN-GS 문서 안내

## 현재 기준 문서

- [Urgent Work Master](Urgent_Work/OSN_GS_Urgent_Work_Master.md): 현재 방향, 활성 작업, 승인 경계의 단일 기준
- [작업로그 보존 정책](worklogs/README.md): 유지 중인 최소 검증 기록 목록
- [Architecture](architecture.md): 프레임워크 수준의 설계 결정
- [현재 구현 파이프라인](current_framework.md): 논문·연구 보조용으로 현재 동작하는 Gaussian–canonical visible-NURBS–ADC lifecycle과 구현 수식을 설명
- [NURBS Construction](nurbs_construction.md): NURBS 중간 표현과 구현 계약
- [Worklog 138: Scale-Separated Visible Surface Representative](worklogs/138_scale_separated_visible_surface_representative_closure.md): raw Visible Surface와 retained-only NURBS representative의 scale-separated continuation audit 및 부분 feasibility 판정 ([output](../output/confirmed/138_scale_separated_visible_surface_representative/README.md))
- [Worklog 139: Physical-Chart-Constrained Surface Representative](worklogs/139_physical_chart_surface_representative_closure.md): frozen physical chart를 보존하는 graph B-spline representative, controlled continuation A/B 및 architecture 판정 ([output](../output/confirmed/139_physical_chart_surface_representative/README.md))
- [Worklog 146: 산출물 폴더 Worklog 번호 정규화](worklogs/146_output_worklog_prefix_normalization.md): `output/` 및 `output/confirmed/`의 Worklog 산출물 루트에 대응 번호 prefix를 적용하고 demo/문서 참조를 동기화
- [output/ 폴더 관리 규약](output_folder_conventions.md): gitignore된 `output/`의 번호 매김·`confirmed/` 이동·preview_png 통합 규칙(참조용 유일 문서)
- [Worklog 126: WL123 Fixed Observed/Occluded Gaussian Visualization](worklogs/126_wl123_fixed_observed_occluded_gaussian_visualization.md): WL125의 고정 Gaussian visualization 계약으로 생성한 실제 Original Scene / Observed/Occluded 결과
- [Worklog 127: Novel-View Observed/Occluded Inspection Correction](worklogs/127_novel_view_observed_occluded_inspection_correction.md): query camera set 밖 novel inspection pose에서 다시 만든 현재 Observed/Occluded 결과
- [Worklog 128: Real-scene Parametric Surface Continuation Feasibility Demo](worklogs/128_real_scene_parametric_surface_continuation_feasibility_demo.md): canonical 경로와 분리된 WL127 Visible Surface holdout/continuation meeting demo 및 negative verdict ([output](../output/128_demo_parametric_surface_continuation/README.md))
- [Worklog 129: Corrected First-order Parametric Continuation Revalidation](worklogs/129_corrected_first_order_parametric_continuation_revalidation.md): WL128의 underscaled continuation defect를 historical baseline으로 보존한 corrected analytic Taylor Arm B 재검증 및 negative verdict ([output](../output/129_demo_corrected_first_order_parametric_continuation/README.md))

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

## 2026-08-18 Latent midsurface 회수 가능성 귀속

- [Worklog 93](worklogs/93_latent_midsurface_recoverability_attribution.md)은 Worklog 92의 `LOCALLY_THICK_UNIMODAL_SHEET`/`LOCALLY_SINGLE_CURVED_SHEET`(Worklog 90 `MULTILAYER_OR_VOLUMETRIC` evidence 대부분)가 recoverable latent 2D midsurface를 갖는지 center position만으로 측정했다. Local kNN diagnostic 평면에 quadratic curvature 추정을 추가하고, raw 대 diagnostic-projected position-only adjacency로 manifold topology를 비교했다.
- 4개 checkpoint(2900/3000/3100/final) 전부에서 manifold 개선 비율 99.8~100%, curvature 보존 비율 86.6~91.9%, valid face incidence가 거의 2배(30.5~36.3%→63.0~72.7%)로 늘고, support band fidelity 86.0~91.5%로 회수된 표면이 관측 evidence를 벗어나지 않았다.
- **Decision A: LATENT_SURFACE_RECOVERABLE** — raw Gaussian center 자체가 잘못된 geometry representation이며, 다음 architecture target은 boundary 추출 이전의 명시적 latent-surface evidence representation이다.

## 2026-08-18 Bounded surface-evidence representation architecture gate

- [Worklog 94](worklogs/94_surface_evidence_representation_gate.md)는 root-cause 진단에서 architecture 결정으로 전환하는 단일 배치다. RAW_CENTER_BASELINE/CENTER_LATENT_SURFACE/COVARIANCE_SURFEL_SUPPORT/HYBRID_LATENT_PLUS_SUPPORT 4개 representation을 같은 7개 real region·같은 unmodified Worklog 89 constructor 체인으로 fallback 없이 비교했다.
- Checkpoint 2900/final 실측: COVARIANCE_SURFEL_SUPPORT는 RAW_CENTER_BASELINE과 수치가 정확히 동일했고, HYBRID는 CENTER_LATENT_SURFACE와 거의 동일했다. Latent 기반 두 representation은 recoverable evidence를 2.8~5.1배 늘렸지만 valid_supported는 네 representation 전부 0.2% 미만이었고 unresolved는 83.7~88.0%로 여전히 압도적이었다. Latent representation은 held-out p95도 2~6.9배 악화시켰다.
- **Decision 3**: 네 representation 모두 coherent evidence 대다수를 unresolved/unsafe로 남긴다. Constructor-level 재설계를 중단하고, 다음 architecture target은 training 중 visible geometric evidence 생성 자체(upstream)로 옮긴다.

## 2026-08-19 2DGS surface-evidence 병렬 아키텍처 실험 (exp/2dgs-nurbs-surface-evidence, 병합 안 함)

- [Worklog 95](worklogs/95_2dgs_surface_evidence_branch.md)는 Worklog 94 Decision 3이 지목한 upstream 방향(training 중 visible geometric evidence 생성 자체)의 첫 후보로 2DGS(arXiv:2403.17888v3, `hbb1/2d-gaussian-splatting` @ 335ad61, `diff-surfel-rasterization` @ e0ed020)를 별도 브랜치에 이식하고, **vanilla `baseline_compatible` 3DGS와 같은 downstream constructor 계약**으로 비교했다.
- 이식 충실도: primitive는 `scale_dim=2`로 세 번째 scale 텐서 자체가 없고, CUDA rasterizer는 upstream과 byte-identical(perspective-correct ray-splat intersection, object-space low-pass filter 포함)이며, depth distortion/normal consistency는 공식 `train.py` staging(3000/7000)을 재조정 없이 30k 완주했다. paper와 official code가 갈리는 두 지점(depth distortion의 depth 좌표, normal consistency의 aggregation)은 명시 기록 후 **하나의 정합적인 official 형식**만 구현했다.
- 실측(648x420, 동일 2M primitive 예산, 양 arm 모두 cap 미도달): 2DGS는 primitive 40% 적게·학습 25% 빠르게 쓰면서 held-out PSNR을 0.87 양보한다(29.11→28.24). 구조 evidence는 크게 개선 — normal coherence 0.748→**0.966**, `needle_like` 16.9%→**1.75%**, ADC anisotropy 38,979→**22.5**, affinity `same_surface` 2.63%→**12.56%**, constructor 수용 evidence **37.4배**, usable curve network region 3→**43**, structural curve 9→**153** segment.
- **그러나 downstream NURBS 계약은 여전히 미충족이다**: `valid_supported`가 처음으로 0이 아닌 값(vanilla 0 → 2DGS 35)이 나왔지만 `unresolved`는 오히려 67.4%→89.2%로 올랐고 materialization rate는 0.077%다. coherent evidence 대비 회수율은 0.44%→0.087%로 **떨어졌다** — evidence는 49배 늘었는데 회수 비율은 5배 나빠졌으므로, 다음 병목 후보는 evidence 생성이 아니라 **cut-boundary 회수 단계의 scaling**이다.
- 이 과정에서 기존 OSN-GS 결함 하나를 고쳤다: `--adc_max_gaussians`가 걸릴 때 남은 예산을 clone에 먼저 전부 배정해 split이 **완전히 사라지던** 문제(demand 비례 분배 + 최고 gradient 우선 선택으로 수정, `tests/test_density_control_budget.py`). 이 결함 때문에 첫 비교 시도는 폐기했다.
- 브랜치는 병합하지 않는다. 2DGS를 NURBS 성공 방향으로 튜닝하지 않았다.

## 2026-08-19 2DGS Coverage-first Surfel Subset partition — 신규 canonical 방향 (arch/2dgs-coverage-first-surface)

- 사용자 지시로 `voxel-surface-regions`의 Worklog 105/106(volumetric 3DGS covariance-minor-axis normal 기반 coverage-first partition) 방향을 중단하고, 이미 검증된 `exp/2dgs-nurbs-surface-evidence` 브랜치의 실제 2DGS surfel primitive를 canonical surface evidence로 채택했다. 신규 브랜치 `arch/2dgs-coverage-first-surface`(base `exp/2dgs-nurbs-surface-evidence`@`54b72c2`)에서 진행하며 `voxel-surface-regions`에는 병합하지 않는다.
- [Worklog 96](worklogs/96_2dgs_coverage_first_surfel_partition.md): 신규 `torch_surfel_surface_orientation.py`가 surfel의 intrinsic `t_u`/`t_v`/`t_w`(학습된 rotation quaternion의 열)를 그대로 읽는다 — eigen-decomposition·covariance 구성이 전혀 없음을 AST로 강제 검증. Coverage-first partition 구현은 `voxel-surface-regions`에서 코드 변경 없이 이식(orientation 타입 힌트만 구조적 Protocol로 완화, 3DGS/2DGS 어느 쪽 orientation이든 받되 어느 쪽도 import하지 않음). 신규 export 스크립트가 fail-closed로 volumetric checkpoint를 거부함을 실증.
- 학습된 2DGS checkpoint(Worklog 95가 만든 30k-iteration 결과)를 이 로컬 머신에서 찾지 못해 사용자에게 질의했고, **사용자가 이 머신(RTX 5080)에서 재학습을 선택**했다. `scripts/build_surfel_extension.bat`으로 벤더링된 surfel CUDA 확장을 빌드(21개 CUDA 테스트 skip→pass)하고, Worklog 95와 동일 설정으로 30k 재학습해 **최종 surfel 수(1,197,331 vs 1,193,268)와 held-out PSNR/SSIM(28.256/0.8997 vs 28.24/0.899)이 원본과 사실상 일치**함을 확인했다.
- **실측 결과(혼재)**: 정정된 3DGS 기준(Worklog 106, `osn_gs_scene/3000`) 대비 2DGS는 최대 subset 비율(82.94%→74.70%)과 normal-cut edge 비율(25.13%→22.12%)이 소폭 개선됐지만, singleton/fallback ownership 비율은 오히려 소폭 악화(2.09%→3.38%)했고 local unsigned normal agreement 분포는 3DGS와 거의 동일했다. 시각 검토 결과 최대 subset은 여전히 평평한 바닥과 굴곡진 산울타리(서로 다른 orientation)를 가로질러 이어져 있다. Architecture 판단은 이 배치에서 내리지 않는다 — 사용자가 `output/osn_gs_2dgs_coverage_first_subset_partition/`의 4개 view를 직접 검토한다.

## 2026-08-20 Region-level anti-chaining Surfel Subset partition (arch/2dgs-coverage-first-surface)

- [Worklog 97](worklogs/97_region_coherent_surfel_partition.md)은 Worklog 96이 밝힌 병리(intrinsic 2DGS normal로도 local pairwise connected component가 single-linkage chaining으로 74.70% 거대 subset을 만듦)를 **partition union rule 하나만** 바꿔 재실측했다. Primitive/local candidate graph는 완전히 동일 — sign-invariant orientation scatter `M_R=sum n_i n_i^T`로 region 전체의 concentration을 측정하고, 기존 `a=0.85`에서 대수적으로 유도한(새 자유 파라미터 없는) floor `(1+a)/2=0.925`를 병합 조건으로 추가했다.
- **실측: 최대 subset 비율 74.70%→21.20%, subset 수 58,646→104,548, region-coherence rejected merge 553,357개.** 옛 894,378-surfel 거대 subset이 31,564개로 분해됐고(최대 후손은 원래의 28.4%), 남은 최대 subset은 시각 검토상 진짜 평평한 바닥 하나. WL96 singleton 40,410개는 동일 local graph 때문에 전부 isolated fallback으로 그대로 남음. Coverage identity 100% 유지.
- **이 worklog도 architecture 성공/실패 판단을 내리지 않는다** — 사용자가 6개 review export view(`output/osn_gs_region_coherent_surfel_partition/`)를 검토한 뒤 다음 단계(subset-local Trust)를 결정한다.

## 2026-08-20 Discontinuity-first Surfel Subset partition — 부정적 실측 (arch/2dgs-coverage-first-surface)

- [Worklog 98](worklogs/98_discontinuity_first_surfel_partition.md)은 Worklog 97의 region-concentration 방식이 곡률 있는 테이블 옆면을 정상 normal 회전만으로 쪼갠다는 지적에 따라, local shape operator 기반 smooth-surface residual + positional/parallel-sheet 기준으로 union rule을 다시 교체했다.
- **합성 fixture에서는 성공**(원통형 밴드 180° 회전 유지, 크리즈 정확히 절단, 평행 시트 분리) — 그러나 **실제 scene 재실측에서는 실패**: 개별 edge 판정만으로는 dense kNN 그래프의 percolation을 막지 못해 **94.51%짜리 거대 subset**이 재발했다(WL97의 20.84%보다 훨씬 나쁨, WL96의 74.70%보다도 나쁨). 테이블은 더 이상 다리별로 쪼개지지 않지만 바닥·산울타리와도 분리되지 않는다.
- Checkpoint 유실(사용자의 `output/confirmed/` 정리 과정 추정)로 동일 설정 재학습 필요했고, 재현성 확인됨(surfel 수·PSNR 오차범위 내 일치).
- **이 worklog는 부정적 실측 결과를 정직하게 보고한다** — architecture 최종 판단은 사용자가 review export(`output/osn_gs_discontinuity_first_surfel_partition/`) 검토 후 결정한다.

## 2026-08-20 Interface-coherent Surfel Region merge — 혼재된 실측 (arch/2dgs-coverage-first-surface)

- [Worklog 99](worklogs/99_interface_coherent_region_merge.md)는 WL97(largest 20.84%, 과분열)과 WL98(largest 94.51%, percolation 재발)을 각각 최종 union rule로 쓰지 않고, "WL97의 안전하지만 과분열된 초기 region → region 간 interface 전체를 WL98의 미분 증거로 집계 평가 → 광범위하게 지지되는 매끄러운 접점만 merge"라는 2단계 파이프라인(`torch_interface_coherent_region_merge.py`)으로 교체했다.
- 구현 중 WL97 자체의 candidate edge 승인에 positional 검사가 없어 평행 시트를 그대로 초기 region으로 합쳐버린다는 것을 발견 — WL97에 opt-in 필드 `require_positional_continuity`(기본값 `False`, 기존 동작 불변)를 추가해 정정.
- 합성 fixture(원통 과분열 복원, 크리즈/평행시트/zigzag 체인 분리 유지, 단일 edge 지지 부족)는 전부 설계 의도대로 통과.
- **실제 scene 재실측(같은 checkpoint): 테이블 상판 곡면은 하나의 region으로 성공적으로 복원됐으나(다만 정확한 공로 귀속은 미확정), 배경(파티오+잔디+울타리)이 다시 53.86%짜리 거대 subset으로 합쳐졌다** — WL98(94.51%)보다는 낫지만 WL97(20.84%)보다는 나쁘다.
- 신규 focused 테스트 14개 + WL97 회귀 테스트 1개, 전체 regression 1144 passed 1 skipped. **혼재된 결과를 정직하게 보고 — architecture 판단 없음.**

## 2026-08-21 Region-conditioned bilateral interface Surfel Region merge — 뚜렷한 개선 (arch/2dgs-coverage-first-surface)

- [Worklog 100](worklogs/100_bilateral_interface_region_merge.md)은 Worklog 99가 재발시킨 percolation이 WL98의 `min(r_i->j, r_j->i)`(편측 허용) residual을 region merge라는 더 강한 주장에 그대로 재사용했기 때문이라는 가설을 검증했다. 나머지(초기화·candidate graph·지지/extent floor·threshold 공식·과반 0.5)는 전부 Worklog 99와 동일하게 고정한 채, per-edge 증거만 **region-conditioned**(자기 region 이웃만으로 local shape operator 재적합) + **bilateral**(양방향 모두 통과해야 smooth)로 교체했다(`torch_bilateral_interface_region_merge.py`).
- 구현 중 실제 버그 발견·수정: threshold를 매 라운드 재계산하면 완전히 균일한 크리즈(분산 0)가 median+MAD 붕괴로 100% smooth로 오분류됨을 합성 fixture로 발견 — WL98/99처럼 **한 번만, 큰(same-region 내부 포함) 모집단에서** 계산하도록 정정.
- **실제 scene 재실측(같은 checkpoint): 최대 subset 비율 53.86%(WL99) → 22.91%로 개선**, 초기화 자체의 20.62%에 근접. Accept된 interface가 6,051→1,742로 급감.
- **WL99의 patio→hedge 연결 lineage(5개 merge 전부)를 직접 추적**해 신규 bilateral 인증서로 재평가한 결과, **5개 전부 기각**(2개는 명시적 편측 지지)됨을 확인 — 실제로 두 seed가 최종적으로 분리됨도 확인.
- 신규 focused 테스트 14개, 전체 regression 1158 passed 1 skipped(+14). Worklog 99 문서의 coverage identity 표기 오탈자도 정정. **architecture 최종 판단은 이 배치에서 내리지 않는다.**

## 2026-08-21 Rejected-interface attribution 및 region-adaptive support 실험 — support starvation 기각 (arch/2dgs-coverage-first-surface)

- [Worklog 101](worklogs/101_region_support_starvation_attribution.md)은 Worklog 100의 낮은 merge 수(1,742/1,763,096)가 "conservative 초기화 → 작은 fragment의 same-region 이웃 부족 → support 불가 → 영원히 병합 못함"이라는 순환 의존 때문인지 측정했다. Rejected interface를 non-overlapping이 아닌 multi-label로 완전 귀속(`insufficient_region_support_*`, `interface_unique_support_failure`, `directional_residual_failure_*`, `bilateral_smooth_fraction_failure` 등)하고 region 크기별(1/2-4/5-8/9-16/17-32/33-64/>64) support 통계를 냈다.
- **측정: support만 해결되면 나머지 기하 테스트를 전부 통과했을 interface는 0.29%(5,186/1,761,354)뿐** — support 부족은 흔하지만(특히 singleton region은 구조적으로 0%) 대부분의 rejection은 interface 자체가 너무 작아서(unique surfel count 98.5%, extent 87.0% 위반) support와 무관하게 이미 기각 대상이다. **Support starvation은 material하지 않다고 판정.**
- 그럼에도 가설을 직접 검증하기 위해 region-adaptive support(고정 global k=8 마스킹 대신, 기존 `spatial_connect_spacing_multiplier`로 bounded된 same-region 지역 검색)를 구현·실측했다(`torch_region_adaptive_support_merge.py`, Worklog 100 모듈은 전혀 수정 안 함, merge threshold 전부 동일).
- **실제 scene 재실측: 최대 subset 비율이 22.91%(WL100)→42.13%로 거의 두 배가 되는 실질적 percolation 위험을 발견.** Worklog 100이 막았던 특정 patio↔hedge 연결(WL99의 5-merge lineage)은 여전히 차단됐지만, 산울타리(hedge) 내부에서 새로운 거대 병합이 발생했다(시각 검토로 확인).
- **두 독립적 증거(material하지 않음 + percolation 재발)가 같은 결론을 가리켜 adaptive support를 채택하지 않는다.** Worklog 100(`SUPPORT_MODE_FIXED_MASKED_KNN`)이 유일한 baseline으로 유지된다.
- 신규 focused 테스트 12개(FIXED 모드가 WL100과 완전히 동일함을 재현 테스트로 고정 포함), 전체 regression 1170 passed 1 skipped(+12). **architecture 최종 판단은 이 배치에서 내리지 않는다.**

## 2026-08-21 Maximal Visible Surface Components — 실제 scene 부정적 결과 (arch/2dgs-coverage-first-surface)

- [Worklog 102](worklogs/102_maximal_visible_surface_components.md)는 Worklog 97-101의 "conservative 초기화 → bilateral 증명 후 merge" 철학을 전면 교체해 "완전한 관측-가시 evidence에서 시작 → 명시적 관측/occlusion/불연속 증거가 있을 때만 CUT"하는 새 아키텍처(`torch_maximal_visible_connectivity.py`)를 구현했다. Canonical Phase-C observation 상태(`torch_observation_evidence.py`, 전혀 수정 안 함)를 그대로 재사용하되, 수백만 edge 규모를 위해 동일한 per-view 규칙을 벡터화하고 `classify_world_samples`와 완전히 동일한 출력임을 직접 검증했다.
- **곡률-오판 문제(지시 §5)를 실제로 발견·수정**: 처음 설계한 "두 endpoint 깊이의 선형보간"이 사실상 금지된 직선 3D chord와 구조적으로 동일하다는 것을 fixture를 돌리기도 전에 깨닫고, 카메라 자신의 화면-공간 depth를 RANGE(기하학적으로 필연적인 상하한, `edge_length` 기반) 판정으로 교체했다. WL98의 positional-offset 공식에서 실제 부호 버그도 발견·수정(이 모듈 자신의 복사본만; WL100은 지시대로 보존).
- 합성 fixture 8종(완전 가시 평면/곡면, occluder로 분리된 벽, occluded gap이 있는 같은 곡면, known free-space gap, 평행 시트, 진짜 discontinuity, 증거 없는 unobserved gap) 전부 통과, 신규 focused 테스트 14개.
- **실제 scene 재실측(161개 전체 train camera, 2DGS surfel rasterizer로 직접 depth 렌더링): 최대 component 비율 92.69%** — WL96(74.70%)보다 나쁘고 WL98(94.51%)에 근접한다. 관측 기반 CUT(occlusion/free-space)은 전체 cut의 1.4%뿐 발동했고(local 이웃 사이에는 실제 occluder/gap이 드묾), WL98 재사용 geometric cut은 광범위하게(29.3%) 발동했음에도 dense kNN 그래프의 percolation을 막지 못했다 — WL98이 이미 증명한 실패 모드가 완전히 다른 architecture에서도 재현됐다. Patio-측/hedge-측 seed가 동일 component로 확인(WL99/100이 막았던 연결이 재발), 테이블의 독립성도 사라졌다.
- 전체 regression 1184 passed 1 skipped(+14). **Architecture 성공을 주장하지 않는다 — 정직한 부정적 결과.**

## 2026-08-21 Positive Visible Adjacency — 혼재된 결과 (arch/2dgs-coverage-first-surface)

- [Worklog 103](worklogs/103_positive_visible_adjacency.md)은 Worklog 102의 92.69% percolation을 "connectivity-by-default 원칙의 실패"가 아니라 "그 원칙이 적용된 그래프의 실패"로 재해석해, spatial kNN을 순수 후보 관계 생성기로 격하하고 "양성 관측 지지가 있어야만 가시 인접성이 성립"(`torch_positive_visible_adjacency.py`)하는 것으로 반전했다. WL102의 `_per_view_status_codes`/`_project_to_camera`, WL98/102의 shape-operator/residual 기계를 그대로 import 재사용하고, `torch_observation_evidence.py`/`torch_maximal_visible_connectivity.py`는 전혀 수정하지 않았다(WL102는 baseline 비교로 그대로 재실행).
- 관계 상태는 7-way(`POSITIVE_VISIBLE_CONTINUATION`/`CUT_KNOWN_FREE_SPACE`/`CUT_OCCLUDED_DOMAIN`/`CUT_VISIBLE_GEOMETRIC_DISCONTINUITY`/`CUT_POSITIONAL_SHEET_SEPARATION`/`UNRESOLVED_OBSERVATION_CONFLICT`/`UNKNOWN_NO_POSITIVE_OBSERVATION`)로 상호 배타적이며, 다중 시야 취합에 퍼센트 threshold가 전혀 없다. 합성 fixture 10종(A-J) 전부 통과, 신규 focused 테스트 14개.
- **실제 scene 재실측(같은 checkpoint, 161개 train camera): 최대 component 비율 92.69%(WL102) → 10.50%로 극적 개선**, 테이블/패티오가 실제로 분리됐다(시각 확인). **그러나 전체 surfel의 63.4%가 고립된 singleton이 되고 component 수가 13,585 → 768,829로 급증**했다 — spatial edge 5,132,180개 중 25.3%만 co-observation을 받았고(WL102와 정확히 일치), 그중 최종 채택된 양성 edge는 20.3%뿐이라 "양성 지지만" 원칙을 이 evidence 밀도 위에 적용하면 구조적으로 과소-연결이 불가피함을 실측이 보여줬다. hedge/배경은 스페클(수천 개 극소 component)로 심하게 단절된 반면 테이블/패티오처럼 넓고 반복 관측되는 평면은 깨끗하게 분리된 단일 component로 남았다.
- 전체 regression 1198 passed 1 skipped(+14). **WL102·WL103 둘 다 채택하지 않는다 — 같은 축의 양 극단 모두 실제 scene에서 원하는 결과를 주지 못한, 혼재된 정직한 결과.**

## 2026-08-21 Node-Level Observability Accounting — Branch A (arch/2dgs-coverage-first-surface)

- [Worklog 104](worklogs/104_node_level_observability_accounting.md)는 Worklog 103을 전혀 수정하지 않고 그대로 재실행(baseline replay, 커밋된 리포트와 완전 일치)한 뒤, canonical Phase-C 규칙을 각 surfel의 자기 CENTER에 대해 161개 학습 뷰 전부와 대조하는 신규 node-level 관측성 회계(`torch_node_level_observability_accounting.py`)를 추가했다.
- **실측: WL103 singleton의 94.5%는 자기 CENTER가 어떤 뷰에서도 `on_observed_surface`가 된 적이 없다** — pairwise edge 판정이 너무 엄격해서가 아니라 surfel 자체가 애초에 positive observed-visible evidence가 아니라는 뜻(**Branch A**). renderer-native `radii>0`와 비교하면 이 surfel의 99.98%가 평균 48개 뷰에서 여전히 투영/기여는 하고 있었으나(occlusion-aware 신호 아님, 2DGS 커널이 노출하는 유일한 per-surfel 신호), directive 지시에 따라 그 신호만으로 결론을 뒤집지 않고 한계로 기록했다.
- Branch A에 따라 새 adjacency는 만들지 않고, `torch_primitive_ownership_visible_topology_separation.py`로 **primitive ownership(전량 보존)과 visible topology membership(component 크기>=2인 36.6%만 구조적 component)을 별개 계약으로 분리**했다 — singleton-owned surfel은 버려지지 않지만 더 이상 "Visible Surface Component"라 부르지 않는다.
- 신규 focused 테스트 18개, 전체 regression 1216 passed 1 skipped(+18).

## 2026-08-21 Renderer Contribution Diagnostics — Worklog 104 Branch A 기각 (arch/2dgs-coverage-first-surface)

- [Worklog 105](worklogs/105_renderer_contribution_diagnostics.md)는 벤더된 2DGS CUDA 커널을 전혀 수정하지 않고, 모든 학습 스텝이 이미 실행하는 공식 backward pass를 진단 목적으로만 `torch.autograd.grad`로 재사용해(`.backward()` 아님, `.grad` 비변경) surfel별 실제 alpha-compositing 기여를 측정했다(`osn_gs/render/torch_surfel_contribution_diagnostics.py`).
- **실측: Worklog 104가 "한 번도 positively observed 안 됨"(713,540개)으로 분류한 surfel의 95.4%(680,527개)가 실제로는 렌더러의 공식 alpha-compositing에 기여하고 있었다**(중앙값 20개 뷰). **Case B: Worklog 104의 Branch A가 기각된다** — Phase-C의 point-sample CENTER 질의는 학습된 2DGS 표현에 부적절한 primitive-level visibility 정의임이 실증됐다.
- Directive 지시대로 여기서 멈췄다: 새 threshold/adjacency 없음. Rendering 불변성과 오탐 방지(진짜 가려진 surfel)를 실제 CUDA fixture로 검증했다.
- 신규 focused 테스트 9개, 전체 regression 1225 passed 1 skipped(+9).

## 2026-08-21 Renderer-Grounded Visible Adjacency — 통제된 대조, 부정적 결과 (arch/2dgs-coverage-first-surface)

- [Worklog 106](worklogs/106_renderer_grounded_visible_adjacency.md)은 Worklog 105의 renderer-contribution 신호로 WL103의 Phase-C center endpoint eligibility를 대체한 신규 `torch_renderer_grounded_visible_adjacency.py`를 WL103과 정확히 동일한 candidate graph/corridor/기하 게이트로 통제 대조했다.
- **실측: singleton 비율 63.4%→83.8%로 악화, 최대 component 10.50%→2.91%로 더 작아짐** — percolation 재발이 아니라 훨씬 심한 파편화. WL103-singleton-and-renderer-contributing 720,052개 중 11.4%만 edge를 얻었고, 남은 88.6%의 96.9%는 다중 뷰 관측 모순(co-contributing 이웃 부재는 0.14%뿐)이 원인이었다.
- 시각적으로도 테이블·패티오·hedge 전부 파편화됨을 확인. Directive 지시대로 threshold 조정 없이 멈췄다 — 결과는 다음 아키텍처가 camera-induced adjacency로 이동해야 함을 시사하지만 구현은 다음 배치로 미룬다.
- 신규 focused 테스트 14개, 전체 regression 1239 passed 1 skipped(+14).

## 2026-08-21 Camera-Induced Visible Adjacency — 뚜렷한 개선, 최종 채택 아님 (arch/2dgs-coverage-first-surface)

- [Worklog 107](worklogs/107_camera_induced_visible_adjacency.md)은 카메라 자신의 렌더링된 표면이 직접 surfel 인접성을 생성하는 아키텍처로 전환했다(3D edge에 카메라가 승인하는 방식에서 벗어남). 벤더 CUDA를 직접 읽어 이미 내부에 존재하는 `median_contributor`(T>0.5 crossing)를 노출하는 **진단 전용 CUDA 빌드**(원본 미수정 형제 디렉터리)를 만들고, rendering 불변성과 오탐 방지를 실제 CUDA로 검증했다.
- 신규 `torch_camera_induced_visible_adjacency.py`: 이미지-격자 인접 → 3D 국소성 필터(제약만) → 2차 기하 게이트(재사용) → **모든 뷰의 양성 관계 합집합**(occlusion이 다른 뷰의 양성을 부정하지 않음, conflict 상태 자체가 구조적으로 없음).
- **실측(WL106 대조): singleton 83.8%→45.0%, 최대 component 2.9%→36.8%.** 시각 검토로 테이블이 패티오와 분리된 단일 component로 남고, 최대 component는 패티오 바닥의 정당한 연속 표면으로 보임을 확인(WL96-102 시절 percolation과 질적으로 다름). Hedge는 여전히 대부분 파편화.
- Directive 지시대로 최종 architecture 성공을 자동 선언하지 않는다. 신규 focused 테스트 15개, 전체 regression 1254 passed 1 skipped(+15).

## 2026-08-23 Renderer-Native Surface Representative Backbone — 조건부 통과 (arch/2dgs-coverage-first-surface)

- [Worklog 108](worklogs/108_camera_induced_representative_backbone_audit.md)은 Worklog 107의 adjacency 알고리즘을 전혀 수정하지 않고, "renderer-contributing 전체"가 아니라 "renderer surface representative"라는 올바른 구조적 모집단으로 재평가하는 architecture gate/회계 감사다.
- **회계 불일치 해소**: contributing과 representative 사이 정확히 36,051개의 예상 밖 교차 카테고리를 실측 확인(WL107의 385,998 자체는 정확, 단순 뺄셈이 discrepancy를 만듦) — 서로 다른 두 CUDA 빌드의 부동소수점 재구성 차이로 추정, 버그 증거 없음.
- **Representative-only 회계: singleton 16.7%(vs 전체 기준 45.0%), 연결 83.3%** — 훨씬 결합력 있는 backbone. 결정론적 픽셀 anchor로 테이블-패티오 분리, hedge 3지점 모두 다른 component임을 재확인.
- **그러나 패티오 거대 component 멤버의 20.4%가 hedge 인접 영역과 겹침**을 발견 — "패티오만의 순수한 표면"이라는 WL107 결론을 부분 수정. Bridge 감사: 최대 component edge의 5.5%가 구조적 bridge, 19.2%가 단일-뷰 지지.
- **Gate: CONDITIONAL PASS** — representative backbone을 canonical topology 후보로 제안하되 패티오-hedge 경계 caveat 동반, non-representative contributor는 attach하지 않고 retained evidence로만 분류.
- Production 코드 미변경, 신규 focused 테스트 9개, 전체 pytest 재실행 안 함.

## 2026-08-24 Renderer-Native Surface Representative Graph — Gate Closure, GATE PASS (arch/2dgs-coverage-first-surface)

- [Worklog 109](worklogs/109_renderer_native_topology_gate_closure.md)는 Worklog 108의 두 caveat를 실측으로 닫았다. WL107 adjacency 알고리즘은 이번에도 무수정.
- **CAVEAT 1**: 진단 CUDA 빌드(canonical 무수정)에 `out_forward_accepted`를 추가해 representative 캡처와 **같은 forward 실행**에서 기록되게 함. 실측: `representative_and_not_forward_accepted = 0`(전수 확인) — WL108의 36,051 discrepancy는 100% forward_accepted였고, WL105 별도 backward 진단이 실제 기여를 놓친 것임을 부동소수점 추측 없이 확정.
- **CAVEAT 2**: WL107 무수정 재실행으로 재현성 확인(36.77%/45.02% 완전 일치) 후, 89,502개(hedge의 26.2%) 패티오/hedge 중첩의 실제 그래프 프론티어(1,110개 엣지 전수)를 추출, Tarjan bridge-finding + DFS 서브트리 크기로 정확한 split-impact 계산. **frontier와 교차하는 bridge는 56,816개 중 67개(0.118%)뿐이고 최대 분리는 56개(0.013%)** — 패티오-hedge 중첩은 취약한 단일 다리가 아니라 1,043개 중복 경로로 얽힌 견고한 다중 뷰 지지(80.7%) 연결이며, 최대 component의 진짜 취약점(고영향 bridge 상위 10.7% 분리)은 경계가 아니라 각 영역 내부에 있다.
- **GATE: PASS** — Renderer-Native Surface Representative Graph를 canonical Visible Surface Topology Backbone으로 정식 채택. 단, 최대 component가 patio-인접 구조와 hedge 식생을 다중 뷰로 함께 포함한다는 사실은 순수 geometric 위상 구성의 정직한 한계로 기록(의미론적 분리는 Trust 단계 과제). Non-representative contributor는 이번에도 attach하지 않았다.
- Production/runtime 공유 코드 미변경(진단 CUDA 빌드만 확장), 신규 focused 테스트 5개 + 기존 5개 재검증, 전체 pytest 재실행 안 함.

## 2026-08-24 Non-Representative Renderer Evidence — Role Attribution, AMBIGUOUS/LAYERED SUPPORT (arch/2dgs-coverage-first-surface)

- [Worklog 110](worklogs/110_nonrepresentative_evidence_attribution.md)은 WL109가 GATE PASS로 확정한 canonical topology(`torch_camera_induced_visible_adjacency.py`)를 전혀 수정하지 않고, representative가 되지 못한 채 forward-accept된 395,676개 서펠의 역할을 귀속(attribution)만 했다 — attachment는 이번 배치에서 하지 않았다.
- 진단 CUDA 빌드(canonical 무수정)에 픽셀당 bounded slot 배열(K=16, `contrib_count`로 truncation 항상 감지 가능)을 추가해, 커널이 이미 median-crossing 체크에 쓰는 running transmittance `T`를 그대로 재사용해 새 threshold 없이 PRE_MEDIAN/POST_MEDIAN을 분류하고, 같은 픽셀의 representative를 통해 contributor↔canonical-component co-support를 스트리밍 방식(전체 픽셀×서펠 행렬 없음)으로 집계했다.
- **핵심 발견 — 심각한 truncation**: 전체 픽셀·뷰 슬롯의 **97.4%(42,660,905/43,817,760)**가 K=16 캡을 넘었고, 슬롯은 depth 오름차순으로 채워지므로 truncation은 항상 "더 단순한 쪽"(단일 컴포넌트, PRE_MEDIAN)으로만 표본을 편향시킨다 — 휴리스틱으로 대체하지 않고 이 한계를 그대로 보고했다(directive Section 5).
- **실측(그 편향에도 불구하고)**: 정확히 하나의 컴포넌트를 일관되게 co-support하는 인구는 26.2%뿐이고 48.0%는 2개 이상(중앙값 4개, 최대 609개)에 걸쳐 있으며, 63.3%가 최소 1회 POST_MEDIAN(PRE_MEDIAN 43.8%보다 큼)이다 — truncation이 이 방향을 과소평가하는 쪽으로만 작용하므로 실제 모호성은 더 클 가능성이 높다. 테이블이 가장 깨끗, 헤지/배경이 가장 layered(POST-계열 64.7%, 다중-컴포넌트 54.4%). table↔patio, 최대컴포넌트↔hedge 동시 접촉 표본은 0건.
- **건축 결정: AMBIGUOUS/LAYERED SUPPORT** — non-representative 증거를 일괄 Visible Surface Support Evidence로 부르지 않는다. Trust는 patio/hedge 객체 정체성 분리를 담당하지 않는다는 것도 명시적으로 재확인(WL109 프레이밍 교정).
- Production/runtime 공유 코드 미변경, 신규 focused 테스트 25개, 전체 pytest 재실행 안 함.

## 2026-08-24 Representative-Only Visible NURBS Scaffold — NOT VIABLE (arch/2dgs-coverage-first-surface)

- [Worklog 111](worklogs/111_representative_only_visible_nurbs.md)은 WL107/109 canonical topology(무수정)와 WL110의 AMBIGUOUS/LAYERED SUPPORT 판정(non-representative 증거 배제)을 그대로 얼려두고, 785,937개 representative만으로 scene-covering continuous visible NURBS를 만들 수 있는지 실측했다.
- 신규 `torch_camera_observed_chart_domains.py`: 카메라 자신의 픽셀 좌표를 chart UV로 직접 사용(3D PCA/kNN 아님), scipy 정확 connected-components로 chart 후보를 만들되 서로 다른 canonical 컴포넌트는 구조적으로 절대 같은 chart를 공유할 수 없다. NURBS fit은 기존 프로젝트 기본값(8×4 control grid)을 그대로 재사용, 최소 chart 멤버 수(32)는 수학적으로 유도(튜닝 없음).
- **실측(전체 161개 뷰, 위상 재생 완전 일치 확인)**: representative 멤버십 커버리지 68.4%, 픽셀 커버리지 93.1%로 양호해 보이지만 **컴포넌트 단위 커버리지 분포는 중앙값·p95 모두 0%** — 커버리지가 소수의 거대 컴포넌트에만 집중된다. fitting residual 중앙값 0.032이지만 최대 7.9, overlap 법선 불일치 중앙값 5.04°이지만 p95 57.9°/최대 180°에 근접 — 거대/미분화 chart의 fitting 실패.
- 영역별: table_legs 88.9%(최고), table_side_curved 57.3%(최저), patio 77.8%, hedge 49.0%(최저 영역).
- **건축 판정: NOT VIABLE**(현재 chart 구성 방식으로는) — 주원인은 동결된 위상 자체의 파편화(CANONICAL_TOPOLOGY_ISSUE), 부차 원인은 거대 blob이 미분화된 채 하나의 control grid로 강제 fit됨(CHART_PARAMETERIZATION_FAILURE). NURBS readiness 주장하지 않음.
- 신규 focused 테스트 15개(directive 11절 A-F 계약 포함), 프로덕션/CUDA/트레이닝 코드 무수정으로 전체 pytest 재실행 안 함.

## 2026-08-24 Renderer-Native Pixel Surface as NURBS Fitting Geometry — NO (arch/2dgs-coverage-first-surface)

- [Worklog 112](worklogs/112_renderer_native_pixel_surface_nurbs.md)는 WL111을 정확히 보존(chart 구성/UV/고정 8×4 NURBS 무변경)하고, 3D fitting 대상만 "대표 서펠 중심"에서 "렌더러가 계산한 픽셀별 median-depth 언프로젝션 표면 점"으로 바꾼 통제 A/B 비교를 실행했다. 기존 공식 2DGS 코드-충실 언프로젝션 함수(`depths_to_points`)를 재사용, CUDA 재빌드 불필요.
- **실측**: 유효 chart 3,963→14,900개(pixel-count 기준 지지 조건으로 완화), representative 커버리지 68.4%→71.5%, 픽셀 커버리지 93.1%→94.7%로 소폭 개선 — 그러나 **컴포넌트 단위 커버리지 분포는 중앙값·p95 모두 여전히 정확히 0%**. fitting residual 최댓값 7.9→1517.2, overlap 위치 불일치 중앙값 0.030→0.055(악화)·최댓값 8.1→1514.4로 극단 악화.
- 영역별 커버리지는 소폭 개선(curved rim +4.9%p, hedge +5.8%p)됐으나 기하 품질 개선은 동반하지 않았다.
- **건축 판정: NO** — 대표-중심/픽셀-표면 불일치는 WL111 실패의 주 원인이 아니다. 남은 실패는 NURBS chart capacity/granularity(거대 미분화 chart) 문제이며 이번 배치는 그것을 해결하지 못하고 악화시켰다.
- 신규 focused 테스트 9개(실제 CUDA 언프로젝션 계약 포함), 전체 pytest 재실행 안 함.

## 2026-08-25 Chart Representation Contract Diagnostic — 4개 원인 각기 다른 증상에 배정, 건축 결정 없음 (arch/2dgs-coverage-first-surface)

- [Worklog 113](worklogs/113_chart_representation_contract_diagnostic.md)은 WL112를 튜닝하지 않고 chart 세분화도 구현하지 않은 채, WL107/109 위상·WL111 blob 구성·WL112 픽셀-표면 기하·고정 8×4 NURBS를 전부 동결하고 "카메라-관측 blob 하나 = 사각형 NURBS chart 하나"가 왜 실패하는지 순수 진단했다.
- **zero-coverage 컴포넌트 153,600개 전부(100%)가 `NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES`**(A. SUPPORT_LIMITED, fit 실패 경로 0건). fitted chart의 bbox occupancy 중앙값 0.5, 43.5%가 구멍 보유, 구멍 있는 chart의 residual이 없는 chart보다 2.6배 높음(B. RECTANGULAR_DOMAIN_FAILURE). full-rank chart의 residual이 rank-deficient chart보다 오히려 높음(C. FIXED_NURBS_CAPACITY_FAILURE, 소수 거대 컴포넌트에 국한).
- **극단값 역추적**: residual 극단(최대 1517)은 patio 최대 컴포넌트의 거대·holey chart(B+C 결합)에서, overlap 극단(최대 1514)은 작은 컴포넌트의 chart 내부 렌더러 depth 국소 이상값(D. NUMERICAL/GRAZING_SURFACE_FAILURE)에서 — 서로 겹치지 않는 별개 메커니즘.
- 영역별: table_top 75.5%, table_side_curved 62.2%(순수 A), patio 79.9%(최고지만 극단값 대부분 발생), hedge 54.8%(순수 A, 최저).
- 4개 원인을 단일로 강제하지 않고 서로 다른 통계량/컴포넌트 규모대에 정확히 배정. 건축 결정/새 표현 메커니즘 구현 없음.
- 신규 focused 테스트 12개(순수 로직), 프로덕션/CUDA 코드 무수정으로 전체 pytest 재실행 안 함.

## 2026-08-25 Local Rank-Complete NURBS Chart Network — LOCAL_CHART_UNIT_NOT_VIABLE (arch/2dgs-coverage-first-surface)

- [Worklog 114](worklogs/114_local_rank_complete_chart_network.md)는 WL107/109 위상·WL112 픽셀-표면 기하·고정 8×4 NURBS를 동결하고, chart **단위**만 "blob=chart 하나"(WL112)에서 "blob → pole-of-inaccessibility 시드 + BFS 성장 → 고정 8×4 design matrix가 처음 full column rank에 도달하는 지점에서 닫힘 → 여러 local chart"로 교체했다.
- **통제 비교(위상/대표는 전체 161개 뷰, chart 성장+fit만 8개 뷰 stride 표집 — 명시적 범위 축소)**: chart 수 889→14,137(15.9배), residual 중앙값 9배·p95 8배 개선, 구멍 있는 chart 비율 46.1%→15.7%, aspect ratio p95 3.36→1.22로 도메인 모양 뚜렷이 개선.
- **그러나 representative 커버리지가 11.7% 감소(다섯 영역 전부)하고 overlap 법선 불일치가 크게 악화(중앙값 5.8°→18.2°, p95 59.3°→96.5°)했다.** WL113의 D(렌더러 median-depth 국소 불안정) 이상치가 새 방법에서도 동일 패턴으로 지속됨을 확인.
- **건축 판정: LOCAL_CHART_UNIT_NOT_VIABLE** — fit 품질/도메인 모양 개선이 coverage 하락·overlap 악화라는 대가 없이 오지 않았다.
- 신규 focused 테스트 9개, 프로덕션/CUDA 코드 무수정으로 전체 pytest 재실행 안 함.

## 2026-08-25 Design-Intent/Specification/Implementation Traceability Audit — 감사만 수행, 구현 없음 (arch/2dgs-coverage-first-surface)

- [Worklog 115](worklogs/115_design_intent_specification_implementation_traceability_audit.md)는 WL107-113의 RESEARCH INTENT → SPECIFICATION → IMPLEMENTATION → OBSERVED RESULT 인과 사슬을 감사했다(구현/튜닝 없음).
- **결론: 감사 범위에서 순수 INTENT-level 실패나 IMPLEMENTATION DEVIATION은 발견되지 않았다.** 관측된 실패는 압도적으로 SPECIFICATION-INDUCED(chart 단위 정의, 사각형 UV 도메인 — `torch_nurbs.py`에 이미 존재하나 미사용인 `uv_support_mask` 발견, per-view 비병합)이거나 의도적 통제 조건(고정 8×4 용량)이거나 렌더러 자체 현상(D)이다.
- WL114의 rank-complete local chart 제안을 사전 감사해 "full column rank(대수적)"가 "유효한 local geometric chart(기하적)"를 보장하지 않는다는 간극을 지적했고, WL114의 실측이 이를 직접 확인했다.
- Design debt 목록과 증명된 canonical contract vs 미해결 표현 선택을 분리 정리. 코드 변경 없음.

## 2026-08-25 Visible-NURBS Representation Contract Recovery Audit — 기존 fitter/trimming/multi-patch 재발견, 코드 변경 없음 (arch/2dgs-coverage-first-surface)

- [Worklog 116](worklogs/116_visible_nurbs_representation_contract_recovery_audit.md)은 WL114를 그 정확한 후보에 대한 유효한 negative 결과로 받아들이되 일반화하지 않고, `osn_gs/surface/torch_nurbs.py`를 정밀 감사했다.
- **핵심 발견**: 기존 정규화 fitter는 full-rank 관측 지지를 요구한 적이 없다(Tikhonov 항이 항상 solvable하게 만듦) — "full column rank == 유효한 chart"라는 WL114의 폐쇄 규칙을 명시적으로 폐기.
- 이미 존재하지만 미사용인 두 메커니즘 발견: `uv_support_mask`(UV trimming, materialization만 해결·fitting coupling은 미해결임을 코드 자체 docstring으로 확인) / `fit_coupled_patch_graph_lsq`(shared-boundary 공동 fitting multi-patch, 옛 annulus 계보에만 존재).
- 고정 8×4는 코드베이스 전체에 6개의 서로 다른 resolution 기본값이 이미 있어 architecture 법칙이었던 적이 없다.
- WL113 A/B/C/D 재분류, WL114를 "locality는 유용하다" vs "disjoint rank-closed 추출은 실패했다"로 분리 재해석 — overlap 악화를 "seam 증가" 대신 "coupling 부재"로 더 정확히 귀속.
- 다음 배치를 이끌 단일 질문 도출: `uv_support_mask` 적용만으로 실패 B가 충분한가, coupled fitting이 필요한가. 코드 변경 없음.

## 2026-08-26 Holey-Chart Fitting-Coupling Attribution — MIXED/INCONCLUSIVE, 거대 chart는 정반대 방향 (arch/2dgs-coverage-first-surface)

- [Worklog 117](worklogs/117_holey_chart_fitting_coupling_attribution.md)은 WL113의 "구멍 있는 chart 2.6배 나쁜 residual" 상관관계가 진짜 fitting-coupling 실패(B2)인지 chart 규모의 대리 변수였는지 전체 161개 뷰로 가렸다.
- **B1 검증**: 14,900개 chart 전부에서 `uv_support_mask` 부여가 fit을 비트단위로 전혀 바꾸지 않음을 확인(0건 위반).
- **B2**: 일반 chart는 약한 실재 hole-근접 상관(중앙값 -0.055~-0.098)이 있으나, 규모로 층화하면 원시 비율은 중간 구간에서 아티팩트로 부풀고 최대 규모 구간에서는 1.3배로 줄거나 0.85배로 **역전**된다.
- **핵심 발견**: residual 극단값을 지배해 온 거대 patio chart 10개 중 8개는 residual이 hole 경계에서 **멀수록 나쁘다** — B2와 정반대.
- 합성 대조군: 평면 무정보, 곡면은 꼬리 품질만 유의미하게 나빠짐.
- **판정: MIXED/INCONCLUSIVE** — 스크립트의 자동 판정을 그대로 따르지 않고 신호 크기를 재검토해 방향성 있게 교정: 일반 chart엔 약한 B2 신호, 최대-영향 거대 chart는 scale/capacity가 더 유력. Coupled fitting 미구현.
- 신규 focused 테스트 15개, 프로덕션 코드 무수정.

## 2026-08-26 Visible-NURBS Evidence Contract Closure — domain mismatch 확인, ARM A(현재)가 일관되게 우수 (arch/2dgs-coverage-first-surface)

- [Worklog 118](worklogs/118_visible_nurbs_evidence_contract_closure.md)은 WL117을 조건부 수용한 뒤, WL117의 hole-거리 분석(post-fit UV)과 WL113의 원 관측(camera-raster domain)이 서로 다른 도메인을 측정했다는 점을 명시적으로 닫았다.
- Sibling 진단 CUDA에 median 이벤트의 low-pass provenance(rho3d/rho2d/s) 4개 필드 추가(canonical 무수정, 렌더링 불변성 재검증 통과).
- **핵심 발견**: 같은 해상도에서 camera-domain·fitted-domain hole 판정이 51.1% 불일치(15.8% vs 64.4%); 고정 UV(ARM B) 대비 현재 foot-point 보정(ARM A)이 모든 영역에서 residual 2.9~3배(최대 1500배) 개선; normal 부호 반전은 불일치의 2.7%만 설명; chart 내부 representative spread가 cross-chart 변위의 85.7%에 달함; low-pass 지배는 작은-chart D-outlier(44.3%)에서만 뚜렷하고 거대 chart(20-29%)에서는 아님.
- Equal-count 합성 대조군으로 "hole topology가 단순 샘플 수 감소를 넘어서는 효과가 있는가"에 곡면·foot-point 경로 한정 "예"로 답함.
- 신규 focused 테스트 15개(CUDA 3개 포함) 전부 통과, canonical 코드 무수정.

## 2026-08-26 Visible-NURBS Geometry / UV Control Correction — METRIC G/C가 정반대 방향, 진짜 trade-off 발견 (arch/2dgs-coverage-first-surface)

- [Worklog 119](worklogs/119_visible_nurbs_geometry_uv_control_correction.md)는 WL118의 fixed-UV A/B가 solve 횟수(2 vs 1)와 evaluation UV 의미론이 arm마다 달라 공정한 통제가 아니었음을 정정했다.
- **핵심 발견**: solve 횟수를 동일하게 맞추고 METRIC G(기하 오차)/METRIC C(camera-correspondence 오차)를 분리 평가하자, foot-point 보정(ARM A)은 METRIC G에서 모든 영역 우수(0.79~0.86배)하지만 METRIC C에서는 고정 UV(ARM B)가 모든 영역 우수(1.09~1.69배) — 방향이 갈리는 진짜 trade-off이며, WL118의 "ARM A가 일관 우수"는 서로 다른 metric을 섞어 비교한 인공물이었다.
- Renderer의 median-surfel 직접 교차점(G2) 재구성이 G0/G1과 소수점 6자리까지 일치함을 실측 검증(canonical 무수정, 순수 Python).
- Pixel 단위 D-outlier 귀속: 모집단 수준 low-pass 지배 pixel은 residual 2.1배 높고 top-1000의 54%를 차지하지만, 역사적으로 반복 지목된 단일 극단 chart(10592)는 branch가 섞여 있어 chart 단위 fit 퇴화가 원인임을 확인.
- 실행 시간 73.7분(WL118의 1.6배)은 directive가 요구한 arm별 독립 재평가 비용임을 CPU 통제 벤치마크로 확인(버그 아님; 발견된 실제 버그 1건은 전체의 ~1%만 기여).
- 신규 focused 테스트 14개 전부 통과, canonical 코드 무수정.

## 2026-08-26 Worklog 119 GPU Utilization / Host Serialization Audit — CURRENT EXECUTION-GRANULARITY LIMIT

- [별도 감사 worklog](worklogs/119-1_gpu_utilization_host_serialization_audit.md)는 활성 full run을 보호한 뒤 동일 checkpoint/GPU 조건의 8-view/512-chart OLD/FIXED A/B와 1초 nvidia-smi sampling을 수행했다.
- known pixel-record scalar serialization을 bulk NumPy helper로 유지한 결과 exact-equivalence focused test 포함 14개가 통과했고, OLD/FIXED pixel_records count는 각각 42,998로 동일했다.
- chart loop는 43.840초에서 43.087초로 약 1.7% 개선됐지만 GPU util은 평균 15.21%/15.76%, p95 16%로 낮은 수준이 유지됐다. total wall time 개선은 약 0.5%였다.
- profiler에서는 topology replay _knn이 141.312초로 지배했고, named pixel-record helper 128 calls cumulative는 0.015초였다. G0/G1/G2 및 WL118 representative spread는 vectorized path로 확인했다.
- 최종 귀속은 CURRENT EXECUTION-GRANULARITY LIMIT이다. 여러 chart solve batching은 별도 architecture 배치의 가설로 남겼으며 이번 배치에는 구현하지 않았다. 1.5~2.5배 wall-clock 개선은 현재 근거로 확정하지 않는다.
## 2026-08-26 Worklog 119-2 Exact-Semantics Performance Optimization — 약 2배 단축

- [Worklog 119-2](worklogs/119-2_exact_semantics_performance_optimization.md)는 WL119 수학/topology/chart/NURBS/solve 계약을 유지하면서 완전 중복 연산만 제거했다.
- 동일 exact KNN을 2회에서 1회로 재사용하고, ARM A 중복 Metric G projection, 불필요 basis derivative, projector 동일-UV 재평가, ARM B normal-system 재조립, 중복 blob labeling/전체 image scan, chart scalar sync를 제거했다.
- 동일 8-view/512-chart에서 total 185.494→93.006초(1.994배), chart loop 43.087→21.342초(2.019배), 11.883→23.990 charts/s. 주요 report section은 exact 동일하고 focused tests 83개 통과.
- chart GPU median은 여전히 16%이며, 최종 profile의 89.2%는 한 번 남은 O(N²) exact KNN이다. 남은 chart 병목은 execution-granularity limit이고, 다음 구조적 후보는 exact spatial-neighbor architecture 또는 projection/basis batching이다.

## 2026-08-27 Worklog 119-3 Performance Track Phase 1~4 — 잠재 성능 확인, equivalence gate로 미채택

- [Worklog 119-3](worklogs/119-3_performance_track_phase_1_4.md)은 Main Architecture Track 및 WL119 scientific result와 독립된 Performance Track이다. 기존 serial chart path와 `torch.cdist` KNN을 immutable reference로 유지했다.
- 실제 8-view/512-chart corpus는 206,889 pixels(min/median/p95/max 32/60/486.2/114,571)이며 deterministic bucket plan과 top-10 pathological chart를 기록했다. Runtime OOM splitting과 silent fallback은 사용하지 않았다.
- Dependency-terminal chart batching은 2.06배 잠재 speedup을 보였지만 509/512 charts가 초기 continuous 기준을 넘고 8개 chart에서 research winner/tie 관계가 바뀌어 미채택했다. 통과한 3개는 애초 oversize serial charts였다.
- SciPy cKDTree exact 후보는 full scene에서 70.552→0.647초(108.98배)였지만 neighbor row 136,277개, candidate edges, accepted edges, partition roots 1,251개가 reference와 달라 contract mismatch로 미채택했다.
- Production/default backend 변경은 없고, focused suite 89개 및 전체 회귀 1,422 passed/22 skipped/18 subtests가 통과했다. CUDA Graph·multi-stream·custom chart CUDA kernel은 구현하지 않았다.

## 2026-08-27 Worklog 119-4 Reference-Exact KNN Semantics Attribution — outcome C

- [Worklog 119-4](worklogs/119-4_reference_exact_knn_semantics_attribution.md)는 WL119 exact `torch.cdist` KNN을 변경하지 않고 full-scene 136,277 mismatch 행을 완전 귀속했다.
- 동일 환경 reference는 neighbor/order/distance/graph/root까지 exact 반복됐고 production default는 explicit MM과 exact였다. 그러나 114,812행은 order-only, 10,466행은 raw MM tie membership 교환, 10,999행은 derived float32 MM error bound 내 membership 역전이었다(`F=0`, `H=0`).
- reference ranking은 production GEMM shape/chunk와 정확한 `topk(K)` tie 선택에 결합돼 있다. 축소 candidate/pair-MM은 이를 깨끗하게 재현할 correctness proof가 없어 Attribution Gate를 실패했으며, accelerated backend와 production 변경은 구현하지 않았다.
- 최종 판정: `C. REFERENCE SEMANTICS TOO IMPLEMENTATION-COUPLED FOR A CLEAN EXACT REPLACEMENT`. focused tests `28 passed`; attribution-only 지시대로 full suite는 실행하지 않았다.

## 2026-08-27 Query-contract closure

- [Worklog 123: Volumetric Frontier Query Contract Closure](worklogs/123_volumetric_frontier_query_contract_closure.md)
  - world-space 3D를 canonical volumetric query로 유지하고 renderer median-event provenance를 source-view exact identity로 보존하는 계약을 닫았다. exhaustive source identity 0/43,817,760 contradiction, cross-view historical global contradiction 19/19 rescue, generic P1-excluded stability 1,118/1,590,240 diagnostic disagreement을 확인했으며 최종 verdict는 A다. Candidate B·topology·epsilon/tolerance는 변경하지 않았다.

## 2026-08-27 Canonical architecture reduction audit

- [Worklog 124: Canonical Architecture Reduction Audit](worklogs/124_canonical_architecture_reduction_audit.md)
  - Worklog 96–123 evidence와 실제 current call graph를 분리했다. smallest defensible canonical core는 world-space x + optional exact renderer-event provenance + stored median frontier + frozen aggregation이다. visible topology/NURBS는 provisional, occluded-side/uncertain materialization은 premature/open으로 남겼고 코드와 exports는 변경하지 않았다.


## 2026-08-27 Fixed Gaussian Visualization Contract

모든 Gaussian visualization은 고정적으로 Original Scene과 Observed/Occluded를 함께 포함한다. Original Scene은 해당 환경의 Gaussian만 원래 색상으로 렌더링하고, Observed/Occluded는 **동일한 Gaussian들의 색상만** Observed Space/Occluded Space 상태에 따라 바꾼다. 조명·shading·emissive 효과·추가 marker Gaussian·geometry 변경은 금지한다.

고정 색상은 OBSERVED=(0.10,0.85,0.35) green, OCCLUDED=(0.92,0.18,0.18) red, UNRESOLVED=(0.60,0.60,0.62) gray다. Occluded Gaussian/volumetric representation이 실제로 없으면 marker를 발명해 표시하지 않으며, validated Occluded volumetric render가 승인된 경우에만 고정 두 결과와 함께 추가한다. 이전 WL123 EVENT_IDENTITY_EFFECT 출력은 이 contract 이전의 historical diagnostic output으로 보존하지만 canonical 비교 결과로 사용하지 않는다. 상세 계약은 [Worklog 125](worklogs/125_fixed_gaussian_visualization_contract.md)와 [AGENTS.md](../AGENTS.md) 문서에 기록했다.

## 2026-08-31 Worklog 130 Parameterization / Termination / Target-Coherence Attribution

- [Worklog 130](worklogs/130_parametric_continuation_attribution_before_occluded_surface.md)은 Worklog 128/129의 frozen fit과 prediction을 바꾸지 않고, final footpoint UV·실제 mesh-face interface·withheld face connectivity를 별도 분석했다.
- 두 ROI의 frozen control grid replay는 tolerance 안에서 재현됐지만, terminal final-u support가 curved rim 51.6%, thin structure 58.2%이고 thin structure inversion fraction은 11.412%였다. fitted `u=1`의 실제 interface agreement도 thin structure에서 coverage `<=2h` 1.6%로 무너졌다.
- primary withheld target의 interface-connected 비율은 vertices 76.64%, faces 85.68%로 단일-sheet gate를 통과하지 못했고 competing sheets가 공존했다. 원래 Worklog 129 metric은 frozen population과 재계산 결과가 일치했다.
- 최종 판정은 **A. PARAMETERIZATION CONTRACT FAILED**다. second-order continuation, true-occluded prototype, canonical production 변경은 수행하지 않았다. 분석 모듈/출력은 `devtools/demo/parametric_continuation_attribution.py` 및 `output/130_demo_parametric_continuation_attribution/`에 격리했다.

## 2026-08-31 Worklog 131 Explicit Geometric Termination Mapping

- [Worklog 131](worklogs/131_explicit_geometric_termination_mapping.md)은 `u=1` rectangular NURBS edge를 termination으로 부르지 않고, fixed physical holdout plane과 frozen observed-side NURBS의 교차곡선 `GEOMETRIC_TERMINATION_CURVE`를 사용해 같은 first-order continuation을 재검증했다.
- curved rim은 32/32 plane roots, `u_gamma=0.9910–0.9976`, `d local_u/dl=1.0`을 얻었지만 full population median `3.179h→3.275h`, coverage `9.23%→8.83%`로 materially 개선되지 않았다. supported termination attribution도 median `3.308h`, coverage `8.83%`였다.
- thin leg/brace는 fixed plane root가 0/32라 explicit termination을 구성하지 못했다. mesh fragmentation은 physical sheet 의미로 해석하지 않았다.
- 최종 판정은 **C. EXPLICIT TERMINATION DOES NOT MATERIALLY HELP**다. second-order, true-occluded, canonical production 변경은 수행하지 않았고 결과는 `output/131_demo_explicit_geometric_termination_continuation/`에 격리했다.

## 2026-08-31 Worklog 132 Supported-Termination Attribution Contract Closure

- [Worklog 132](worklogs/132_supported_termination_attribution_contract_closure.md)는 Worklog 131의 `Gamma`/ROI/plane/physical direction/first-order prediction을 동결한 채, 모든 withheld row를 정확히 하나의 nearest-v Gamma column에 배정하는 별도 attribution을 수행했다.
- curved rim은 supported Gamma `23/32`이고 target은 `11,640 supported / 360 unsupported`로 분리됐다. 지원 target의 correspondence-restricted Arm B는 median `4.012h`, coverage `<=h 5.52%`, distance-bin median `2.208h→4.895h`로 여전히 명확히 실패했다.
- residual curvature 진단은 `median cosine(R,A)=-0.617`, `R·A>0=18.23%`라 candidate gate가 통과하지 못했다. second-order candidate, true-occluded prototype, canonical production 변경은 수행하지 않았다.
- thin leg/brace는 32개 fixed v row 모두 `definitely_no_intersection`으로 분류됐고 새 Gamma는 만들지 않았다. 출력은 `output/132_demo_supported_termination_attribution/`에 격리했으며 최종 판정은 **C. FIRST-ORDER STILL FAILS ON CORRECTLY ISOLATED SUPPORTED TERMINATION; VISIBLE CURVATURE DOES NOT EXPLAIN IT**다.

## 2026-08-31 Worklog 133 Physical Correspondence / Curvature Identifiability Closure

- [Worklog 133](worklogs/133_physical_correspondence_curvature_identifiability_closure.md)는 WL132의 Gamma UV/XYZ, physical direction, ROI, holdout, first-order prediction, support mask를 재생하고 identity를 `PASS`로 확인한 뒤, fitted parametric-v와 Gamma XYZ에서 계산한 physical-v의 대응을 분리 비교했다.
- curved rim에서 Gamma-v의 Pearson/Spearman은 `0.9983/1.0000`이지만 target assignment `3,919/12,000 (32.66%)`가 바뀌었고 supported target은 `11,640→11,920`으로 변했다. 따라서 WL132의 curvature attribution은 correspondence-confounded로 닫혔다.
- physical correspondence의 supported target은 median `3.717h`, p95 `10.866h`, coverage `<=h 6.69% / <=2h 21.75%`, normal median/p95 `24.39°/80.04°`였다. frozen face-interface에서 정의한 B 바닥은 median `0.660h`, p95 `30.298h`이고, bias-corrected residual의 overall curvature cosine은 `-0.674`, positive fraction `22.23%`였다.
- fixed `0–2h, 2–4h, 4–8h, 8–16h, >16h` bins와 raw/bias-corrected residual을 기록했다. thin leg/brace는 32/32 fixed v row가 `definitely_no_intersection`이며 Gamma가 없다.
- **최종 판정: B. PARTIAL FEASIBILITY DEMO — physical-v correspondence confounds the WL132 attribution.** second-order/third-order/q fitting, true-occluded prototype, canonical production 변경은 수행하지 않았다. 산출물은 `output/confirmed/demo_physical_correspondence_curvature_identifiability/`에 격리했다.

## 2026-08-31 Worklog 134 Meeting Feasibility Demo

- [Worklog 134](worklogs/134_meeting_occluded_surface_feasibility_demo.md)는 frozen Worklog 127 Visible Surface에서 curved table side/rim boundary holdout을 만들고, retained frontier 기반 first-order continuation과 observed top-side junction transfer를 별도로 검증한 비정규 meeting demo다.
- H1은 median `6.988h`, p95 `13.488h`, coverage `<=h 1.57% / <=2h 8.81%`, H2는 `AMBIGUOUS` branch로 단일 prediction을 만들지 못했다. H1의 pseudo-volume 위반도 있어 controlled gate는 **C. CONTROLLED HOLDOUT FAILS**로 닫혔다.
- 요청된 visualization-scope에 따라 raw fixed-view PNG overlay와 NPZ/PLY geometry를 `output/134_meeting_occluded_surface_feasibility/`에 출력했다. true-occluded prototype은 gate 실패로 실행하지 않았고 canonical research code는 변경하지 않았다.
- [Worklog 135](worklogs/135_meeting_demo_visualization_scope_correction.md)는 Worklog 134의 실험·gate·metric을 유지한 채 local `u/v/n` surface skin, generated grid surface, `u-v` footprint, `u-n` profile을 추가해 raw geometry를 직접 읽을 수 있도록 시각화 범위만 보정했다.
- [Worklog 136](worklogs/136_semantically_aligned_occluded_surface_feasibility_demo.md)는 실제 leg/brace와 실제 tabletop-side source/target pair를 사용한 semantic-alignment feasibility demo를 별도 `devtools/demo`·output 경로에 추가했다. H1은 withheld median `4.684h`, coverage `<=h 3.12%`, H2는 measured source angle `77.63°`의 두 branch 모두 geometry gate reject로 **C. NEGATIVE FEASIBILITY RESULT**를 기록했다.

## 2026-08-31 Worklog 140 Real Gaussian Scene 정성적 Surface Construction 검증

- [Worklog 140](worklogs/140_real_gaussian_scene_surface_construction_validation.md)은 frozen trained 2DGS checkpoint, canonical renderer, WL127 raw Visible Surface, WL139 graphness/physical-chart representative를 실제 Gaussian Scene에서 정성적으로 대조하는 격리 평가다.
- WL139 모듈·canonical renderer·checkpoint·WL127 결과는 변경하지 않았다. continuation, pseudo-occlusion, Candidate B, true-occluded prototype은 실행하지 않았다.
- 실제 `DSC08111.JPG` raw scene에서 수동 camera-aligned curved table-side/rim seed를 추가했고, 기존 WL139 curved-rim 좌표는 paver/ground로 투영되는 historical alignment control로 분리했다. graphness pass 자체를 semantic alignment나 성공으로 해석하지 않는다.
- 7개 ROI 중 graphness pass는 camera-aligned rim, historical control, adjacent side, patio이며 tabletop/leg/hedge는 materially multivalued로 fail-closed했다. primary representative proximity는 진단용이고 qualitative macro-shape는 `USER REVIEW REQUIRED`다.
- raw/representative PLY·NPZ와 same-camera A/B/C/D PNG, 3D raw/representative/overlay/normal/boundary 출력을 `output/confirmed/140_real_gaussian_scene_surface_validation/`에 생성했다. 상세 결과·남은 정합성 위험은 worklog를 따른다.

## 2026-08-31 Worklog 141 Oracle Single-Surface Support / Renderer-Native Appearance Evidence

- [Worklog 141](worklogs/141_oracle_single_surface_support_renderer_native_appearance_evidence.md)은 자동 Surface Membership 없이, WL127 raw Visible Surface에 고정된 3-camera polygon oracle support를 만들고 baseline spatial population과 동일한 WL139 graphness/fitter를 비교했다.
- tabletop 후보는 oracle support 후에도 graphness fail, curved rim과 ground는 graphness-PASS 대표면이 생성됐지만 unsupported chart domain과 정성 alignment가 남아 최종 Stage A는 **F. MIXED / INCONCLUSIVE**로 기록했다.
- WL127 PLY에 renderer primitive/contributor provenance가 없어 `NO_VALID_PRIMITIVE_PROVENANCE`로 닫았고, nearest-Gaussian proxy나 SH appearance membership score는 계산하지 않았다. 자동 membership, region growing, continuation, Occluded Surface, canonical renderer 변경은 수행하지 않았다.
- 산출물은 `output/141_oracle_single_surface_support_appearance_evidence/`에 격리했으며 focused tests 23개가 통과했다.

## 2026-08-31 Worklog 142 Multi-view Support Lifting / Projection / Depth / Physical-Sheet Attribution

- [Worklog 142](worklogs/142_multi_view_support_lifting_projection_depth_physical_sheet_attribution.md)는 WL141의 MASK_ONLY_BASELINE을 그대로 재생하고, projection/coordinate contract와 canonical renderer depth_median 기반 depth-layer contamination을 분리 진단했다.
- checkpoint Gaussian center projection은 세 control camera에서 renderer alpha alignment 0.998481/0.998726/0.998684로 기계적 PASS였고, WL141 support는 세 ROI 모두 row-ID count/hash 기준 exact 재현됐다.
- 고정 WL139 mu depth rule 적용 결과 tabletop 1,367→0, curved rim 17,842→0, paver ground 6,220→0으로 MASK_PLUS_DEPTH_SUPPORT가 남지 않았다. curved rim은 세 camera에서 consistent/behind/in-front depth relation이 혼합됐다.
- 최종 attribution은 F. MIXED / INCONCLUSIVE다. depth consistency는 layer contamination evidence이지만 physical-sheet identity proof가 아니므로 automatic membership, representative replay, continuation, Occluded Surface는 실행하지 않았다.
- canonical renderer/checkpoint/161 cameras/WL127/WL139/WL141/Candidate B는 변경하지 않았고, 결과는 output/142_multi_view_support_lifting_projection_depth_attribution/에 격리했다. focused tests 10개가 통과했다.


## 2026-09-01 Worklog 143 Renderer median-depth 의미론 및 multi-view evidence aggregation 감사

- [Worklog 143](worklogs/143_renderer_median_depth_semantics_multi_view_evidence_aggregation.md)는 canonical renderer의 `depth_median`을 실제 CUDA event 정의까지 추적하고, pixel/depth renderer-native self-consistency를 검증한 격리 진단이다.
- 6개 deterministic camera에서 각 6,000개 sample의 reprojection p95가 `1.14e-13` 이하, absolute renderer-z residual p95가 `1.78e-15` 이하로 identity gate를 통과했다. `depth_median`은 Gaussian center z나 Euclidean ray length가 아니라 renderer event의 camera/view-space z다.
- WL141/WL142 `MASK_ONLY_BASELINE`은 세 ROI에서 row-ID count/hash 기준 exact replay됐다. `tabletop 1,367`, `curved rim 17,842`, `paver 6,220`의 `>=2 NEAR`가 모두 0이어서 hard veto가 all-zero 원인이 아니었다. curved rim은 D1 676개만 남고 D2/D3는 0이었다.
- 최종 attribution은 **C. DEPTH REPRESENTATION AND AGGREGATION ARE VALID, BUT THE HISTORICAL MASK SUPPORT HAS LITTLE DIRECT DEPTH CONSISTENCY**다. Surface Membership, representative/SH, continuation, true-occluded prototype, canonical production은 실행하거나 변경하지 않았다.
- 산출물은 `output/143_multi_view_support_lifting_depth_semantics_evidence_aggregation/`에 격리했고, focused tests `14 passed` 및 실제 CUDA `failures=[]`를 확인했다.


## 2026-09-02 Worklog 144 Per-view renderer surface correspondence 및 physical-sheet oracle audit

- [Worklog 144](worklogs/144_per_view_renderer_surface_correspondence_physical_sheet_oracle_audit.md)는 WL141 frozen polygon/camera/ROI를 변경하지 않고, 각 camera의 독립 renderer `depth_median` event cloud를 복원해 surface correspondence를 감사했다.
- 세 ROI 모두 per-view cloud를 별도 저장하고 common-world 3D view, all-target raw reprojection, continuous pairwise distance, fixed-k local differential agreement를 생성했다. WL127 point identity를 multi-view에서 직접 요구하지 않는다.
- tabletop pairwise reciprocal median은 `118.67h` 이상, paver는 `62.70h` 이상이며, curved rim도 한 pair만 median `1.84h`이고 나머지는 `33.07h/34.64h`였다. WL127 MASK_ONLY point에 대한 nearest/second/third cloud distance도 별도 diagnostic으로 기록했다.
- 직접 3D/reprojection review와 정량 분포를 결합한 세 case classification은 모두 **C. SEMANTIC_MASK_MISASSOCIATION**이다. frozen masks가 한 physical sheet가 아니라 ground/tabletop/brace/front-side 구조를 함께 선택한다.
- 자동 Surface Membership, 새 vote/threshold, KNN selection, NURBS/continuation/Occluded Surface, Candidate B 및 canonical code는 변경하지 않았다. 산출물은 `output/144_per_view_renderer_surface_correspondence_physical_sheet_oracle_audit/`에 격리했다.


## 2026-09-02 Worklog 145 Genuine physical-sheet oracle 및 clean-support representative 검증

- [Worklog 145](worklogs/145_genuine_physical_sheet_oracle_clean_support_representative.md)는 WL141 historical mask를 고치지 않고, 새로 수동 동결한 image-space interior polygon에서 canonical renderer `depth_median` event를 per-view로 복원해 genuine physical-sheet oracle을 검토했다.
- `tabletop_broad_planar_clean`은 세 독립 cloud의 pairwise reciprocal median `1.31h~1.83h`와 common-world/reprojection review를 바탕으로 `CLEAR_PHYSICAL_SHEET_ORACLE`로 승격되고, frozen WL139 graphness `PASS_GRAPH_LIKE` 뒤 unchanged representative를 한 번 실행했다. raw→representative median/p95는 `1.33h/2.17h`였다.
- 다만 full representative rectangle의 supported vertex는 `248/3840 (6.46%)`뿐이고 representative→raw median/p95는 `32.40h/77.94h`였다. 결과는 clean observed patch에 대해서만 유효하고 unsupported domain은 검증되지 않은 **B. VALID ONLY ON SUPPORTED DOMAIN**으로 분류했다.
- curved rim과 near-vase 후보는 각각 `PARTIAL / MIXED`로 비승격했다. automatic Surface Membership, WL141 mask repair, continuation, Occluded Surface, SH/appearance completion, canonical production 변경은 없었다. 산출물은 `output/145_genuine_physical_sheet_oracle_clean_support_representative_audit/`에 격리했다.
