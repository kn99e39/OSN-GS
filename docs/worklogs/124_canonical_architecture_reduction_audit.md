# Worklog 124: Canonical Architecture Reduction Audit

상태: **완료**  
범위: Worklog 96–123의 증거와 현재 저장소의 실제 import/call graph를 분리한 bounded architecture audit  
코드 변경: 없음 (문서와 색인만 추가)

## Agent Interpretation of Intent

### DIRECTION

Worklog 96–123의 역사적 결론을 보존한 상태에서 현재 OSN-GS의 표면·renderer·topology·visible NURBS·observation·occlusion·continuation·uncertain 계층을 inventory하고, 현재 실행 경로와 역사적/진단/실험 경로를 분리한다. 목표는 새 표면 알고리즘을 만드는 것이 아니라 가장 작은 방어 가능한 canonical architecture를 닫는 것이다.

### PURPOSE

현재 구현이 실제로 무엇을 실행하는지와 논문 수준에서 무엇을 canonical이라고 부를 수 있는지를 구분한다. 특히 WL123의 `world-space x + optional exact renderer-event provenance` query contract가 현재 실행 그래프에 어떤 자리를 갖는지, downstream occluded-side materialization이 아직 정당화되지 않았음을 명시한다.

### CENTRAL INTENT

세계 좌표의 3D 위치를 global volumetric representation으로 유지한다. renderer provenance는 query의 위치·소유권·신뢰도·component·surface continuation·global visibility를 정의하지 않고, 오직 “이 `x`는 이 renderer median frontier event와 정확히 동일하다”는 source-view identity certificate로만 사용한다. WL122의 18.62% closure failure을 일반 arbitrary-3D query의 불안정성으로 확대 해석하지 않는다.

### PRESERVE

- canonical renderer, pre-update `T > 0.5` median rule, Candidate B `classify_view`, global aggregation, checkpoint, 161 cameras
- WL107/109 topology evidence, WL120 original query bank, WL121 supplemental true-fragmentation bank, WL122 median-frontier evidence, WL123 query-contract evidence
- Candidate B의 historical outputs와 현재 생산 경로
- Worklog 96–123의 측정값과 문서. 특히 과거 번호를 사용한 `docs/worklogs/18_canonical_visible_nurbs_training_integration.md`도 현재 WL124로 재번호화하지 않는다.

### CHANGE ONLY

- 현재 repository의 module/import/call graph를 읽기 전용으로 분류
- WL123에서 이미 진단된 query-representation contract를 architecture layer로 위치시킴
- public export와 premature downstream materialization의 상태를 보고
- 이 Worklog와 `docs/README.md`, `docs/worklogs/README.md`의 색인 링크 추가

### DO NOT

- 파일 이동·삭제·이름 변경·refactor·production query class 추가
- Candidate B, median rule, alpha, topology, NURBS, continuation, Trust, visibility boundary policy 변경
- epsilon, ULP band, `nextafter`, view-count rule, dense voxel architecture 도입
- median depth를 physical first hit으로 부르기
- renderer contribution, surface representative, surface identity, volumetric observation을 동일시하기
- post-median contribution을 redundant 또는 independent physical surface evidence로 라벨링하기

### PROMPT-REQUIRED DECISION

가장 작은 방어 가능한 architecture가 `world-space x`를 canonical query로 유지하면서 exact renderer median-event provenance를 선택적으로 보존할 수 있는지, 그리고 WL123의 18.62% closure failure이 exact event round-trip에 국한되는지를 판정한다. 최종 선택지는 A–D 중 하나이며 자동으로 A를 선택하지 않는다.

### AGENT-INTRODUCED OPERATIONAL CHOICE

현재 tree를 2026-08-27 checkout의 static import/call graph로 audit하고, 과거 worklog의 scientific status와 현재 runtime reachability를 별도 축으로 기록했다. 별도 production branch나 candidate implementation은 만들지 않았고, 공통 math utility 여부는 reusable API인지와 canonical scientific status를 혼동하지 않도록 별도 boolean으로 표시했다.

## Scientific Contract Inventory

| 계약 | 현재 증거 | WL124 판정 |
|---|---|---|
| Renderer observation | canonical Gaussian/Surfel renderer와 pre-update alpha/transmittance | 보존된 입력 계약 |
| Visible frontier | Candidate B, stored per-view median depth | 현재 operational frontier; physical first hit 아님 |
| Visible topology | WL107/109 renderer-native topology evidence | 유효한 입력/제약 후보이나 최종 surface truth 아님 |
| Visible surface geometry | current visible-NURBS construction path | 실행되는 provisional geometry layer |
| Volumetric query | WL123 `world_position` | canonical representation으로 수용 |
| Event provenance | `(camera_id, pixel_id, stored_median_depth, optional representative_id)` | exact source-event identity certificate만 허용 |
| Global observation | frozen ANY-OBSERVED aggregation | 변경 없음; provenance가 대체하지 않음 |
| Occluded partition | WL120/121/122에서 B 외의 독립 winner 없음 | downstream open/premature |
| Continuation/materialization | current optional code와 historical experiments 존재 | 과학적 canonical으로 승격하지 않음 |
| Trust/uncertain promotion | current optional adapters 존재 | 본 batch 범위 밖; premature |

WL96–123의 evidence chain은 “현재 실행되는 pipeline이 있다”는 것과 “각 downstream semantic claim이 입증됐다”는 것을 분리한다. WL123의 query contract verdict는 A였지만, 그것은 query representation 계약이며 occluded surface나 topology repair의 승인과 동일하지 않다.

### Worklog 122 해석 교정

WL122의 측정값은 역사적 evidence로 그대로 보존한다. 다음 해석만 현재 architecture에 명시적으로 carry forward한다.

1. `renderer-visible somewhere`는 `same-surface redundant contribution`을 뜻하지 않는다.
2. Renderer Contribution, Surface Representative, Surface Identity, Volumetric Observation은 별도 개념이다.
3. `same frozen visible component`은 강한 provenance이지만 physical redundancy의 증명이 아니다.
4. “post-median evidence is overwhelmingly redundant representation”은 WL122로 확립되지 않았다.
5. WL122의 2.24%는 stated marginal assumptions 아래에서, 어디에서도 median representative가 아닌 surfel의 post-median contribution에 대한 upper bound일 뿐이다. 독립 hidden-surface evidence의 정확한 양이 아니다.
6. `39.06% * 72.35% = 28.26%`는 weight accounting으로는 유효하다. post-median accepted contributors는 1,150,990,609개이고, front-of-median은 248,820,747개로 **count fraction 21.617965%, weight fraction 27.646166%**이다. at/behind-median은 **count fraction 78.382035%, weight fraction 72.353834%**이다. post-median total contribution weight fraction은 39.054929%이며 `0.39054929 * 0.72353834 = 0.28257739`, 즉 28.257739%이다. 따라서 27.65%를 count fraction으로 읽는 해석은 INVALID지만, 실제 WL122 집계의 27.65%가 weight fraction이므로 historical 28.26% 계산은 VALID하다.

## Historical Frontier Preservation

다음은 이 batch에서 동결된 historical frontier contract이다.

- canonical renderer의 기존 결과와 pre-update `T > 0.5` median selection은 수정하지 않았다.
- Candidate B `classify_view`와 global ANY-OBSERVED aggregation은 수정하지 않았다.
- 161 cameras, WL107/109 topology, WL120 4,712-query bank, WL121 908-query supplemental bank, WL122 median-frontier corpus를 재설계하지 않았다.
- WL123에서 43,817,760 source median events의 historical float32 contradiction은 8,157,322 (18.62%), provenance-preserved contradiction은 0으로 기록되어 있다. 이 수치는 WL124에서 재측정하거나 덮어쓰지 않는다.
- WL123 generic audit의 P1-excluded relevant 1,590,240 query-view pairs 중 diagnostic float32/reference disagreements는 1,118이었다. 이 수치는 “전역 생산 classifier의 실패율”이 아니라 reference attribution audit 결과다.

## Repository Module Inventory

### 현재 tree 규모

- `osn_gs/surface`: `*.py` 107개 (`__init__.py` 포함)
- `scripts/devtools/observed_occluded`: `*.py` 16개
- `osn_gs/gaussian`: model/density/adapter 계층이 존재하며 uncertain append 관련 optional module이 포함됨
- `osn_gs/render`: canonical Gaussian/Surfel renderer와 query-depth/representative/contribution diagnostic sibling이 분리되어 있음
- 별도 `osn_gs/topology` package는 현재 tree에서 확인되지 않았고, visible topology 구현은 주로 `osn_gs/surface`의 historical/current families에 있다.

### 기능별 inventory

| 계층 | 현재 존재하는 대표 모듈/family | 실제 의미 |
|---|---|---|
| renderer | `osn_gs/render/gaussian_rasterizer.py`, `surfel_rasterizer.py`, vendored `diff_gaussian_rasterization` | canonical renderer |
| renderer diagnostics | `torch_surfel_contribution_diagnostics.py`, `torch_surfel_query_depth_diagnostics.py`, `torch_surfel_representative_diagnostics.py` | WL evidence용 sibling; canonical output 변경 없음 |
| Gaussian model | `torch_model.py`, `torch_surfel_model.py`, density-control 계열 | current training/model path |
| visible evidence | covariance frame, manifold affinity, structural reliability, surface region formation, orientation, tangent frame | current visible construction 입력 |
| boundary/topology | world boundary halfedges, ordered world boundary graph, directed ordering, support termination, region status | current visible-NURBS construction path의 boundary stages |
| visible NURBS | `torch_visible_surface_construction.py`, `torch_nurbs.py`, materialization adapter, chart boundary helpers | 실행되는 visible geometry layer; scientific finality는 provisional |
| full-evidence/ADC | density-preserving representative selection, full-neighborhood evidence, region-owned full evidence | optional post-ADC current path |
| old visible mechanisms | voxel, annulus, bilateral, old boundary builders, dense chart/partition families | historical/diagnostic/legacy; canonical reduction 대상 |
| occluded/continuation | `torch_candidate_evidence.py`, `torch_continuation_domain.py`, `torch_occluded_*`, continuation bridge/shell | current code에 남아 있으나 canonical evidence로 승격할 근거 없음 |
| uncertain materialization | `torch_safe_uncertain_proposal_production.py`, `torch_uncertain_gaussian_proposal.py` 및 `osn_gs/gaussian` adapters | optional/manual route; premature |
| WL120–123 audit harness | `scripts/devtools/observed_occluded` 16개 | diagnostic contract audit; production path 아님 |

## Current Canonical Execution Graph

실제 current path는 다음과 같이 축약된다.

```text
train.py
  -> TorchPipelineConfig / TorchOSNGSTrainer
  -> OSNGaussianRasterizer (default gaussian_3d)
  -> visible_nurbs_update_schedule=initialize
  -> construct_visible_nurbs_from_gaussians
       covariance frame
       -> reliability / manifold affinity
       -> visible surface regions
       -> orientation / canonical tangent frames
       -> world-space boundary halfedges
       -> support termination / ordering / recovery
       -> visible boundary region status
       -> visible boundary materialization
       -> TorchNURBSSurface binding (cluster_ids, surface_uv)
  -> normal training / ADC
  -> optional post-ADC full-evidence reconstruction
       representative selection -> full-neighborhood evidence
       -> same visible constructor -> region-owned full evidence fit
```

정적 import audit의 실제 핵심 결과는 다음과 같다.

```text
surface_py_files=107
surface_public_exports=54
surface_init_export_names_unique=True
core_direct_surface_imports=
  torch_density_preserving_representative_selection,
  torch_full_cloud_continuation_shell,
  torch_full_neighborhood_evidence,
  torch_gaussian_covariance_frame,
  torch_gaussian_structural_reliability,
  torch_nurbs,
  torch_region_owned_full_evidence,
  torch_visible_surface_construction
core_imports_premature_occluded=False
core_imports_renderer_native_topology=False
```

위의 direct-import 목록은 `torch_full_cloud_continuation_shell`이 optional post-ADC 경로에 실제로 연결되어 있다는 뜻이지, 그 shell이 scientific canonical으로 승인됐다는 뜻이 아니다. main training loop는 uncertain activation을 호출하지 않는다. `scripts/devtools/observed_occluded`의 `volumetric_query.py`와 query-contract harness도 WL123 diagnostic contract를 실행할 뿐 production trainer의 canonical query object를 추가하지 않는다.

## Module Classification Matrix

분류 기준은 다음과 같다. A는 현재 기본 또는 명시된 optional current training path에 도달 가능한 모듈, B는 현재 visible architecture에 연결되거나 재사용 가능하지만 scientific closure가 남은 provisional module, C는 과거 baseline, D는 진단/attribution/성능 측정, E는 독립 실험 후보, F는 superseded 또는 premature downstream이다. `R=Y`는 독립 reusable math/geometry utility로 볼 수 있음을 뜻하며 canonical scientific status와는 별개이다. 아래 목록은 현재 `osn_gs/surface`의 107개 파일 중 `__init__.py`를 제외한 106개와 audit harness 16개를 빠짐없이 한 category에만 배정한다.

| 범위 | category | reusable_math_utility |
|---|---|---|
| `torch_boundary_eligibility.py`, `torch_boundary_support_network.py`, `torch_boundary_support_termination.py`, `torch_canonical_region_tangent_frame.py`, `torch_density_preserving_representative_selection.py`, `torch_directed_boundary_ordering.py`, `torch_full_neighborhood_evidence.py`, `torch_gaussian_covariance_frame.py`, `torch_gaussian_manifold_affinity.py`, `torch_gaussian_structural_reliability.py`, `torch_gaussian_surface_orientation.py`, `torch_gaussian_surface_region_formation.py`, `torch_nurbs.py`, `torch_ordered_boundary.py`, `torch_ordered_world_boundary_graph.py`, `torch_region_owned_full_evidence.py`, `torch_region_parametric_chart_boundary.py`, `torch_structural_normal.py`, `torch_surfel_surface_orientation.py`, `torch_termination_neighborhood_scale.py`, `torch_visible_boundary_materialization_adapter.py`, `torch_visible_boundary_region_status.py`, `torch_visible_surface_construction.py`, `torch_world_space_boundary_halfedges.py` | **A CANONICAL**: current execution path | `Y`: covariance/affinity/reliability/NURBS/normal/tangent/ordering/termination helpers; `N`: trainer-facing orchestration/materialization |
| `torch_boundary_source_fidelity.py`, `torch_boundary_support_spacing.py`, `torch_boundary_surface_quality.py`, `torch_component_boundary.py`, `torch_constrained_chart_lsq.py`, `torch_full_region_surface_face_topology.py`, `torch_gaussian_support_continuity.py`, `torch_region_coherent_surfel_partition.py`, `torch_region_owned_boundary_materialization.py`, `torch_region_owned_dense_boundary_support.py`, `torch_sampled_surface_geometry.py`, `torch_surface_candidate_graph.py`, `torch_surface_components.py`, `torch_surface_decomposition.py`, `torch_surface_region_validation.py`, `torch_trimmed_component_fitter.py` | **B PROVISIONAL CANONICAL**: visible geometry/topology candidates still used or reusable, but not architecture proof | `Y`: local geometry/fitting/graph helpers; `N`: evidence assembly and policy modules |
| `torch_bilateral_interface_region_merge.py`, `torch_boundary_component_recovery.py`, `torch_boundary_first_visible_builder.py`, `torch_boundary_planar_partition.py`, `torch_boundary_reconciliation.py`, `torch_boundary_refinement.py`, `torch_boundary_review_geometry.py`, `torch_boundary_self_intersection.py`, `torch_coverage_first_subset_partition.py`, `torch_discontinuity_first_surfel_partition.py`, `torch_interface_coherent_region_merge.py`, `torch_maximal_visible_connectivity.py`, `torch_positive_visible_adjacency.py`, `torch_primitive_ownership_visible_topology_separation.py`, `torch_region_adaptive_support_merge.py`, `torch_voxel_regions.py`, `torch_patch_boundary.py` | **C HISTORICAL BASELINE**: WL96–109 lineage or older visible mechanisms preserved as evidence | `Y`: standalone partition/geometry routines; `N`: historical policy/experiment wrappers |
| `torch_aabb_broad_phase.py`, `torch_camera_induced_visible_adjacency.py`, `torch_camera_observed_chart_domains.py`, `torch_chart_topology.py`, `torch_chart_unit_evidence_scale_boundary.py`, `torch_chart_unit_face_incidence_partition_boundary.py`, `torch_chart_unit_latent_midsurface_attribution.py`, `torch_chart_unit_local_center_geometry_attribution.py`, `torch_chart_unit_partition_seam.py`, `torch_chart_unit_surface_topology_attribution.py`, `torch_chart_unit_surface_topology_temporal_lineage.py`, `torch_chart_unit_topology_partition_boundary.py`, `torch_dense_boundary_connectivity_diagnostics.py`, `torch_dense_boundary_scale_diagnostics.py`, `torch_dense_chart_unit_assembly.py`, `torch_dense_parametric_chart_support.py`, `torch_dense_surface_consistency_components.py`, `torch_exact_knn_performance.py`, `torch_intrinsic_boundary_parameterization.py`, `torch_knn_reference_attribution.py`, `torch_local_orientation_folding.py`, `torch_node_level_observability_accounting.py`, `torch_nonrepresentative_evidence_attribution.py`, `torch_nurbs_performance_batch.py`, `torch_parametric_diagnostics.py`, `torch_renderer_grounded_visible_adjacency.py`, `torch_single_chart_uv_validity.py`, `torch_surface_evidence_representation_gate.py`, `torch_surface_proxy.py`, `torch_voxel_hierarchy.py` | **D DIAGNOSTIC**: attribution, topology audit, performance, or old mechanism measurement | `Y`: broad phase, parameterization, topology bookkeeping, KNN/NURBS measurement helpers; `N`: report/diagnostic orchestration |
| `torch_boundary_constrained_surface.py`, `torch_boundary_central_cap.py`, `torch_boundary_multi_loop.py`, `torch_boundary_role_evidence.py`, `torch_coons_patch.py`, `torch_local_rank_complete_chart_growth.py` | **E EXPERIMENTAL CANDIDATE**: independent visible candidate not adopted by current canonical path | `Y`: patch/curve/graph primitives; `N`: candidate selection or policy |
| `torch_annulus_chart.py`, `torch_candidate_evidence.py`, `torch_chart_conflict.py`, `torch_continuation_domain.py`, `torch_eligible_boundary_continuation_bridge.py`, `torch_full_cloud_continuation_shell.py`, `torch_observation_evidence.py`, `torch_occluded_chart.py`, `torch_occluded_chart_hardening.py`, `torch_occluded_region_candidate.py`, `torch_region_owned_full_evidence_boundary_topology.py`, `torch_safe_uncertain_proposal_production.py`, `torch_uncertain_gaussian_proposal.py` | **F SUPERSEDED/PREMATURE**: old visible annulus/boundary semantics or unvalidated occluded/continuation/uncertain materialization | `Y`: generic chart/patch primitives where independently reusable; `N`: semantic proposal, ownership, hardening, continuation, or uncertain policy |
| `osn_gs/surface/__init__.py` | **D DIAGNOSTIC/API façade** for this audit; no scientific status inferred from re-export | `N` |

`torch_full_cloud_continuation_shell.py`은 현재 optional call graph에 존재하지만 WL96–123의 closure를 넘어서는 continuation construction이므로 A가 아니라 F로 분류한다. 반대로 현재 visible constructor가 실제로 호출하는 support/ordering/materialization module은 A로 분류한다. 이 구분은 “실행된다”와 “과학적으로 최종 승인됐다”를 분리하기 위한 것이다.

### Audit harness 분류

`scripts/devtools/observed_occluded/__init__.py`, `candidate_a_surface_hit.py`, `candidate_b_median_depth.py`, `candidate_c_geometric_visibility.py`, `candidate_d_renderer_reachability.py`, `engine.py`, `frontier_synthetic_contracts.py`, `frontier_validation.py`, `query_bank.py`, `query_contract_synthetics.py`, `shared.py`, `synthetic_contracts.py`, `synthetic_value_contracts.py`, `topology_gap_bank.py`, `value_diagnostics.py`, `volumetric_query.py`는 모두 **D DIAGNOSTIC**이다. 이 중 `shared.py`만 `R=Y`이고 나머지는 `R=N`이다. `candidate_b_median_depth.py`의 historical decision function도 production Candidate B를 변경하지 않는 audit mirror이다.

## Public API Audit

현재 `osn_gs.surface.__init__`은 54개의 unique export를 노출하며 import 자체는 통과한다.

```text
surface_import=PASS exports=54 unique=54
```

이 export 목록은 현재 동작을 보존하기 위해 당장 변경하지 않는다. architecture reduction 관점의 후속 상태는 다음과 같다.

| 상태 | 현재 export |
|---|---|
| `KEEP_CANONICAL` | `TorchCurveSet`, `TorchNURBSSurface`, `build_torch_surface`, `fit_torch_visible_surface`, `fit_torch_visible_surface_lsq` |
| `KEEP_INTERNAL` | `boundary_control_indices`, `classify_world_samples`, `fit_torch_base_curves`, `pca_parameterize_points`, `project_torch_points_to_nurbs`, `extract_trimmed_patch_boundaries` |
| `MOVE_TO_LEGACY_LATER` | `EmptyVoxelSupportResult`, `TorchVoxelSurfaceRegions`, `build_torch_voxel_surface_regions`, `query_empty_voxel_support`, `PatchBoundarySegment`, `PatchEdgePair`, `PatchReconciliationResult`, `build_rectangular_patch_edge` |
| `DIAGNOSTIC_ONLY` | `BroadPhasePair`, `CameraViewEvidence`, `CorrespondenceEdge`, `SampleEvidence`, `SharedBoundaryConstraint`, `SupportChain`, `sweep_and_prune_pairs`, `validate_candidate_observation_evidence` |
| `REMOVE_FROM_PUBLIC_API_LATER` | `ConflictEdge`, `ContinuationDomain`, `ContinuationDomainBuildError`, `ObservationEvidence`, `OccludedChartFitConfig`, `OccludedChartHardeningConfig`, `OccludedChartSafetyResult`, `OccludedChartConflictEdge`, `OccludedChartResult`, `GaussianToSafeUncertainProposalResult`, `SafeUncertainProposalAttempt`, `SafeUncertainProposalProductionResult`, `build_safe_uncertain_proposals_from_bridge`, `run_safe_uncertain_proposals_from_gaussians`, `OccludedRegionCandidate`, `attach_conflict_edges`, `build_candidate_conflicts`, `build_occluded_chart_conflicts`, `build_continuation_domain`, `build_geometric_region_candidates`, `build_observation_evidence`, `evaluate_occluded_chart_safety`, `fit_coupled_patch_graph_lsq`, `fit_occluded_chart`, `predict_torch_occlusion_curves`, `sample_torch_occluded_surface` |

상태 표는 cleanup 명령이 아니다. 이 batch에서는 export를 옮기거나 제거하지 않았고, public compatibility와 architecture recommendation을 분리했다. 특히 `ObservationEvidence`와 `CandidateEvidence` 계열은 이름이 observation을 포함하더라도 WL123 query provenance의 대체가 아니다.

## Premature Occluded-Side Audit

다음 경로는 현재 tree에 존재하거나 optional call path에 연결되어 있지만 이 batch의 canonical architecture로 승격할 수 없다.

| premature family | 관찰된 상태 | WL124 결론 |
|---|---|---|
| `ObservationEvidence`, `CandidateEvidence` | evidence/candidate proposal 객체와 export 존재 | renderer-event identity 또는 global observation의 shortcut으로 사용 금지 |
| `ContinuationDomain`, continuation bridge/shell | optional continuation input과 historical bridge가 존재 | surface continuation을 추론할 근거 없음; F |
| `OccludedRegionCandidate`, `OccludedChart`, `OccludedChartHardening` | fit/safety/conflict/hardening pipeline 존재 | occluded NURBS materialization 미승인; F |
| `ChartConflict`, `SafeUncertainProposal` | conflict graph와 uncertain proposal production 존재 | trust/ownership/uncertain promotion을 도입하지 않음; F |
| `region-owned full-evidence boundary topology` | post-ADC optional code에 materialization helper 존재 | current execution reachability만으로 scientific closure가 생기지 않음; B/F 경계 중 downstream은 F |
| continuation boundary reconciliation / uncertain generation | 여러 old helper와 gaussian adapters 존재 | topology/continuation/uncertain batch로 넘기지 않음 |

현재 `TorchOSNGSTrainer.activate_and_train_uncertain_step` API가 존재한다는 사실과 main `train.py` loop가 그것을 호출한다는 사실은 다르다. static call audit상 main loop는 uncertain activation을 실행하지 않으며, 이를 canonical architecture에 포함시키지 않았다.

WL123의 provenance contract도 이 family들을 정당화하지 않는다. provenance는 source view에서 exact `ON_FRONTIER` identity만 인증한다. 다른 view는 frozen geometric frontier comparison을 사용하고, global은 frozen ANY-OBSERVED를 사용한다. provenance로 unrelated-view visibility, global observation, component ownership, surface continuation, trust를 결정하지 않는다.

## Minimal Canonical Architecture

이 audit에서 방어 가능한 최소 개념 구조는 다음이다.

```text
RendererObservation
  -> VisibleTopology                 (WL107/109 evidence; provisional)
  -> SurfaceGeometry                 (current visible NURBS; provisional)
  -> VisibleObservationFrontier      (Candidate B stored median depth)

VolumetricQuery {
    world_position x                 # canonical, always present
    optional renderer_event_provenance {
        camera_id
        pixel_id
        stored_median_depth
        representative_id             # only when already available
    }
}

VisibleObservationFrontier + VolumetricQuery
  -> per-view frontier state
  -> frozen global ANY-OBSERVED

later, only after new evidence:
  -> ObservedOccludedPartition
  -> continuation/materialization
```

`renderer_event_provenance`는 `world_position`을 수정하지 않는다. provenance가 valid한 source view에 대해서만 “이 query는 이 exact stored median event”를 답한다. provenance를 제거한 동일 world coordinate는 ordinary float32 geometric evaluation으로 진단하며 답을 강제하지 않는다. Q3/Q4처럼 clearly camera-side/behind인 점에는 provenance를 적용하지 않는다.

이 구조는 event-native representation으로 global volume을 대체하지 않는다. renderer event가 아닌 arbitrary 3D 위치는 동일한 `world_position` query path를 사용한다. Occluded partition은 이 구조의 downstream placeholder일 뿐이며 이번 batch에서 구현하거나 materialize하지 않는다.

## Complexity Accounting

현재 complexity를 algorithmic big-O 하나로 합치지 않고 역할별로 분리했다.

| 비용/복잡성 | 현재 위치 | architecture 의미 |
|---|---|---|
| renderer observation | 161-camera render와 stored per-pixel events | 입력 비용; query representation이 대체하지 않음 |
| visible evidence/topology | Gaussian/surfel covariance, KNN/affinity, region/boundary graph | 현재 runtime의 주요 scientific/compute path |
| visible NURBS | chart fitting, basis/projection, boundary materialization | 실행되지만 provisional geometry layer |
| volumetric query | world-space position projection/depth compare, optional source identity lookup | 최소 canonical query cost; camera/pixel tensor를 global key로 만들지 않음 |
| global aggregation | per-view state의 frozen ANY-OBSERVED reduction | WL120–123 보존, view-count rule 없음 |
| diagnostics | WL120–123 exhaustive banks, reference arithmetic, synthetic contracts | report-only; production cost에 포함하지 않음 |
| occluded/continuation/uncertain | multiple candidate/helper graphs and optional adapters | 현재 tree complexity는 크지만 canonical contract에는 포함하지 않음 |

따라서 canonical reduction의 핵심은 새 asymptotic optimization이 아니라 semantic dependency reduction이다. 107 surface files와 54 exports 전체를 canonical core로 취급하면 visible legacy와 premature occluded mechanisms가 scientific dependency처럼 보이는 문제가 생긴다. 반대로 최소 core는 renderer observation, visible topology/geometry evidence, stored median frontier, world-space query, exact event identity certificate, frozen aggregation으로 제한된다.

## Paper-Level Architecture

OSN-GS는 canonical renderer에서 얻은 `RendererObservation`을 이용해 visible topology와 visible surface geometry를 구성한다. visible observation frontier는 각 view/pixel의 pre-update `T > 0.5` event들을 이용해 저장한 Candidate B median depth로 정의한다. arbitrary volumetric location은 world-space position `x`로 표현한다. 어떤 query가 renderer median event에서 직접 생성된 경우에만 optional provenance `(camera_id, pixel_id, stored_median_depth, representative_id?)`를 붙이며, 이 provenance는 source view의 exact frontier identity certificate로만 해석한다. 동일한 `x`의 다른 view 판정은 통상의 frozen frontier comparison으로 수행하고 global state는 ANY-OBSERVED aggregation으로 계산한다. Surface representative, renderer contribution, surface identity, volumetric observation은 분리한다. visible topology/NURBS는 현재 실행되는 provisional geometry layer이고, observed/occluded partition·continuation·occluded NURBS·Trust는 별도 evidence가 생길 때까지 downstream open이다.

이 문장이 현재 구현과 논문 claim 사이의 최소 공통분모다. median depth는 physical first hit이 아니며, post-median contribution은 direct surface observation으로 부르지 않는다.

## Implementation Fidelity Statement

이번 Worklog 124에서는 production behavior를 변경하지 않았다.

- Candidate B decision function, renderer, global aggregation, topology, NURBS, alpha, `T > 0.5`, checkpoint, 161 cameras는 수정하지 않았다.
- WL96–123 historical worklog와 output corpus는 수정하지 않았다.
- WL123 query representation은 diagnostic contract로만 audit했으며 production `VolumetricQuery` class나 tolerance를 추가하지 않았다.
- `osn_gs.surface.__init__`의 54 exports는 그대로 두었다.
- static import audit와 public import check를 수행했고, focused regression은 `26 passed, 1 warning in 8.18s`였다. warning은 `torch_voxel_hierarchy.py`의 existing `requires_grad=True` tensor conversion warning이며 이번 변경으로 생긴 것이 아니다.
- 위 결과는 implementation completion이 아니라 architecture inventory completion이다. 후속 topology repair, Occluded Surface, Trust, uncertain promotion을 자동으로 시작하지 않는다.

## Architecture Verdict

### A. WORLD-SPACE VOLUMETRIC QUERY + EVENT PROVENANCE IS A VIABLE CANONICAL QUERY CONTRACT

이 verdict는 **query contract에 한정**한다.

근거는 다음과 같다.

1. WL123 exhaustive source median-event corpus 43,817,760건에서 provenance-preserved exact identity contradiction은 0건이었다. historical float32 world round-trip contradiction 8,157,322건(18.62%)은 provenance로 round-trip 재판정을 제거하면 사라진다.
2. generic arbitrary-3D audit에서 P1 zero-thickness frontier 이벤트를 제외한 relevant pair 1,590,240건 중 reference disagreement은 1,118건으로 diagnostic attribution되었고, majority는 same-pixel frontier arithmetic 및 6건의 discrete pixel boundary였다. 이 audit는 measured margin을 production tolerance로 바꾸지 않았다.
3. WL123 cross-view replay에서 3,400 historical anchors의 19 global contradictions가 source-event identity 보존으로 0이 되었고, 다른 view는 ordinary frozen geometric evaluation을 유지했다.
4. WL121 true-fragmentation 300 contexts에서 endpoint provenance identity를 보존해 endpoint contradiction을 제거하면서 midpoint(no provenance) classifications는 변경하지 않았다. midpoint OBSERVED를 continuity나 component merge로 사용하지 않았다.

따라서 WL122의 18.62%는 **exact renderer-event round-trip closure problem**으로 해석하는 것이 가장 작은 설명이다. 일반 arbitrary-3D volumetric classification이 “event-native representation만 가능하다”고 결론낼 증거는 없다. 동시에 visible topology/NURBS와 occluded semantics 전체가 canonical으로 닫혔다고 주장하지 않는다.

## Remaining Architecture Question

남은 질문은 query representation이 아니라 downstream semantic evidence다. 즉, world-space query와 exact event identity certificate를 유지한 상태에서 어떤 추가 관측이 `ObservedOccludedPartition`과 continuation/materialization을 정당화하는지 아직 결정되지 않았다. 이 batch는 그 질문을 topology repair, Occluded Surface construction, Trust 구현으로 확장하지 않는다.

## Exact Branch / Commit / Commands / Outputs

### Branch and commit

- branch: `arch/2dgs-coverage-first-surface`
- base before this batch: `273bd50 Record Worklog 123 commit reference`
- Worklog 124 documentation commit: `TO-BE-RECORDED`
- 다른 agent의 기존 변경: `M docs/agent_memory/MEMORY.md`, `?? docs/agent_memory/feedback_view_readme_required.md` — 보존했고 staging하지 않았다.

### Commands and outputs

```text
git status --short --branch
## arch/2dgs-coverage-first-surface...origin/exp/2dgs-nurbs-surface-evidence
M docs/agent_memory/MEMORY.md
?? docs/agent_memory/feedback_view_readme_required.md

static module/import audit
surface_py_files=107
surface_public_exports=54
surface_init_export_names_unique=True
core_imports_premature_occluded=False
core_imports_renderer_native_topology=False

.venv\Scripts\python.exe -B -c "import osn_gs.surface as s; ..."
surface_import=PASS exports=54 unique=54

.venv\Scripts\python.exe -B -m pytest -q tests\test_torch_pipeline_smoke.py tests\test_visible_surface_construction.py tests\test_observation_evidence.py tests\test_primitive_ownership_visible_topology_separation.py
26 passed, 1 warning in 8.18s
```

문서 링크 추가와 commit 후에는 `git diff --check`, `git status --short --branch`, `git log -2 --oneline`을 다시 실행한다. 이 Worklog의 commit placeholder는 첫 문서 commit SHA로 갱신한다. 최종 commit SHA는 다음 단계의 commit-reference update에서 확인한다.

