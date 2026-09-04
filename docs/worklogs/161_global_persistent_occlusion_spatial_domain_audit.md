# Worklog 161 — Global Persistent-Occlusion Spatial Domain 및 Occluded-Region Contract Audit

## 상태

W160 pointwise observability contract의 relevance/evidence semantics를 executable source와 frozen real replay로 재조정했다. 그 결과 historical semantics는 일관되게 확인됐지만, Gaussian이 없는 hidden location까지 포함하는 승인된 pre-latent spatial query domain과 discretization/indexing contract는 발견되지 않았다. 따라서 spatial field를 만들지 않고 **`OCCLUSION_DOMAIN_CONTRACT_GAP`**에서 중단했다.

## 1. W160 reconciliation

W160과 동일한 canonical 2DGS checkpoint, frozen W153 median-depth cache, 161개 training camera, camera calibration, projection convention, query-depth convention, historical Candidate-B 및 `aggregate_global`을 사용했다. Gaussian center `1,190,469`개는 W161 spatial domain으로 간주하지 않고 pointwise semantics audit population으로만 사용했다.

- per-view pair: `1,190,469 × 161 = 191,665,509`
- W153 depth map: 161개, `depth_sha256 = 77c78ae9feb19b3f63bc9cd0145d35944d3d0a8afd06a516f973ddeb17f70c8e`
- `h = 0.012105485424399376`, `mu = 0.03631645627319813`
- historical Candidate-B 변경: 없음
- W154–W159 변경: 없음

## 2. Geometric relevance와 renderer evidence availability

실행 가능한 source를 직접 추적했다.

- `scripts/devtools/observed_occluded/shared.py::project_queries`는 `w > 0`, camera-space `z >= 0.2`, rounded pixel이 image 내부인지로 **GEOMETRICALLY_RELEVANT**를 결정한다.
- `scripts/devtools/observed_occluded/candidate_b_median_depth.py::classify_view`는 geometric relevance를 먼저 유지하고, relevant query의 median-depth sentinel(`median_depth > 0`)이 없으면 **UNRESOLVED**를 부여한다.
- valid median event가 있을 때만 `query_depth <= median_depth`를 `OBSERVED`, `query_depth > median_depth`를 `OCCLUDED`로 분류한다.

따라서 historical executable contract는 다음의 Semantics A다.

> geometrically relevant + renderer evidence unavailable → `UNRESOLVED`

Semantics B인 “no valid median event → `NON_RELEVANT`”는 historical code의 동작이 아니다.

실제 161개 camera 전체 결과는 다음과 같다.

- geometrically relevant pairs: `55,452,404`
- renderer evidence available pairs: `55,452,404`
- geometrically relevant but evidence unavailable pairs: `0`
- evidence availability / geometric relevance: `100%`

이번 scene에서는 median map이 모든 geometrically relevant pixel에 valid event를 가지고 있어 두 semantics의 실제 결과 차이가 나타나지 않았다. 이는 Semantics B를 채택했다는 뜻이 아니다.

## 3. Historical global-state impact

실제 Gaussian-center population에서 Semantics A와 hypothetical Semantics B를 모두 계산했다.

| Global state | Semantics A | Semantics B | A→B |
|---|---:|---:|---:|
| `OBSERVED` | 798,304 | 798,304 | 798,304→798,304 |
| `OCCLUDED` | 391,457 | 391,457 | 391,457→391,457 |
| `UNRESOLVED` | 708 | 708 | 708→708 |

global-state disagreement은 `0`이다. 실제 no-evidence pair가 0이기 때문이다.

그러나 synthetic D에서 `[UNRESOLVED, OCCLUDED]`는 historical Semantics A에서 global `UNRESOLVED`이고, Semantics B에서 첫 camera를 `NON_RELEVANT`로 제거하면 global `OCCLUDED`가 된다. 따라서 “absence of evidence must not silently cast an OCCLUDED vote” 원칙에 맞는 historical Semantics A를 보존했다.

## 4. Spatial query-domain audit

기존 후보를 source/artifact 계약으로 구분했다.

1. `osn_gs/surface/torch_voxel_regions.py::build_torch_voxel_surface_regions`
   - Gaussian point 입력의 min/max에서 AABB를 만들고 occupied cell만 보존한다.
   - output type 자체가 surface-aligned adaptive cells 및 visible NURBS patch candidate partition이다.
   - Gaussian이 없는 hidden location을 query할 수 있는 complete scene domain이 아니므로 부적격이다.

2. `osn_gs/surface/torch_voxel_hierarchy.py::build_voxel_gaussian_hierarchy`
   - module docstring상 retained experimental `voxel_patch_stage1` ablation이다.
   - Gaussian center-derived root AABB, raw Gaussian indices, occupied hierarchy만 제공한다.
   - 승인된 all-space pre-latent occlusion domain이 아니므로 부적격이다.

3. W153 sparse authoritative TSDF field
   - renderer event에서 파생된 authoritative voxel key/value support다.
   - missing key는 `UNKNOWN`이며 complete spatial domain의 empty cell semantics가 아니다.
   - W153 field를 Occlusion Domain으로 재해석하지 않았다.

4. `osn_gs/surface/torch_continuation_domain.py::build_continuation_domain`
   - 기존 boundary에 대한 boundary-local continuation strip이며 production path에 연결되지 않은 future continuation diagnostic이다.
   - Gate V/C와 latent geometry를 이 batch의 spatial bounds로 사용할 수 없다.

`docs/current_framework.md`도 현재 기본 경로를 Gaussian 및 canonical visible-NURBS lifecycle로 기술하고, uncertain Gaussian 자동 생성·append 및 uncertain-to-certain promotion을 기본 경로에서 수행하지 않는다고 명시한다. `docs/worklogs/120_observed_occluded_volumetric_operationalization_audit.md`의 dense volumetric grid를 canonical 표현으로 도입하지 않았다는 기록도 유지했다.

## 5. Discretization / resolution contract

승인된 general scene-domain resolution 또는 indexing contract는 존재하지 않는다. W153의 `h`는 TSDF surface-field discretization이며 general occlusion query spacing으로 승인된 값이 아니다. 그러므로 W161에서 `h`를 재사용해 spatial lattice를 만들지 않았고, 새 voxel size, scene bounds, sample resolution, cell key 또는 empty-cell semantics를 선택하지 않았다.

## 6. Stop-condition result

W160 pointwise classifier는 유효하지만 다음 계약이 빠져 있다.

- Gaussian-center AABB와 독립된 canonical scene/query bounds
- Gaussian 없는 hidden/empty location을 포함하는 승인된 query population
- general spatial discretization resolution 및 indexing
- complete pre-latent domain의 native cell-complex semantics

따라서 Stop Condition은 **`OCCLUSION_DOMAIN_CONTRACT_GAP`**이다. `RELEVANT_EVIDENCE_SEMANTIC_MISMATCH`는 아니다. 실제 replay에서 relevance/evidence가 일치했고, executable historical semantics도 하나로 결정됐다.

## 7. Global occlusion field

real spatial field는 구성하지 않았다. total domain queries, spatial `GLOBAL OBSERVED / GLOBAL OCCLUDED / GLOBAL UNRESOLVED` counts, relevant-camera count distribution은 Gaussian-center counts와 혼동하지 않도록 산출하지 않았다. Gaussian-center global counts는 3절의 semantic reconciliation 증거일 뿐 W161 spatial-field accounting이 아니다.

## 8. Occluded-region accounting

canonical spatial cell complex가 없으므로 native face adjacency를 적용할 대상이 없다. `GLOBAL OCCLUDED` region 수, size distribution, largest-region fraction, singleton count, world AABB를 발명하지 않았다. KNN, radius, 18/26-neighbor, morphological closing, smoothing, dilation, bridging도 사용하지 않았다.

## 9. Gaussian / TSDF independence

W161 spatial query population 자체가 성립하지 않아 Gaussian proximity와 TSDF-authority independence를 정량화하지 않았다. 새 near-Gaussian radius를 만들지 않았고, Gaussian center 또는 W153 authoritative voxel의 부재를 spatial `OCCLUDED` 근거로 사용하지 않았다. 이 미실행은 누락이 아니라 domain gap으로 인한 의도된 중단이다.

## 10. Synthetic contracts A–F

pointwise/global mechanics를 synthetic으로 확인했다.

- A: visible bounded volume → false `GLOBAL OCCLUDED` 없음
- B: 모든 relevant view에서 hidden → `GLOBAL OCCLUDED`
- C: 여러 view 중 하나에서 visible → `GLOBAL OBSERVED`
- D: relevant camera의 renderer event 없음 → historical Semantics A에서 `UNRESOLVED`, `OCCLUDED`로 silent promotion 없음
- E: 모든 camera relevance 밖 → `UNRESOLVED`, `GLOBAL OCCLUDED` 아님
- F: Gaussian membership 없는 hidden query → arbitrary-query classifier mechanics는 유지됨

모두 통과했다. Synthetic PASS는 mechanics만 검증하며 real spatial domain의 존재나 hidden surface의 존재를 증명하지 않는다.

## 11. Real-scene quantitative result

real replay는 pointwise semantic audit까지만 수행했다. 161개 camera accounting을 모두 기록했고 각 camera에서 geometrically relevant pairs와 evidence-available pairs가 같았다. spatial domain classification과 spatial population accounting은 Stop Condition 때문에 시작하지 않았다.

## 12. Human qualitative review exports

요청된 canonical review cameras `DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`를 future spatial review target으로 기록했지만, 현재는 표시할 canonical spatial field가 없다. 따라서 `global_state`, `global_occluded_only`, `global_unresolved`, `occluded_region_ids`, `relevant_view_count`, `common_world` PNG를 만들지 않았다. 빈 field를 임의 bounds로 채운 이미지를 architecture evidence로 남기지 않았다.

`output/161_global_persistent_occlusion_spatial_domain_audit/review_views/README.md`에는 이 중단 이유와 향후 고정 palette를 기록했다. 이 batch는 `HUMAN_REVIEW_REQUIRED` 상태이며 qualitative success를 자동 선언하지 않는다.

## 13. Human review questions

다음 질문은 canonical spatial domain이 승인된 뒤에만 사람이 답할 수 있다.

1. `GLOBAL OCCLUDED`가 out-of-FOV가 아니라 physically plausible hidden location인가?
2. missing camera evidence가 `UNRESOLVED`로 보존되는가?
3. 어떤 `GLOBAL OCCLUDED` location이 canonical camera에서 직접 reachable한 것처럼 보이지 않는가?
4. Gaussian center가 없는 위치에도 spatial `OCCLUDED`가 존재하는가?
5. major Occluded Region이 coherent한가, sparse isolated cell 집합인가?
6. region boundary가 fused-TSDF coverage가 아니라 persistent observability 변화에 대응하는가?

현재는 domain contract가 없어 이 질문들에 답할 수 있다고 주장하지 않는다.

## 14. Visible-surface branch와의 관계

- Gate O1: pointwise per-view/global classifier → W160에서 닫혔고 W161에서 relevance/evidence semantics를 재확인했다.
- Gate O2: spatial `GLOBAL OCCLUDED` domain → **열림**, `OCCLUSION_DOMAIN_CONTRACT_GAP`.
- Gate V: visible source surface / support / boundary / representative → 여전히 열림.
- Gate C: continuation → 시작하지 않음.

W154–W159의 Gaussian Region, TSDF ownership, zero-set topology, Boundary First, NURBS 및 continuation 결과는 변경하지 않았다. Gate O2와 Gate V가 닫히기 전에는 continuation으로 진행하지 않는다.

## 15. Architecture verdict

### INTENT ALIGNMENT

`PASS`. 새로운 bounds, resolution, visibility heuristic, latent evidence 또는 hidden geometry를 만들지 않았고, pointwise classifier와 spatial-domain 존재 여부를 분리했다.

### IMPLEMENTATION FIDELITY

`PASS`. executable `project_queries`, `candidate_b.classify_view`, `aggregate_global`과 confirmed W153 cache를 사용했다. Semantics A/B를 실제 pair-level 및 global-level로 비교했고, synthetic A–F도 통과했다.

### ARCHITECTURE RESULT

**`OCCLUSION_DOMAIN_CONTRACT_GAP`**. W160 pointwise contract는 임의 3D query에 적용 가능하지만, 이를 Gaussian/TSDF/NURBS에 의존하지 않는 canonical pre-latent spatial domain으로 확장할 승인된 bounds·population·resolution·indexing·topology contract가 현재 없다.

## 16. Retained / rejected / open

### Retained

historical Candidate-B, W160 pointwise classifier, W153 median-depth maps, canonical checkpoint, 161 cameras, camera calibration, projection/depth convention, historical global aggregation, W153 fused TSDF artifact, W154–W159 artifacts를 유지했다.

### Rejected

Gaussian-center population을 spatial domain으로 사용, Gaussian-center AABB 확장, W153 authoritative support를 complete domain으로 사용, W153 `h`의 general-domain 재사용, fused TSDF sign을 occlusion으로 사용, `UNRESOLVED`를 `OCCLUDED`로 promotion, latent/NURBS/continuation bounds, 새 connectivity 및 region repair를 거부했다.

### Open

canonical pre-latent scene/query bounds, 승인된 all-space spatial discretization/indexing, complete domain의 native topology, renderer median event를 넘어서는 independent physical first-hit/hidden-surface evidence가 남아 있다.

## 산출물 및 검증

- [W161 output README](../../output/161_global_persistent_occlusion_spatial_domain_audit/README.md)
- [W161 report](../../output/161_global_persistent_occlusion_spatial_domain_audit/worklog_161_report.json)
- [W161 pointwise audit NPZ](../../output/161_global_persistent_occlusion_spatial_domain_audit/w161_pointwise_relevance_audit.npz)
- [W161 audit script](../../devtools/demo/worklog_161_global_persistent_occlusion_spatial_domain_audit.py)
- [W161 focused test](../../tests/test_worklog_161_global_persistent_occlusion_spatial_domain_audit.py)

focused tests는 `4 passed`이며, CLI import/help와 confirmed-cache local CUDA replay가 성공했다. production behavior와 historical output은 변경하지 않았다.
