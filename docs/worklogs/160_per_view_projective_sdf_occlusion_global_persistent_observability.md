# Worklog 160 — Per-View Projective-SDF Occlusion 및 Global Persistent-Observability Contract 복구

## 1. 작업 목적

이번 작업은 latent continuation 이전 단계에서 필요한 per-view projective-SDF / ray-order occlusion evidence가 실제로 존재하는지 감사하고, 여러 camera에서 persistent하게 `OCCLUDED`인 query만 global `OCCLUDED`로 승격되는지 확인하는 diagnostic-only audit이다. W127, W139, W145, W148–W159의 결과와 frozen artifact는 변경하지 않았다.

## 2. 현재 observability architecture

현재 경로는 frozen checkpoint와 canonical renderer가 만든 camera별 `median_depth`를 기준으로 query를 각 camera에 투영하고, query depth와 renderer event depth의 순서를 비교한 뒤, 그 결과를 all-relevant-view 규칙으로 global aggregation한다. 이 경로는 새 latent surface, NURBS, Gaussian region 또는 topology 표현을 추가하지 않는다.

## 3. Historical global-state path audit

기존 Candidate-B는 다음 state를 사용한다.

- `NON_RELEVANT`: projection/depth/pixel 조건상 해당 camera가 query를 평가하지 않음
- `UNRESOLVED`: relevant하지만 유효한 renderer median event가 없음
- `OBSERVED`: valid median event에 대해 `query_depth <= median_depth`
- `OCCLUDED`: valid median event에 대해 `query_depth > median_depth`

기존 `shared.aggregate_global`은 relevant camera 중 하나라도 `OBSERVED`이면 global `OBSERVED`로 두고, relevant camera가 하나 이상이며 모든 relevant camera가 `OCCLUDED`인 경우에만 global `OCCLUDED`로 둔다. `NON_RELEVANT`는 all-relevant 판정에서 제외되며 `UNRESOLVED`는 relevant evidence로 남는다. majority, confidence, percentage vote는 사용하지 않는다.

## 4. Projective-SDF per-camera contract

각 camera에 대해 다음 항목을 보존·재생했다.

- camera ID와 순서: W153 `replay_input_runtime.json`의 161개 `camera_names`
- renderer event: W153 `replay_cache/renderer_median_depth_maps.npz`의 camera별 median-depth map
- valid/relevant pixel: `w > 0`, camera-space `z >= 0.2`, rounded pixel이 image 내부이고 해당 pixel의 median depth가 양수
- camera-space query depth: frozen camera의 `world_view_transform`에서 계산한 `z`
- projective signed distance: `s_v(x) = d_v(pi_v(x)) - z_v(x)`
- sign: `s > 0`은 median event보다 camera 쪽, `s < 0`은 event 뒤쪽, `s = 0`은 surface alignment
- truncation: W153의 TSDF fusion에서만 `|s| <= mu`를 authoritative 조건으로 사용하며, `mu = 3h = 0.03631645627319813`, `h = 0.012105485424399376`이다. per-view classifier의 새 threshold로 사용하지 않았다.
- fusion weight: W153 frozen fusion과 동일하게 authoritative view당 정확히 1이다.

W153 depth cache의 camera 수는 161이고 replay array hash가 runtime manifest와 일치했다. 따라서 Stop Condition A인 `PER_VIEW_OCCLUSION_CONTRACT_GAP`은 발생하지 않았다.

## 5. Surface alignment와 occlusion ordering의 분리

`s = 0`은 query가 renderer median event와 같은 projective depth에 있다는 뜻일 뿐이며, 물리적인 surface membership 또는 실제 first-hit truth를 증명하지 않는다. `s > 0`과 `s < 0`은 renderer event에 대한 camera-relative ordering만 나타낸다. 따라서 surface alignment count는 accounting으로 별도 보존하고, alignment 자체를 독립적인 physical-surface claim으로 승격하지 않았다.

## 6. Relevant-view contract

relevant-view는 query가 camera 앞에 있고(`w > 0`, `z >= 0.2`), rounded projection이 image 안에 있으며, 그 pixel에 valid renderer median event가 있는 경우다. `NON_RELEVANT`는 global occlusion의 분모에 넣지 않는다. relevant인데 median event가 없으면 `UNRESOLVED`이며 `OCCLUDED`로 추정하지 않는다.

## 7. Isolated per-view classifier

구현은 `classify_projective_sdf_evidence`와 historical Candidate-B를 함께 사용한다. 공개 state label은 기존 contract를 보존한 `NON_RELEVANT / OBSERVED / OCCLUDED / UNRESOLVED`다. `SURFACE_ALIGNED`는 state를 추가하지 않고 `s == 0` accounting으로만 기록했다. report의 `directly_reachable_observed_count`는 historical `OBSERVED` count이며 surface-aligned row를 포함한다.

이 함수는 Gaussian row membership를 전제하지 않고 임의의 3D query를 받을 수 있는 projective query contract로 작성되었다. 이번 real-scene population은 checkpoint Gaussian center 전체를 사용했을 뿐이다.

## 8. Global persistent observability aggregation

global `OCCLUDED`는 하나의 camera에서 뒤에 보였다는 뜻이 아니라, relevant camera가 하나 이상 존재하고 모든 relevant camera의 local state가 `OCCLUDED`인 경우에만 부여된다. 반대로 relevant camera 중 하나라도 `OBSERVED`이면 global `OBSERVED`이며, relevant `UNRESOLVED`가 남아 있고 observed가 없으면 global `UNRESOLVED`다. 이 exact all-relevant rule을 synthetic test와 real replay에서 확인했다.

## 9. Arbitrary-query 및 Gaussian membership 독립성

projective projection, median-depth lookup, signed-distance sign, relevant-view mask는 query 좌표만으로 평가된다. Gaussian identity, region ID, TSDF ownership, NURBS chart 또는 topology membership가 classifier의 선행 조건이 아니다. 이 독립성은 후속 surface/region/topology 단계와 observability evidence를 분리하는 핵심 계약이다.

## 10. Synthetic contract A–H

요청된 synthetic A–H 계약을 실행했고 모두 통과했다. valid/invalid projection, median event 부재, front/behind ordering, exact alignment, non-relevant camera 제외, unresolved 보존, persistent all-relevant occlusion을 각각 확인했다. synthetic 결과는 `worklog_160_report.json`에 보존했다.

## 11. Historical Candidate-B reconciliation

real checkpoint의 1,190,469개 Gaussian center와 161개 camera에 대해 총 `191,665,509`개 per-view pair를 비교했다.

- exact agreement: `191,665,509 / 191,665,509 = 1.0`
- total disagreement: `0`
- `OBSERVED ↔ OCCLUDED` disagreement: `0`
- `UNRESOLVED` disagreement: `0`
- historical Candidate-B source 또는 global aggregation code 변경: 없음

따라서 이번 결과는 새로운 global-state path를 도입한 것이 아니라, 기존 path가 요구된 projective-SDF ordering contract와 의미적으로 동일함을 확인한 것이다.

## 12. Per-camera accounting

161개 camera 각각에 대해 총 query 수, relevant/non-relevant 수, invalid projection, renderer-near 미만, image 밖, valid median, observed, surface alignment, occluded, unresolved 수를 report JSON에 기록했다. 예를 들어 matched review camera `DSC07960.JPG`는 1,190,469개 query 중 relevant `441,843`, non-relevant `748,626`, valid median `441,843`, historical observed `96,989`, surface aligned `3`, occluded `344,854`, unresolved `0`이다. 모든 camera에서 per-view reconciliation disagreement는 0이었다.

## 13. Real-scene matched-camera review

동일 checkpoint, iteration, resolution, background, renderer, Gaussian row count 조건으로 `DSC08043.JPG`, `DSC07960.JPG`, `DSC08003.JPG`를 matched camera로 검토했다. `per_camera_state`를 먼저 저장하고, 그 다음 161개 camera의 global aggregation 결과를 `global_state`로 저장했다. 고정 review annotation인 `tabletop`, `table_side_lower_geometry`, `vase_foreground_structure`, event 1527(`DSC08003.JPG`, pixel `(259,169)`, radius 12)의 historical context도 유지했다.

20개의 sample query에서 일부 camera가 `OCCLUDED`이고 다른 camera가 `OBSERVED`인 사례를 확인했으며, 이 경우 global state가 `OBSERVED`가 되는 all-relevant semantics를 직접 확인했다. 이는 한 camera의 occlusion을 global hidden-surface truth로 오인하지 않게 하는 review evidence다.

## 14. Mandatory Gaussian visualization contract

`mandatory_gaussian_visualization_pair`에 `Original Scene`과 `Observed-Occluded`를 생성했다. 두 결과는 동일한 Gaussian rows와 geometry를 공유하고, Original Scene은 learned SH appearance를 유지하며, Observed-Occluded만 fixed state color를 적용한다.

- `OBSERVED = (0.10, 0.85, 0.35)` green
- `OCCLUDED = (0.92, 0.18, 0.18)` red
- `UNRESOLVED = (0.60, 0.60, 0.62)` gray

marker Gaussian, 추가 geometry, shading, recolor는 사용하지 않았다. 모든 review visualization은 PNG로 저장했고 PPM은 생성하지 않았다. 각 visualization directory에는 의미, 입력/state semantics, palette/legend, shared rendering condition, review limitation을 설명하는 UTF-8 `README.md`가 있다.

## 15. Fused TSDF와의 비동치성

W153 fused field에 대해 scalar sign shortcut을 별도 diagnostic으로 계산했다. authoritative voxel 수는 `76,720,314`, Gaussian center query 중 field voxel이 없어 `UNKNOWN`인 수는 `298,189`, global per-view state와 shortcut의 disagreement는 `711,376`이었다. shortcut은 field value sign을 `OBSERVED/OCCLUDED`로 직접 해석했을 뿐이며, 이 결과를 occlusion oracle로 승격하지 않았다.

이 차이는 fused TSDF가 여러 camera의 evidence를 geometry reconstruction용 scalar field로 압축한다는 사실과, global persistent observability가 camera별 relevance와 ordering을 보존해야 한다는 사실이 비동치임을 보여준다. 따라서 field의 missing/negative/positive를 camera-dependent global occlusion으로 재해석하지 않는다.

## 16. W154–W159와의 관계

이번 W160은 W153의 renderer event를 upstream observability evidence로 정리한다. W154의 Gaussian identity → direct TSDF zero-surface sample → region ownership → native TSDF component → Boundary First/NURBS 경로와, W155 intrinsic-normal region viability, W156 support fragmentation, W157 component separation, W158 mesh-free zero-set connectivity, W159 topology ambiguity는 모두 downstream geometry/support/topology diagnostics로 유지된다. W160은 그 어느 단계의 membership, connectivity, field, Boundary First, NURBS 또는 latent surface를 변경하지 않았다.

## 17. Retained / rejected / open

### 유지한 것

W127/W139/W145/W148–W159 artifacts, W153 per-camera median maps, historical Candidate-B output, event 1527 review status, frozen renderer/checkpoint/camera calibration을 유지했다.

### 거부한 것

fused TSDF sign을 occlusion oracle로 쓰는 것, TSDF `UNKNOWN`을 `OCCLUDED`로 바꾸는 것, Gaussian Region을 classifier prerequisite로 요구하는 것, majority/percentage/confidence vote, 새 threshold, topology/Boundary First/NURBS/latent 변경을 거부했다.

### 남은 위험

renderer median event 자체가 물리적 first-hit surface라는 독립적인 ground truth인지는 여전히 open이다. renderer-relative contract를 넘어서는 hidden-surface evidence가 필요하다면 별도 작업으로 검증해야 한다.

## 18. 판정 및 검증

최종 architecture verdict는 **`HISTORICAL_GLOBAL_STATE_ALREADY_VALID`**이다. 의미는 historical Candidate-B가 이미 deterministic renderer-relative per-camera ordering을 제공하고, existing `aggregate_global`이 all-relevant persistent `OCCLUDED`를 정확히 enforce한다는 것이다. W160은 이를 projective-SDF 관점에서 재현·감사했으며 production path를 교체하지 않았다.

- intent alignment: `PASS`
- implementation fidelity: `PASS`
- synthetic A–H: all pass
- focused tests: `4 passed`
- CLI/import check: pass
- real replay: `COMPLETE_WL160_PER_VIEW_PROJECTIVE_SDF_OCCLUSION_GLOBAL_PERSISTENT_OBSERVABILITY_AUDIT`
- Gaussian center: `1,190,469`
- camera: `161`
- replay runtime: 약 `4.52 s` (local cached diagnostic replay)

## 산출물

- [W160 output README](../../output/160_per_view_projective_sdf_occlusion_global_persistent_observability/README.md)
- [W160 report](../../output/160_per_view_projective_sdf_occlusion_global_persistent_observability/worklog_160_report.json)
- [W160 audit script](../../devtools/demo/worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit.py)
- [W160 focused test](../../tests/test_worklog_160_per_view_projective_sdf_occlusion_global_persistent_observability_audit.py)

