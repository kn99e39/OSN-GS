# Worklog 71: Region-Owned Full-Evidence Boundary Topology Reconstruction

## 목적

worklog 70의 representative-edge densification(3~4개 representative edge를 evidence로 채우는 접근)은 canonical boundary 방식으로 더 이상 채택하지 않는다. worklog 70의 진단 결과 자체는 그대로 남긴다.

이번 목표는 각 region이 소유하는 full-cloud observed evidence와 기존 typed boundary half-edge evidence(physical termination/crease/observation frontier/ambiguous)로부터 실제 ordered boundary topology를 직접 복원하는 것이다. Region formation, representative membership, ownership gating은 변경하지 않는다. Representative boundary는 provenance 참조용으로만 쓰고, 최종 topology를 강제하는 seed로 쓰지 않는다.

## 방법

### 1차 설계와 실측 기반 폐기

처음에는 "evidence 점 하나하나를 typed half-edge candidate에 반경 기준으로 직접 붙여서 evidence-density 그래프를 통째로 구성"하는 설계로 구현·테스트(22개 단위 테스트 전부 통과)까지 마쳤다. 그러나 실제 checkpoint에 돌려보니 **모든 region이 branching으로 붕괴**했다(예: region 0에서 260개 노드가 하나의 branching component로 뭉침). 원인은 evidence가 조밀하기 때문에 한 typed candidate 주변에 붙는 evidence 수가 수십~수백 개에 달하고, 그 evidence들끼리도 서로 반경 안에 들어와 "체인"이 아니라 "덩어리"를 만들었기 때문이다. 이 결과를 그대로 채택하지 않고 설계를 교체했다(아래).

### 최종 설계: SEED-level topology를 기존 production 함수로 복원 + edge별 evidence densification

1. **Typed seed candidate**: `torch_boundary_support_termination.extract_support_termination_candidates`(physical termination/reliability frontier/sampling gap)와 `torch_world_space_boundary_halfedges.extract_world_space_boundary_halfedge_candidates`(crease/parallel-sheet conflict/ambiguous continuation) — 둘 다 기존 production 함수, 미변경. **발견**: 후자는 production 파이프라인(`torch_visible_surface_construction.py`)이 계산은 하지만 `VisibleSurfaceConstructionResult.boundary_halfedge_candidates`에 병합하지 않는 dead-code 상태였다(`relation_halfedges` 변수가 계산 후 전혀 쓰이지 않음). Production을 고치지는 않고, 이 스크립트가 두 함수를 직접 호출해 crease/ambiguous 증거까지 전부 복원했다.
2. **Seed-to-seed adjacency**: 기존 production `torch_ordered_world_boundary_graph.build_boundary_compatibility`/`recover_ordered_boundary_components`를 **그대로(미변경) import**해서 재사용했다. 이 함수는 representative 밀도에서 이미 검증된 tangent/normal alignment + 동일 reason + 고정 거리(0.15) 기준으로 seed 후보들을 연결한다 — 1차 설계가 실패한 "evidence 밀도" 문제를 representative 밀도로 되돌려 피한 것이다.
3. `recover_ordered_boundary_components`의 `ordered_source_ids`가 실제로는 **geometric walk가 아니라 gaussian_id로 정렬된 목록일 뿐**이라는 것을 직접 구성으로 확인했다(정렬 순서상 인접한 두 id가 실제로는 그래프상 연결돼 있지 않은 경우가 나옴). 그래서 이 함수는 topology **상태**(closed/open/branching/ambiguous/isolated) 판정에만 쓰고, 실제 순서는 같은 `build_boundary_compatibility`의 accepted edge로 직접 adjacency를 재구성해 결정론적 walk로 복원했다.
4. `ordered_closed_loop`/`ordered_open_chain`으로 판정된 component만, region-owned full evidence(worklog 67과 동일한 gate, REGION 단위로 재집계 — 기존 patch 단위 대신 `RegionFormationResult.node_region_id`로 직접 클러스터링해 `_propagate_with_evidence_gating`(worklog 129, 미변경)에 넣음)로 **edge 단위 densification**을 수행한다. 알고리즘은 worklog 70과 동일한 철학(3D world 좌표, edge당 evidence 소유권 판정, `local_evidence_scale`(worklog 32의 representative `mean_spacing`, region 소속 representative들의 median) 폭으로 binning해 edge당 여러 점을 순서대로 삽입, 각 신규 segment는 원래 edge의 `boundary_reason`을 그대로 물려받음)이지만, worklog 70의 `materialize_dense_boundary`를 직접 재사용하지 않고 **새로 구현**했다 — 그 함수는 내부적으로 `validate_simple_closed_loop`를 호출해 nonplanar를 즉시 실패로 처리하는데, 이번 라운드는 그 결합을 명시적으로 풀어야 하기 때문이다(worklog 70 모듈·테스트는 그대로 둔다).
5. Densify 결과는 `evaluate_closed_loop_geometry()`(신규)로 별도 검증한다. `compute_planarity`(기존, 미변경)로 평면성을 항상 보고하고, `NONPLANAR_AMBIGUOUS`일 때는 2D proper-crossing 검사를 **수행하지 않고**(`not_checked_nonplanar`) 그 자체를 실패로 간주하지 않는다. Planar-enough/mildly-curved인 경우에만 `_project_to_local_plane`/`_segments_intersect`(기존, 미변경)로 실제 proper crossing을 검사한다. 실제 3D surface self-intersection은 이번에도 어디서도 검사하지 않는다(`"not_checked"` 명시).
6. Branch/ambiguous seed junction은 evidence를 붙이기도 전에 typed fail-closed(`boundary_topology_branch_detected`/`boundary_topology_ambiguous_junction`)로 분리한다. Open chain은 강제로 닫지 않고 `boundary_topology_open_fragment`로 남긴다. 한 region 안에 독립된 closed loop이 여럿이면 전부 별개로 보존하고(합치지 않음), outer/inner 역할은 추정하지 않는다.
7. `boundary_topology_closed_loop_recovered`에 도달한 loop만, worklog 70과 동일한 reduced gate(uv_near_collision/neighborhood_preservation/accepted_edge_uv_crossing/interior_outside_boundary만 승인 기준, `parallel_sheet_suspected`/raw triangle fold는 진단 전용)로 single-chart UV validity를 재검증하고, 통과한 경우에만 6×6 NURBS로 fitting해 worklog 66의 기존 임계값(`UNDER_SUPPORTED_MIN_EVIDENCE=4`, `EXTRAPOLATION_NORMALIZED_DISTANCE_BOUND=4.0`)으로 `valid_supported`/`extrapolative`를 판정한다. 실패하면 그때만 `partition_materialization_required`로 승격한다(지시대로, densify 이전 상태에서는 승격하지 않음).

## 결과 (baseline_compatible@2900/3100 + baseline@2900/3100 참조, 전체 37개 region · 282개 seed-level connected component)

### Topology 복원 결과 (component 단위)

| 상태 | 개수 |
|---|---:|
| `boundary_topology_insufficient_evidence` | 156 |
| `boundary_topology_open_fragment` | 79 |
| `boundary_topology_branch_detected` | 30 |
| `boundary_topology_closed_loop_recovered` | **17** |
| `boundary_topology_self_intersecting` | 0 |
| **합계** | **282** |

조건별 region 수: baseline_compatible@2900 7 / @3100 19, baseline@2900(참조) 8 / @3100(참조) 3.

**worklog 70과 달리 이번에는 실제로 evidence-topology 기반 closed loop가 17건 복원됐다** — 일부 region(예: baseline@2900의 region 0)에서는 서로 독립된 closed loop 2개가 동시에 복원되어 병합하지 않고 각각 typed 결과로 남겼다. `boundary_topology_self_intersecting`은 0건 — 복원된 17개 loop 전부 `geometry_crossing_check="checked"`, `proper_crossing_count=0`, `planarity_class="planar_enough"`였다(즉 이번 실측 데이터에서는 nonplanar-disclosure 경로가 실제로 발동하지는 않았지만, 분리된 검증 로직 자체는 정상 동작함을 확인했다).

### 복원된 17개 closed loop의 특징

전부 `seed_vertex_count=3`(가장 작은 가능한 닫힌 loop인 삼각형)이었고, `densified_extension_count`는 1(15건)/2(1건)/4(1건)로 거의 항상 1건만 삽입됐다. `interior_outside_boundary` 비율은 17건 중 16건이 정확히 100%, 1건만 96.1%로, **densify 이후에도 evidence가 사실상 전혀 포함되지 않는다.**

원인: typed half-edge seed candidate(`extract_support_termination_candidates`가 만드는 물리적 termination gap 등) 자체가 희소하고 국소적이라, representative 밀도에서 검증된 기존 `build_boundary_compatibility`(고정 거리 0.15 + tangent/normal alignment + 동일 reason)로 연결하면 서로 아주 가까운 소수의 candidate끼리만 묶여 **가장 작은 형태인 삼각형**을 이룬다. 삼각형의 각 변 길이가 `local_evidence_scale`과 비슷하거나 작아 binning이 사실상 1개 구간(`bin_count≈1`)으로 collapse되고, 결과적으로 densify가 edge당 최대 1점만 추가한다.

### 이 결과가 의미하는 것 — worklog 69/70과의 관계

worklog 69는 "representative 3~4점 boundary가 evidence 범위보다 작다"고 진단했고, worklog 70은 그 고정된 3~4점 boundary를 densify해도 `interior_outside_boundary`가 여전히 100%에 가깝게 남는다는 것을 실측으로 보였다. 이번 라운드는 **완전히 다른 방법(worklog 61의 leftmost-turn parametric chart boundary를 전혀 참조하지 않고, typed half-edge evidence 자체의 그래프에서 topology를 직접 복원)으로도 결과가 수렴한다는 것**을 보였다: 복원되는 boundary가 여전히 최소 크기(삼각형)에 머무른다. 이는 문제가 "boundary를 만드는 특정 알고리즘의 한계"가 아니라 **typed half-edge evidence 자체가, 어떤 방법으로 조립하든, region의 실제 evidence 범위에 비해 본질적으로 희소하고 국소적**이라는 더 근본적인 사실을 가리킨다.

### Reduced-gate 재검증

17개 전부 `partition_materialization_required`로 귀결됐다(dense boundary 자체는 성공했으므로 지시대로 densify **이후**에만 승격). Gate 위반 사유: `interior_outside_boundary` 17/17(100%), `uv_near_collision_count` 13/17, `accepted_edge_uv_crossing_count` 7/17, `neighborhood_preservation_mean<0.5`는 1/17만 — worklog 69와 마찬가지로 국소 이웃 구조 자체는 대체로 건강하고, 지배적 실패 원인은 여전히 containment다.

`valid_supported`/`extrapolative`는 0건(fitting 단계에 도달한 loop가 없음).

## 테스트

신규 `tests/test_region_owned_full_evidence_boundary_topology.py`(14개: edge densification의 outward/inward/open-chain 처리, provenance 상속, planarity와 crossing 분리 검증(평면/bowtie/nonplanar/점 부족), 그리고 seed 사각형+evidence로 closed loop 복원·branch/ambiguous/open fragment/self-intersection 전체 오케스트레이션) 전부 통과. 신규 모듈 `osn_gs/surface/torch_region_owned_full_evidence_boundary_topology.py` 1개만 production 코드이고, worklog 70의 기존 모듈·테스트는 변경하지 않았다. 지시대로 focused pytest만 실행했고 full pytest는 수행하지 않았다.
