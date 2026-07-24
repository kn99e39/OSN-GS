# Worklog 74: Boundary-Conditioned Occluded Surface 방향 전환 감사

날짜: 2026-07-23

상태: **문서·코드 인터페이스 감사 및 migration plan 초안 작성 완료. Production 코드와 기본값은 변경하지 않았고, 새 Phase 5 구현은 시작하지 않았다.**

## 1. 기존 방향성 요약

기존 Boundary-First 방향은 adaptive voxel leaf를 global visible-surface component로 복구한 뒤 component boundary/topology를 판정하고, topology별 visible NURBS chart를 만든 다음 Phase 5 occluded extension으로 이동하는 순서였다.

실제 진행은 Phase 5까지 도달해 annulus wedge의 shared boundary를 하나의 선형계에서 푸는 Step 5-A를 production 기본값으로 채택했다. 이후 `curved_annulus`의 과분할과 `mild_curved_sheet`의 spurious annulus 때문에 Phase 1/2 remediation으로 되돌아갔다. Worklog 60-B–65의 quadratic proxy, spatial candidate graph, merge-only agglomeration, Gaussian-native covariance 신호는 diagnostics-only로 검증됐지만 broad negative control을 통과하지 못해 production 적용이 기각됐다.

## 2. 새 방향성 요약

새 active methodology는 global visible-surface decomposition의 완전성을 선행조건으로 두지 않는다.

```text
local visible NURBS patches
-> artificial patch-boundary reconciliation
-> remaining open boundary
-> parametric continuation domains
-> bounded multi-sided occluded candidate
-> jointly constrained occluded NURBS
-> result-based validity / uncertainty
-> uncertain Gaussian proposal
```

Voxel은 local partition과 neighborhood provenance로 유지한다. Facing, normal similarity, tangent alignment는 candidate를 조기에 제거하는 hard gate가 아니라 soft evidence로 사용한다. 초기 범위는 two-sided 또는 multi-sided visible support가 있는 bounded occlusion으로 한정한다.

## 3. 근본적 차이

| 항목 | 기존 방향 | 새 방향 |
|---|---|---|
| 핵심 성공조건 | global component/topology 복구 | local boundary-conditioned occluded chart 생성 |
| voxel 역할 | global component membership의 출발점 | local patch partition과 spatial provenance |
| 경계 역할 | visible component topology 판정 | reconciliation 후 continuation 조건 |
| candidate 정책 | pairwise merge admissibility | high-recall bounded region formation |
| geometry 판정 | 입력 pair를 먼저 분류 | 생성된 chart의 validity와 uncertainty를 평가 |
| Phase 1 blocker | global curved connectivity 해결 필요 | local artificial seam만 식별·조정하면 됨 |

이 전환은 프로젝트의 원래 목표인 “관측 Gaussian에서 파생된 NURBS를 비관측 영역으로 연장하고 uncertain Gaussian을 생성한다”는 `docs/architecture.md`의 one-way 계약과 더 직접적으로 일치한다.

## 4. 유지 가능한 모듈

- `osn_gs/surface/torch_nurbs.py`
  - control grid, rational weights, degree, clamped knot 생성, first derivative, normal, LSQ fit, foot-point projection을 재사용할 수 있다.
  - `fit_coupled_wedge_ring_lsq`의 local-control-index를 global shared variable로 매핑하는 방식은 generic constrained fit의 기반으로 재사용할 가치가 높다.
- `osn_gs/surface/torch_voxel_hierarchy.py`
  - local neighborhood, AABB, leaf plane, explicit empty child provenance를 유지한다.
  - empty leaf는 관측 부재일 뿐 occluded surface 증거로 사용하지 않는다.
- `osn_gs/surface/torch_component_boundary.py` 및 boundary refinement/eligibility 모듈
  - support mask, contour segment, outer/hole loop 표본, local frame을 초기 boundary evidence로 재사용한다.
- `osn_gs/surface/torch_annulus_chart.py`
  - shared-boundary fit, seam gap, tangent/normal mismatch, Jacobian singular-value/flip 진단을 재사용한다.
- `osn_gs/data/torch_scene.py`, `osn_gs/render/gaussian_rasterizer.py`
  - 카메라 pose/projection과 per-view rendered depth를 observation evidence의 원천으로 사용할 수 있다.
- `nurbs_constructor_benchmark/`
  - deterministic scene/metric/runner 구조는 유지하되 occluder, camera, known-free-space oracle을 가진 새 fixture가 필요하다.
- `TorchGaussianModel`의 `is_uncertain`, `surface_uv`, `cluster_ids`, `confidence`
  - 검증된 occluded chart에서 Gaussian proposal을 만든 뒤 기존 metadata 계약으로 연결할 수 있다.

## 5. 중단 또는 deprecated 처리할 범위

다음 모듈은 삭제하지 않고 **production 미적용 deprecated diagnostics branch**로 고정한다.

- `torch_surface_proxy.py`
- `torch_surface_candidate_graph.py`
- `torch_surface_decomposition.py`
- `torch_gaussian_support_continuity.py`
- 관련 Stage 0–3/3-R 분석 script, test, artifact

이 branch의 결과는 실패 근거와 regression reference로 보존하되, 새 continuation candidate나 production component membership에 연결하지 않는다. Global component correctness, all-pair sparse classification, quadratic proxy threshold tuning, covariance principal-axis 기반 pairwise merge는 active blocker에서 제외한다.

`build_surface_components`와 Phase 2 boundary 추출은 제거하지 않는다. Local visible patch bootstrap과 boundary evidence를 공급하는 용도로 유지하되, scene 전체의 정답 component를 완벽히 복구해야 한다는 요구만 내린다.

## 6. 새 methodology에 필요한 데이터

- Stable patch ID와 patch/component/chart provenance
- 순서와 방향이 있는 parametric boundary segment
- boundary UV/world curve, endpoint, loop orientation
- control points, weights, degree, knot vector
- first/second surface derivative와 normal
- boundary의 patch-interior 방향 및 adjacent inner isocurve
- 인접 patch 사이 boundary correspondence와 UV reversal 정보
- camera pose/projection, 관측 depth/coverage, known-free-space query
- local empty/unobserved spatial query와 그 증거 출처
- continuation strip의 finite extent, confidence, supporting boundary
- candidate topology, endpoint correspondence, conflict graph
- constrained fit 결과와 C0/Jacobian/self-intersection/penetration/visibility diagnostics

## 7. 현재 코드에 존재하는 데이터

- `TorchNURBSSurface`에 control grid, weights, degree와 `evaluate_with_derivatives()`가 있다.
- clamped knot vector는 구조별 lazy cache로 생성된다.
- Gaussian별 patch ID와 UV binding이 있다.
- Phase 2에 support mask, contour UV/world segment, outer/hole loop world sample과 local frame이 있다.
- voxel hierarchy에 parent/child, AABB, leaf state, local plane, explicit empty child가 있다.
- annulus slice에 chart coordinate semantics와 cyclic seam 관계가 있으며 Step 5-A joint solve가 있다.
- `TorchScene`에 카메라가 있고 rasterizer가 per-view `depth`와 visible Gaussian index를 반환한다.
- Jacobian flip, minimum singular value, condition, seam gap, tangent/normal mismatch 진단의 상당 부분이 이미 구현돼 있다.

## 8. 누락된 interface와 data

| 요구사항 | 현재 판정 | 필요한 변경 |
|---|---|---|
| boundary curve | 부분 존재 | boundary cell 집합/contour segment를 ordered, oriented parametric segment로 승격 |
| knot vector | 내부 생성만 존재 | read-only public accessor와 export 계약 추가 |
| first derivative | 존재 | 그대로 재사용 |
| second derivative | 없음 | rational second partials 또는 안정적인 derivative API 추가 |
| UV outward/inward orientation | 일반 patch에는 없음 | support-mask/loop orientation 기반 명시적 방향 저장 |
| inner isocurve | 없음 | boundary offset과 patch support를 이용한 생성 API 추가 |
| patch adjacency | leaf/component provenance만 부분 존재 | patch-boundary correspondence graph 추가 |
| open-boundary 상태 | 없음 | `reconciled_internal` / `unsupported` / `extension_candidate` 상태 추가 |
| boundary lifecycle | Phase 2에서 일시적으로 생성 | Boundary-First state와 export에 stable record로 보존 |
| visibility evidence | scene/trainer에는 존재 | constructor/state로 전달하는 별도 observation context 추가 |
| free-space query | 없음 | multi-view ray/depth 기반 query 추가 |
| empty-space query | explicit empty voxel만 존재 | spatial query API 추가. 단, empty voxel을 occlusion truth로 해석하지 않음 |
| generic coupled fit | annulus ring 전용 | arbitrary patch graph와 boundary index map을 받는 constrained solver로 일반화 |

추가로 CUDA rasterizer의 `depth`는 실제로 inverse-depth 출력이고 fallback은 weighted depth를 반환해 의미가 같지 않다. Occlusion 판단에 사용하기 전에 depth convention과 coverage/alpha를 backend-independent contract로 정규화해야 한다. 현재 `visibility_filter`는 Gaussian이 화면에 기여했는지만 나타내며 free-space 또는 occluded-region 증거가 아니다.

## 9. 최소 end-to-end prototype

Production model에 Gaussian을 추가하지 않는 isolated benchmark로 시작한다.

1. 카메라와 analytic occluder를 가진 two-sided/oblique synthetic scene에서 visible patch 두 개 이상을 준비한다.
2. ordered boundary segment와 inner isocurve를 만들고, 인공 chart seam은 먼저 `reconciled_internal`로 처리한다.
3. 남은 두 개 이상의 boundary에서 finite first-order continuation strip을 만든다.
4. strip overlap과 camera-known-free-space 배제를 이용해 bounded candidate를 만든다.
5. 두 visible boundary와 명시적인 finite connector 두 개로 닫힌 quadrilateral domain을 정의한다.
6. visible boundary는 C0 shared geometry로 묶고 interior는 jointly constrained LSQ로 푼다.
7. fold, singular value, self-intersection, visible penetration, free-space 침범, extension 크기, candidate conflict를 평가한다.
8. 통과 결과만 `UncertainGaussianProposal`로 샘플링하고 production `TorchGaussianModel`에는 아직 append하지 않는다.

두 경계 곡선만으로는 하나의 표면이 유일하게 결정되지 않는다. 따라서 최소 prototype부터 endpoint correspondence와 connector 생성 규칙을 candidate topology의 일부로 명시해야 한다.

## 10. 단계별 구현 난이도

| 단계 | 난이도 | 주된 이유 |
|---|---|---|
| patch/boundary record와 export | 중 | 기존 데이터가 있으나 transient·비정렬 상태 |
| knot/second derivative/inner isocurve | 중 | NURBS 기반 함수 확장과 수치 검증 필요 |
| artificial boundary reconciliation | 상 | arbitrary segment correspondence와 false reconciliation 방지 |
| observation/free-space context | 상 | backend depth 의미 통일과 multi-view ray consistency 필요 |
| continuation/candidate builder | 중상 | high recall과 bounded candidate explosion 제어 필요 |
| arbitrary-angle constrained bridge | 상 | topology, connector, shared-variable solver 일반화 필요 |
| validity/self-intersection/penetration | 상 | robust geometric predicates와 scale normalization 필요 |
| uncertain Gaussian proposal | 중 | 기존 model metadata는 재사용 가능 |

## 11. 예상 위험

- Geometry가 유효해도 서로 무관한 표면을 잇는 의미적으로 잘못된 bridge일 수 있다. Result-based validation만으로 semantic correctness를 완전히 판별할 수 없으므로 multi-view free-space와 candidate conflict가 필요하다.
- Sparse trim boundary에서 NURBS derivative가 불안정할 수 있다. “Gaussian에서 geometry를 재추정하지 않는다”는 원칙을 유지하되 fit residual, Jacobian condition, boundary support를 derivative confidence에 포함해야 한다.
- High-recall continuation은 후보 수가 급증할 수 있다. Finite extent, local neighborhood, two-sided support, visible occupancy 배제는 최소 hard constraint로 유지해야 한다.
- Artificial decomposition seam과 실제 occlusion boundary를 구분하지 못하면 같은 표면을 불필요하게 extension하거나 실제 gap을 잘못 reconciliation할 수 있다.
- Camera가 없는 현재 synthetic benchmark만으로 visibility consistency를 검증할 수 없다.
- Uncertainty는 초기에는 진단값이어야 하며 Gaussian opacity/confidence나 pruning 정책에 바로 결합하면 안 된다.

## 12. 새 plan 파일

구현 초안은 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`에 작성했다. 상태는 `DRAFT / USER APPROVAL REQUIRED`이며 production 기본값, 기존 Phase 1, Phase 5 본편을 변경하지 않는다.

방향 승인 후에는 다음 문서 정리가 필요하다.

- `docs/architecture.md`: algebraic curve prediction 중심 설명을 boundary-conditioned continuation 중심으로 수정
- `TODO.md`: global component recovery를 최우선 blocker로 둔 항목을 local reconciliation/visibility/constrained bridge 순서로 재편
- 기존 Final Boundary-First/Phase 5 plan: 구현 기록·배경 문서로 강등하고 새 plan이 active gate를 소유하도록 정리

## 13. 구현 착수 전 승인 질문

다음 승인 범위는 production 통합이 아니라 새 plan의 **Phase A–B: data contract와 artificial-boundary reconciliation isolated prototype**으로 제한하는 것이 안전하다. 이 범위가 승인되기 전에는 코드 구현, production default 변경, uncertain Gaussian append, Phase 5 본편을 시작하지 않는다.