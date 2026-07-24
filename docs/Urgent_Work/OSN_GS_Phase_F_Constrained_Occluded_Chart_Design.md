# OSN-GS Phase F — Minimal Constrained Occluded NURBS Bridge: Design

상태: **설계 확정, 구현 진행 승인됨(2026-07-24).** 상위 계획 `OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §8. 사용자가 코드/API 감사 결과를 검토하고 아래 계약을 최종 구현 기준으로 확정했다. 구현 결과는 `docs/worklogs/83_phase_f_constrained_occluded_chart_implementation.md`(Gate F)로 보고한다.

## 0. 감사 근거 (실존 API)

- `TorchNURBSSurface`: `control_grid (U,V,3)`, `weights`, `degree_u/v`(부족 시 `_effective_degree` 자동 강등), clamped uniform knot `[0,1]`(확장/삽입 헬퍼 없음), `evaluate`/`evaluate_with_derivatives`/`evaluate_with_second_derivatives`, `knots_u/v`. fitting 시점 weights=1 → control point에 대해 선형.
- 재사용 primitive: `_lsq_normal_system`, `_solve_control_grid_lsq`, `_second_difference_penalty`, `boundary_control_indices`(전부 `torch_nurbs.py`, 명시적으로 재사용 승인됨), `compute_parametric_jacobian_metrics`/`compute_orientation_consistency`(`torch_parametric_diagnostics.py`).
- **self-intersection / visible-surface-penetration 구현은 코드에 없음** — Phase F 최소에서도 `checked=False` + cheap proxy까지만.
- `fit_coupled_patch_graph_lsq`는 point-cloud-per-patch 피터라 점군 없는 occluded chart에 직접 부적합 → **직접 개조하지 않고** primitive만 재사용.

## 1. 역할과 canonical 관계

Phase E pairwise bounded ribbon candidate → 실제 occluded NURBS chart.

```
ContinuationDomain ≠ Occluded NURBS surface
OccludedRegionCandidate ≠ Occluded NURBS surface
OccludedChartResult.surface = 실제 constrained occluded NURBS chart (TorchNURBSSurface)
```

## 2. 초기 canonical scope

open pairwise quadrilateral ribbon만 지원: planar / unequal support sample counts / reversed correspondence / orthogonal / oblique / curved two-sided / degenerate·rejected candidate handling. **미지원**: cyclic annular fitting, multi-sided joint topology, multiple-candidate global selection, periodic NURBS, topology-changing fitting. Cyclic candidate는 `state=unsupported, reason="cyclic_topology_deferred"`로 정직하게 반환(억지로 사각 patch로 펼치지 않음).

## 3. 입력 계약

```python
def fit_occluded_chart(
    candidate: OccludedRegionCandidate,
    domains_by_id: Mapping[str, ContinuationDomain],
    boundaries_by_id: Mapping[str, PatchBoundarySegment],
    surfaces_by_patch_id: Mapping[int, TorchNURBSSurface] | None = None,
    *, config: OccludedChartFitConfig,
) -> OccludedChartResult
```

global singleton 없음(caller registry). solver 실행 전제: `candidate.state == "candidate"`, supporting domain 정확히 2개(둘 다 `domains_by_id`에 존재), open/non-cyclic, finite support chain, valid correspondence ordering, nonzero support interval, nondegenerate connector, finite bridge. candidate가 `unsupported`/`rejected`이면 solver 미실행, 대응 Phase F 상태로 전달.

## 4. Geometry source of truth

canonical source는 `ContinuationDomain`(`world`, `sample_valid_mask`, `s_world`, `outward_tangent_world`, `normal`, source IDs). candidate의 `support_chain_a/b` + `correspondence_edges`는 correspondence와 selected `(s,t)`만 제공. `PatchBoundarySegment`는 provenance/confidence/state/closed-consistency/ID validation 전용 — `PatchBoundarySegment.world`로 domain geometry를 덮어쓰지 않는다(closed duplicate-endpoint 유입 방지).

## 5. Support boundary의 의미 (핵심 정책)

두 geometry를 분리 보존:
```
visible source boundary       = ContinuationDomain.world[s, 0]      (provenance / validation reference)
continuation-supported chain  = ContinuationDomain.world[s, t_selected]  (chart의 opposite boundary)
```
**초기 canonical policy: chart는 두 continuation-supported matched chain 사이의 occluded chart를 만든다.** visible boundary `t=0`까지 되늘려 포함하지 않는다(그러면 Phase D/E bounded candidate보다 넓은 영역을 자의 생성). `surfaces_by_patch_id`가 있으면 analytic source surface와의 tangent/normal 관계를 **diagnostic으로만** 계산(solver hard constraint 아님).

## 6. Parameter correspondence — paired resampling

Phase E correspondence-edge 기반 paired resampling으로 sample 수/spacing 불일치 정규화:
1. candidate의 ordered correspondence edges를 읽는다.
2. 각 edge `(s_a,t_a)↔(s_b,t_b)`의 world pair를 domain에서 가져온다.
3. 누적 paired arclength(양측 평균 세그먼트 길이)를 정규화해 공통 parameter `u∈[0,1]`로 만든다.
4. 공통 `u` grid에서 support A/B를 piecewise-linear resample한다.
5. reversed correspondence는 edge order(s_a 오름차순) 기준으로 정방향 canonicalize(결정성).
6. 원래 correspondence + resampling map을 provenance에 저장한다.
**미사용**: independent arclength resample 후 임의 pairing, knot refinement, degree elevation, post-hoc boundary overwrite. (knot insertion/degree elevation API가 없으므로 최소 구현에 도입하지 않음.)

## 7. Canonical chart topology

unit-square: `u∈[0,1]` support chain 방향, `v∈[0,1]` support A→B 방향.
```
v=0: support chain A     v=1: support chain B
u=0: connector start     u=1: connector end
```
두 support chain은 opposite boundaries. connector는 observed가 아니라 provisional topology bound.

## 8. Solver — Coons/transfinite seed + new single-chart constrained LSQ

`fit_coupled_patch_graph_lsq` 직접 개조 금지. 재사용: `_lsq_normal_system`, `_solve_control_grid_lsq`, `_second_difference_penalty`, `boundary_control_indices`, degree/knot/control-grid convention, `TorchNURBSSurface`, parametric diagnostics. 필요 시 single-chart helper로 안전 추출하되 기존 fitter 동작 불변.

모듈 분리:
```
osn_gs/surface/torch_coons_patch.py            coons_bilinear_patch(...)
osn_gs/surface/torch_constrained_chart_lsq.py  solve_constrained_chart(...)
osn_gs/surface/torch_occluded_chart.py         OccludedChartResult / OccludedChartFitConfig / fit_occluded_chart
```

## 9. Coons/transfinite seed

네 boundary curve(support A=v0, support B=v1, connector start=u0, connector end=u1)로 bilinear Coons transfinite 초기 sampled surface 생성. connector는 선형 보간 `C_u0(v)=(1-v)A[0]+vB[0]`, `C_u1(v)=(1-v)A[-1]+vB[-1]` → corner가 support endpoint와 정확히 일치(코너 호환 자동). support A/B가 geometry 지배, connector는 topology/extent 안정화. corner 불일치 시 support chain endpoint를 canonical corner 우선. Coons seed 자체를 최종 surface로 승인하지 않음(이후 constrained LSQ + diagnostics 필수).

## 10. Constraint weight 정책

```
support boundary : exact/near-exact high-weight constraint
connector        : low-weight soft regularizer  (seed + low-weight)
interior         : fairness/smoothness regularizer + Coons-seed tikhonov
```
**금지**: connector exact constraint, support==connector weight, connector seed-only 완전 무시, fit 후 control-point boundary overwrite. config 분리: `support_weight`, `connector_weight`, `fairness_weight`, `interior_seed_weight`. invariant `support_weight >> connector_weight`. 값은 provisional configurable, raw value를 payload에 기록. (구현: support/connector 샘플을 각자 weight로 하나의 weighted-data normal system(`_lsq_normal_system`)에 넣고 fairness penalty + Coons-seed tikhonov를 더해 단일 solve — support_weight≫connector_weight가 normalize 후에도 상대비로 보존.)

## 11. C0 / G1 / G2 정책

- C0: support boundaries에서 exact/near-exact, fit 후 boundary conformance로 실제 residual 측정, tolerance 초과 시 validated 승격 안 함(hard).
- G1/G2: hard constraint 아님. tangent/normal/second-order mismatch는 diagnostic으로만. orthogonal/oblique의 큰 mismatch만으로 reject 안 함. **occluded chart가 어느 한 visible patch normal을 그대로 따르도록 강제하지 않음.**

## 12. Evidence 정책

Phase E evidence(ObservationEvidence, candidate conflict, known-free contradiction)를 solver weight로 사용 금지. 용도: fit 후 chart validation, raw uncertainty metadata, rejection provenance. hidden scalar/geometry weight로 압축 금지. candidate가 이미 full known-free contradiction으로 `rejected`면 solver 미실행.

## 13. `OccludedChartResult` 상태

```
fitted     : solve 성공, validation gate 미통과/미평가
validated  : 최소 Phase F validity gate 통과
unsupported: 미지원 topology/cyclic candidate 등
rejected   : solve 실패, fold/collapse, C0 위반, evidence hard contradiction 등
```
`candidate`/`chart`를 상태 문자열로 쓰지 않음(Phase E `OccludedRegionCandidate.state`와 충돌 회피).

## 14. `OccludedChartResult` 최소 계약

`chart_id`, `source_candidate_id`, supporting domain/boundary/patch IDs, `topology`, `surface: TorchNURBSSurface|None`, `common_parameter`, `support_samples_a/b`, `connector_samples_start/end`, `parameter_correspondence`, `constraint_config`, `fit_diagnostics`, `boundary_conformance`, `tangent_mismatch`, `normal_mismatch`, `second_order_mismatch`, `jacobian_diagnostics`, `orientation_diagnostics`, `parameter_quality`, `self_intersection_status`, `visible_surface_penetration_status`, `evidence_consistency`, `conflict_provenance`, `state`, `reason`, `provenance`, `payload()`. deterministic `chart_id` = `source_candidate_id` + solver/config fingerprint + parameterization fingerprint.

## 15. Validity gate

검사: solver finite success, support C0 residual, support-chain deviation, connector deviation, local Jacobian singular value, orientation flip, condition number, parameter distortion, tangent/normal/second-order mismatch, evidence consistency, conflict provenance, deterministic payload.
- **Hard reject**: non-finite solve, Jacobian collapse, orientation flip/fold, support C0 residual > tolerance, zero-area chart, Phase E full known-free contradiction.
- **Soft diagnostic only**: G1/G2 mismatch, connector deviation, candidate conflict, partial evidence contradiction.

## 16. Self-intersection / visible-surface penetration

`self_intersection_status.checked=False`, `visible_surface_penetration_status.checked=False`. cheap proxy만 허용(sampled grid normal 급반전, nonadjacent cell AABB overlap count, chart AABB vs source visible patch AABB overlap, support 반대 방향 fold). proxy를 complete 판정으로 표현 금지. 완전 검사는 Phase F+1/별도 hardening.

## 17. Cyclic candidate

solver 미실행, `state=unsupported, reason="cyclic_topology_deferred"`. Phase E cyclic candidate fixture로 unsupported propagation 검증.

## 18. Fixture (21항목)

1 planar→validated, 2 unequal counts→validated, 3 reversed→동일 chart/payload, 4 orthogonal→validated(G1 mismatch만 큼), 5 oblique→validated, 6 curved→validated, 7 high-curvature/fold→rejected, 8 zero-area→rejected/전달, 9 connector-sensitive→support 우선, 10 connector weight 낮춰도 C0 유지, 11 known-free→미실행/rejected, 12 partial evidence→fit+metadata, 13 conflict→chart별 fit+provenance, 14 cyclic→unsupported, 15 nonuniform resampling, 16 boundary conformance residual, 17 Jacobian collapse, 18 orientation flip, 19 deterministic ID/payload, 20 기존 regression 없음, 21 전체 suite 회귀 없음. threshold fixture는 numeric probe 선행.

## 19. 구현 파일 / export

신규: `torch_coons_patch.py`, `torch_constrained_chart_lsq.py`, `torch_occluded_chart.py`. export: `OccludedChartResult`, `OccludedChartFitConfig`, `fit_occluded_chart`(+ 재사용용 `coons_bilinear_patch`, `solve_constrained_chart`은 선택). 내부 solver/helper는 private.

## 20. 구현하지 않을 것

cyclic NURBS fitting, multi-sided chart, candidate global selection/ranking, robust self-intersection/penetration, evidence-based solver weighting, uncertain Gaussian proposal/append, production pipeline/trainer integration, certain-Gaussian reverse gradient, candidate conflict resolution.
