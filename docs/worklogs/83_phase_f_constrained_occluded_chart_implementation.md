# Worklog 83: Phase F — Constrained Occluded NURBS Chart 구현 및 Gate F 보고

날짜: 2026-07-24

상태: **Phase F 구현·검증 완료. 승인 게이트 F 보고. Phase G Gaussian proposal, global candidate selection/ranking, multi-sided topology, production integration 모두 미착수.**

## 1. 배경

사용자가 Phase F 코드/API 감사 결과를 검토하고 상세 계약(`OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md`)을 확정, 설계+구현 연속 진행을 지시했다. Phase E pairwise bounded ribbon candidate를 실제 occluded NURBS chart(`TorchNURBSSurface`)로 변환한다.

## 2. 변경 파일

**신규:**
- `osn_gs/surface/torch_coons_patch.py` — surface-agnostic bilinear Coons(transfinite) seed.
- `osn_gs/surface/torch_constrained_chart_lsq.py` — weighted single-chart constrained LSQ(`_lsq_normal_system`/`_second_difference_penalty` 재사용).
- `osn_gs/surface/torch_occluded_chart.py` — `OccludedChartResult`/`OccludedChartFitConfig`/`fit_occluded_chart` + 진단/validity gate.
- `tests/test_occluded_chart.py`(21개).

**수정:** `osn_gs/surface/__init__.py`(export), 상세 설계 문서, impl plan(§8/상태선/§11 표/다음 승인 요청).

**미변경(회귀 방지):** `torch_nurbs.py`의 `fit_coupled_patch_graph_lsq` 등 기존 fitter 동작 불변(primitive만 재사용, import만). Production 경로 미변경.

## 3. 최종 API/data contract

```python
fit_occluded_chart(candidate, domains_by_id, boundaries_by_id, surfaces_by_patch_id=None, *, config) -> OccludedChartResult
```
- global singleton 없음(caller registry). `OccludedChartResult.state ∈ {fitted, validated, unsupported, rejected}` — Phase E `OccludedRegionCandidate.state`(`candidate/unsupported/rejected`)와 문자열 충돌 회피(`candidate`/`chart` 미사용).
- `OccludedChartResult`는 설계 §14 필드 전부 포함(surface, common_parameter, support/connector samples, parameter_correspondence, fit/boundary/tangent/normal/second-order/jacobian/orientation/parameter diagnostics, self-intersection·penetration status, evidence_consistency, conflict_provenance, deterministic chart_id).

## 4. candidate → parameter domain 변환

- state 전파: candidate `rejected`/`unsupported` → solver 미실행, 대응 상태로 전달. cyclic(`connector_end is None` 또는 `provenance["cyclic"]`) → `unsupported, "cyclic_topology_deferred"`. candidate가 이미 full known-free contradiction으로 `rejected`면 미실행.
- unit-square: `u`=support chain 방향, `v`=support A→B. `v=0`=support A, `v=1`=support B, `u=0`=connector start, `u=1`=connector end.

## 5. paired resampling (필수 교정 준수)

correspondence-edge 기반: ordered edges의 `(s_a,t_a)↔(s_b,t_b)` world pair → 누적 paired arclength 정규화 → 공통 `u` grid에서 support A/B piecewise-linear resample. reversed correspondence는 edge order(s_a 오름차순) 기준 정방향 canonicalize(결정성). 원래 correspondence + resampling map을 `parameter_correspondence`에 보존. knot refinement/degree elevation/independent-arclength/post-hoc overwrite **미사용**.

## 6. Coons seed

네 boundary(support A=v0, support B=v1, connector start=u0, connector end=u1)로 bilinear transfinite seed 생성. connector는 support endpoint 사이 선형 → corner 자동 일치. Coons는 control-grid 해상도 seed로만 쓰고 최종 surface로 승인하지 않음.

## 7. constrained LSQ

support 샘플(고weight) + connector 샘플(저weight)을 하나의 weighted-data normal system(`_lsq_normal_system`)에 넣고 fairness penalty(`_second_difference_penalty`) + Coons-seed tikhonov를 더해 단일 solve(weights=1이라 control point에 선형 → 정규방정식 정확). `fit_coupled_patch_graph_lsq` 미개조.

## 8. support/connector weight 정책

`support_weight`(기본 1.0) ≫ `connector_weight`(기본 0.03), `fairness_weight`(1e-3), `interior_seed_weight`(1e-4). connector = seed + 저weight soft regularizer(exact 아님, seed-only도 아님). raw value를 `constraint_config` payload에 기록. connector weight를 1e-4로 낮춰도 C0 유지 확인(test 10).

## 9. C0/G1/G2 결과

- C0: support boundaries near-exact, fit 후 `boundary_conformance`로 실제 residual 측정, tolerance(= `c0_residual_rel_tolerance × local_surface_scale`) 초과 시 validated 승격 안 함. planar fixture 실측 C0 residual ≈ 0.0028 ≪ tolerance 0.0375.
- G1/G2: `tangent_mismatch`/`normal_mismatch`/`second_order_mismatch` diagnostic으로만. orthogonal/oblique candidate에서 큰 mismatch가 있어도 reject 아님(test 4). occluded chart가 어느 visible normal도 강제로 따르지 않음(undirected angle stat).

## 10. validity/rejection 결과

- Hard reject: non-finite solve, jacobian collapse(`min_jacobian_singular_value_normalized ≤ eps`), orientation flip(`orientation_flip_count > 0`), C0 residual > tolerance, zero-area chart(`min_area_jacobian ≤ eps`), Phase E full known-free contradiction.
- Soft diagnostic only: G1/G2 mismatch, connector deviation, candidate conflict, partial evidence contradiction.
- crossed-correspondence fold fixture(수동 candidate)에서 orientation_flip/jacobian_collapse로 rejected 확인(test 7·18), mid-span pinch에서 collapse 확인(test 17).

## 11. cyclic unsupported 처리

closed annular candidate(Phase E가 `connector_end=None`으로 생성) → solver 미실행, `state=unsupported, reason="cyclic_topology_deferred"`(test 14). 사각 patch로 억지로 펼치지 않음.

## 12. Self-intersection / penetration

`self_intersection_status.checked=False`, `visible_surface_penetration_status.checked=False`. cheap proxy만: sampled grid normal 급반전 count, chart AABB vs source visible patch AABB overlap count. complete 판정으로 표현하지 않음 — Phase F+1/별도 hardening.

## 13. Fixture별 결과 (설계 §18, 21항목 + Coons 2개)

| # | 항목 | 결과 |
|---|---|---|
| 1 | Planar quadrilateral bridge | validated |
| 2 | Unequal support sample counts | validated |
| 3 | Reversed correspondence | 동일 chart_id/payload |
| 4 | Orthogonal support | validated, G1 mismatch diagnostic만 |
| 5 | Oblique support | validated |
| 6 | Curved two-sided support | validated |
| 7 | High-curvature fold-over(crossed corr.) | rejected(orientation_flip/collapse) |
| 8 | Zero-area candidate | rejected 전달, 미실행 |
| 9 | Connector-sensitive | support 우선(support C0 < 0.02) |
| 10 | Connector weight 낮춰도 C0 유지 | validated, C0 유지 |
| 11 | Known-free contradiction | 미실행, rejected |
| 12 | Partial evidence contradiction | validated, metadata 보존 |
| 13 | Candidate conflict | chart별 fit + conflict provenance |
| 14 | Cyclic candidate | unsupported |
| 15 | Nonuniform resampling | validated(unequal counts) |
| 16 | Boundary conformance residual 측정 | 측정 확인 |
| 17 | Jacobian collapse 감지 | rejected |
| 18 | Orientation flip 감지 | rejected |
| 19 | Deterministic chart ID/payload | 반복 호출 동일 |
| — | Coons planar/corner-mismatch | pass |
| 20 | 기존 NURBS/annulus/Phase D/E regression | 없음 |
| 21 | 전체 suite 회귀 | 없음 |

threshold/pathological fixture(7·17·18)는 crossed-correspondence·mid-span-pinch 기하를 수동 candidate로 구성해 numeric probe 후 확정했다(Phase E nearest-correspondence는 fold를 만들지 않으므로 수동 구성이 필요 — 이 자체가 Phase E가 sane candidate만 만든다는 방증).

## 14. config/threshold와 근거

기본값은 **잠정 configurable**이며 raw value를 payload에 보존한다: `resolution_u/v=7/5`, `degree_u/v=3`, `support_weight=1.0`, `connector_weight=0.03`, `fairness_weight=1e-3`, `interior_seed_weight=1e-4`, `c0_residual_rel_tolerance=0.15`(local scale 비율). 확정 scientific constant 아님 — scene/seed sweep으로 재검증할 parameter.

## 15. 전체 suite 결과

- 신규: `tests/test_occluded_chart.py` 21 passed.
- 전체 pytest: `311 passed, 1 skipped, 8 subtests passed`(Gate E 시점 290 → +21).
- `torch_nurbs.py`/annulus/Phase D/E 회귀 없음(primitive 재사용, import만; 기존 fitter 미변경).

## 16. Phase G/production 미착수 확인

cyclic NURBS fitting, multi-sided chart, candidate global selection/ranking, robust self-intersection/penetration, evidence-based solver weighting, uncertain Gaussian proposal/append, production pipeline/trainer integration, certain-Gaussian reverse gradient, candidate conflict resolution 모두 미착수. `osn_gs/core/` 어디에도 신규 모듈 import 없음.

## 17. Reset 이후 Phase D–F가 기존 Phase 5 목표를 어떻게 대체했는가 (대응표)

| 기존 Boundary-First Phase 5 목표 | Reset 이후 대체 |
|---|---|
| visible boundary에서 boundary-aligned extension chart 생성 | Phase D: boundary-local world-space **sampled continuation strip**(analytic chart 아님, first-order only) |
| extension을 위한 boundary frame·확장 방향 | Phase D: world-arclength `s`, world-space `N×T` outward direction, second-order는 diagnostic-only |
| extension chart의 support/topology 판정 | Phase E: two continuation strip 간 **pairwise bounded-region candidate**(broad-phase pairing + `(s,t)` correspondence + monotonic component), evidence는 별도 validation |
| G1/G2 seam으로 이어지는 확장 surface fitting | Phase F: **constrained single-chart occluded NURBS**(Coons seed + weighted LSQ), C0 hard·G1/G2 soft, connector는 provisional bound |
| Phase 5 blocker(curved_annulus 과분할, spurious annulus 등 global topology) | Reset가 global component correctness를 선행조건에서 제거 — Phase D–F는 local boundary 기반이라 해당 blocker에 의존하지 않음 |
| occluded surface = 최종 목표 | Phase F chart는 아직 occluded **hypothesis chart**이며, Gaussian proposal(Phase G)·production adoption(Phase H)은 별도 게이트로 남음 |

## 18. 중단 및 다음 승인

계획대로 Phase F 구현과 Gate F 보고까지 완료하고 멈춘다. 다음 승인 요청은 **Phase G — Uncertainty와 Gaussian Proposal**로 한정하며, 별도 사용자 승인 없이는 Phase G, global candidate selection/ranking, multi-sided topology, production integration을 시작하지 않는다.
