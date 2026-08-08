# Worklog 78: Constructor Provenance 및 Chart-Frontier Semantics

## 범위와 수정

Worklog 75 covariance normal, Worklog 76 full-evidence spacing, Worklog 77 angular-gap bias correction과 dense connectivity certificate는 유지했다. production audit에서 `torch_visible_surface_construction.py`가 relation half-edge(crease/parallel/ambiguous)를 계산하면서 physical termination loop에는 올바르게 제외하지만, Worklog 61의 **parametric chart boundary provenance에도 전달하지 않는** semantic gap을 확인했다.

수정은 parametric chart construction에만 `termination_halfedges + relation_halfedges`를 전달한다. 물리 boundary candidate/closure/materialization은 여전히 termination halfedge만 사용한다. 따라서 existing accepted region topology가 만든 chart edge는 typed `crease`/`observation_frontier`를 정확히 보존하고, physical termination으로 위장되지 않는다. geometry, membership, ownership, threshold, connectivity는 변경하지 않았다.

## Constructor-wide attribution matrix (baseline-compatible@2900)

| region class | region | attribution | downstream |
|---|---|---|---|
| topology-supported chart frontier | 0,1,2,3,6 | `chart_frontier` + existing typed boundary evidence | parametric chart materialized, full evidence fit evaluated |
| no eligible chart topology | 4,5 | `local_geometry_ambiguous` / open-or-branching accepted topology | no chart input; no forced closure |
| dense physical support | all 7 | mixed: corrected predicate adds candidates, but closed physical loop 0 | physical path intentionally does not consume chart frontier |

Ownership gate is not a demonstrated cross-region merge defect in this batch: full-evidence fitting remains the existing strict propagated region-owned set and no foreign evidence is added. Worklog 77 establishes that the remaining dense candidate discontinuity is dominated by unobserved support, not merely predicate rejection. Thus absence from a region-owned set is not promoted to scene absence; it remains typed unassigned/other-region evidence outside this constructor's ownership contract.

## End-to-end real replay

`baseline_compatible@2900`: 7 regions, 5 materialized parametric charts, all full-evidence fits materialized. Classification: `valid_supported=1`, `extrapolative=4`, `partition_materialization_required=0` (no dense recovered loop reached the reduced UV gate). Dense physical-support topology remains closed 0. The valid chart is region 3 (surface-to-evidence p95 exactly 4.0); extrapolative charts are regions 0/1/2/6 with p95 8.57/9.68/14.07/12.91.

## Verdict

The current constructor is viable for **existing topology-supported parametric chart boundaries** and must keep that semantics distinct from physical termination. It is not viable as a universal dense physical-boundary reconstructor on the current real evidence: corrected predicate works on continuous synthetic support, but real dense physical loops remain absent without forbidden bridging. The remaining work requires region/chart-boundary representation or upstream observed evidence density, not another normal/scale/connectivity variant.

## Verification

Focused: parametric boundary, parametric materialization, region-owned evidence, visible construction: **27 passed, 4 subtests passed**. A full regression is run after this constructor contract correction.
