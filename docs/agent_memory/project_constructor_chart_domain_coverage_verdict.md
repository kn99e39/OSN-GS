---
name: project_constructor_chart_domain_coverage_verdict
description: "worklog 79 - VERDICT: the visible NURBS constructor is NOT usable on real data; region/chart representation needs redesign. All 5 real charts fail identically (boundary_chart_extent_mismatch, before parameterization) because chart topology (3-7 representatives) and fitting evidence (93-1035 points) are 31-301x mismatched. Added a fail-closed chart-domain coverage contract."
metadata:
  type: project
---

worklog 79 ran ONE constructor-wide pass (chart topology -> domain -> UV -> NURBS fitting together, not another one-factor ablation) over all 7 real `baseline_compatible@2900` regions, after worklog 78 (other session) restored physical-termination vs parametric-chart-frontier provenance.

**Worklog 78 eligibility contract verified as structurally sound**: `construct_region_parametric_chart_boundaries` builds `adjacency` ONLY from `region.internal_accepted_edge_ids`; relation half-edges supply node REASON LABELS and can never create an edge. So ambiguous relation evidence cannot become a chart boundary without topology support (measured: all 5 charts are physical_termination/crease only, 0 ambiguous promotions). But a SEPARATE contract gap exists: eligibility checks only "is the accepted topology a simple cycle", never whether the chart domain contains the evidence it will be fit to — so a 3-node triangle is trivially eligible.

**Seven-region matrix — all 5 materialized charts share ONE cause**: `boundary_chart_extent_mismatch`, dominant stage **before parameterization**. Representative topology is 3-7 nodes spanning only 0.15-0.67x the owned-evidence extent, while ownership propagation (`_propagate_with_evidence_gating`) has no in-plane distance bound, so a 3-representative region owns 93-1035 points (evidence/member 31-301x). Result: **89.1-99.8% of owned evidence lies OUTSIDE the chart domain**. Jacobian near-degenerate 0 and local fold <=0.63% everywhere — **fitting geometry is healthy, so no fitting/UV improvement can remove this failure**.

Corrected worklog 78's numbers: it reported region 3 as `valid_supported` at p95 4.0, but that was measured on evidence the fit had seen. Under deterministic spatial holdout region 3 is 6.43 — **held-out `valid_supported` is 0 of 5**, not 1.

Regions 4/5 (no chart), classified from their own accepted-edge graph without forcing closure or inventing a partition: region 5 has a pendant (degree-1) node and a 2-core below 3 -> `genuinely_open_or_unsupported_topology`; region 4 has 7 nodes with a single 2-core component and cyclomatic number >=2 -> `ambiguous_branching_topology` (no basis to claim it needs multiple charts).

**Correction applied**: `fit_region_owned_full_evidence_patch` gained a chart-domain coverage contract — if a majority of the evidence lies outside the boundary loop, it fails closed as `chart_domain_does_not_cover_evidence` BEFORE fitting. Containment is measured in the boundary loop's OWN best-fit plane (refitting to the union would let distant evidence rotate the frame it is tested against) and reuses `interior_within_boundary`. Not threshold tuning: measured violations are 82.6-99.6%, and a dedicated test pins that every bound in 0.5-0.85 gives the identical verdict. Also found the existing test fixture used `grid[:4]` — four COLLINEAR points, a zero-area "boundary" — which encoded the very defect being fixed; replaced with a real perimeter loop.

**Before/after (real, full replay)**: full-evidence fit `materialized` 5->0, `extrapolative` 5->0, `chart_domain_does_not_cover_evidence` 0->5. Eligible chart boundaries (5), representative-scale chart surfaces (5), physical surfaces (0), and no-chart regions (2) all unchanged — only the false claim that a chart represents its region's owned evidence now fails closed.

**VERDICT: the visible NURBS constructor is not usable on real data; the region/chart representation itself requires redesign.** The mismatch is structural, not a local bug: the unit carrying topology (3-7 representatives) and the unit carrying evidence (93-1035 points) differ by 31-301x, and the former covers only 0.15-0.67x the latter's spatial extent. worklog 79's contract discloses this honestly; it cannot remove it.

**How to apply:** the redesign must make chart domain and the evidence it represents live at the SAME scale — raise representative topology to evidence density, restrict ownership to the chart domain, or construct justified multiple charts per region. worklog 79 deliberately chose none of these. Do not reopen orientation ([[project_normal_source_architecture_decision]]), scale ([[project_boundary_support_spacing_scale_decision]]), predicate ([[project_boundary_support_predicate_bias_fix]]), or boundary-algorithm variants without new evidence.
