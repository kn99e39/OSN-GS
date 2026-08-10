---
name: project_chart_unit_coherence_audit_evidence_scale_boundary
description: "worklog 84 -- coherence audit (over-merge rejected) + evidence-scale boundary topology for worklog 83 assembled chart units, verdict B"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-10T04:50:40.500Z
---

Worklog 84 (docs/worklogs/84_chart_unit_coherence_audit_and_evidence_scale_boundary.md): closed two coupled questions left by [[project_dense_chart_unit_assembly]] in one batch.

**Coherence audit**: factored out worklog 82's own `internal_normal_disagreement_fraction` (0.15 bound, same formula) as a reusable function, reapplied at ASSEMBLED-unit scale (evidence-only, never fit quality). Real result: 178/178 units coherent (max disagreement 0.04) -- over-merging via orientation incoherence essentially does not occur.

**Evidence-scale boundary topology** (new `osn_gs/surface/torch_chart_unit_evidence_scale_boundary.py`): first design (reusing `extract_dense_boundary_support`'s own `_connect` closed-loop as the order) measured 0/178 materialized on real data -- directly reproduces worklog 71's already-documented limitation (17/282 closure rate). Redesigned: use `extract_dense_boundary_support` for candidate ADMISSION only (worklog 77 predicate, reliable), order admitted candidates by ANGLE around centroid in the unit's own PCA tangent plane (evidence-scale, no sparse macro dependency), validate via `evaluate_closed_loop_geometry` (self-intersection) AND `measure_edge_support_occupancy` (worklog 76, used here as the closing gap-bridging safety gate -- angular ordering always closes SOME polygon including a chord across an open arc's empty end, which self-intersection alone can't catch; verified directly with a half-ring fixture).

Real result (7 regions, evidence-weighted, 3526 points): assembled/coherent 88.1%, materialized only 1.5%, valid_supported 0.9%, no_chart 86.7% (unsupported_closure 95, no_dense_support 71, coverage_failed 2). Directly traced all 11 units containing a previously-valid micro-component: 5 stayed valid_supported, 6 failed -- ALL 6 via `unsupported_closure`, ZERO via `ambiguous_or_over_merged`. This decisively answers worklog 83's 16->4 valid-chart-loss question: legitimate exposure of downstream boundary-evidence insufficiency, NOT over-merging, NOT a mixture.

Verdict: (B) -- chart units can be assembled and over-merging is not occurring (rejects C), but observed evidence is insufficient to establish safe chart-scale boundaries for most units (rejects A: only 1.5% of evidence reaches materialization). The sparse-macro-node dependency was successfully removed, but candidate density itself doesn't densely enough surround a large unit's full perimeter.

Related: [[project_dense_chart_unit_assembly]], [[project_dense_surface_consistency_chart_unit_decomposition]]
