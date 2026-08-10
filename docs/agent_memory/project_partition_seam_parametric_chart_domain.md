---
name: project_partition_seam_parametric_chart_domain
description: "worklog 86 -- partition-seam chart-domain contract, FINAL NO-GO verdict closing the worklog 79-86 boundary-first visible-constructor line"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-10T05:47:08.107Z
---

Worklog 86 (docs/worklogs/86_partition_seam_parametric_chart_domain.md): final round of the chart-domain/boundary-first constructor investigation (worklog 79-86). Redesigned the chart-domain contract on the premise that physical surface termination and parametric chart boundary are NOT equivalent -- a chart boundary may legitimately include an evidence-backed partition_seam (an intrinsic cut through the unit's own coherent interior) alongside physical_termination/crease/observation_frontier, as long as the seam is determined from topology/evidence alone, never fit quality.

New `osn_gs/surface/torch_chart_unit_partition_seam.py` wraps [[project_evidence_scale_local_surface_topology_boundary]]'s `materialize_chart_unit_boundary` completely unchanged: tries physical-only reconstruction first; only when it finds zero closed loops but exactly ONE open physical fragment does it attempt a seam -- the shortest path through the unit's own interior same_surface adjacency graph (worklog 82's original interior-mesh defaults k=8/cap=12, reused, distinct from worklog 85's curve-specific unrestricted/cap=2 graph) connecting the fragment's two loose ends, excluding other boundary candidates as intermediates, with the same crease veto applied so seams never cross an existing crease. Two or more disjoint fragments are disclosed `STATE_MULTI_FRAGMENT_UNRESOLVED`, never guessed at. Seam-combined loops pass through the SAME self-intersection + occupancy safety checks worklog 85 uses.

Real result (evidence-weighted, 3526 points, 7 regions): coherent chart-unit coverage 88.1%, partitioned parametric-domain coverage only 1.0%, valid_supported 0.5%, unresolved 87.1%. Materialized units DOUBLED (4 -> 8: 4 physical_only + 4 physical_plus_seam) and valid_supported doubled (2 -> 4) -- the seam mechanism is real, sound, and non-fabricated (verified via direct tests of `_find_open_paths`/`_find_partition_seam`). But breakdown of all 178 units shows 87% of remaining failures are causes seams cannot address: no_dense_support (71, no physical evidence exists at all), multi_fragment_unresolved (23, ambiguous stitching correctly declined), unsupported_closure+coverage_failed+self_intersecting (67, loops form but evidence density/consistency fails final safety checks regardless of seam).

**Verdict: NO-GO.** The mechanism works but yields far too little (1.0%) for production adoption. Per explicit instruction, this closes the ENTIRE worklog 79-86 boundary-first visible-constructor redesign line with a direct conclusion: the current trained Gaussian evidence is insufficient for the intended parametric visible-surface representation. Not integrated as canonical; no full regression. Do not reopen this line without new evidence -- next direction (if resumed) must address WHY evidence density/continuity is insufficient (e.g. training/ADC-stage evidence distribution) or reconsider the boundary-first assumption itself, not another boundary heuristic.

Related: [[project_evidence_scale_local_surface_topology_boundary]], [[project_chart_unit_coherence_audit_evidence_scale_boundary]], [[project_dense_chart_unit_assembly]], [[project_dense_surface_consistency_chart_unit_decomposition]]
