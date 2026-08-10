---
name: project_evidence_scale_local_surface_topology_boundary
description: "worklog 85 -- replaced centroid-angle boundary ordering with local 2-manifold adjacency graph; CLOSED verdict, observed evidence insufficient for boundary-first constructor"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-10T05:17:01.133Z
---

Worklog 85 (docs/worklogs/85_evidence_scale_local_surface_topology_boundary.md): replaced [[project_chart_unit_coherence_audit_evidence_scale_boundary]]'s centroid-angular-sort boundary ordering (not a general perimeter-topology reconstruction, could silently misorder concave boundaries or close open fragments via a chord) with a genuine local 2-manifold adjacency graph.

Design: `extract_dense_boundary_support` still does candidate ADMISSION only (worklog 77, unchanged). `build_same_surface_adjacency` (worklog 82's own criterion, 0.85/0.35, unchanged, newly factored out as a reusable function) is applied directly to the admitted candidates. A connected component where every vertex has degree exactly 2 is, by graph theory, exactly one simple cycle -- order comes purely from walking the graph, no angle/projection/hull-like operation. Degree>=3 -> `branch_detected`; degree<=1 -> `open_fragment`; both disclosed, never forced. Multiple valid degree-2 components = multiple genuine independent loops.

**Self-caught bug**: initially reused worklog 82's interior-mesh k=8/cap=12 for this graph, which conflated "search pool width" with "final degree" -- measured directly: 0/178 real units materialized (a clean synthetic ring became a dense clique instead of closing). Fixed by keeping degree cap=2 (a true topological invariant: a curve cannot have vertex degree >2) while making the search pool unrestricted (every other candidate is a legal match target) -- re-verified the synthetic ring still closes perfectly, real materialization improved from 0 to some but stayed very low.

**Real result (evidence-weighted, 3526 points, 7 regions)**: true lack of boundary-support evidence 4.7%, evidence present but no valid manifold topology 83.0%, successfully recovered supported perimeter 0.5% -- LOWER than worklog 84's flawed centroid-sort method (1.5%), which is not a regression but an honest result: the flawed method sometimes passed topologically-unproven orderings by luck. Real units directly inspected show admitted boundary candidates fragment into many small loops (3-5 vertices) and open fragments rather than one dominant perimeter.

**Verdict: current observed Gaussian evidence is insufficient for the boundary-first constructor via evidence-scale local topology.** The sparse-macro-topology dependency (worklog 84's stated bottleneck) was fully removed, but the evidence that replaced it is not dense/continuous enough to form a chart-unit-scale perimeter. Per explicit instruction, chart-unit boundary topology (worklog 82-85) is now CLOSED -- do not reopen without new evidence. Not integrated as canonical; no full regression run.

Related: [[project_chart_unit_coherence_audit_evidence_scale_boundary]], [[project_dense_chart_unit_assembly]], [[project_dense_surface_consistency_chart_unit_decomposition]]
