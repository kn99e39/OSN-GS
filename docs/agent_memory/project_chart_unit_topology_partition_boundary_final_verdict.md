---
name: project_chart_unit_topology_partition_boundary_final_verdict
description: worklog 88 -- chart-unit membership-cut boundaries derived directly from full evidence-scale same-surface topology; actual final NO-GO
metadata:
  node_type: memory
  type: project
  modified: 2026-08-10
---

Worklog 88 (`docs/worklogs/88_chart_unit_topology_partition_boundary_final_go_no_go.md`) supersedes only Worklog 87's global conclusion, not its accepted implementation. The Worklog 87 stable-ID daisy-chain is rejected as an architecture test because it still requires Worklog 77 candidates and stable ID does not establish geometric adjacency.

New experimental module: `osn_gs/surface/torch_chart_unit_topology_partition_boundary.py`. It builds the unchanged Worklog 82 full-region same-surface graph once, takes each Worklog 83 chart unit's induced subgraph, constructs a PCA-UV tangent rotation system, and traces ordered outer half-edge loops using actual graph edges only. Stable IDs are tie-break/canonicalization only. Worklog 77 candidates provide diagnostics and physical provenance but are never an admission prerequisite. Non-typed loop edges are first-class `partition_seam`; closed topology, self-intersection, occupancy, and Worklog 79 coverage all fail closed.

Real 7-region replay (`output/extent_ab/val88/chart_unit_topology_partition_boundary_replay.json`, 3526 evidence): coherent 3108/3526 (88.15%); cut recoverable 43/3526 (1.22%, only 1.38% of coherent); physical-only/mixed/seam-only evidence 40/0/3; valid/extrapolative/unsafe/unresolved 31/0/12/3065; evidence-weighted held-out p95 3.754. Candidate-0 was genuinely tested and a seam-only domain was produced, proving the contract works, but yield remains negligible.

Correction to Worklog 87 prose: the unchanged artifact/code recomputes 71 candidate-0 units (164 evidence), not 64. All are size 2 or 3 because Worklog 77 returns no candidates for n<4. Fate: 49 two-node units have one induced edge; 12 three-node units have two induced edges; 5 closed triangles fail Worklog 79 coverage; 3 closed triangles fail occupancy; 1 typed physical-only and 1 seam-only triangle materialize, both unsafe. Thus 2/71 units (6 evidence) recover.

This Worklog 88 verdict is superseded by Worklog 89 because global PCA rotation and induced-subgraph largest-outer-face selection did not implement the requested full-region face-incidence contract. See [[project_full_region_face_membership_incidence_final_verdict]].
