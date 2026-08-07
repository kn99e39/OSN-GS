---
name: project_normal_source_architecture_decision
description: "worklog 75 - CLOSED decision: KEEP covariance_normal, reject structural_normal (positions-only local PCA). A/B showed 0 vs 0 closed loops everywhere; tangent rejections drop but relocate to the normal stage and net edge survival worsens. Normal source is NOT the binding constraint on boundary closure."
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T08:43:30.436Z
---

worklog 75 closed the normal-source architecture question with a single bounded A/B (explicitly not another diagnostic round), answering worklog 74's call for an explicit structural-normal experiment.

New experiment-only `osn_gs/surface/torch_structural_normal.py` (no production module imports it): `compute_structural_normals` = local PCA over region-owned observed point POSITIONS only (no scale/rotation/covariance/SH/opacity/renderer/optimizer), `rebuild_candidate_orientation` re-derives only the orientation frame of an ALREADY-FROZEN candidate set using the existing missing-sector/outward-direction logic. Frozen across A/B: candidate ids+xyz, region ownership, boundary reasons, candidate extraction, connectivity scale/distance thresholds, ambiguity/mutuality, topology acceptance — both modes run the SAME unmodified `_connect` and worklog 73/74 diagnostics. Scenes: existing fixtures only (`box_face`, `cylinder`) plus real `baseline_compatible@2900` (4 regions).

**Verdict: KEEP covariance_normal.** Evidence: (1) final closed loops 0 vs 0 in all 8 regions — structural normal does not fix boundary closure; (2) tangent rejections fall 33→22 but the loss RELOCATES to the normal stage (5→40, 8x) and net survival worsens — tangent edges 240→198 (−17.5%), mutuality edges 94→86 (−8.5%), worse in 3 of 4 real regions; (3) new instability — under the structural normal, 30–60% of mode A's own already-admitted candidates (12/12 on cylinder side) would fail boundary admission, so it is not a drop-in swap; (4) synthetic surfaces are byte-for-byte indistinguishable (disagreement 0.16–0.42° flat, 7.31° curved) while real data disagrees by median 30.8–63.6° — positions-only PCA is unstable specifically on real trained Gaussian clouds, which are locally volumetric/noisy rather than clean surface sheets.

Cost was never the issue: structural normal is CHEAPER (<3ms/region, N×3×4 bytes, no persistent state) and rendering was verified `torch.equal` bitwise identical before/after on the real CUDA rasterizer.

**Why:** the important conclusion is NOT that covariance normals are good. worklog 74's hypothesis (covariance-derived tangent breaks cycles) is partially confirmed — swapping the normal really does cut tangent rejections — but it buys no final topology because the loss just moves upstream. **The normal source is not the binding constraint on boundary closure.** Two orientation sources disagreeing by median 30–64° on real data yet producing identical 0-loop outcomes localizes the bottleneck upstream, in candidate/support density and scale domain.

**How to apply:** do NOT propose another orientation/normal-quality variant for boundary closure — that axis is closed by measurement. Remaining candidates are candidate/support density and the scale domain itself (worklog 74 left evidence for a `boundary_support_spacing` contract, not activated in 75), or moving off a boundary-loop representation. Related: [[project_region_owned_full_evidence_boundary_topology_reconstruction]], [[project_region_owned_full_evidence_boundary_materialization]].
