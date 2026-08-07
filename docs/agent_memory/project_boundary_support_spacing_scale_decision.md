---
name: project_boundary_support_spacing_scale_decision
description: "worklog 76 - CLOSED decision: KEEP full_evidence_spacing as the production connectivity scale. Both independent boundary-support scales genuinely fix the units (no_candidate 67%->1.3%) but buy that continuity by bridging observed gaps (unsupported edges 4.9%->~49%). Scale is conclusively NOT the bottleneck."
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T09:35:59.294Z
---

worklog 76 closed the scale-domain question for the dense region-owned boundary-support certificate, following worklog 72-74's finding that 68% of half-lines died on "no continuation candidate within the local scale".

New `osn_gs/surface/torch_boundary_support_spacing.py` makes three spacings semantically separate — `full_evidence_spacing`, `representative_spacing` (report-only, never a connectivity scale), `boundary_support_spacing` — and supplies exactly three estimators: current full-evidence (baseline), region-level robust candidate spacing, per-candidate robust local spacing. The 2.5x distance multiplier and 0.1x ambiguity tolerance were held FIXED across modes (no per-mode tuning), and the connectivity certificate was not redesigned. Production got additive-only parameters (`connectivity_scale` on `_connect` and the worklog 73 diagnostics, `boundary_support_spacing_mode` on `extract_dense_boundary_support`), all defaulting to previous behavior — verified by reproducing worklog 72's published real numbers exactly (open_or_ambiguous 621, closed 0).

**Verdict: KEEP `full_evidence_spacing`.** The units error was REAL and both independent scales fix it: `no_candidate_within_local_scale` falls 1108/1652 (67%) → 254 (region) → 21 (local, 1.3%), neither-direction coverage 486→46, with zero branch explosion and zero proper crossings. But `measure_edge_support_occupancy` (new, disclosure-only) shows the continuity is purchased by connecting across observed empty space: edges containing an empty interior bin rise 9/185 (4.9%) → 191/397 (48.1%) → 211/427 (49.4%), and the longest unsupported run reaches 75–92% of a single edge's length in every real region (baseline median 0.0, p90 0.0). That is prohibited gap bridging. The payoff is negligible anyway — exactly 1 extra real closed loop (region6), whose `interior_outside_boundary` is 99.78%, the same degenerate outcome as worklog 70/71.

Because production is unchanged, the real dense boundary still yields 0 closed loops, so nothing reaches the reduced UV gate or 6x6 NURBS fitting: `valid_supported`/`extrapolative`/`partition_materialization_required` are all 0 — because there is no gate INPUT, not because the gate rejected anything. Do not conflate those two states when reporting.

**Why:** worklog 73's "68% distance failure" was not a units bug hiding recoverable continuity. Fixing the units makes the failure disappear, but what the candidates then reach is unsupported distant evidence — the boundary-support candidates are simply not spatially adjacent. **Scale is conclusively removed as the bottleneck**, just as [[project_normal_source_architecture_decision]] (worklog 75) removed the normal source.

**How to apply:** do NOT start another scale-tuning round or propose another spacing estimator — that axis is closed by measurement, and the module docstring records the rejection. The remaining candidates are the boundary-support PREDICATE itself (which points get admitted as boundary support) and its evidence density, or moving off a boundary-loop representation entirely. Related: [[project_region_owned_full_evidence_boundary_topology_reconstruction]].
