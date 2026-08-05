---
name: occluded-candidate-safe-uncertain-proposal-integration
description: "Worklog 57 (other-session work, indexed here for continuity) — connects worklog 56's eligible-boundary bridge to existing Phase F (constrained occluded-NURBS fit) -> F.1 (sampled safety gate) -> G (uncertain Gaussian proposal) as one production orchestration; no model append/appearance/opacity"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-05T03:35:15.660Z
---

Worklog 57 (docs/worklogs/57_occluded_candidate_safe_uncertain_proposal_production_integration.md) built `osn_gs/surface/torch_safe_uncertain_proposal_production.py` connecting [[project_eligible_boundary_continuation_bridge]] (worklog 56)'s `OccludedRegionCandidate`s to the existing (unmodified) Phase F constrained occluded-NURBS fit, Phase F.1 sampled safety gate, and Phase G uncertain-Gaussian proposal generation — `run_safe_uncertain_proposals_from_gaussians()` (raw evidence entry point) and `build_safe_uncertain_proposals_from_bridge()` (from an already-built bridge result).

**Fail-closed candidate-ready contract**: Phase E's own candidate states are `valid`(pair)/`degenerate`(pair-but-record-provenance)/`rejected`(exclude) — a provenance-preservation contract, NOT a Phase F/G approval condition. Phase F only runs when: candidate.state=="candidate", supporting domain/boundary/patch IDs satisfy pairwise cardinality+provenance, BOTH `ContinuationDomain.state=="valid"` (not merely non-rejected), and the domain->boundary->patch registry chain resolves. `degenerate` domains, unsupported/rejected candidates, provenance mismatches, fit failures, non-validated charts, unsafe/ambiguous safety results, and non-eligible proposals are all typed-rejected with 0 proposals — no status promotion, no synthetic fallback.

**Fixture results**: candidate-ready planar fixture (the same `_facing_planar`/`_manual_bridge` synthetic fixture worklog 58/59 also reuse) produces a real non-empty `proposed` proposal batch with full provenance. Box (cap 64, 7 candidates) and Thin-slab (cap 64, 3 candidates) run the real orchestration end-to-end but every one hits a `degenerate` supporting domain or rejected candidate state -> 0 proposals (not bypassed). Sphere stays at 0 candidates/proposals. Real 5k's 0 candidates (worklog 56's AABB non-contact) and real 3k/10k's 0 candidates stay 0 proposals. Model append, appearance, opacity are NOT performed here — every proposal keeps `append_state="not_appended"`, `appearance_state="unset"`, `opacity_state="unset"`. Full pytest 739->744.

**How to apply:** [[project_safe_uncertain_proposal_atomic_append_integration]] (worklog 58) consumes ONLY this module's `proposed`-status attempts. Never treat `degenerate`/non-`valid` continuation-domain state as good enough for Phase F input — that's Phase E's own provenance-preservation policy, not an approval signal.
