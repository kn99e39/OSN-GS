---
name: safe-uncertain-proposal-atomic-append-integration
description: "Worklog 58 (other-session work, indexed here for continuity) — connects worklog 57's `proposed` safe proposals to the existing model-only UncertainGaussianAppendAdapter transaction; caller must supply real appearance/opacity, no synthesis"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-05T03:35:28.953Z
---

Worklog 58 (docs/worklogs/58_safe_uncertain_proposal_atomic_append_production_integration.md) built `osn_gs/gaussian/torch_safe_uncertain_append_production.py` forwarding only [[project_occluded_candidate_safe_uncertain_proposal_integration]] (worklog 57)'s `proposed`-status attempts to the pre-existing (unmodified) `UncertainGaussianAppendAdapter.append()` 4-way transactional commit (model tensors + adapter-owned provenance sidecar + model-owned occluded-chart owner registry + model-owned batch-ID ledger).

**Key entry points**: `run_safe_uncertain_proposals_and_append_from_gaussians()` (raw evidence through append) and `append_safe_uncertain_proposals()` (from an already-built worklog-57 result). `initialization_provider` MUST return an explicit `UncertainAppendInitialization` (features_dc/features_rest/opacity_logits/uncertain_confidence_logits) per attempt or the attempt is a typed `appearance_initialization_required` rejection — appearance/opacity are never synthesized here.

**Critical precondition inherited from the (unmodified) adapter**: `UncertainGaussianAppendAdapter.append()` requires `model.optimizer is None` at call time (`model_only_append_requires_no_optimizer`) — this is why [[project_appended_uncertain_gaussian_trainer_activation]] (worklog 59) exists as a separate follow-up round: the adapter's own docstring explicitly deferred "optimizer state expansion, trainer/renderer/checkpoint integration" as future work, and worklog 59 is that work.

**Verified**: candidate-ready planar fixture appends real rows (tensor count 0->receipt's `appended_sample_count`). Same batch re-run -> `duplicate_proposal_batch`, zero tensor growth (model-owned ledger, tied to the model object's lifetime, no checkpoint persistence yet). Injected `_commit_ledger` failure -> `rolled_back`, model/sidecar/owner-registry/ledger all confirmed byte-identical to pre-transaction snapshot. Worklog 57's already-rejected candidates, Box/Thin-slab degenerate paths, Sphere's no-candidate path all append 0. Full pytest 744->751.

**How to apply:** never call `UncertainGaussianAppendAdapter.append()` directly on ad-hoc data outside this production path — always go through `append_safe_uncertain_proposals()` so provenance/ledger/registry stay consistent. Any caller with an ACTIVE optimizer must detach it (`model.optimizer = None`) before invoking this path and is responsible for reattaching/extending it afterward — that responsibility now lives in [[project_appended_uncertain_gaussian_trainer_activation]], never inline here.
