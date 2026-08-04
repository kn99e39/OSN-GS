---
name: visible-boundary-eligibility-contract
description: "Worklog 54 — formalized a 5-state per-region production contract (eligible_closed_boundary/open_observed_fragment/insufficient_observation/ambiguous_boundary/rejected_unsafe) gating NURBS materialization, plus fail-closed 2-cycle branch-budget exhaustion tracking"
metadata:
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-04T03:00:40.998Z
---

Worklog 54 (docs/worklogs/54_visible_boundary_eligibility_and_unsupported_state_contract.md) formalized the conclusion of worklogs 47-53 ([[project_downstream_valid_directed_matching_repair]] etc.) into an explicit production contract, without touching candidate/continuation thresholds, the matching objective, gap interpolation, or NURBS fitting.

**New module** `osn_gs/surface/torch_visible_boundary_region_status.py`: `classify_all_region_boundary_statuses()` assigns exactly one of 5 states per region:
- `eligible_closed_boundary` — `ordered_closed_loop` component that passes `validate_simple_closed_loop`, non-branching, `outer_boundary_candidate` role. Only this state's `eligible_component_ids` are ever passed to `materialize_visible_boundary_component`.
- `open_observed_fragment` — real `observed_support_termination` candidates linked into an `ambiguous_ordering` (>=2 node) open chain — the solver's best downstream-valid non-closed result.
- `insufficient_observation` — zero boundary evidence, or physical candidates exist but are all `isolated_boundary_candidate` (never linked).
- `ambiguous_boundary` — typed non-physical evidence exists (crease/parallel/frontier/sampling-gap/cross-region) but zero `observed_support_termination` candidates were ever generated.
- `rejected_unsafe` — `ordering_capacity_exceeded`, a closed loop failing self-intersection, or 2-cycle branch-budget exhaustion.

**Fail-closed addition**: `torch_directed_boundary_ordering.py`'s worklog-53 2-cycle branch-and-bound now tracks whether `_MAX_TWO_CYCLE_BRANCH_EXPANSIONS` was exhausted while a 2-cycle remained unresolved (`_max_weight_one_in_one_out_matching_with_diagnostics` returns `(matched, exhausted)`; the public `_max_weight_one_in_one_out_matching` is now a thin wrapper, unchanged for existing callers). When exhausted, every component from that region gets `"two_cycle_branch_budget_exhausted"` in `unresolved_reasons`, which the region-status classifier treats as an unconditional `rejected_unsafe` override regardless of `ordering_state`.

**Wiring**: `construct_visible_nurbs_from_gaussians` now computes `region_boundary_statuses` and restricts the `attempts`/materialization loop to only components whose ID is in some region's `eligible_component_ids` (previously any `ordered_closed_loop` was attempted). Exposed via `VisibleSurfaceConstructionResult.region_boundary_statuses` and `diagnostic_summary["region_boundary_statuses"]` + per-state counts.

**Result:** real 5k retains exactly its 2 known regions (130, 141) as `eligible_closed_boundary`; real 3k (155 regions: 65 insufficient / 65 ambiguous / 25 open / 0 eligible) and 10k (136 regions: 60/55/21/0) get zero eligible regions, fully accounted for in the other 3 typed states — no silent empty/failed collapse. Negative-control fixtures (Box 6/Cylinder 2/Sphere 0/Thin-slab 3 at cap 64) byte-identical to worklog 53 baseline. Full pytest 720->730 passed (10 new tests: 3 for budget-exhaustion fail-closed behavior including a mocked forced-exhaustion end-to-end case, 7 for the 5-state classifier covering every state deterministically, including a hand-constructed bowtie self-intersection fixture since the real solver rarely produces one).

**How to apply:** any future change to `torch_directed_boundary_ordering.py`'s matching or `torch_visible_boundary_region_status.py`'s classification rules must re-verify: (1) the negative-control fixture table, (2) that 5k's region 130/141 stay `eligible_closed_boundary`, (3) `tests/test_visible_boundary_region_status.py` in full. The five states are now the ONLY vocabulary for describing region-level boundary outcomes in diagnostics/exporters — don't reintroduce ad hoc "empty result" or generic "failed" reporting for boundary regions.
