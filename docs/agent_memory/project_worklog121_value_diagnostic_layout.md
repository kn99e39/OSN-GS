---
name: project_worklog121_value_diagnostic_layout
description: "Where WL121's value-diagnostic layer lives, and how the WL120 baseline replay gate works"
metadata: 
  node_type: memory
  type: project
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-27T05:20:29.227Z
---

Worklog 121 added a supplemental value layer over worklog 120's frozen candidates.
Nothing in `candidate_a..d.py` or `shared.aggregate_global` was touched.

- `scripts/devtools/observed_occluded/value_diagnostics.py` —
  `evaluate_with_values` runs ONE sweep that produces worklog 120's state arrays
  (by calling the frozen `classify_view` functions) plus the value table;
  `candidate_c_blocker_values` computes the corrected camera-nearest (MIN t) /
  query-nearest (MAX t) blocker provenance, world gaps, opacities and
  same-component attribution, reusing the frozen `SEGMENT_EPSILON` and
  `GeometricSceneSupport` so the value pass cannot drift from the decision;
  `assert_historical_state_replay` / `bank_replay_check` are the baseline gate.
- `scripts/devtools/observed_occluded/topology_gap_bank.py` — replays WL107/109
  read-only and collects raster-local adjacencies spanning a FINAL component
  separation; also builds verified zero-relevant-view controls.
- `scripts/devtools/observed_occluded/synthetic_value_contracts.py` — S-D1
  (accepted-event depth inversion), S-C1 (same-surface overlapping footprints),
  S-B1 (median-event round trip).
- `scripts/devtools/observed_occluded_value_space_comparison.py` — driver. It
  STOPS with SystemExit if the baseline gate fails, writing the report first.
  `--allow-replay-failure` exists for smoke tests ONLY and must never be used for
  a reported run.
- `tests/test_observed_occluded_value_space_comparison.py` — 38 tests.

**CUDA**: five PURELY ADDITIVE fields were added to the existing `_qdepth` sibling
(`query_resolution_depth`, `query_termination_alpha`, `query_late_front_count`,
`pixel_inversion_count`, `pixel_max_backward_jump`). Extending in place was allowed
because bit-identity of every pre-existing output is asserted by
`TestQDepthWorklog121Additivity`.

**Build gotcha**: adding outputs changes `RasterizeGaussiansCUDA`'s tuple arity, and
ninja will link a STALE `ext.o` against the new signature (unresolved-external
LNK2019). Always wipe the JIT build dir first:
`rmdir /s /q %TEMP%\osn_gs_diff_surfel_rasterization_qdepth` then
`scripts\build_surfel_extension_qdepth.bat 12.0`.

Full 161-view run: ~430 s (two representative sweeps, the WL107/109 KNN graph, the
double-C value pass over 4,712x161, and the 908-query supplemental bank).

Results in [[project_observed_occluded_value_space_supplement]]; the WL120 baseline
this gate compares against lives at
`output/confirmed/120_osn_gs_observed_occluded_volumetric_audit/observed_occluded_per_view_states.npz`.
