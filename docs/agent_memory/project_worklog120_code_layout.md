---
name: project_worklog120_code_layout
description: "Where WL120's Observed/Occluded audit code lives, and the third diagnostic CUDA sibling it added"
metadata: 
  node_type: memory
  type: project
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-26T09:54:12.517Z
---

Worklog 120 added a THIRD vendored CUDA sibling next to the canonical
`diff_surfel_rasterization` and worklog 107's `diff_surfel_rasterization_diag`:

- `osn_gs/render/vendor/diff_surfel_rasterization_qdepth/` — copy of the `_diag`
  sources plus one optional input (`query_depths`, (H,W,8) camera-space z) and
  four outputs (`query_T`, `query_terminated`, `query_reached`,
  `query_prefix_count`). It only OBSERVES the canonical `test_T < 0.0001f`
  termination event at the canonical site; no new threshold, no canonical value
  changed. Loaded via `osn_gs/render/torch_surfel_query_depth_diagnostics.py`.
- A separate sibling was chosen over editing `_diag` so every earlier replay
  (WL107/109/110/112–119) keeps calling a bit-identical build.

**Critical implementation lesson**: the probe must resolve ONLY at contributors
the kernel ACCEPTS. The kernel computes `depth` for every tile-list candidate
from the surfel's UNBOUNDED plane, so an unaccepted candidate can report an
arbitrary intersection depth — the first revision resolved before the acceptance
checks and ~99% of probes then resolved at the first list entry with T=1.0,
degenerating candidate D. Regression guard:
`test_probe_resolves_only_at_accepted_contributors`.

Audit code (all new, nothing tracked was modified):
- `scripts/devtools/observed_occluded/` — `shared.py` (query representation,
  projection, relevant-view contract, frozen aggregation, metrics),
  `engine.py`, `query_bank.py`, `synthetic_contracts.py` (S1–S7), and one
  module per candidate (`candidate_a_surface_hit`, `candidate_b_median_depth`,
  `candidate_c_geometric_visibility`, `candidate_d_renderer_reachability`).
- `scripts/devtools/observed_occluded_volumetric_audit.py` — driver.
- `tests/test_observed_occluded_volumetric_audit.py` — 60 tests, incl. AST-based
  isolation tests asserting `engine.py`/`query_bank.py` contain no
  `STATE_OBSERVED`/`STATE_OCCLUDED` token at all.

**Environment gotcha**: `torch.utils.cpp_extension.load` runs `where cl` on EVERY
load, including cache hits, so any script touching either diagnostic sibling must
run under vcvars64. Use `scripts\run_with_msvc_env.bat <cmd>`; build with
`scripts\build_surfel_extension_qdepth.bat 12.0` (JIT, deliberately not
pip-installed — an installed copy would shadow it, and the installed `_diag`
package already fails its DLL load).

Full 161-view replay takes ~110 s. See
[[project_observed_occluded_volumetric_operationalization]] for the results.
