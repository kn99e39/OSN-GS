---
name: project-renderer-grounded-visible-adjacency
description: "Worklog 106: replaced WL103's Phase-C center endpoint eligibility with WL105's official-renderer contribution signal, controlled-replayed against WL103 (same shared candidate graph/corridor/geometric gates) -- NEGATIVE result: singleton 63.4%->83.8%, largest component 10.50%->2.91% (worse fragmentation, not percolation); of WL103's 720,052 singleton-but-actually-contributing surfels only 11.4% gained an edge, remaining 88.6%'s dominant cause (96.9%) is multi-view observation conflict/contradiction, not lack of co-contributing neighbors (0.14%). Stopped per directive -- points toward camera-induced adjacency as the next architecture, not implemented this batch"
metadata:
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-22T15:36:31.442Z
---

Worklog 106 (2026-08-21, `arch/2dgs-coverage-first-surface`): built on WL105's finding (Phase-C center query inadequate for 2DGS primitive visibility) by replacing WL103's endpoint eligibility (Phase-C center `on_observed_surface`) with WL105's renderer-contribution signal (unmodified, reused via `compute_renderer_contribution_for_view`), keeping everything else -- the range-based corridor test, depth_epsilon, interior sample count, geometric residual/positional gates -- byte-identical to WL103's own parameters (directive's "controlled replay" requirement).

New module `osn_gs/surface/torch_renderer_grounded_visible_adjacency.py` is a structural sibling of WL103 (duplicates its corridor/geometry loop rather than importing it, since WL103 has no eligibility-substitution hook), not a subclass. WL103/104/105 all left completely unmodified and separately replayable. New relation states per directive naming: `POSITIVE_RENDERER_VISIBLE_CONTINUATION`, `UNKNOWN_NO_RENDERER_SUPPORTED_RELATION`, others unchanged from WL103's names.

**Controlled real-scene comparison (same checkpoint/cameras, ONE shared candidate graph built once for both arms):**

| | A. WL103 (center-grounded) | B. Renderer-grounded (new) |
|---|---|---|
| components | 768,829 | 1,004,080 |
| largest fraction | 10.50% | **2.91%** |
| singleton fraction | 63.4% | **83.8%** |
| final positive edges | 1,043,908 (20.3%) | 409,620 (8.0%) |
| CUT_OCCLUDED_DOMAIN | 1,380 | 1,175,611 |
| UNRESOLVED_OBSERVATION_CONFLICT | 18,107 | 2,834,719 |

**Result: NEGATIVE, worse fragmentation, not percolation.** Loosening eligibility to renderer-contribution (95.4% of all surfels vs WL103's narrow center test) made far more candidate edges co-eligible (~1M/camera vs WL103's ~140K/camera), but the overwhelming majority of the newly-eligible pairs resolved to occlusion/conflict rather than positive continuity -- because many render-contributing surfels are redundant/overlapping depth layers that genuinely disagree across views, not because the corridor test itself is broken.

**Key causal test:** of WL103's 720,052 singleton-and-actually-contributing surfels, only 81,974 (11.4%) gained a renderer-grounded edge; 638,078 (88.6%) remain singleton. Remaining-singleton cause breakdown: `OBSERVATION_CONFLICT` 65.0%, `HARD_OBSERVATION_CONTRADICTION` 31.9%, `POSITIONAL_SHEET_SEPARATION` 2.56%, `GEOMETRIC_DISCONTINUITY` 0.36%, `NO_SAME_VIEW_COCONTRIBUTING_SPATIAL_NEIGHBOR` only 0.14%. The dominant failure (96.9% combined) is multi-view contradiction/conflict, NOT lack of co-contributing neighbors -- meaning the exact WL103 pairwise-corridor-relation architecture itself (not sparse candidacy) is the bottleneck.

Visual confirmation: `RENDERER_GROUNDED_VISIBLE_COMPONENTS` shows fine rainbow speckle across table, patio, AND hedge/background alike (the table, cleanly single-colored in WL103, is now also fragmented); `OCCLUDED_FREE_SPACE_TERMINATION_VIEW` shows most of the scene (including table/patio) marked occluded, not just hedge.

Per directive: stopped here, no threshold tuning. This result is framed as justifying (but NOT implementing) a next architecture shift from "3D candidate edge -> camera approves pair" to "camera-visible surface support -> camera GENERATES adjacency" (camera-induced adjacency) -- explicitly deferred to a future batch.

Full regression 1239 passed, 1 skipped (+14 from WL105's 1225: `tests/test_renderer_grounded_visible_adjacency.py`, includes an AST-based test proving the new module never imports Phase-C center classification). See [[project_renderer_contribution_diagnostics]] (WL105, the primitive evidence reused unmodified here) and [[project_positive_visible_adjacency]] (WL103, replayed unmodified as baseline A).
