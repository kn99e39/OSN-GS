---
name: project-node-level-observability-accounting
description: "Worklog 104: replayed WL103 unmodified, added node-level (surfel-center) observability accounting -- 94.5% of WL103 singletons never had their CENTER classified on_observed_surface in any of 161 views (Branch A: surfel itself lacks positive evidence, not pairwise-edge over-strictness); but renderer-native radii>0 shows 99.98% of those same surfels DO project in ~48 views on average (weak, occlusion-unaware signal, does not overturn Branch A per directive). Implemented primitive-ownership vs visible-topology-membership representational separation (36.6% structural, no new adjacency, nothing discarded)"
metadata:
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-21T08:11:43.084Z
---

Worklog 104 (2026-08-21, `arch/2dgs-coverage-first-surface`): the user accepted Worklog 103 as a successful correction of WL102's percolation mechanism but rejected it as canonical (768,829 components, 63.4% singleton, 10.50% largest). Directive: do NOT loosen WL103's threshold; instead determine whether WL103 applies positive-observation semantics at the WRONG GRANULARITY -- separate SURFEL VISIBILITY (node-level) from PAIRWISE VISIBLE ADJACENCY (edge-level), and check whether a point-sample Phase-C query is an adequate visibility proxy for a finite-support 2DGS surfel.

`torch_positive_visible_adjacency.py` (WL103) was left completely unmodified -- replayed via its own already-public `build_candidate_graph`/`compute_positive_visible_adjacency_evidence`/`_connected_component_roots`, cross-checked bit-for-bit against the committed WL103 report (768,829 components, largest=0.1050, exact match).

New module `osn_gs/surface/torch_node_level_observability_accounting.py`: for every surfel, count training views where its own CENTER (not any edge) is canonically `on_observed_surface` (`_per_view_status_codes`, reused unmodified from WL102). 4-way node category (A never-observed / B observed-no-positive-edge / C observed-with-positive-edge / D observed-conflict-only) and a 6-way WL103-singleton cause breakdown (NODE_NEVER_POSITIVELY_VISIBLE / NODE_VISIBLE_BUT_NO_COOBSERVED_CANDIDATE_EDGE / COOBSERVED_EDGE_EXISTS_BUT_CORRIDOR_POSITIVE_TEST_FAILS / POSITIVE_OBSERVATION_EXISTS_BUT_GEOMETRIC_GATE_CUTS / OBSERVATION_CONFLICT / OTHER).

**Real-scene measurement (same checkpoint/cameras as WL96-103, 1,190,469 surfels, 161 views):**
- All surfels: A=713,540 (59.9%), B=40,786 (3.4%), C=435,481 (36.6%), D=662 (0.06%).
- WL103 singletons only (754,988): A=94.5%, B=5.4%, D=0.09%, C=0% (structurally impossible for a singleton).
- Singleton cause breakdown: NODE_NEVER_POSITIVELY_VISIBLE 94.5%, NODE_VISIBLE_BUT_NO_COOBSERVED_CANDIDATE_EDGE 3.5%, POSITIVE_OBSERVATION_EXISTS_BUT_GEOMETRIC_GATE_CUTS 1.78%, COOBSERVED_EDGE_EXISTS_BUT_CORRIDOR_POSITIVE_TEST_FAILS 0.18%, OBSERVATION_CONFLICT 0.02%.
- **Branch A decisively triggered** (>=50% threshold, actual 94.5%): most WL103 singletons genuinely lack node-level positive-visible evidence -- not a pairwise-edge-strictness artifact.

**Renderer-native visibility signal investigation (directive section 3):** `osn_gs/render/surfel_rasterizer.py`'s own docstring already documents that per-pixel-per-surfel alpha-compositing weights (`omega_i`, paper eqs. 12-14) exist only inside the vendored CUDA kernel and are never returned to Python -- exposing them would require editing the vendored kernel (forfeits OFFICIAL_CODE_FAITHFUL). The only actually-available per-surfel signal is `radii`/`visibility_filter`/`visibility_mask` (`radii>0`), a PROJECTION/CULLING signal, not occlusion-aware contribution proof.

**Bounded center-vs-renderer comparison:** of the 713,540 surfels never center-visible, 713,434 (99.98%) DO have `radii>0` in a training view -- median 33 views, mean 48 views out of 161. `center_positive_renderer_negative_count = 0` (one-directional containment, as expected). This is a genuine, honestly-reported nuance: these surfels are not "nothing" -- they project/render in dozens of views -- but per directive's explicit instruction ("do not call frustum inclusion visible surface evidence if it does not imply actual visible contribution"), this weak signal does NOT overturn the Branch A decision. No stronger renderer signal exists in the current API without touching the vendored kernel (documented, not attempted).

**Branch A action taken (per directive: stop before inventing new adjacency):** new module `osn_gs/surface/torch_primitive_ownership_visible_topology_separation.py` -- pure re-reading of WL103's own (unmodified) `PositiveVisibleAdjacencyResult`, no new adjacency, no new threshold. `PrimitiveOwnershipAccounting`: all 1,190,469 surfels retained unconditionally (never discarded). `VisibleTopologyAccounting`: only components with size>=2 count as "structural Visible Surface Component" membership -- 435,481 surfels (36.6%); the remaining 754,988 (63.4%, exactly WL103's singletons) are retained/owned but explicitly NOT called Visible Surface Components (no Trust score, no discard).

Visual review (`SINGLETON_CAUSE_VIEW`, `NODE_OBSERVABILITY_CATEGORY_VIEW`, `RENDERER_PROJECTABILITY_VIEW`): table/patio solid green (Category C) in the category view and mostly dark (non-singleton) in the singleton-cause view; hedge/background is uniformly bright red (NEVER_POSITIVELY_VISIBLE) in the singleton-cause view, yet shows meaningful cyan/blue (nonzero radii) in the projectability view -- visually confirming both the Branch A attribution and its caveat simultaneously.

Full regression 1216 passed, 1 skipped (+18 from WL103's 1198: 14 node-accounting tests + 4 ownership/topology-separation tests). No camera-induced adjacency (Branch B) was implemented -- not required per directive when Branch A is taken. See [[project_positive_visible_adjacency]] for WL103 (the baseline this batch replayed unmodified).
