---
name: project-camera-induced-representative-backbone-audit
description: "Worklog 108: architecture-gate/accounting audit of WL107 (algorithm unmodified) using the correct structural population (renderer surface REPRESENTATIVES, not all contributors). Resolved the exact 36,051-surfel discrepancy (representative is NOT a strict subset of contributing; likely forward-vs-backward CUDA floating-point reconstruction asymmetry, not a bug). Representative-only topology: singleton 16.7% (vs WL107's raw 45.0%), connected 83.3%. Deterministic pixel-anchor verification confirms table/patio separation and hedge non-absorption at 3 sampled points, BUT quantitatively 20.4% of the largest (patio) component's membership overlaps a hedge-adjacent region -- partial correction to WL107's 'pure patio' claim. Gate: CONDITIONAL PASS with explicit caveat; non-representative contributors remain unattached support evidence"
metadata:
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-24T04:23:31.832Z
---

Worklog 108 (2026-08-23, `arch/2dgs-coverage-first-surface`): the user accepted WL107 as the strongest candidate but directed an architecture-gate/accounting audit -- do NOT modify WL107's adjacency algorithm or tune thresholds; instead reevaluate it using the correct structural population (RENDERER SURFACE REPRESENTATIVES, i.e. surfels that at least once become the renderer's own `median_contributor` at T>0.5), since a surfel that never becomes a representative can never appear in a camera-induced representative-ID edge and is therefore structurally always singleton.

**§1 accounting discrepancy resolved exactly.** Directive predicted a 36,051-surfel gap between WL107's reported numbers; measured cross-tab (reproduced identically across two independent reruns):

|  | REP+ | REP- |
|---|---|---|
| CONTRIB+ | 749,886 | 385,998 |
| CONTRIB- | **36,051** | 18,534 |

Representative is NOT a strict subset of contributing. WL107's own reported 385,998 was correct; the "discrepancy" was WL107's own naive subtraction (`1,135,884 - 785,937 = 349,947`) ignoring the unexpected 36,051 cross-category. Likely explanation (plausible, not proven): the two signals come from two SEPARATE CUDA builds -- WL105's contribution signal derives from the canonical package's backward kernel (which reconstructs running transmittance `T` via DIVISION from `T_final`, and independently re-checks `alpha < 1/255`), while WL107's representative signal is captured directly in the forward pass (division-free) of a separate diagnostic build. Floating-point asymmetry between forward-direct and backward-reconstructed alpha near the 1/255 boundary is a known class of issue in this kind of renderer; the small size (3-4.6%) is consistent with a boundary-case effect, not a logic bug. Does not change the overall architecture conclusion.

**§2 representative-only topology accounting:** 785,937 representatives, degree0 (singleton) 131,378 (16.7%), degree>=1 654,559 (83.3%) -- markedly more coherent than WL107's raw all-surfel statistic (45.0% singleton), which is diluted by structurally-always-singleton non-representatives.

**§4 remaining representative-singleton causes** (131,378 total): `CAMERA_PAIR_GENERATED_BUT_FAILS_3D_LOCALITY` 57.7%, `PASSES_LOCALITY_BUT_POSITIONAL_SEPARATION` 34.4%, `PASSES_LOCALITY_BUT_GEOMETRIC_DISCONTINUITY` 8.0%, `REPRESENTATIVE_HAS_NO_DISTINCT_PIXEL_NEIGHBOR_RELATION` 0%.

**§5 locality-rejection distance distribution** (7,538,912 rejected pairs): distance/local-spacing ratio median 2.96x (mean 12.49x, p95 73.2x, max 1979x) -- most rejections are near-miss silhouette/depth-boundary cases, not wildly non-local coincidences.

**§6 largest-component identity, deterministic (not color-only):** subset 0, 437,751 members (36.77%, matches WL107 exactly), bounding box unusually tall (y-range 16.3 units, inconsistent with a pure flat floor). Deterministic pixel-anchor lookup (actual `representative_id` queried at known screen pixels on the established preview camera, not just color sampling): table anchor -> subset 1 (degree 9, clearly separate); both patio anchors -> subset 0; all three hedge anchors -> three DIFFERENT subsets (448039 singleton, 9, 479), none matching subset 0. HOWEVER a precise overlap computation (added after visual review flagged it) shows **89,502 surfels (20.4% of the largest component's own membership, 26.2% of a crude hedge-region mask) actually belong to subset 0** -- i.e. WL107's "the 36.8% component is patio's own pure surface" claim needs partial correction: some ground-level hedge/foliage adjoining the patio genuinely shares the giant component (plausible real-world ground-to-vegetation continuity), while the 3 specific sampled hedge anchor points happen to sit in separate components.

**§7 bridge/connectivity robustness:** rank-0 (patio) component: 1,034,050 edges, 56,816 bridges (5.5%, articulation edges whose removal splits the component), 19.2% of edges supported by exactly 1 training view. Ranks 1-2 similar (5.7%/8.9% bridges, 16.8%/22.2% single-view). Not catastrophic (94.5%+ of edges are redundant/cyclic) but a real, non-trivial reliance on single-view-supported and structurally fragile connections -- not individually audited for silhouette-proximity at the full 56,816 scale (reported as aggregate statistics only, out of this batch's scope).

**§8 hedge representative-backbone reassessment:** of 342,085 hedge-region surfels (crude nearest-anchor heuristic), 319,281 (93.3%) contributing, but only 204,164 (59.7%) representative; of those, 160,290 (78.5%) are actually connected. WL107's visual impression of "hedge mostly fragmented" conflated two different things: (a) genuine representative-backbone fragmentation (21.5% of hedge representatives are singleton) and (b) 115,117 hedge contributing-but-never-representative surfels that structurally cannot define topology at all (rendering-support primitives, not a fragmentation failure).

**Gate decision: CONDITIONAL PASS.** All PASS criteria hold except "largest component quantitatively consistent with ONE legitimate visible surface region" -- that one is only partially met (20.4% hedge overlap). Formally propose the Renderer-Native Surface Representative Graph as the leading canonical Visible Surface Topology Backbone CANDIDATE, with Renderer-Contributing Non-Representative Surfels (404,532 total: 385,998 contributing-only + 18,534 neither) classified as retained Visible Surface Support Evidence -- NOT attached to any component in this batch (attachment strategy explicitly deferred).

No production code (`osn_gs/`) was modified in this batch (per directive: reuse WL107's algorithm unchanged) -- only a new analysis script `scripts/devtools/camera_induced_representative_backbone_audit.py` (with 9 new focused tests for its own bridge-finding/distribution helper logic, pure-Python, no CUDA needed). Full pytest regression was deliberately NOT rerun (directive: only rerun if shared production behavior changes; it did not).

Known script bug found and fixed after the first run: a loop variable named `report` shadowed the outer report dict, so the first run's JSON was incomplete (only the last bridge-loop entry + views/render_ppm) -- fixed by renaming to `bridge_report_entry`; all print-logged diagnostics were unaffected and the full JSON was recovered by a second (identical, reproducible) rerun that also added the new hedge-overlap statistic.

See [[project_camera_induced_visible_adjacency]] (WL107, the baseline replayed unmodified here).
