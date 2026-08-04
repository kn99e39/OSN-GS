---
name: project-renderer-depth-sort-perf-regression
description: "WebRenderer depth-sort-during-movement perf regression (2026-07-31) — fixed by scoping continuous 50ms re-sort to Gaussian Composition mode only, restoring 160ms idle-delay sort for every other mode"
metadata: 
  node_type: memory
  type: project
  originSessionId: 60561b53-5326-495c-b10d-4f5edc8152a5
  modified: 2026-07-31T04:43:42.719Z
---

The `WebRenderer` (`WebRenderer/main.js`) CPU depth sort (`O(N log N)` over all Gaussians +
full sorted-index GPU buffer rewrite) used to only run 160ms after camera movement stopped
(`sortIdleDelayMs`). Commit `9fc4780` ("renderer improved", 2026-07-20) replaced that with a
continuous throttled re-sort every 50ms **even while the camera is still moving**
(`depthSortIntervalMs`), applied to ALL render modes — intentional per
`GAUSSIAN_COMPOSITION_IMPROVEMENT_PLAN.md` P3, but only actually needed for the `composition`
mode (SH-based Gaussian Composition needs correct back-to-front order mid-motion to avoid
transparency popping — plan doc item 5). Applying it renderer-wide made every other mode
re-sort+re-upload continuously during camera movement, which is the perf regression the user
reported ("성능이 저하됐다, 예전에 있었던 모든 버퍼 매 프레임 재구성 문제가 다시 생긴 것 같다").

**Fix (2026-07-31):** `render()`'s `canUpdateSort` now branches on
`state.renderer.mode === "composition"` — composition keeps the 50ms continuous-during-movement
sort, every other mode reverted to the original `!dragging && !isMovingByKeyboard() && idle for
sortIdleDelayMs(160ms)` gate (restored `isMovingByKeyboard()`/`lastInteractionTime` machinery that
9fc4780 had removed). Updated `Architecture.md` §7.4/§12 and
`GAUSSIAN_COMPOSITION_IMPROVEMENT_PLAN.md` P3 to describe the mode-scoped behavior instead of the
renderer-wide one.

**Why:** the regression was invisible in normal dev/testing because it only shows up as frame-rate
drop *during active camera movement* with a nontrivial Gaussian count — static frames were always
fine, and this environment has no Node.js/browser to load-test the renderer directly.

**How to apply:** if the renderer feels slow again during camera movement, check `main.js`'s
`canUpdateSort` branch first before assuming it's a GPU/shader problem — this is the second time a
"make it sort more" change here has regressed as "sorts too often for every mode." Any FUTURE mode
that genuinely needs continuous mid-motion resort (like `composition`) should be opted in
explicitly by name, not by making the idle-delay gate looser for everyone.
