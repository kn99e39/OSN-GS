---
name: feedback_include_subset_visualization
description: "Always include a canonical subset/component-membership visualization as a fixed, standard review export, the same way ORIGINAL_2DGS_SCENE is always included"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-26T04:49:41.817Z
---

The user said (2026-08-26, mid-Worklog-118): "앞으로는 생성된 subset들을 visualization하는 것을 고정적으로 중간 산출물에 포함시켜. 지금 original scene을 포함시키는 것처럼." (From now on, always include a visualization of the generated subsets as a fixed part of intermediate outputs, the same way you currently include the original scene.)

**Why:** `ORIGINAL_2DGS_SCENE` has been a fixed baseline reference view in every worklog's export batch since early in this session. The user wants the canonical visible-topology subset/component membership (colored by `subset_ids`, the WL107/109 canonical topology -- the same hash-color-by-component-id technique used throughout WL107-118) to be an equally standard, always-present view, not something added ad hoc only when a batch happens to be about topology.

**How to apply:** every future worklog's devtools export script must include, alongside `ORIGINAL_2DGS_SCENE`, a view showing the current canonical subset/component membership (hash-colored by `subset_ids`) as a fixed, standing export -- regardless of whether that batch's own focus is topology-related. Apply starting with Worklog 119 onward; for Worklog 118 specifically (already mid-run when this feedback arrived), add it as a small supplementary export appended after the main run completes (topology replay is cheap and frozen, reusable standalone) rather than restarting the long chart-fitting run already in progress.
