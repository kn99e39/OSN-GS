---
name: feedback-verify-before-reacting
description: "When given a multi-part correction/instruction, read current code state first and report what's already satisfied before doing any new work"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ab75852-24f1-41ab-9e47-206246eb025c
  modified: 2026-07-23T09:59:31.495Z
---

Before acting on a multi-item correction or follow-up instruction, re-read the actual current code/file state and report which items are already satisfied — do not assume prior work (even your own, from a few turns ago) left something undone just because the user is asking for it again.

**Why:** During OSN-GS's Phase C Gate C review ([[project_boundary_conditioned_occlusion]]), the user gave a second round of corrections and explicitly said "다음 최신 지시에서 이미 완료한 부분이 있으면 작업을 수행하지 말고 보고부터 해" (if any part of the latest instructions is already done, report first instead of just doing the work). Checking first found that most of the requested camera-fingerprint fields (`world_view_transform`, `full_proj_transform`, `image_height`, `image_width`, `image_name`) were already added in the previous round — only one sub-item (an explicit identity-fallback rule) was actually missing. Redoing all of it blind would have wasted a full pass and obscured what genuinely changed.

**How to apply:** Whenever a review/correction lists several required changes (especially a second or third round on the same file), grep/read the current implementation for each item BEFORE editing, and open with a short status report (done / not done / partially done) before touching any code. This applies generally to iterative review cycles, not just this one — the cost of checking is small; the cost of redundant or contradictory edits (or reporting stale "still to do" status) is not.
