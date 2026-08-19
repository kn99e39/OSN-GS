---
name: feedback-correct-replay-checkpoint
description: "User corrected the checkpoint used for real-scene replays — use output/osn_gs_scene/3000/point_cloud.ply, not output/extent_ab/val64/baseline_compatible/final"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-19T09:22:56.336Z
---

The user confirmed (2026-08-19, after Worklog 105) that `output/extent_ab/val64/baseline_compatible/final` — the checkpoint every real-scene replay from Worklog 94 through 105 used — is **not a properly trained 3DGS scene**. Its `render.ppm` looked visibly wrong/degraded to the user.

**Use `output/osn_gs_scene/3000` going forward** for real-checkpoint replays. Verified visually: PSNR 23.92 (vs. the old checkpoint's 20.1), clean sharp render.ppm showing a garden table/planter scene with no visible degradation. `output/osn_gs_scene/final` is actually iteration 10000 with LOWER PSNR (19.8) than 3000 — do not assume "final" is best; the user specifically named `3000`.

**Why this matters:** every quantitative measurement in Worklogs 94-105 (latent coverage %, subset counts, cut-edge ratios, etc.) was produced on the wrong scene and should be treated as unverified until re-measured on `output/osn_gs_scene/3000`. This is not itself a re-litigation of those worklogs' code/logic findings (bugs fixed, architecture decisions) — only their checkpoint-dependent NUMBERS are suspect.

**How to apply:** before running any new real-scene replay/export script, check whether the checkpoint path was inherited from a prior worklog's example command — those examples still say `output/extent_ab/val64/baseline_compatible/final` and must be updated to `output/osn_gs_scene/3000` (or whatever checkpoint the user names next) rather than copy-pasted. See [[project_coverage_first_subset_partition]] and [[project_coverage_first_partition_measurements]] for the specific worklog affected so far.
