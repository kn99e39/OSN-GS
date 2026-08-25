---
name: feedback_output_folder_numbering
description: Number worklog output folders under output/ by worklog number; move superseded ones into output/confirmed/
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T07:30:16.360Z
---

Starting 2026-08-25 (during Worklog 113), every worklog's real-scene export folder under `output/` must be prefixed with its zero-padded 3-digit worklog number (e.g. `113_osn_gs_chart_contract_diagnostic`), so folder order is visible without opening each report. Only the CURRENT (latest) worklog's folder stays directly under `output/`; every other worklog's folder (and its paired `*_run.log`) moves into `output/confirmed/` with the same numbered name once superseded. `output/osn_gs_scene` and `output/arch_2dgs_coverage_first_surface` (the live training-checkpoint roots, not worklog exports) are EXCLUDED and never renamed or moved.

**Retroactive pass done in Worklog 113**: renamed/moved worklogs 96-112's folders into `output/confirmed/0NN_...`, and the loose root-level `*_run.log`/`*_smoketest.log` files into `output/confirmed/_run_logs/0NN_...`. A few older, ambiguous-lineage confirmed folders (`extent_ab`, `osn_gs_2dgs_scene_3k_renderer_ply`, `osn_gs_coverage_first_subset_partition`, `osn_gs_coverage_first_subset_partition_v2`, `osn_gs_scene_latent_coverage_audit`, `osn_gs_scene_latent_coverage_audit_subdivided`) had no worklog-number `batch` field and were left unprefixed rather than guess wrong.

**Why**: `output/` had accumulated 8+ worklogs' raw exports flat with no ordering, making it hard to tell what's current vs. historical at a glance.

**How to apply**: when a worklog's devtools script writes its `--out` folder, name it `output/0NN_<descriptive_name>` directly (skip the extra move step next time). When starting a NEW worklog, first move the previous worklog's still-unprefixed `output/<name>` folder (if any) into `output/confirmed/0NN_<name>`.

**Caveat**: some devtools scripts have CLI defaults pointing at the old unprefixed paths (e.g. `renderer_native_pixel_surface_nurbs.py --wl111-export-dir` defaults to the old `output/osn_gs_rep_only_nurbs`, now `output/confirmed/111_osn_gs_rep_only_nurbs`) — these are just optional convenience defaults, not broken code, but pass the new path explicitly if replaying.
