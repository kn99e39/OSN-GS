# `output/` Folder Conventions

`output/` is gitignored (see `.gitignore`), so nothing under it is visible in
the repository or discoverable via git history. This document is the
canonical, tracked record of how it is organized — read it before creating a
new worklog's real-scene export folder, and update it if the convention
changes.

## Top-level layout

```
output/
  osn_gs_scene/                        # EXCLUDED -- live 3DGS baseline checkpoint root, never renamed/moved
  arch_2dgs_coverage_first_surface/    # EXCLUDED -- live 2DGS training checkpoint root, never renamed/moved
  1NN_osn_gs_<name>/                    # the CURRENT (latest) worklog's export -- stays directly under output/
  confirmed/
    0NN_osn_gs_<name>/                  # every OLDER worklog's export, moved here once superseded
    _run_logs/
      0NN_<name>_run.log                # loose stdout/stderr logs paired with the folder above
    <a few pre-2026-08 folders with ambiguous/no worklog-number lineage, left unprefixed>
```

- **Numbering**: every worklog's export folder is prefixed with its
  zero-padded 3-digit worklog number, e.g. `113_osn_gs_chart_contract_diagnostic`.
  This makes chronological order visible in a plain directory listing without
  opening any report.
- **Only the current worklog's folder lives directly under `output/`.** As
  soon as a new worklog's batch starts, the previous worklog's folder (and
  its paired `*_run.log`) is moved into `output/confirmed/0NN_...` — see
  [feedback_output_folder_numbering](agent_memory/feedback_output_folder_numbering.md).
- **Excluded from numbering/moving**: `output/osn_gs_scene/` and
  `output/arch_2dgs_coverage_first_surface/` are training-checkpoint roots
  (`checkpoint.pt`, `point_cloud.ply`, `nurbs_surface.json` per saved
  iteration), not worklog exports. They are referenced by many devtools
  scripts' `--checkpoint` argument and must never be renamed or relocated.
- A handful of folders under `output/confirmed/` predate this convention or
  belong to an earlier, different branch lineage (`extent_ab`,
  `osn_gs_2dgs_scene_3k_renderer_ply`, `osn_gs_coverage_first_subset_partition`,
  `osn_gs_coverage_first_subset_partition_v2`,
  `osn_gs_scene_latent_coverage_audit(_subdivided)`) — these were left
  unprefixed rather than guessing a wrong worklog number onto them.
- `output/confirmed/` is where the user reviews and marks exports as
  "checked" — see [project_output_confirmed_convention](agent_memory/project_output_confirmed_convention.md).
  "Confirmed" means simply reviewed, not an architecture/gate approval.

## Per-worklog export folder layout

Each worklog's devtools script (`scripts/devtools/<name>.py`, `--out
output/0NN_osn_gs_<name>`) writes one subfolder per named "view" — a
distinct visualization/diagnostic export, e.g. `ORIGINAL_2DGS_SCENE`,
`ZERO_COVERAGE_CAUSE`:

```
output/0NN_osn_gs_<name>/
  <VIEW_NAME_A>/
    iteration_0000001/point_cloud.ply   # WebRenderer-compatible PLY (see RENDERER_INPUT_FORMAT.md)
    render.ppm                          # raw raster preview from OSNSurfelRasterizer
    README.md                           # Korean, explains this view's color coding and meaning
  <VIEW_NAME_B>/
    ...
  preview_png/
    <VIEW_NAME_A>.png                   # PNG conversion of that view's render.ppm
    <VIEW_NAME_B>.png
    ...
  <name>_report.json                    # full measured numbers for the worklog
```

**PNG previews are bundled into ONE shared `preview_png/` folder at the
batch's top level, named `<VIEW_NAME>.png` per view — never scattered as a
separate `preview_png/` subfolder inside each view's own directory.** (This
convention drifted during worklogs ~109-113 to one `preview_png/render.png`
subfolder per view; it was corrected back to the original combined-folder
form during worklog 113. If you ever find per-view `preview_png/` subfolders
again, consolidate them the same way.) See
[feedback_png_preview_in_output](agent_memory/feedback_png_preview_in_output.md)
for the standing instruction to always produce these PNGs proactively.

Devtools scripts themselves only write `render.ppm` (via `write_ppm`); the
PNG conversion into the shared `preview_png/` folder is a manual post-step
(PIL `Image.open(ppm).save(...)`) run once per batch after the script
finishes.
