---
name: project-uncertain-confidence-rename
description: "model._confidence/get_confidence renamed to _uncertain_confidence/get_uncertain_confidence (2026-07-31) to disambiguate from unrelated construction-time \"confidence\" scores in osn_gs/surface/*"
metadata: 
  node_type: memory
  type: project
  originSessionId: 60561b53-5326-495c-b10d-4f5edc8152a5
  modified: 2026-07-31T06:36:26.111Z
---

`TorchGaussianModel`'s per-Gaussian reliability field was renamed codebase-wide:
`_confidence` -> `_uncertain_confidence`, `get_confidence` -> `get_uncertain_confidence`,
`confidence_lr` -> `uncertain_confidence_lr`, and every `confidence=`/`"confidence"` kwarg/dict-key
touching this field (torch_model.py, torch_pipeline.py, torch_trainer.py stream payload key
`confidences`->`uncertainConfidences`, torch_checkpoint.py, torch_losses.py, torch_density_control.py,
torch_uncertain_append_adapter.py's `confidence_logits`->`uncertain_confidence_logits`, PLY property
`confidence`->`uncertain_confidence`, and the matching WebRenderer JS fields/tests).

**Why:** the user was confused because `osn_gs/surface/*` (region/boundary/affinity construction
modules) ALSO has a bunch of unrelated "confidence" names (`relation_confidence`, `region_confidence`,
`PatchBoundary.confidence`, half-edge `confidence`) that are geometric/statistical evidence scores
computed fresh during visible-surface construction. Investigation confirmed ZERO data dependency
between the two — `model.get_confidence` was never read anywhere in `osn_gs/surface/*`. The user asked
for the codebase to make this distinction nameable, not just documented.

**How to apply:** when you see `uncertain_confidence` anywhere in `osn_gs/gaussian/`, `osn_gs/core/`,
`osn_gs/losses/`, `osn_gs/utils/torch_checkpoint.py`, or WebRenderer, it is the per-Gaussian structural
reliability of an UNCERTAIN Gaussian (sigmoid of a learned logit, used in density-control pruning and
`uncertain_confidence_loss`/`uncertain_anchor_loss`). Bare `confidence`/`region_confidence`/
`relation_confidence` in `osn_gs/surface/*` is a completely different, construction-time-only concept —
do not conflate them, and do not "helpfully" rename surface/* confidence fields to match; they were
deliberately left alone. This is a BREAKING change to the checkpoint (`format_version=2`) and PLY
schema — old checkpoints/PLY files saved before 2026-07-31 will not load (key `confidence` no longer
exists; no back-compat shim was added, per this repo's no-compat-hack convention). Full repo pytest
(597 passed) and the renamed WebRenderer JS smoke tests were reviewed, but the JS smoke tests could not
actually be *executed* in this environment (no Node.js) — same gap already disclosed in worklog 128 (the WebRenderer-diagnostics one, now `docs/worklogs/22_webrenderer_gaussian_diagnostics.md` after the 2026-07-31 renumbering — note there were two same-numbered `128_*.md` files at the time; the OTHER one, about the benchmark's own dataset, is now `docs/worklogs/23_osn_gs_benchmark_volumetric_solid_dataset_replacement.md`).

**Follow-up (same day):** the user then asked the renderer's "Confidence Heatmap" mode to actually
display the CONSTRUCTION-time confidence instead of `uncertain_confidence`. Added
`TorchPipelineState.surface_patch_confidence: tuple[float, ...]` (one entry per `surface_patches`
index, populated from `SurfaceRegionCandidate.region_confidence` at all three construction sites —
`_initialize_canonical`, the ADC post-commit reconstruction, and `maintain_surface_from_certain` — via
a new `TorchOSNGSPipeline._patch_confidence_from_regions` helper), persisted it through
`torch_checkpoint.py`, streamed it per-Gaussian as `surfaceConfidences` in `_stream_payload` (looked up
by `cluster_ids` via `_surface_patch_confidence_lookup`, -1.0 sentinel for unassigned — NOT NaN, since
NaN is not valid JSON), added it to `save_ply`/PLY as `surface_confidence`, and pointed
`GaussianMemory.js`'s `mode === "confidence"` at `surfaceConfidence` instead of `uncertainConfidence`.
`uncertainConfidences`/`uncertain_confidence` remain on the wire and in the PLY, just no longer driving
that particular renderer mode. Full pytest green (597 passed) both before and after adding new coverage
(`test_surface_patch_confidence_matches_region_confidence`, `surfaceConfidences` assertions in
`test_stream_payload_includes_renderer_diagnostic_arrays`/`test_checkpoint_round_trip_restores_raw_state`/
`test_vectorized_ply_preserves_renderer_header`). JS-side changes (`gaussian_catcher.js`, both smoke
tests, `RENDERER_INPUT_FORMAT.md`) again reviewed but not executed (no Node.js here).
