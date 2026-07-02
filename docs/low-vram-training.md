# OSN-GS Low VRAM Training Patch

## Goal

This patch reduces peak VRAM use for OSN-GS training on 16 GB GPUs without changing the default training path for existing users.

## What Was Implemented

1. Training images can now stay on CPU memory and move to GPU only for the active batch.
2. Training-time render resolution can be lowered with `--train_resolution_scale`.
3. The number of uncertain Gaussians can be capped with `--max_uncertain_gaussians`.
4. A convenience preset `--low_vram` applies safer defaults for 16 GB GPUs.

## New CLI Options

- `--image_device cpu`
  - Keeps loaded training images in system RAM instead of VRAM.
- `--train_resolution_scale 2`
  - Trains at half resolution on each axis during rendering and loss computation.
- `--max_uncertain_gaussians 128`
  - Limits how many uncertain surface samples are turned into trainable Gaussians.
- `--low_vram`
  - Preset equivalent to keeping images on CPU when `--image_device` is not explicitly set, forcing at least `--train_resolution_scale 2`, limiting `uncertain_samples_u` to `8`, limiting `uncertain_samples_v` to `2`, and defaulting `--max_uncertain_gaussians` to `128` when not explicitly set

## Files Changed

- `train.py`
- `scripts/train_osn_gs_torch.py`
- `osn_gs/interop/colab_args.py`
- `osn_gs/data/colmap_scene.py`
- `osn_gs/data/torch_scene.py`
- `osn_gs/core/torch_pipeline.py`
- `osn_gs/core/torch_trainer.py`

## Recommended Usage

Notebook or CLI runs can start with:

```bash
python train.py -s DATASET -m output/osn_gs_scene --low_vram
```

If VRAM is still high, try:

```bash
python train.py -s DATASET -m output/osn_gs_scene --image_downscale 2 --train_resolution_scale 2 --max_uncertain_gaussians 96
```

If quality drops too much, relax in this order:

1. Increase `--max_uncertain_gaussians`
2. Set `--train_resolution_scale 1`
3. Lower `--image_downscale`

## How To Remove This Patch Later

Delete or revert the changes in the files listed above, then remove this document:

- `docs/low-vram-training.md`

The feature is intentionally isolated to CLI/config/data-loading code paths so it can be removed without touching the core math or checkpoint format.

## Ongoing Context Log

- 2026-07-01: User requested that whenever the environment, project situation, or task direction changes, the relevant `.md` files should be updated with that context instead of relying only on chat history.
- 2026-07-01: NURBS is an intermediate representation, not a replacement final output. Training should keep Gaussian primitives as the main output while preserving visible NURBS reconstruction data for later visualization tools.
- 2026-07-01: The Colab training notebook should pass NURBS-related configuration alongside OSN-GS training/Gaussian primitive output handling so downstream visualization can consume both Gaussian and NURBS artifacts.

- 2026-07-01: WebRenderer PLY compatibility request. Renderer requires Graphdeco-style Gaussian fields `x`, `y`, `z`, `f_dc_0..2`, raw `opacity`, optional raw log `scale_0..2`, and `rot_0..3`. OSN-GS has corresponding primitives in `TorchGaussianModel`, so `save_ply` should emit those names instead of debug-only RGB/`scale_x` fields.
- 2026-07-01: Notebook output packaging now includes NURBS visualization data. OSN-GS output inspection creates `visualization_manifest.json` under `MODEL_ROOT`, pairing each `point_cloud.ply` with its sibling `nurbs_surface.json` so external tools can load Gaussian primitives and the visible NURBS intermediate together.
- 2026-07-02: Added `visible_surface_resolution_scale` so Stage 1 visible NURBS control-grid density can be increased from the notebook Train cell without changing the base U/V parameters. Final resolution is computed from `visible_surface_resolution_u/v * scale`.
