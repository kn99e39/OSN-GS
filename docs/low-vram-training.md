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
- 2026-07-02: High `visible_surface_resolution_scale` can increase NURBS fitting memory. Added `visible_surface_fit_device` and `visible_surface_fit_chunk_size` so fitting can run on CPU and process the uv grid in chunks while keeping the final NURBS intermediate available for visualization.
- 2026-07-02: Implemented basic 3DGS-style Adaptive Density Control for OSN-GS. ADC now accumulates viewspace gradients/radii, clones or splits certain Gaussians, prunes low-opacity/oversized certain Gaussians, and is wired to `densify_until_iter`, `densification_interval`, and `densify_grad_threshold`. Uncertain-to-certain promotion is explicitly disabled; uncertain cleanup may prune only.
- 2026-07-02: OSN-GS saved iteration output folders now use plain numeric names such as `1000` and `10000` instead of `iteration_001000`. Notebook output inspection sorts numeric iteration folders and still treats `final` as the latest consolidated output.

## 2026-07-06 Training throughput note

- Notebook training defaults no longer save full outputs at iteration 1; explicit save iterations are treated as exact output checkpoints.
- Image staging and visible-surface NURBS fitting default to the selected training device, with chunked fitting kept configurable for VRAM control.
- ADC growth is capped by default in the notebook so density control cannot accidentally exhaust VRAM during short experiments.

## 2026-07-06 Runtime NURBS Chunk Sizing

- Visible-surface NURBS fitting now treats chunk size `0` as auto mode.
- Auto mode samples available CUDA VRAM once at pipeline initialization, fixes the chosen chunk size for the run, and logs the selected value.

## 2026-07-06 Automatic Image Placement

- Training image storage now supports `auto`, which loads images on CPU first, estimates the full stack size, and moves the stack to CUDA only when current free VRAM can safely hold it.
- If the image stack exceeds the runtime VRAM budget, images remain on CPU while CUDA still handles Gaussian tensors, rasterization, NURBS fitting, and training math.

## 2026-07-06 Per-View Image Staging

- Training images now remain as CPU-staged per-view tensors instead of one full stacked tensor.
- Each iteration samples the required view batch and transfers only that small batch to the training device, matching the original 3DGS memory pattern more closely.

## 2026-07-06 ADC Gradient Fallback

- Adaptive Density Control now falls back to Gaussian xyz gradients when the CUDA rasterizer does not populate screen-space point gradients.
- ADC passes always log tracked gradient statistics, even when clone/split/prune counts are zero, so disabled or ineffective density control is visible in training output.

## 2026-07-06 Streaming NURBS Snapshots

- OSN-GS training can now stream packed Gaussian snapshots over WebSocket directly from `train.py`.
- Streamed snapshots can include the visible NURBS intermediate as `nurbs_surface`; the payload is sent when the surface is first available or rebuilt.
- Notebook OSN-GS training exposes streaming knobs and can disable slow PLY/NURBS/checkpoint file output when using the renderer stream.

## 2026-07-06 Covariance Initialization Pipeline

- OSN-GS now initializes Gaussian covariance scale from chunked nearest-neighbor point spacing, following the original 3DGS scale+rotation covariance convention without requiring `simple-knn`.
- Notebook and CLI controls expose covariance initialization mode, KNN chunk sizing, min scale, max scene-scale ratio, and scale multiplier.

