# OSN-GS

OSN-GS is an experimental 3D Gaussian Splatting framework that predicts occluded surface structure from observed Gaussian geometry. It fits base curves on observed Gaussian centers, extrapolates occlusion curves, builds a NURBS-like parametric surface, and places uncertain Gaussians on the inferred surface.

## Current Training Path

The active training implementation is the Torch path:

```bash
python scripts/train_osn_gs_torch.py \
  -s /path/to/scene_root \
  --device cuda \
  --iterations 30000 \
  --output outputs/osn_gs_run
```

The project also exposes a notebook-compatible wrapper:

```bash
python train.py -s /path/to/scene_root -m outputs/osn_gs_run --iterations 30000
```

To train from a COLMAP/Graphdeco-style dataset:

```bash
python train.py \
  -s /path/to/scene_root \
  -m outputs/osn_gs_colmap \
  --iterations 30000 \
  --image_downscale 2
```

If `diff_gaussian_rasterization` is installed, OSN-GS uses it automatically. Otherwise it uses a differentiable fallback renderer that is useful for debugging the OSN-GS pipeline, but not for final 3DGS-quality results.

## Colab Notebook

`../3DGS_Renderer/colab_train_3dgs.ipynb` now has an `osn_gs` framework mode. In the project setup cell, keep:

```python
FRAMEWORK_MODE = 'osn_gs'
```

The notebook will discover an uploaded `OSN-GS` project zip by its top-level `train.py`, discover/upload a COLMAP-style dataset, and pass that dataset to OSN-GS through `train.py -s DATA_ROOT`.

## Inputs

OSN-GS now supports a COLMAP/Graphdeco-style scene directly:

```text
scene_root/
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
```

Text exports are also accepted:

```text
scene_root/
  images/
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.txt
```

## Outputs

Each save directory contains:

- `point_cloud.ply`: trained Gaussian cloud with an `uncertain` vertex property
- `render.ppm`: rendered image preview
- `checkpoint.pt`: Torch checkpoint
- `metrics.txt`: iteration, loss, PSNR, Gaussian counts, and rasterizer backend flag

## Main Modules

- `osn_gs/core/torch_pipeline.py`: observed curve fitting, occlusion curve prediction, surface construction, uncertain Gaussian initialization
- `osn_gs/core/torch_trainer.py`: differentiable training loop and output saving
- `osn_gs/gaussian/torch_model.py`: 3DGS-style Torch Gaussian parameter container
- `osn_gs/gaussian/torch_density_control.py`: uncertain pruning and promotion policy
- `osn_gs/surface/torch_nurbs.py`: Torch NURBS surface representation (Cox-de Boor rational B-spline evaluator)
- `osn_gs/surface/torch_voxel_regions.py`: pre-NURBS voxel surface regioning (batched normal estimation + boundary detection)
- `osn_gs/render/gaussian_rasterizer.py`: CUDA rasterizer bridge with chunked torch fallback renderer
- `osn_gs/losses/torch_losses.py`: image, surface, uncertainty, and anchor losses

## CUDA Dependencies

For full 3DGS-quality training, install the standard 3DGS CUDA submodules in the Python environment:

- `diff_gaussian_rasterization`
- `simple_knn`

The workspace already contains a reference `gaussian-splatting` checkout, so those submodules can be installed from there on the target Linux/CUDA machine.

## Ongoing Context Log

- 2026-07-01: User requested that whenever the environment, project situation, or task direction changes, the relevant `.md` files should be updated with that context instead of relying only on chat history.
- 2026-07-01: NURBS is an intermediate representation, not a replacement final output. Training should keep Gaussian primitives as the main output while preserving visible NURBS reconstruction data for later visualization tools.
- 2026-07-01: The Colab training notebook should pass NURBS-related configuration alongside OSN-GS training/Gaussian primitive output handling so downstream visualization can consume both Gaussian and NURBS artifacts.

- 2026-07-01: WebRenderer PLY compatibility request. Renderer requires Graphdeco-style Gaussian fields `x`, `y`, `z`, `f_dc_0..2`, raw `opacity`, optional raw log `scale_0..2`, and `rot_0..3`. OSN-GS has corresponding primitives in `TorchGaussianModel`, so `save_ply` should emit those names instead of debug-only RGB/`scale_x` fields.
- 2026-07-01: Notebook output packaging now includes NURBS visualization data. OSN-GS output inspection creates `visualization_manifest.json` under `MODEL_ROOT`, pairing each `point_cloud.ply` with its sibling `nurbs_surface.json` so external tools can load Gaussian primitives and the visible NURBS intermediate together.
- 2026-07-02: Added `visible_surface_resolution_scale` so Stage 1 visible NURBS control-grid density can be increased from the notebook Train cell without changing the base U/V parameters. Final resolution is computed from `visible_surface_resolution_u/v * scale`.
- 2026-07-02: High `visible_surface_resolution_scale` can increase NURBS fitting memory. Added `visible_surface_fit_device` and `visible_surface_fit_chunk_size` so fitting can run on CPU and process the uv grid in chunks while keeping the final NURBS intermediate available for visualization.
- 2026-07-02: Notebook Train cell should show live RAM/VRAM usage, current training iteration, and a bounded train.py output tail so long runs do not leave the user waiting without feedback.
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


## 2026-07-10 Multi-Agent Follow-Up Context

- The user is now collaborating with both Codex and Claude. Keep implementation decisions, environment constraints, and current task direction in Markdown so another agent can continue without relying on chat history.
- `docs/README.md` is the primary handoff/worklog document. `Agent.md` stores agent workflow rules and Windows Remote-SSH caveats. `docs/architecture.md` stores framework-level design decisions.
- Do not treat NURBS or voxel processing as optional debug extras. The user clarified that NURBS and voxel regioning should remain strongly integrated into the OSN-GS learning framework, even while optimizing throughput.
- Training throughput work should focus on reducing blocking I/O, stream/cache overhead, and optimizer-state churn rather than disabling NURBS/Voxel reconstruction.

## 2026-07-10 ADC Schedule and Optimizer Update

- Notebook ADC schedule was moved closer to original 3DGS defaults: `densify_until_iter=15000`, `densification_interval=100`, `densify_grad_threshold=0.0002` for both OSN-GS and Graphdeco-style branches.
- `OSN_ADC_MAX_GAUSSIANS` was changed to `0` in the notebook during follow-up so ADC is not capped at 200k by default. Reconfirm VRAM expectations before using uncapped growth on large scenes.
- ADC clone/split/prune runs on CUDA tensors when training device is CUDA. It was not intentionally moved to CPU.
- `TorchGaussianModel.replace_tensors()` now attempts to preserve Adam optimizer state row-wise across Gaussian append/prune operations. Newly appended Gaussian rows start with zero optimizer moments; pruned rows are removed from optimizer state.
- `TorchOSNGSTrainer` no longer calls `training_setup()` immediately after normal ADC or uncertain cleanup changes. Surface rebuild still calls `training_setup()` because that path may change broader model/surface state.

## 2026-07-10 Streaming and Cache Throughput

- Notebook default `STREAM_EVERY` was changed from `200` to `1000` to reduce frequent full Gaussian JSON/WebSocket/cache overhead during training.
- OSN-GS streaming now uses a background worker queue for stream-cache writes and WebSocket sends. The training loop still builds a detached CPU payload snapshot, but file write and network send are no longer performed inline.
- The stream worker queue is bounded. If the renderer/cache path falls behind, snapshots may be skipped with a `[WS] stream queue full` message rather than blocking training.
- `STREAM_TO_RENDERER = False` should be treated as "disable live WebSocket only." Bulk streaming after training remains possible if `STREAM_CACHE_DIR` is still passed and `STREAM_EVERY` or `STREAM_ITERATIONS` cause cache snapshots to be produced.
- Before assuming bulk streaming works, verify that the Train cell always passes `--stream_cache_dir` even when `STREAM_TO_RENDERER` is false, and only gates `--stream_url` behind `STREAM_TO_RENDERER`.

## 2026-07-10 Current Known Risks

- `docs/architecture.md` has previously suffered Korean mojibake. Preserve UTF-8 carefully and avoid rewriting the whole file through console encodings.
- Notebook output/execution metadata is noisy and may contain large stale outputs. Prefer targeted JSON edits to specific source lines.
- Training performance bottlenecks seen so far include stream/cache JSON serialization, WebSocket send, output save, NURBS/voxel rebuild/export, and surface rebuild optimizer reset. Do not disable NURBS/Voxel to solve this unless the user explicitly asks.

## 2026-07-10 NURBS Made Real, Voxel Regions Vectorized, Legacy Prototype Removed

- `TorchNURBSSurface.evaluate()` now evaluates a real rational tensor-product NURBS (Cox-de Boor basis on a clamped uniform knot vector, weighted by `weights`) instead of bilinear interpolation over the control grid. Degree auto-clamps when a grid axis has fewer control points than the configured degree.
- `osn_gs/surface/torch_voxel_regions.py` normal estimation and boundary detection are fully vectorized (batched/chunked SVD, 6 vectorized neighbor-offset passes) instead of a Python loop per region; behavior verified equivalent on CPU, and the win grows substantially on GPU where the old code paid per-region kernel-launch overhead.
- `--low_vram` was fixed in `scripts/train_osn_gs_torch.py` (the flag was referenced but not registered in that script's own argparse, causing an immediate `AttributeError`; `train.py`'s parser in `osn_gs/interop/colab_args.py` was unaffected).
- The original numpy-only prototype framework (non-`torch_*` files across `osn_gs/core`, `osn_gs/gaussian`, `osn_gs/surface`, `osn_gs/losses`, `osn_gs/optim`, parts of `osn_gs/data`, `osn_gs/render/prototype_renderer.py`, parts of `osn_gs/utils`) was deleted along with its two already-broken consumers (`scripts/train_osn_gs.py`, `tests/test_framework_smoke.py`) and two already-executed one-off migration scripts under `scripts/devtools/`. See `Agent.md` for the exact file list. All `osn_gs/**/__init__.py` now export only `torch_*` symbols.

## 2026-07-10 Surface/ADC/I-O Stabilization

- Periodic global surface rebuild has been removed. Initial voxel/curve topology stays frozen while NURBS control points remain continuously trainable; only persistently failing patches may receive local correction.
- Visible NURBS control grids and rational weights now have a dedicated optimizer and receive Gaussian-to-surface fitting plus curvature gradients.
- Gaussian bindings now persist local UV and a normal-connected voxel patch ID. Voxel boundaries split the 6-neighbor graph; base curves and visible NURBS are fit per patch.
- Total NURBS control points are bounded by `max_surface_control_points` (notebook default 65,536). Notebook resolution scale default was reduced from the unsafe experimental value 100 to 4.
- Certain ADC no longer forces a top-10-percent fallback. It uses tracked screen/xyz gradients, original-style start/end scheduling, rotated anisotropic split offsets, opacity reset, delayed screen-size pruning, metadata inheritance, and exponential xyz LR decay.
- Stream/cache JSON list conversion now runs in the background worker. Payloads include all NURBS patches and voxel patch IDs.
- PLY remains renderer-compatible ASCII but uses a vectorized NumPy writer rather than a Python loop per Gaussian.
- Checkpoint format v2 preserves raw Gaussian tensors, both optimizer states, ADC accumulators, SH degree, UV/patch bindings, and all NURBS patches. Resume with `--resume_checkpoint PATH`.
- `STREAM_TO_RENDERER = False` still allows later bulk streaming because `--stream_cache_dir` is passed independently.
- Windows notebook setup no longer hardcodes CUDA 13.3 or a specific MSVC toolset directory; it respects a valid existing `CUDA_HOME`.
- Uncertain-to-certain promotion remains forbidden. Current Stage 1 still creates visible surfaces only.
- Detailed implementation reports live under `docs/worklogs/`.

## 2026-07-10 ADC Parity Expansion

- Rechecked ADC against the official Graphdeco `gaussian-splatting` `train.py` and `scene/gaussian_model.py`.
- Densification now uses the original open iteration boundaries, and opacity reset is independent from the densification interval.
- Screen-space and world-space oversized pruning activate together only after the configured size-pruning threshold.
- Gaussian optimizer order now follows backward -> ADC/prune -> optimizer step. Gradient rows and Adam rows survive append/prune; new children start with zero gradients/moments.
- Position LR uses scene spatial scaling and opacity LR defaults to 0.05.
- Notebook/CLI controls now expose `adc_percent_dense`, `adc_prune_opacity_threshold`, `adc_split_samples`, `adc_max_screen_size`, and `adc_max_scale_ratio`.
- OSN-GS extensions remain deliberate: optional Gaussian cap, persistent UV/patch metadata inheritance, uncertain prune-only behavior, and no uncertain-to-certain promotion.
- See `docs/worklogs/06_adc_parity_expansion.md`.


## 2026-07-10 Density-Adaptive Voxel NURBS

- Visible-surface regions now use a 16-base-grid adaptive voxel hierarchy; occupied cells above the configured weighted-density quantile are subdivided.
- Rebuild density combines Gaussian opacity with bounded inverse covariance volume. Initial construction falls back to Gaussian count density.
- Mixed-resolution face adjacency preserves normal-boundary patch splitting.
- NURBS control points are budgeted per patch using density and boundary complexity while retaining the global `max_surface_control_points` limit.
- Streaming payloads include voxel level, weighted density, and finest-grid bounds.
- Notebook/CLI controls: `adaptive_voxel_density`, `voxel_max_subdivision_depth`, `voxel_density_quantile`, and `voxel_density_covariance_weight_cap`.
- See `docs/worklogs/07_density_adaptive_voxel_nurbs.md`.

## 2026-07-10 Persistent Surface Lifecycle

- Density-adaptive voxel regioning now runs once as the initialization bootstrap.
- The former rebuild interval is a NURBS quality-inspection interval exposed as --surface_update_interval; --surface_rebuild_interval remains a CLI compatibility alias.
- NURBS control grids and rational weights continue updating every iteration through the surface optimizer.
- Scene-normalized patch residuals use patience before triggering correction.
- Local correction voxelizes only the failed patch and appends significant split components without replacing the initial voxel snapshot or existing patches.
- Existing Adam state is retained; only newly appended patch tensors are registered.
- Maintenance counters and topology version are checkpointed and included in NURBS output/stream metadata.
- See docs/worklogs/08_persistent_surface_lifecycle.md.

## 2026-07-11 Parametric NURBS Fitting (Derivatives, Foot-Point UV, Least-Squares)

- `TorchNURBSSurface` now evaluates analytic first derivatives and surface normals (`evaluate_with_derivatives`, `normals`) alongside positions.
- Foot-point projection (`project_torch_points_to_nurbs`) binds Gaussians to their true closest surface parameter: dense-grid seeding plus damped Gauss-Newton, guaranteed never worse than the seed.
- Surface maintenance refreshes certain-Gaussian UV bindings by foot-point projection before measuring patch quality, so residuals now measure real point-to-surface distance. Report/log includes `uv_refreshed`.
- Visible NURBS fitting is now a regularized least-squares solve (control grid linear in the fit because rational weights are 1 at fitting time) alternated with foot-point UV reprojection. The old inverse-distance fill remains as the seed and as `--surface_fit_mode idw` fallback. Voxel region density acts as point weights.
- On an analytic sheet, normalized RMS surface distance improved ~3x versus the IDW seed (0.0087 -> 0.0028); regression-tested with threshold 0.005.
- Default NURBS degree is now (2, 2) — degree_v was previously 1 (piecewise linear). New knobs in config/CLI/notebook: `surface_fit_mode`, `surface_degree_u/v`, `surface_fit_smoothness` (default 1e-4), `surface_fit_tikhonov` (1e-4), `surface_fit_rounds` (2), `surface_projection_iterations` (4). Both `train.py` and `scripts/train_osn_gs_torch.py` share these via `osn_gs/interop/colab_args.py`; the notebook exposes them as `OSN_SURFACE_*`.
- Remaining follow-ups: per-patch UV occupancy (trimming) mask, cross-patch UV reassignment, smoothness default revalidation on real COLMAP scenes. See docs/worklogs/09_nurbs_derivatives_footpoint.md and docs/worklogs/10_least_squares_nurbs_fit.md.

## 2026-07-13 Synthetic NURBS Constructor Validation

- Added the isolated root-level `nurbs_constructor_benchmark/` framework. It generates deterministic plane, sine-sheet, and sharp-crease Gaussian-center scenes, then calls the production `TorchOSNGSPipeline.initialize()` path directly; no NURBS constructor code is copied.
- Each run records input-point foot-point RMS, analytic chart residual, normal error, patch/control-point counts, and finite-value status to `report.json`. Optional error thresholds make it usable as a regression gate.
- Usage and extension instructions live in `nurbs_constructor_benchmark/README.md`. See `docs/worklogs/11_synthetic_nurbs_constructor_benchmark.md` for scope and verification status.

## 2026-07-14 CUDA Build Preflight

- CUDA rasterizer를 사용할 때 train.py와 scripts/train_osn_gs_torch.py는 scene loading 전에 MSVC, INCLUDE/LIB, CUDA nvcc, Ninja를 검사한다.
- Windows에서는 preflight가 x64 MSVC environment를 현재 training process에 활성화한다. 따라서 cl.exe가 notebook kernel의 초기 PATH에 없더라도 JIT build 전에 복구된다.
- Preflight then prepends the resolved compiler directory and requires PyTorch's exact where cl probe to succeed.
- 준비가 안 된 경우 iteration 0 renderer error 대신 실행 가능한 원인을 포함한 preflight error로 즉시 종료한다.
- --skip_cuda_build_preflight는 진단 우회용이며 기본값은 검증 실행이다.
- See docs/worklogs/09_cuda_build_preflight.md.

## 2026-07-14 Surface Loss Patch Minibatch

- Notebook timing showed normal iterations were dominated by backward at about 0.30s while rasterization was about 0.005-0.007s.
- The cause was full multi-patch NURBS loss evaluation: every patch performed a Python bool(mask.any()) GPU synchronization per iteration.
- NURBS loss now uses a deterministic round-robin patch minibatch. Default surface_loss_patch_budget=16; 0 keeps the full-patch behavior.
- Active patches still receive anchor fitting plus smoothness gradients, and all patches rotate through the loss schedule.
- Timing now reports surface_loss separately from backward.
- See docs/worklogs/10_surface_loss_patch_minibatch.md.

## 2026-07-14 Surface Loss Runtime Audit

- Stored notebook timing showed the old full-patch NURBS loss path: render forward was about 0.006s while the combined surface-loss/backward phase was about 0.30s.
- The current trainer uses a round-robin surface patch budget of 16 by default; 0 explicitly restores full-patch evaluation.
- Training startup now prints the effective patch budget, and timing separates surface_loss from backward for the next run.
- See docs/worklogs/10_surface_loss_runtime_audit.md.


## 2026-07-14 Training Bottleneck Audit

- A completed notebook run with the current timing split reports steady NURBS surface loss at 0.056s and backward at 0.123s; renderer forward is 0.013s.
- The recurring large cost is ADC: at approximately 190k Gaussians, the density stage measured 4.644s because clone/split/prune each rebuild Gaussian tensors and preserve Adam state.
- Per-iteration CUDA-to-CPU metric extraction currently serializes the training stream. Full GPU-to-CPU streaming snapshots and periodic global surface maintenance are checkpoint-bound costs.
- See docs/worklogs/11_training_bottleneck_audit.md.
