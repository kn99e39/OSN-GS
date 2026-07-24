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
- 2026-07-20: Stream snapshots now include `shDegree` and coefficient-major RGB `shCoefficients` for the active SH degree, so the WebGPU renderer can reproduce view-dependent color in Gaussian Composition. Degree-3 payloads are substantially larger than prior DC-RGB snapshots; use a lower stream cadence or `stream_max_gaussians` when live streaming becomes bandwidth-bound. See `docs/worklogs/38_renderer_sh_streaming.md`.
- Streamed snapshots can include the visible NURBS intermediate as `nurbs_surface`; the payload is sent when the surface is first available or rebuilt.
- Notebook OSN-GS training exposes streaming knobs and can disable slow PLY/NURBS/checkpoint file output when using the renderer stream.

## 2026-07-06 Covariance Initialization Pipeline

- OSN-GS now initializes Gaussian covariance scale from chunked nearest-neighbor point spacing, following the original 3DGS scale+rotation covariance convention without requiring `simple-knn`.
- Notebook and CLI controls expose covariance initialization mode, KNN chunk sizing, min scale, max scene-scale ratio, and scale multiplier.


## 2026-07-10 Multi-Agent Follow-Up Context

- The user is now collaborating with both Codex and Claude. Keep implementation decisions, environment constraints, and current task direction in Markdown so another agent can continue without relying on chat history.
- `docs/README.md` is the primary handoff/worklog document. `AGENTS.md` stores agent workflow rules and Windows Remote-SSH caveats. `docs/architecture.md` stores framework-level design decisions.
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
- The original numpy-only prototype framework (non-`torch_*` files across `osn_gs/core`, `osn_gs/gaussian`, `osn_gs/surface`, `osn_gs/losses`, `osn_gs/optim`, parts of `osn_gs/data`, `osn_gs/render/prototype_renderer.py`, parts of `osn_gs/utils`) was deleted along with its two already-broken consumers (`scripts/train_osn_gs.py`, `tests/test_framework_smoke.py`) and two already-executed one-off migration scripts under `scripts/devtools/`. See `AGENTS.md` for the exact file list. All `osn_gs/**/__init__.py` now export only `torch_*` symbols.

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
- Remaining follow-ups: per-patch UV occupancy (trimming) mask, cross-patch UV reassignment, smoothness default revalidation on real COLMAP scenes. See docs/worklogs/10_nurbs_derivatives_footpoint.md and docs/worklogs/11_least_squares_nurbs_fit.md.

## 2026-07-13 Synthetic NURBS Constructor Validation

- Added the isolated root-level `nurbs_constructor_benchmark/` framework. It generates deterministic plane, sine-sheet, and sharp-crease Gaussian-center scenes, then calls the production `TorchOSNGSPipeline.initialize()` path directly; no NURBS constructor code is copied.
- Each run records input-point foot-point RMS, analytic chart residual, normal error, patch/control-point counts, and finite-value status to `report.json`. Optional error thresholds make it usable as a regression gate.
- Usage and extension instructions live in `nurbs_constructor_benchmark/README.md`. See `docs/worklogs/14_synthetic_nurbs_constructor_benchmark.md` for scope and verification status.

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
- See docs/worklogs/12_surface_loss_patch_minibatch.md.

## 2026-07-14 Surface Loss Runtime Audit

- Stored notebook timing showed the old full-patch NURBS loss path: render forward was about 0.006s while the combined surface-loss/backward phase was about 0.30s.
- The current trainer uses a round-robin surface patch budget of 16 by default; 0 explicitly restores full-patch evaluation.
- Training startup now prints the effective patch budget, and timing separates surface_loss from backward for the next run.
- See docs/worklogs/13_surface_loss_runtime_audit.md.


## 2026-07-14 Training Bottleneck Audit

- A completed notebook run with the current timing split reports steady NURBS surface loss at 0.056s and backward at 0.123s; renderer forward is 0.013s.
- The recurring large cost is ADC: at approximately 190k Gaussians, the density stage measured 4.644s because clone/split/prune each rebuild Gaussian tensors and preserve Adam state.
- Per-iteration CUDA-to-CPU metric extraction currently serializes the training stream. Full GPU-to-CPU streaming snapshots and periodic global surface maintenance are checkpoint-bound costs.
- See docs/worklogs/15_training_bottleneck_audit.md.

## 2026-07-15 Direction Correction: NURBS must not move visible Gaussians

- **Corrects a wrong premise that had propagated from `docs/architecture.md` into the code.** The doc described the NURBS as the "single geometric source of truth" whose updates move Gaussian positions. The actual intent is the opposite and one-way: visible structure -> derive NURBS -> infer the occluded surface -> generate Gaussians there. Visible/certain Gaussians are optimized by the image loss alone and must never be pulled by the surface.
- `nurbs_surface_loss` selected `certain = ~is_uncertain` and let gradient flow back into their `_xyz`, so the surface was dragging visible Gaussians. Fixed by detaching the observed Gaussian positions: the term now fits the surface to the Gaussians only. Verified: grad to `model._xyz` is 0 while grad to `control_grid` is > 0. Per-iteration NURBS updates are retained (intended).
- `docs/architecture.md` rewritten accordingly (data-flow, Core Principles, ADC, training-loop pseudocode). Prior worklogs are intentionally left as-is; **`docs/worklogs/23_nurbs_direction_correction.md` states the go-forward direction and takes precedence over any conflicting older text.**
- This also removes, by design, the `TODO.md` candidate "NURBS anchor constrains certain Gaussians". Note `lambda_surface` now means "how fast the surface follows the Gaussians", not how hard Gaussians are constrained.

## 2026-07-15 D-SSIM Image Loss

- OSN-GS image loss now matches original 3DGS: `(1 - lambda_dssim)*L1 + lambda_dssim*(1 - SSIM)` with `lambda_dssim=0.2` (was `0.8*L1 + 0.2*MSE`, no SSIM). This was the #1 suspected cause of the baseline quality gap in `TODO.md`.
- Added a pure-torch `ssim` in `osn_gs/losses/torch_losses.py` ported from `gaussian-splatting/utils/loss_utils.py` (window 11, sigma 1.5, C1/C2 identical) — verified numerically identical to the original (diff 0.0). MSE stays only for PSNR. `TorchTrainingConfig.lambda_l1/lambda_mse` replaced by `lambda_dssim`.
- Tests (26) pass; 6-iteration smoke shows the D-SSIM loss decreasing and differentiable. Still needs a resolution-matched 10k A/B re-train to quantify the gap reduction (see `TODO.md`). See `docs/worklogs/22_ssim_image_loss.md`.

## 2026-07-15 SSH Stream Server Split

- OSN-GS live streaming is split from the training loop again. Training remains a WebSocket client using `--stream_url`; it no longer opens a trainer-owned WebSocket server.
- `scripts/start_trainer_stream.ps1` starts only the loopback stream server at `127.0.0.1:8080` by default. It does not start training or require dataset/output paths.
- Remote renderers should use SSH local port forwarding, for example `ssh -N -L 8080:127.0.0.1:8080 user@trainer-host`, then connect the browser renderer to `ws://localhost:8080` on the renderer machine.
- Notebook/training on the trainer machine should stream to `ws://127.0.0.1:8080` when the local stream server is running.
- See docs/worklogs/16_ssh_stream_server_split.md.

## 2026-07-15 UV Trimming (Surface Support)

- Each NURBS patch now carries a UV support (trim) mask so the rectangular chart is not drawn/measured past the observed point footprint. Computed in `TorchOSNGSPipeline.initialize` from bound Gaussian UVs (occupancy grid + dilation); config knobs `surface_trim_resolution` (default 24, 0 disables) / `surface_trim_dilation` (default 1). `TorchNURBSSurface.uv_support_mask` + `.support(uv)`; exported per-patch as `uv_support` in `nurbs_surface.json`; persisted in checkpoint v2.
- Benchmark support metric samples only the trimmed region. Extrapolation dropped without opening coverage holes (uncovered unchanged): plane 0.239→0.089, sine 0.184→0.092, crease 0.010→0.004. `density_gradient` (0.759→0.659) is limited by the support threshold being calibrated to the dense cluster's spacing, not by trimming. Training math is unaffected (mask is metadata / renderer hint only). See `docs/worklogs/21_uv_trimming.md`.

## 2026-07-15 Ground-Truth NURBS Benchmark Metrics

- `nurbs_constructor_benchmark` now scores the generated NURBS against ground truth on three independent concerns instead of one conflated residual: **Surface Fitting Accuracy** (`chamfer_rms`/`accuracy_rms`/`completeness_rms`), **Surface Support** (`support_coverage_uncovered_fraction`, `support_extrapolation_fraction`), and **Patch Topology** (`topology_label_ari`, patch-count match). Each result carries a `ground_truth` block; optional gates `--max-chamfer-rms`, `--max-extrapolation`, `--min-topology-ari`.
- Each scene exposes its analytic surface + true patch topology (`scenes.py`); metrics in `metrics.py`; a ground-truth NURBS is emitted as `NURBS_output/<scene>/nurbs_surface_gt.json` (renderer format, correct topology — 2 patches for `crease`) for visual overlay.
- These separate failure modes the chart RMS hid: `crease` fits with low residual but low topology ARI (over-segmentation), `density_gradient` shows a high extrapolation fraction. See `docs/worklogs/20_ground_truth_nurbs_metrics.md`.

## 2026-07-15 Baseline 3DGS Comparison Enabled

- The local `gaussian-splatting/` folder (original Graphdeco 3DGS) is now selectable and runnable on this system for side-by-side comparison. Notebook `FRAMEWORK_MODE='graphdeco_3dgs'` now resolves `GS_ROOT` to the local `gaussian-splatting/` folder (previously both modes pointed at the OSN-GS root).
- Its CUDA extensions (`diff_gaussian_rasterization`, `simple_knn`) build on torch 2.12+cu130 / RTX 5080 (sm_120) with `TORCH_CUDA_ARCH_LIST=12.0` and `CL=/Zc:preprocessor` (required by CUDA 13 CCCL headers). `fused_ssim` is optional (train.py falls back to torch SSIM). Reusable build script: `scripts/build_baseline_extensions.bat`. The notebook build cell now injects these flags so a fresh run can compile them.
- Verified end-to-end: 30-iteration `gaussian-splatting/train.py` run on `DATASET` (loss decreasing, train PSNR ~16.8 at iter 30). Installing `diff_gaussian_rasterization` in the venv makes OSN-GS use it as the installed backend (same source as vendored, so behavior is unchanged and faster). Tests (26) pass.
- Fairness: OSN-GS defaults to half-resolution (`--low_vram`) while the baseline trains near full resolution, and the two use different image losses (L1+MSE vs L1+D-SSIM). Match resolution (`--no-low_vram` on OSN-GS or `-r` on the baseline) before comparing. See `docs/worklogs/19_baseline_3dgs_comparison_setup.md` and `TODO.md`.

## 2026-07-15 Notebook/CLI Training Parity

- Fixed a silent divergence where a bare CLI run (`train.py` or `scripts/train_osn_gs_torch.py`) used different defaults than the notebook, most importantly ADC being OFF on the CLI (`densify_until_iter`/`densification_interval` defaulted to 0).
- Both CLI parsers now default to the notebook's **VRAM-safe recipe** so an argument-free run reproduces `colab_train_3dgs.ipynb`: `densify_until_iter=15000`, `densification_interval=100`, `visible_surface_resolution_scale=4.0`, and `--low_vram` on by default (`BooleanOptionalAction`; pass `--no-low_vram` for a full-resolution run).
- Notebook's `--low_vram` forwarding updated to `--low_vram`/`--no-low_vram` so `OSN_LOW_VRAM=False` still opts out under the new default-on semantics.
- Defaults are duplicated across `osn_gs/interop/colab_args.py`, `scripts/train_osn_gs_torch.py`, and the notebook `OSN_*` block — keep all three in sync when changing a training default. Perf-only knobs (`image_device`, `visible_surface_fit_device`, chunk sizes, streaming/log cadence) are intentionally not forced to match since they do not change the trained result. See `docs/worklogs/18_notebook_cli_training_parity.md`.

## 2026-07-15 Hot-Path Metric Scalars Removed

- Implemented priority 1 from `docs/worklogs/15_training_bottleneck_audit.md`: the training loop no longer forces a per-view or per-iteration GPU→CPU synchronization for loss/MSE scalars.
- MSE is accumulated as a device tensor, `mean_mse` feeds the uncertainty loss directly (no CPU→GPU round trip), and `state.last_loss`/`state.last_psnr` are only materialized to host floats when a progress log, stream snapshot, or file save reads them (`_needs_metric_scalars`).
- State dataclass and checkpoint format are unchanged; loss/PSNR remain float fields. Tests (26) pass; a CPU smoke confirms metrics stay finite and correct when intermediate iterations skip materialization.
- Bottleneck audit priorities 2 (ADC single shape transaction) and 3 (snapshot decouple + duplicate-final-snapshot removal) remain open. See `docs/worklogs/17_hot_path_metric_scalars.md`.

## 2026-07-15 NURBS Patch Aspect-Ratio Fix

- Fixed the elongated/sliver-patch NURBS fitting bug tracked in `TODO.md`: `_fit_surface_patches` and `_split_failed_patch` now split each patch's control-point budget using the patch's actual PCA extent aspect ratio (`pca_extent_aspect_ratio` in `osn_gs/surface/torch_nurbs.py`) via the shared `_target_resolution` helper, instead of a fixed global `base_u/base_v` aspect for every patch.
- `crease` benchmark patch grid aspect moved toward the true shape (e.g. patches with true aspect ~5.3 went from grid 2.33 to 2.67-4.00); still bounded by the global `base_u` cap. Tests (26) pass, benchmark shows no regression on the other scenes. See `TODO.md` for the remaining known limitation.

## 2026-07-15 Notebook Interrupt Cleanup

- The notebook Train cell now catches `KeyboardInterrupt` inside `_run_monitored_process()` and terminates the active `train.py` subprocess before re-raising the interrupt.
- Termination first calls `process.terminate()` and waits up to 10 seconds; if the process does not exit, it falls back to `process.kill()`.
- This cleanup affects the training subprocess only. The standalone stream server started by `scripts/start_trainer_stream.ps1` remains a separate process and should still be stopped with Ctrl+C in its own terminal.

## 2026-07-15 Local Graphdeco Notebook Build

`colab_train_3dgs.ipynb` now supports the bundled `gaussian-splatting` project from a Windows local Jupyter kernel. The dependency cell runs `apt-get` only in Colab. Its CUDA extension cell activates the installed MSVC x64 environment, exposes `cl.exe` and `nvcc`, sets `TORCH_CUDA_ARCH_LIST` for the active GPU, and prints complete compiler output on failure. Re-run notebook cells 3, 5, and 6 after a kernel restart before using the Graphdeco training cell. CUDA Toolkit 13.3 with PyTorch CUDA 13.0 emits a minor-version warning but the local diff rasterizer and simple-knn extensions were built and imported successfully.

The local Graphdeco notebook cells read/write patched Python sources with explicit UTF-8 encoding and decode captured subprocess output with `encoding='utf-8', errors='replace'` so Windows kernels do not fall back to `cp949` when upstream files or tool output contain non-ASCII text. The CUDA extension cell also skips rebuilding extensions that are already importable, because Windows keeps imported `.pyd` files locked until the Jupyter kernel restarts. The train cell now uses the same compact live monitor for OSN-GS and baseline Graphdeco runs: resource status, one progress bar with ETA, and one latest iteration/log line.

## 2026-07-15 Renderer Multi-Patch Validation

- Confirmed WebRenderer revision 88477a8 renders all valid NURBS patches, uses deterministic patch color, keeps iso-lines inside each patch, and includes all patches in camera bounds.
- Node is unavailable in this environment, so the included smoke test was not run. Remaining renderer diagnostics, parity, and provenance work are recorded in docs/worklogs/24_renderer_multipatch_validation.md.

## 2026-07-15 Renderer Local-Test Handoff

- Renderer Priority 0 test procedure and pass criteria were moved to docs/worklogs/25_renderer_local_test_handoff.md for execution on a WebGPU-capable local machine.

## 2026-07-15 NURBS Constructor TODO Audit

- Removed already implemented constructor benchmark baselines and metrics from TODO. Remaining NURBS work is recorded in docs/worklogs/26_nurbs_constructor_todo_audit.md.

## 2026-07-15 Support-Domain Constructor Benchmark

- Added deterministic triangle, u_shape, crescent, and planar_hole scenes with analytic GT support predicates and in-domain Gaussian sampling.
- report.json now includes shared-XY support precision/recall/IoU, coverage/unsupported/uncovered, components/holes/Euler topology mismatch, boundary Chamfer/Hausdorff, and artifact paths.
- See docs/worklogs/29_support_domain_benchmark.md.

## 2026-07-16 Notebook Train MSVC Environment

- The OSN-GS Train cell now captures vcvars64.bat into the train.py subprocess environment on Windows, including PATH, INCLUDE, and LIB. This fixes the CUDA rasterizer preflight failure where cl.exe was absent despite the notebook extension-build cell succeeding.
- The subprocess environment is verified with where cl before training starts. See docs/worklogs/30_notebook_train_msvc_env.md.

## 2026-07-16 Unified PowerShell CLI

- After activating .venv, osn-gs --help lists train, benchmark, inspect-surface, and stream-server. The editable package install registers the console script; each subcommand delegates to the existing implementation and preserves its original options.
- Install or refresh it with .venv\Scripts\python.exe -m pip install -e . --no-deps. See docs/worklogs/31_unified_cli.md.

## 2026-07-16 Benchmark GT Renderer Folder Split

- Constructor benchmark renderer exports now keep generated and GT NURBS in separate sibling folders: NURBS_output/scene and NURBS_output/scene_gt. Both use the renderer-standard filename nurbs_surface.json, so loading directories no longer combines two same-type surfaces in one snapshot. See docs/worklogs/32_benchmark_gt_folder_split.md.

## 2026-07-20 Governing NURBS Construction Plan (superseded 2026-07-23)

- The Boundary-First direction (`OSN_GS_Final_Boundary_First_NURBS_Direction.md`) and its Phase 5 extension plan were retired on 2026-07-23 by the direction reset and their now-non-urgent docs were removed from `docs/Urgent_Work/`. The Boundary-First *production code path* (Step 5-A coupled boundary fit, etc.) is preserved unchanged; its completed experiments remain in `docs/worklogs/` (39–56) and the initial retirement in `docs/worklogs/73b_urgent_work_plan_retirement.md`.
- Current governing methodology: [OSN_GS_Direction_Reset_Plan.md](Urgent_Work/OSN_GS_Direction_Reset_Plan.md) (top-level, boundary-conditioned occluded-surface construction) with the active gate in [OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md](Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md). Phases A–E are implemented and Gate A–E approved; Phase F is next.

## 2026-07-20 Phase 2 Component Boundary Baseline

- Fake-hole handling is governed by the adaptive raw-Gaussian-count voxel hierarchy (large uniform root AABB, recursive split only above the maximum leaf count, inactive below the minimum), not by a fixed support-loop cell-size filter. Sparse density-gradient gaps remain calibration diagnostics. See docs/worklogs/34_phase2_component_boundary.md.

## 2026-07-20 Phase 3 Trimmed Component Correctness Baseline

- Phase 3 reuses the existing LSQ/foot-point fitter once per physical component and applies Phase 2 support as a trim mask. Plane, sine, and planar_hole pass the correctness baseline; density_gradient remains a sparse-support calibration risk. See docs/worklogs/35_phase3_trimmed_component_baseline.md.

## 2026-07-20 Boundary-First End-State Decision

- Legacy and voxel_patch_stage1 remain benchmark comparison modes during the approved Boundary-First phases.
- After the final Phase benchmarks pass and the user approves integration, the boundary-first NURBS construction pipeline becomes the only main-training path. The legacy constructor and obsolete two-stage voxel-region path are removed rather than retained as a permanent production fallback.


## 2026-07-21 Notebook 저장 출력 오류 수정

- 5000 iteration 저장 시 voxel region Tensor가 JSON 직렬화를 막아 nurbs_surface.json과 checkpoint 저장 전에 중단되던 문제를 수정했다. 노트북은 subprocess 실패 시 마지막 출력 tail을 표시한다. See docs/worklogs/44_notebook_save_output_error.md.

## 2026-07-22 Phase 4-D 정준 레이아웃 재평가

- 현재 production Phase-2 경계 추정기에서 worst_wedge_optimized는 offcenter 일부 수치를 개선하지만 wedge-wide fold를 제거하지 못하고, elliptical 및 density-gradient 회귀가 남아 기본 경로로 채택하지 않았다. uniform_angle은 계속 production 기준선이며 selector/retry/fallback은 도입하지 않는다. 다음 단계는 사용자 승인 후 단일 결정론적 annulus-layout objective/제약을 재설계하는 것이다. 근거는 docs/worklogs/52_phase4d_production_estimator_multiseed_reevaluation.md 및 OSN_GS_Phase4_Hardening_Plan.md에 기록했다.

## 2026-07-22 인자 없는 inspect-surface 검사

- `osn-gs inspect-surface`는 이제 인자 없이 실행할 수 있다. 로컬 노트북 Dataset 셀의 `DATA_ROOT=NOTEBOOK_ROOT / 'DATASET'`과 OSN-GS Train 셀의 Stage 1/NURBS/voxel/covariance 기본값을 사용한다.
- 기본 출력은 학습 `stream_cache`와 분리된 `output/osn_gs_scene/inspect-surface/`이다. 이 폴더에 `renderer_snapshot.json`, `surface_quality.json`, `surface_quality.txt`를 기록한다. `-s`와 `--output`으로 필요할 때만 재정의한다.
- 상세 검증 및 남은 위험은 `docs/worklogs/54_argument_free_inspect_surface.md`에 기록했다.

## 2026-07-22 Priority 8 학습 성능 및 품질 경로

- ADC clone/split/prune를 단일 shape transaction으로 통합해 parameter tensor와 Adam state 재구성을 pass당 최대 여러 번에서 1번으로 줄였다.
- snapshot capture는 기본 2-slot bounded pinned-memory queue와 CUDA event를 사용한다. `STREAM_QUEUE_SIZE`/`--stream_queue_size`가 대기 snapshot 수를 정하며, 같은 iteration의 final snapshot은 중복 전송하지 않고 cache/WebSocket JSON도 한 번만 직렬화한다.
- surface maintenance는 `OSN_SURFACE_MAINTENANCE_PATCH_BUDGET`/`--surface_maintenance_patch_budget`(기본 16)만큼 patch를 round-robin 검사한다. 0은 모든 patch 검사다.
- training view는 순차 순환 대신 seed 재현 가능한 epoch별 무작위 순열(without replacement)을 사용한다.
- CUDA smoke와 전체 150-test 회귀는 통과했다. 동일 해상도 10k baseline A/B는 남은 acceptance 검증이다. See `docs/worklogs/57_priority8_training_performance.md`.

## 2026-07-22 Urgent_Work 계획 문서 경로

- 루트 계획 문서 참조는 `docs/Urgent_Work/`로 이동했다. TODO, benchmark 문서, source/test docstring, 기존 worklog의 active-plan 참조를 새 위치에 맞췄고 Markdown 링크 검증에서 누락은 없었다. 상세 내용은 `docs/worklogs/58_urgent_work_reference_relocation.md`를 참고한다.

## 2026-07-22 Proxy-Based Surface Decomposition Stage 0/1

- Stage 0에서 현재 Phase 1/2 계약과 기준선을 동결했다. `curved_annulus`는 생성된 edge 14개가 모두 merge됐지만 face-contact가 없는 cross-component AABB-touch pair 때문에 2개 component로 갈리는 것을 재현했다. `mild_curved_sheet`의 spurious annulus는 component가 이미 1개라 Phase 2 문제로 분리했다. See `docs/worklogs/60b_proxy_decomposition_stage0_baseline.md`.
- Stage 1에 diagnostics-only local quadratic proxy와 실제 leaf-pair 분석 도구를 추가했다. 누락 curved pair는 기존 smooth pair와 비슷한 proxy distortion을 보였고, crease/parallel/disconnected는 단일 threshold가 아니라 proxy error·normal variation·layer direction·support gap의 독립 신호가 필요했다.
- Production component builder와 기본값은 변경하지 않았다. 당시 전체 171 tests 통과(1 skip), artifact 반복 생성 hash 일치. Stage 2 Spatial Candidate Graph는 후속 승인 후 완료했으며 아래에 기록했다. See `docs/worklogs/61_proxy_decomposition_stage1_quadratic_diagnostics.md` 및 `artifacts/proxy_decomposition_{baseline,stage1}.json`.

## 2026-07-22 Proxy-Based Surface Decomposition Stage 2

- Diagnostics-only scale-aware spatial candidate graph를 추가했다. Adaptive leaf AABB diagonal에 비례한 반경, deterministic sweep-and-prune, canonical pair ordering, duplicate 제거를 사용하며 face/edge/corner 관계는 결과 provenance로만 기록한다.
- 기본 `curved_annulus`에서 누락 smooth pair 4/4와 기존 face-smooth pair 14/14를 모두 포함했다. 12 nodes, 39 edges, degree mean 6.50, p95/max 9였다. 회전·point count·adaptive leaf resolution·density-gradient sweep에서도 평가 reference recall은 모두 1.0이었다.
- Candidate graph는 의도적으로 broad하다. 8-leaf crease/parallel 및 5–8-leaf density 조건은 complete graph였고, GT-cross false candidate 비율은 crease 57.1%, parallel layer 57.1%, disconnected-close 27.3%였다. 거리 sweep은 edge membership보다 기록된 `support_gap`에 반영되므로 Stage 3 admissibility에서 독립적으로 검증해야 한다.
- 실제 `osn-gs benchmark --constructor boundary_first --bf-candidate-diagnostics`를 실행했으며 production `curved_annulus` component count는 2로 유지됐다. 전체 182 tests 통과(1 skip). 이후 승인된 Stage 3 diagnostics-only prototype 결과는 아래에 기록했다. 상세 근거는 `docs/worklogs/62_proxy_decomposition_stage2_candidate_graph.md`와 `artifacts/proxy_decomposition_stage2.json`, `artifacts/proxy_decomposition_stage2_unified_benchmark.json/report.json`에 있다.

## 2026-07-22 Proxy-Based Surface Decomposition Stage 3

- Stage 2 graph 위에 atomic-leaf 초기 region, local quadratic proxy, normalized support gap, layer consistency를 사용하는 diagnostics-only deterministic merge-only agglomeration을 추가했다. 모든 raw diagnostics를 먼저 계산한 뒤 ordered gate를 적용하며 weighted semantic score는 사용하지 않는다.
- 기본 seed-0에서는 `curved_annulus`를 1 region으로 복원하면서 crease, close parallel sheets, disconnected gap 0.1을 각각 2 regions로 유지했고 plane·planar annulus·density-gradient를 회귀시키지 않았다. 회전, point count, leaf resolution, parallel distance sweep도 통과했다.
- 광범위 sweep에서는 density-gradient 4/5, disconnected gap 0.1은 2/5만 성공했다. Density seed 2 연결은 gap/spacing 5.163 초과가 필요하지만 disconnected seed 1 차단은 3.496 미만이 필요해 현재 signal set에 scene-independent 공통 threshold가 없다. 작은 disconnected gap 0.02/0.05도 오병합됐고 `mild_curved_sheet`는 과분할됐다.
- 반복 실행, reversed candidate ordering, 전체 artifact가 hash-identical했다. 전체 suite는 191 passed, 1 skipped이며 실제 `osn-gs benchmark --constructor boundary_first`에서도 production membership과 기본값은 그대로였다.
- 결론: Stage 3 broad methodology feasibility gate는 실패했다. 후속 승인된 Stage 3-R Gaussian-native diagnostics 결과는 아래에 기록했으며, production integration 및 Stage 4는 여전히 진행하지 않는다. 상세 근거는 `docs/worklogs/64_proxy_decomposition_stage3_merge_only_diagnostics.md`, `artifacts/proxy_decomposition_stage3.json`, `artifacts/proxy_decomposition_stage3_production_benchmark.json/report.json`에 있다.
## 2026-07-22 Proxy-Based Surface Decomposition Stage 3-R

- Gaussian mean/covariance/opacity만 받는 diagnostics-only continuity evaluator를 추가했다. Mahalanobis distance, k-sigma ellipsoid overlap, directional/tangent/normal reach, local bridge density, facing support mass와 기존 point diagnostics를 독립적으로 기록하며 merge decision과 weighted score는 없다.
- 실제 field audit 결과 synthetic scene은 scale/rotation/covariance/opacity를 제공하지 않고, current production initialization도 KNN spacing을 xyz에 반복한 isotropic scale, identity rotation, opacity 0.12다. 따라서 actual 33 pair 모두 covariance principal axis가 surface normal을 의미하지 않았다.
- Stage 3 core conflict에서 best valid signal인 pooled Mahalanobis q0.1은 AUC 0.90이지만 separation margin -0.359였다. Invalid principal-axis 신호를 제외한 두-signal AND도 disconnected gap 0.02를 false positive로 남겼다.
- Opacity weighted/unweighted bridge 차이는 최대 4.44e-16이었다. Bridge density는 sample 17/33/65와 truncation 3/4/6에는 안정적이었지만 covariance scale multiplier 0.5/1/2에 따라 positive와 negative가 함께 0.002 수준에서 0.82 수준까지 이동했다.
- 신규 11 tests 및 전체 202 tests(1 skip)가 통과했다. Artifact 3종은 반복 생성 SHA-256이 byte-identical했고, 실제 boundary-first benchmark의 공통 10 scenes에서 patch/component/topology/chart signature가 Stage 3 기준선과 동일했다.
- 결론: actual pipeline에서 Gaussian-native pairwise signal은 Stage 3 conflict를 해결하지 못한다. Stage 3 재개와 Stage 4 integration은 기각하며, 다음 후보는 별도 승인된 neighborhood/manifold-level connectivity 조사다. 상세 근거는 `docs/worklogs/65_gaussian_native_support_continuity_stage3r.md`, `artifacts/gaussian_support_continuity_stage3r_pairs.json`, `artifacts/gaussian_support_continuity_stage3r_summary.json`에 있다.

## 2026-07-23 NURBS Knot Vector Cache

- `TorchNURBSSurface`가 control-grid 크기, effective degree, dtype, device별 clamped knot vector를 lazy cache한다. Control point와 rational weight가 학습 중 바뀌어도 재사용하며 구조가 바뀌면 자동 무효화한다.
- 생성자, checkpoint, 학습 수학은 변경하지 않았다. 캐시 재사용·무효화·autograd 테스트와 전체 `204 passed, 1 skipped, 8 subtests passed` 회귀를 통과했다.
- Isolated 16-patch forward benchmark에서 forced-uncached 대비 CPU `1.039x`, CUDA `1.063x`였다. 안전한 소폭 개선이지만 전체 `surface_loss` 병목을 해결하는 수준은 아니므로 ragged patch batching이나 UV basis cache로 범위를 확장하지 않았다.
- 상세 결과와 남은 위험은 `docs/worklogs/66_nurbs_surface_loss_knot_cache_opportunity.md`에 기록했다.
## 2026-07-23 Boundary-Conditioned Occlusion 방향 전환 감사

- `docs/Urgent_Work/OSN_GS_Direction_Reset_Plan.md`를 기존 Phase 1–5 문서보다 상위의 methodology 재정의 문서로 검토했다. 새 방향은 global component recovery를 선행 성공조건에서 내리고, local visible NURBS boundary reconciliation → continuation domain → bounded multi-sided occluded candidate → constrained NURBS → uncertainty/validation 순서로 전환한다.
- 현재 코드는 NURBS first derivative, voxel/local boundary provenance, Step 5-A joint solve, camera와 renderer depth를 부분적으로 제공한다. 반면 ordered/oriented open boundary, second derivative, inner isocurve, patch-boundary graph, constructor-level visibility/free-space context, generic constrained fit은 누락돼 있다.
- Proxy Stage 0–3/3-R branch는 production 미적용 deprecated diagnostics로 보존한다. `build_surface_components`는 local patch bootstrap 용도로 유지하되 global component correctness는 새 prototype의 blocker가 아니다.
- Production 코드와 기본값은 변경하지 않았다. 감사 결과는 `docs/worklogs/74_direction_reset_interface_audit.md`, 승인 전용 구현 초안은 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`에 기록했다. 최초 승인 요청 범위는 Phase A–B data contract와 isolated artificial-boundary reconciliation prototype뿐이다.
## 2026-07-23 Boundary-Conditioned Phase A–B

- 승인된 Phase A–B 범위에서 `TorchNURBSSurface`의 read-only knot와 analytic rational second derivative를 추가하고, trim/chart boundary를 ordered/oriented patch record와 inner isocurve로 보존했다. Boundary-First/main NURBS export는 root/per-patch knot를 포함하며 Boundary-First artifact는 stable `patch_boundaries`와 per-patch `boundary_ids`를 포함한다.
- Generic shared-control solver는 explicit local patch graph의 full/partial, same/reversed edge correspondence를 첫 solve부터 하나의 unknown으로 묶는다. Existing production annulus solver는 변경하지 않았고 새 solver는 isolated reconciliation prototype에서만 사용한다.
- Reconciliation은 scale-normalized proximity와 overlap만 hard evidence로 사용하고 tangent/normal은 soft evidence로 기록한다. Coplanar seam과 약 90° orthogonal shared edge는 각각 post C0 max `2.46e-07`, `2.38e-07`로 `reconciled_internal`이 됐으며 disconnected gap `0.2`는 joint fit 없이 남았다.
- 전체 unittest `221 passed, 1 skipped`, pytest `220 passed, 1 skipped, 8 subtests passed`. Known `curved_annulus`/`mild_curved_sheet` blocker와 production component membership은 그대로다. Phase C 이후는 미승인 상태이며 Gate B에서 멈춘다. 상세 근거는 `docs/worklogs/75_boundary_conditioned_phase_ab.md`, active gate는 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md`를 참고한다.

## 2026-07-23 Boundary-Conditioned Phase C

- 사용자가 Gate B 검토 후 Phase C(Observation Evidence와 Free-Space Query)만 승인했다. 신규 `osn_gs/surface/torch_observation_evidence.py`는 카메라별 depth/coverage를 하나의 명시적 view-depth 계약으로 통일하고, world sample마다 5-state per-view 분류(`known_free_space`/`on_observed_surface`/`behind_first_observed_surface`/`unobserved`/`outside_valid_view`)와 별도의 5-state aggregate 분류(`known_free_space`/`occluded_candidate`/`unobserved`/`outside_valid_view`/`conflicting_evidence`)를 제공한다.
- 초안이 "behind_observed_surface" 단일 강한 명칭, depth-epsilon tie를 behind로 처리, "한 카메라라도 behind면 승리"하는 집계 규칙, CUDA invalid depth clamp 순서, backend 공통 `coverage_threshold`를 사용한 점을 사용자가 5가지 근거로 반려했고, 최종 구현은 모두 반영했다(`behind_first_observed_surface`로 개명, `on_observed_surface` 신규 상태, per-camera 5개 리스트를 항상 payload에 보존하는 `conflicting_evidence` 신규 aggregate 상태, mask-먼저-then-invert CUDA depth 복원, backend별 `coverage_kind`/`depth_kind`/`depth_is_approximate` 메타데이터).
- `osn_gs/render/torch_fallback.py`/`gaussian_rasterizer.py`의 render 반환 dict에 `alpha`/`valid_depth_mask` 키만 추가했다(기존 키 값·의미 불변, caller-safety grep으로 dict-key 접근만 확인). Empty-voxel query(`query_empty_voxel_support`)는 `"no_observed_support"` 외의 값을 구조적으로 반환할 수 없고 sample classification과 연결되지 않는다.
- `tests/test_observation_evidence.py` 8개 테스트 전부 통과(Gate C 핵심: 관측 표면 뒤 30-포인트 sweep에서 free-space false acceptance 0/30). 전체 pytest `230 passed, 1 skipped, 8 subtests passed`, 회귀 없음. Production(`torch_pipeline.py`/`torch_trainer.py`) 미변경. 상세 근거는 `docs/worklogs/77a_observation_evidence_phase_c.md`를 참고한다.
- **Gate C 1차 보완 (같은 날)**: 사용자가 두 가지를 추가 요구했다 — (1) multi-view aggregate에서 `on_surface_in`이 있는 sample이 `known_free_space`가 되지 않도록 수정(1차: `free_space_confirmed_by`와 `on_surface_in`의 공존을 `conflicting_evidence`로 처리), (2) `evidence_cache_key()`가 `ObservationEvidence` 전체를 받아 `near`/`far`/`depth_epsilon`과 각 view의 backend/depth convention까지 포함하도록 확장하고, `_topology_version`/`_camera_set_version`도 각각 scale/rotation/opacity와 `full_proj_transform`을 해시에 포함하도록 확장(global cache는 추가하지 않음). 상세 근거는 `docs/worklogs/78_observation_evidence_phase_c_gate_c_followup.md`.
- **Gate C 2차 보완 (같은 날)**: 사용자가 1차 보완도 아직 승인하지 않고 추가 요구했다 — (1) aggregate `STATUS_ON_OBSERVED_SURFACE` 신규 상태 추가(on_surface만 있으면 이 상태로, on_surface가 free/behind 중 하나 이상과 공존하면 `conflicting_evidence`로, 불변식을 docstring에 명시), (2) camera fingerprint에 명시적 `camera_index` identity 성분 추가(`TorchCamera.image_name`이 공유 기본값 `"camera"`라 이름만으로는 카메라를 구분할 수 없다는 점을 문서화), (3) `_tensor_digest()`가 raw byte만 해시하고 shape/dtype을 누락해 서로 다른 텐서가 이론상 충돌할 수 있던 실제 결함을 수정(label/shape/dtype을 해시 입력에 포함), (4) `evidence_cache_key()`가 렌더링 후에만 계산 가능한 post-build result fingerprint라는 계약을 docstring에 명시. 신규 테스트 5개 추가. 전체 pytest `238 passed, 1 skipped, 8 subtests passed`, 회귀 없음. 상세 근거는 `docs/worklogs/79_observation_evidence_phase_c_gate_c_round2.md`를 참고한다.
- **Gate C 최종 승인 (같은 날)**: 사용자가 Gate C를 최종 승인했다. Non-blocking note 2건(duplicate camera 순서 교환 시 fingerprint 동일 — 문제로 보지 않음; synthetic per-view payload 기반 순수 aggregation truth-table 테스트는 향후 추가 권장)을 남겼다. Production pipeline/trainer integration은 아직 미승인.

## 2026-07-23 Phase D — Parametric Continuation Domain 설계

- 사용자가 Phase D의 구현이 아니라 **설계**를 요청했다. `osn_gs/surface/torch_nurbs.py`(NURBS derivative/knot API)와 `osn_gs/surface/torch_annulus_chart.py`(Jacobian singular-value 진단, 과거 seam-placement 실험) + 과거 pre-reset 확장 설계 문서 2개를 먼저 감사한 뒤 설계했다.
- 핵심 감사 결과: knot vector는 `[0,1]`에 하드코딩된 clamped uniform B-spline이라 domain을 넘어 확장하는 헬퍼가 없고, `PatchBoundarySegment`는 실제 `TorchNURBSSurface` 참조를 갖지 않으며, `predict_torch_occlusion_curves`/`build_torch_surface`/`sample_torch_occluded_surface`는 자체 docstring이 "Stage 2 legacy"로 명시하고 호출부가 없는 pre-reset 코드였다.
- 설계 결론: continuation domain은 `TorchNURBSSurface`를 확장하는 대신, boundary의 analytic `S_u`/`S_v`/`S_uu`/`S_uv`/`S_vv`로부터 매번 닫힌형으로 평가되는 별도의 경량 `ContinuationDomain` 객체로 만든다(1차 Taylor를 canonical baseline, 2차는 curvature-aware uncertainty 전용). Outward 방향은 최소자승 기반 축-비의존 general formula로 구하고 `inner_uv`(Phase A 산출물)로 부호를 정한다. `torch_annulus_chart.py`의 Jacobian-SVD 진단을 surface-agnostic 헬퍼로 추출하는 순수 리팩터가 구현 착수 전 prerequisite로 식별됐다. Self-intersection/역침범 전체 검사는 마스터 플랜이 이미 Phase F 소유로 규정하므로 Phase D는 로컬 proxy만 제공한다. Boundary pairing은 Phase E 몫이며 Phase D는 `aabb_min`/`aabb_max` 등 최소 interface만 제공한다. Phase C evidence 결합은 `ContinuationDomain.world.reshape(-1,3)`이 `classify_world_samples`의 입력 형태와 이미 일치한다는 interface 문서화만 하고 호출하지 않는다.
- Phase D continuation domain은 구현·Gate D 승인 완료다. canonical 계약은 `docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md` §6, 구현 근거는 `docs/worklogs/81_phase_d_continuation_domain_implementation.md`를 따른다. Phase C evidence 실제 결합과 production integration은 여전히 별도 범위다.
- **설계 revision 2/3 (같은 날, worklog 80 §7/§8)**: 사용자가 두 차례 더 설계만 교정했다(코드 없음). Revision 2: outward 방향을 UV-space 최소자승에서 순수 world-space 공식(`N=normalize(Su×Sv)`, `C=normalize(N×T)`, inner_world로 부호 선택)으로 전면 교체, sampled grid를 continuous evaluate() 대신 canonical source of truth로 확정, 상태명을 `candidate/degenerate/rejected`에서 `valid/degenerate/rejected`로 변경. Revision 3: `boundary_length` 필드로 closed-loop arclength 계약 완성, 인접 duplicate/zero-length segment를 `ValueError`로 명시적 거부, `ContinuationDomainBuildError` 신규 예외로 pre-grid 실패와 사후 품질 문제(`state=degenerate/rejected`) 분리, `local_surface_scale`의 canonical 집계 공식(`L_boundary`/`L_inner`/`L_control` median, 최소 2개 필요) 확정, second-order diagnostic 명칭을 `curvature_growth_ratio`에서 `second_order_growth_ratio`/`second_order_displacement_at_extent`로 개명(intrinsic curvature 아님을 명시).
- **구현 완료 (같은 날, worklog 81)**: 사용자가 "D 구현 시작해"로 착수를 승인했다. Prerequisite로 `osn_gs/surface/torch_parametric_diagnostics.py`(`compute_parametric_jacobian_metrics`, `compute_orientation_consistency`)를 신설하고 `torch_annulus_chart.py`가 이를 호출하도록 리팩터(필드명/수치 불변, `tests/test_annulus_chart.py` 48개 회귀 없음). 신규 `osn_gs/surface/torch_continuation_domain.py`(`ContinuationDomain`, `ContinuationDomainBuildError`, `build_continuation_domain`, `interpolate_boundary_arclength`)를 설계 그대로 구현했다. 구현 중 설계 §2.3("모든 샘플이 degenerate하면 `ContinuationDomainBuildError`")을 최초 구현이 빠뜨린 간극을 테스트 작성 중 발견해 수정했다(설계 문서는 무변경). `tests/test_continuation_domain.py` 22개 전부 통과, 전체 pytest `260 passed, 1 skipped, 8 subtests passed`, 전체 unittest `261 tests, OK`, 회귀 없음. Production(`torch_pipeline.py`/`torch_trainer.py`) 미변경, Phase C evidence 미호출. 마스터 플랜의 승인 게이트 D(continuation strip 방향·크기·결정성 보고)를 완료했다. 상세 근거는 `docs/worklogs/81_phase_d_continuation_domain_implementation.md`를 참고한다. Phase C evidence 실제 결합, Phase E candidate 생성, Phase F NURBS fitting, production integration은 모두 별도 승인 전까지 시작하지 않는다.

## 2026-07-23 Anisotropy 격차 재평가

- 3k A/B의 핵심 격차는 global scale magnitude가 아니라 OSN-GS 최소축 contraction 부족이다. Anisotropy 중앙값/p90은 OSN-GS `3.3194/8.4219`, baseline `5.3921/19.2291`이며 최대축 중앙값은 비슷하지만 최소축은 `0.005367 vs 0.003339`이다.
- OSN-GS covariance 초기화는 1-NN distance를 사용하지만 Graphdeco `distCUDA2`는 최근접 3개 squared distance의 평균을 사용한다. 실제 DATASET 초기 scale 중앙값은 `0.021216 vs 0.038980`이고, 동일 ADC threshold에서 초기 Gaussian의 16.2857%가 서로 다른 clone/split 영역에 놓인다. 강한 root-cause 후보지만 재학습 ablation 전에는 확정하지 않는다.
- Worklog 73의 OSN-GS `split`은 child 수, baseline `split_candidates`는 parent 수여서 직접 비교할 수 없었다. parent 기준 split 수는 사실상 동일하며 clone 차이를 우선 조사한다.
- 다음 순서는 Graphdeco-compatible 3-NN covariance ablation, ADC gradient source/parent-unit 진단, 필요 시 camera-extent position-LR ablation이다. Production 코드는 변경하지 않았다. 상세 근거는 `docs/worklogs/76_anisotropy_gap_root_candidate_reassessment.md`에 기록했다.

## 2026-07-23 Anisotropy parity ablation 결과

- `graphdeco_knn`(3-NN mean covariance), camera-based position-LR, ADC survivor-gradient drop을 각각 3k 실제 DATASET A/B로 검증했다. 신규 ADC 로그는 clone/split parent와 child 단위, candidate anisotropy, screen-space/fallback gradient source를 분리한다. 세 실행 모두 screen-space gradient만 사용했고 fallback은 0회였다.
- 3-NN covariance는 held-out `PSNR/SSIM`을 `7.9819/0.1202 -> 8.0921/0.1279`로 소폭 개선했으나 anisotropy 중앙값/p90은 `3.3194/8.4219 -> 3.4543/8.6064`로 거의 변하지 않아 주원인으로 기각했다. 기본값은 기존 1-NN으로 유지하고 mode만 ablation으로 제공한다.
- camera-based position LR은 anisotropy 중앙값/p90을 `4.6834/16.4903`까지 올리고 최소축을 baseline에 가깝게 만들었지만 held-out은 `7.8460/0.1114`로 악화됐다. point-cloud scene LR 기본을 되돌리지 않는다.
- ADC survivor gradient drop도 기준과 사실상 같은 anisotropy·held-out 결과여서 주원인이 아니다. 다음 조사 범위는 position LR과 per-axis scaling gradient, position update, clone lineage의 결합 계측이다. 전체 회귀는 `233 passed, 1 skipped`. 상세 결과는 `docs/worklogs/77b_anisotropy_gap_parity_ablation_results.md`를 참고한다.

## 워크로그 번호 규칙

- 병렬 작업 때문에 같은 번호가 발생한 기록은 `NN-A` / `NN-B` suffix를 canonical 식별자로 사용한다. 충돌 목록과 신규 기록 규칙은 `docs/worklogs/README.md`에 정리했다.

## 2026-07-24 Held-out 평가 opacity reset 식별

- `--eval` run의 마지막 iteration이 opacity reset 조건과 겹치면 콘솔 경고와 `held_out_eval.json`의 `post_opacity_reset`으로 명시한다. 학습·평가 결과 자체는 변경하지 않으며, reset 직후 수치를 정상 checkpoint 품질로 해석하지 않게 하는 측정 안전장치다. 상세는 `docs/worklogs/84a_held_out_eval_opacity_reset_warning.md`.


## 2026-07-24 Phase F.1 Gate 보완 진행 상태

- Phase F.1 sampled safety gate는 central-bridge eligibility 정책과 sampled visible-surface 의미를 보완했으나, 필수 fixture 확장 중 발견된 provenance fixture 및 conflict-edge payload 결함을 수정 중이다. Gate F.1은 승인 대기이며 상세 진행 기록은 `docs/worklogs/85_phase_f1_gate_completion_followup.md`.
