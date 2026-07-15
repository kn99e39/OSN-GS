# Synthetic NURBS Constructor Benchmark

This isolated framework validates the **production** OSN-GS visible-NURBS constructor with deterministic synthetic Gaussian scenes. It does not copy the constructor: every run calls `TorchOSNGSPipeline.initialize()` from `osn_gs/core/torch_pipeline.py`.

## Run

From the repository root:

```bash
python -m nurbs_constructor_benchmark
```

This runs `plane`, `sine`, `crease`, and `density_gradient` on CPU and writes `nurbs_constructor_benchmark/results/report.json`.

Useful variants:

```bash
# Exercise one smooth curved scene with the production voxel bootstrap.
python -m nurbs_constructor_benchmark --scenes sine --points 1200

# Compare the direct single-chart path without voxel partitioning.
python -m nurbs_constructor_benchmark --scenes sine --disable-voxel

# See density-adaptive voxel subdivision actually help a non-uniform scene.
python -m nurbs_constructor_benchmark --scenes density_gradient --adaptive-voxel

# Make a CI-style regression gate. The command exits non-zero on failure.
python -m nurbs_constructor_benchmark --max-fit-rms 0.05 --max-chart-rms 0.10
```

## What it tests

- `plane`: baseline fit, UV projection, and normal stability.
- `sine`: smooth-curvature fidelity of the LSQ fitting path.
- `crease`: normal-boundary / multi-patch behavior around two joined planes.
- `density_gradient`: same smooth sheet as `sine`, but Gaussian centers cluster densely near the origin with a sparse background instead of sampling uniformly. Every other scene is uniform, so this is the only one that actually stresses density-adaptive voxel subdivision (`--adaptive-voxel`); with a fixed-resolution grid the dense cluster is under-resolved and the sparse periphery is over-resolved.

Each scene generates deterministic observed Gaussian centers and colors. The production pipeline constructs its normal `TorchGaussianModel`, including its internally initialized opacity, scale, rotation, UV, and patch metadata.

The report contains input-point foot-point RMS, sampled reconstructed-chart residual against the analytic source surface, sampled normal error, patch/control-point counts, and a finite-value check. `surface_chart_rms` is a chart residual (vertical/implicit analytic residual), not a symmetric Chamfer distance.

## Ground-truth NURBS metrics (three construction concerns)

Because each scene knows its true surface `z = f(x, y)` and true patch topology, the generated NURBS is scored against ground truth on three **independent** concerns (in every result's `ground_truth` block; see `metrics.py`). The single chart residual above conflates these, so they are separated:

1. **Surface Fitting Accuracy** — geometric closeness where both surfaces exist. `accuracy_rms` (generated → true surface, precision) and `completeness_rms` (observed true surface → generated, recall), combined as `chamfer_rms`.
2. **Surface Support** — whether the surface exists in the right places. `support_coverage_uncovered_fraction` (holes / under-support over the observed region) and `support_extrapolation_fraction` (surface drawn where there is no input data / over-support). Patches carry a UV trim mask (`surface_trim_resolution`/`--trim-resolution`, `--trim-dilation`), so the support metric measures the trimmed surface the renderer would actually draw. Thresholds scale with the median input point spacing (so the highly non-uniform `density_gradient` reports a strict extrapolation figure — an artifact of the global threshold, not trimming).
3. **Patch Topology** — whether the patch count and boundaries match ground truth. `topology_gen_patch_count` vs `topology_gt_patch_count` and `topology_label_ari` (Adjusted Rand Index of per-Gaussian generated `cluster_ids` vs GT patch labels; permutation-invariant, penalizes over-segmentation). `crease` has 2 GT patches; the other scenes have 1.

These make different failure modes visible that the chart RMS hides — e.g. `crease` fits each sliver with low residual yet scores a low topology ARI (over-segmentation), and `density_gradient` has a high extrapolation fraction (surface drawn far past the clustered data).

Optional regression gates (exit non-zero on breach), one per concern: `--max-chamfer-rms`, `--max-extrapolation`, `--min-topology-ari` (alongside `--max-fit-rms` / `--max-chart-rms`).

## Renderer output

Every run (unless `--skip-renderer-export`) also writes, per scene:

```text
results/NURBS_output/<scene>/point_cloud.ply
results/NURBS_output/<scene>/nurbs_surface.json
results/NURBS_output/<scene>/nurbs_surface_gt.json
```

`point_cloud.ply` + `nurbs_surface.json` are the same pair a real training run writes to its `final` output directory (see `osn_gs/core/torch_trainer.py:save_outputs`), built from the exact same `nurbs_intermediate_payload()` helper so the two never drift apart. `nurbs_surface_gt.json` is the **ground-truth NURBS** in the identical renderer format (`ground_truth.py`) — a degree-1 surface lying on the true `z = f(x, y)`, with the correct topology (two patches for `crease`). Load both at `3DGS_Renderer`/`WebRenderer` per `RENDERER_INPUT_FORMAT.md` to overlay the reconstructed surface on ground truth. Each file's top-level `control_grid` is the primary patch; the full `patches[]` array carries every patch for multi-patch scenes.

## Extending it

Add a scene in `scenes.py`: provide its analytic height `surface_fn`, pointwise `oracle` (residual + normal), and ground-truth topology (`gt_patch_count`, `gt_patch_label`), then add its name to `SCENE_NAMES`. The GT metrics and GT-NURBS export pick these up automatically. Keep constructor changes in the production `osn_gs` modules; this benchmark will automatically evaluate the changed path.

