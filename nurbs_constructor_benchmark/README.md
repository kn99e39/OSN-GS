# Synthetic NURBS Constructor Benchmark

This isolated framework validates the **production** OSN-GS visible-NURBS constructor with deterministic synthetic Gaussian scenes. It does not copy the constructor: every run calls `TorchOSNGSPipeline.initialize()` from `osn_gs/core/torch_pipeline.py`.

## Run

From the repository root:

```bash
python -m nurbs_constructor_benchmark
```

This runs `plane`, `sine`, and `crease` on CPU and writes `nurbs_constructor_benchmark/results/report.json`.

Useful variants:

```bash
# Exercise one smooth curved scene with the production voxel bootstrap.
python -m nurbs_constructor_benchmark --scenes sine --points 1200

# Compare the direct single-chart path without voxel partitioning.
python -m nurbs_constructor_benchmark --scenes sine --disable-voxel

# Make a CI-style regression gate. The command exits non-zero on failure.
python -m nurbs_constructor_benchmark --max-fit-rms 0.05 --max-chart-rms 0.10
```

## What it tests

- `plane`: baseline fit, UV projection, and normal stability.
- `sine`: smooth-curvature fidelity of the LSQ fitting path.
- `crease`: normal-boundary / multi-patch behavior around two joined planes.

Each scene generates deterministic observed Gaussian centers and colors. The production pipeline constructs its normal `TorchGaussianModel`, including its internally initialized opacity, scale, rotation, UV, and patch metadata.

The report contains input-point foot-point RMS, sampled reconstructed-chart residual against the analytic source surface, sampled normal error, patch/control-point counts, and a finite-value check. `surface_chart_rms` is a chart residual (vertical/implicit analytic residual), not a symmetric Chamfer distance.

## Extending it

Add a scene oracle and generator in `scenes.py`, then add its name to `SCENE_NAMES`. Keep constructor changes in the production `osn_gs` modules; this benchmark will automatically evaluate the changed path.

