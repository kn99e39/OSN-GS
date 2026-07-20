# 27. Local Support-Tau Analysis

Date: 2026-07-15

## Work

Added local-density support analysis to the synthetic constructor benchmark.

- metrics.py now records a local support tau for each generated sample using
  the nearest input Gaussian local nearest-neighbour spacing.
- The existing global extrapolation metric remains unchanged for comparison.
- runner.py prints both extrapolation_global and extrapolation_local.

## Result

CPU benchmark command:

    .venv\Scripts\python.exe -m nurbs_constructor_benchmark --scenes density_gradient --points 600 --seed 0 --output C:\tmp\osn_gs_support_tau_check --skip-renderer-export

Result:

- extrapolation_global: 0.659
- extrapolation_local: 0.094
- uncovered: 0.234
- chamfer_rms: 0.027456
- topology ARI: 1.000

## Evaluation

The large global extrapolation value is primarily a global median-spacing
artifact caused by the dense cluster. The local comparison confirms that UV
trimming is not the dominant failure in this scene.

## Remaining work

- Decide the default/gating support tau only after a sensitivity sweep.
- Add the requested occupancy artifact and non-rectangular support scenes.
- Keep global and local values together during the comparison period.
