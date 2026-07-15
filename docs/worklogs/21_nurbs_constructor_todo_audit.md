# 21. NURBS Constructor TODO Audit

Date: 2026-07-15

## Evidence reviewed

Reviewed all prior construction worklogs (01 through 20) and current benchmark code:
nurbs_constructor_benchmark/metrics.py, diagnostics.py, scenes.py, and runner.py.

## Confirmed and removed from TODO

- Synthetic rectangular plane, sine, and density-gradient scenes.
- Input Gaussian to fitted-surface RMS.
- Ground-truth accuracy, completeness, and bidirectional Chamfer RMS.
- Jacobian area-degeneracy and fold-over diagnostics.
- Expected/generated patch-count and label-ARI metrics.

These are implemented in the production constructor benchmark and have recorded
worklog evidence. The metric ownership is separated into fitting accuracy,
support, and topology.

## Still open

- Elongated plane and mild-curved-sheet coverage.
- Jacobian condition number and seed/config stability sweeps.
- Local-density-adaptive support threshold, occupancy artifact, non-rectangular
  support scenes/metrics, and support-mask lifecycle refresh.
- Chartability, topology boundary scoring, and multi-patch improvements.

No NURBS fitting or training code changed in this audit.
