---
name: project_latent_midsurface_recoverability_attribution
description: worklog 93 -- thick/curved Worklog-92 evidence has a recoverable latent 2D midsurface; Decision A LATENT_SURFACE_RECOVERABLE
metadata:
  node_type: memory
  type: project
  modified: 2026-08-18
---

Worklog 93 is a read-only geometry-representation root-cause test, not a new boundary method or a position-PCA-normal replacement. Worklog 89's boundary constructor, Worklog 82 relation semantics, ADC, visible Gaussian training, and NURBS fitting are all fixed and unmodified. No Gaussian xyz is mutated; every projection is diagnostic-only and discarded (verified by a test asserting input-tensor identity).

New read-only module `osn_gs/surface/torch_chart_unit_latent_midsurface_attribution.py`, applied to Worklog 92's LOCALLY_THICK_UNIMODAL_SHEET / LOCALLY_SINGLE_CURVED_SHEET evidence (the ~87-97% majority of Worklog 90's MULTILAYER_OR_VOLUMETRIC evidence). Reuses Worklog 92's own local kNN (k=8) diagnostic plane fit, adds a local quadratic height-field fit (curvature via Hessian trace), and compares raw-center vs. diagnostic-thickness-collapsed local manifold topology using a position-only same-surface adjacency test (Worklog 82's 0.85/0.35 thresholds reused unchanged on a positional analog, never on covariance). Never reads any Gaussian's covariance normal/tangent/scale (AST-verified).

Real replay (baseline_compatible checkpoints 2900/3000/3100/final, 2609-10066 evidence per checkpoint): manifold-improved fraction 99.8-100%, curvature-preserved fraction 86.6-91.9% (not global planarization), valid local face incidence roughly doubles after diagnostic collapse (30.5-36.3% raw -> 63.0-72.7% diagnostic), open/non-manifold fraction nearly halves (63.9-69.5% -> 27.3-37.0%), observed-support-band fidelity 86.0-91.5% (recovered surface stays within observed evidence, not invented). Consistent across all 4 checkpoints.

Decision A: LATENT_SURFACE_RECOVERABLE. Raw Gaussian centers are the wrong geometry representation for visible surface topology. The next architecture target is an explicit latent-surface evidence representation BEFORE boundary extraction -- this batch only diagnosed recoverability, it did not build that representation. See [[project_local_center_geometry_attribution]] for the worklog 92 classification this extends, and [[project_surface_topology_root_cause_attribution]] for the worklog 90 baseline both build on.
