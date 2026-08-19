---
name: project-coverage-first-subset-partition
description: "Worklog 105 replaced the selection-first surface pipeline with a coverage-first Gaussian Subset partition; stage 1 only, awaiting the user's visual review"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-19T09:00:19.319Z
---

Worklog 105 (2026-08-19) is the first implementation batch of a NEW top-level visible-surface construction architecture, replacing the Worklog 95-104 selection-first line (which let scene regions silently leave surface-construction responsibility whenever a support/structural predicate failed).

New order: full trained visible Gaussian scene -> per-Gaussian surface-orientation representation -> coverage-preserving, normal-coherent, spatially-connected Gaussian Subset partition -> (later batches) trust estimation -> latent surface -> one subset : one NURBS patch.

**Central contract: SUBSET OWNERSHIP != TRUSTABILITY.** No Gaussian loses subset ownership for low normal confidence, weak neighbour evidence, or a difficult future latent surface. Trust will later weight influence on latent-surface estimation, never revoke ownership.

New modules (both isolated from Worklog 95-104 — an AST test forbids importing latent-surface/chart/identifiability/NURBS/boundary/held-out modules):
- `osn_gs/surface/torch_gaussian_surface_orientation.py` — principal axes read exactly from the 3DGS parameterization (`Sigma = R diag(exp(scaling)^2) R^T` means `R`'s columns ARE the eigenvectors; no eigh on the production path). The only ambiguity is AXIS ORDER, resolved by descending eigenvalue: `normal=axis(lambda3)`, `tangent_u=axis(lambda1)`, `tangent_v=normal x tangent_u`. Sign is a reproducibility gauge only; comparisons always use unsigned `|dot(n_i,n_j)|`.
- `osn_gs/surface/torch_coverage_first_subset_partition.py` — normal compatibility evaluated ONLY on edges of a local kNN spatial adjacency graph (never global normal clustering), connected components via Shiloach-Vishkin hooking.

Replay: `python scripts/devtools/coverage_first_subset_partition_export.py --checkpoint output/extent_ab/val64/baseline_compatible/final --out output/osn_gs_coverage_first_subset_partition --device cuda --source-path DATASET` (~143 s on RTX 5080). Produces 4 full-scene views, each with its own `render.ppm` from the trainer's own `_preview_camera`.

**Status: NO architecture decision was made. The user must visually review GAUSSIAN_SUBSET_PARTITION before the next stage (subset-local trust estimation) is implemented.** See [[project-coverage-first-partition-measurements]] for the numbers, and [[project_latent_surface_visualization_coverage_completeness]] for the superseded line.
