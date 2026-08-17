---
name: project_surface_topology_temporal_lineage_attribution
description: worklog 91 -- temporal + ADC-lineage split of worklog 90's MULTILAYER_OR_VOLUMETRIC; Decision A
metadata:
  node_type: memory
  type: project
  modified: 2026-08-17
---

Worklog 91 is a temporal + lineage attribution batch, not a new boundary method. Worklog 89's boundary constructor, Worklog 82 relation thresholds, NURBS fitting, and visible Gaussian training are all fixed and unmodified, as is Worklog 90's own covariance-footprint attribution logic (reused, not reimplemented).

New read-only module: `osn_gs/surface/torch_chart_unit_surface_topology_temporal_lineage.py`. For each Worklog 90 `MULTILAYER_OR_VOLUMETRIC` unit it (1) fits a plane to member CENTERS ONLY via SVD (independent of any Gaussian's own covariance eigenvector) and gap-clusters signed depth offsets into layers -- `CENTER_GEOMETRY_LAYERING`; (2) splits Worklog 90's covariance `layer_conflict` node mask by whether centers are single-sheet -- `COVARIANCE_ONLY_AMBIGUITY`. New replay script `scripts/devtools/chart_unit_surface_topology_temporal_lineage_replay.py` adds ADC lineage (stable-Gaussian-ID set diff across checkpoints, since clone/split always allocate fresh IDs and never reuse them, joined against the training log's cumulative `OSN-GS ADC: iteration=N ...` counters) and camera-based visibility/depth-ordering (real train-camera renders via `OSNGaussianRasterizer`, view-space depth via `camera.world_view_transform`).

Real `baseline_compatible` replay across all 5 available checkpoints (600, 2900, 3000, 3100, final; `output/extent_ab/val64/baseline_compatible`): iteration 600 has 0 failed topology units (pre-densification). At 2900/3000/3100/final, true-center-multilayer fraction of `MULTILAYER_OR_VOLUMETRIC` evidence is 93.49%/92.52%/93.91%/95.90% -- stable across a 4x swing in total failed evidence (3073->9889->11368->6497 after the 3100->final screen-prune storm, 405,767 stable IDs died). Covariance-only ambiguity stays a minor 4.10%-7.48% throughout. Lineage: the dominant competing-layer unit (region 3/unit 0, 1386 evidence at `final`) got 239 of its members in the 600->2900 interval alone (vs. 22/49/6 in later intervals) -- most of the structure forms during the initial densification wave and survives subsequent pruning. Visibility: the two dominant layers (1378 vs 3 members) are simultaneously visible (`radii>0`) in 14/161 train cameras, with mean view-space depth separation 0.277 (range 0.211-0.387) -- a real competing-depth structure, not view-conditioned occlusion.

Decision A: true center-distribution multilayer dominates and is stable across every available checkpoint; Decision B (covariance-only representation failure) is rejected. Do NOT rewrite boundary topology. Next investigation target is ADC/densification/pruning behavior -- why this competing-layer structure forms and persists -- not boundary reconstruction. See [[project_surface_topology_root_cause_attribution]] for the worklog 90 baseline this extends.
