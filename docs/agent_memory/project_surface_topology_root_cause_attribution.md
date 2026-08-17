---
name: project_surface_topology_root_cause_attribution
description: worklog 90 -- read-only covariance-footprint attribution for Worklog 89 topology failures; Decision C
metadata:
  node_type: memory
  type: project
  modified: 2026-08-10
---

Worklog 90 is a root-cause attribution batch, not a new boundary method. Worklog 89's full-region local-frame face -> membership-incidence algorithm, Worklogs 82-84, region ownership, visible Gaussian training, PCA-UV, and 6x6 NURBS are fixed and unmodified.

New read-only module: `osn_gs/surface/torch_chart_unit_surface_topology_attribution.py`. For each Worklog 89 `chart_unit_cut_non_manifold` coherent unit it compares the fixed Worklog 82 bounded center-kNN relation with native covariance 1-sigma tangent footprints and normal thickness. It never adds an edge, relation, face, loop, or closure. Primary mutually exclusive causes are CENTER_UNDERSAMPLING, RELATION_FALSE_NEGATIVE, TRUE_SUPPORT_GAP, MULTILAYER_OR_VOLUMETRIC, and GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH.

Real baseline_compatible@2900 replay (7 regions, 3526 total evidence) covers all 167 Worklog 89 failed topology units / 3073 evidence. Primary evidence attribution: multilayer/volumetric 2810 (91.44%), center undersampling 141 (4.59%), true gap 95 (3.09%), graph-to-surface mismatch 27 (0.88%), relation false negative 0. Evidence-weighted metrics: center spacing/equivalent tangent scale 1.247; compatible footprint-overlap coverage 49.20%; missing same-surface edges despite compatible footprints 11.60%; relation FN node fraction 6.87%; layer ambiguity 82.88%; plausible continuous-footprint local complex only 168/3073 (5.47%). Raw compatible pairs: 1494 accepted, 177 missing center graph, 181 ambiguous relation rejection, 0 typed veto, 6789 competing layer conflicts.

Decision C: do NOT replace production topology with a covariance-footprint graph and do NOT tune/redesign Worklog 82 relation thresholds. Multilayer/depth-normal ambiguity, with a smaller true-support-gap contribution, dominates even when footprints are considered. Any next upstream investigation must measure depth/visibility ordering, ADC clone/split/prune covariance and spacing distributions, normal-depth layer multiplicity against confidence/ownership, and photometric-gradient/opacity/radius/birth lineage at conflicting layers. Do not reopen boundary topology heuristics.
