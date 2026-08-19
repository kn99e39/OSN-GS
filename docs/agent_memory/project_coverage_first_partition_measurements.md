---
name: project-coverage-first-partition-measurements
description: "Worklog 105 measured partition numbers on baseline_compatible/final, plus the open one-subset-one-NURBS scale question"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-19T09:00:33.351Z
---

Worklog 105 replay on `output/extent_ab/val64/baseline_compatible/final` (the same checkpoint as Worklogs 94-104; the HANDOFF_2026-08-19 §0 question about whether it is the intended scene is still unresolved).

Coverage: input = all 1,685,549 visible Gaussians (uncertain = 0). assigned 1,685,549 / unassigned 0 / multiply-owned 0 / spatially disconnected subsets 0 / `coverage_identity_holds = true`. Not restricted to the old 7 regions (7,774 evidence) and requiring no latent support.

Partition: 166,585 subsets. min 1, median 1, mean 10.12, p95 10, **max 559,541 (33.2% of the scene)**. Singletons 107,947 (64.8% of subsets, 6.4% of Gaussians); size <= 8 is 156,687 subsets (94.1%) holding 265,533 Gaussians (15.8%). Edges: candidate 8,655,268 / spatial 7,344,950 / **normal-compatibility cut 3,389,357 (46.1% of spatial)** / accepted 3,955,593. Fallback ownership 107,947 (6.40%): 107,215 normal-incompatible neighbourhood + 732 no spatial neighbour.

Parameters (all in `CoverageFirstPartitionConfig`, all reported, none tuned against the render): `neighbor_count=8`, `spatial_connect_spacing_multiplier=2.0`, `normal_compatibility_min_alignment=0.85` (31.79°). The first and third are reused verbatim from the pre-existing `ManifoldAffinityConfig`.

**Open question deliberately left for a later batch (Worklog 105 §9):** the largest subset owns 33.2% of the scene and the top 6 each exceed 27,000 Gaussians, so "must an arbitrarily huge normal-connected component stay one subset?" is a real problem on this scene. Complexity-driven subset refinement was NOT implemented.

Also confirmed while producing ORIGINAL_SCENE: the trained scene DOES contain the central table and planter. The earlier "central objects missing" complaint was a property of the region-owned 7,774-evidence SUBSELECTION (Worklog 103), not of the checkpoint. See [[project-coverage-first-subset-partition]].
