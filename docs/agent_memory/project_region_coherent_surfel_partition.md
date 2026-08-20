---
name: project-region-coherent-surfel-partition
description: "Worklog 97 fixed Worklog 96's single-linkage chaining pathology by adding a region-level orientation-coherence merge gate; giant subset dropped 74.70%->21.20%; no architecture decision yet"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-20T05:44:26.834Z
---

Worklog 97 (2026-08-20, `arch/2dgs-coverage-first-surface` branch): fixed the dominant remaining problem Worklog 96 measured — plain connected-component union over locally-accepted normal-compatible edges allowed **single-linkage chaining** (a flat patio floor and a differently-oriented hedge/background fused into one 894,378-surfel subset, 74.70% of the scene, because every step along the chain was locally plausible).

Fix: keep everything about Worklog 96 (intrinsic 2DGS normals, the same kNN spatial-adjacency + local-spacing + `|dot|>=0.85` candidate graph) and change ONLY the union rule. New module `osn_gs/surface/torch_region_coherent_surfel_partition.py`:

- Each growing region keeps a sign-invariant orientation scatter `M_R = sum_i w_i n_i n_i^T` (uniform weight 1.0 — explicitly NOT a future trust score).
- A merge is accepted only if the union's concentration `C_R = lambda_max(M_R)/trace(M_R)` stays `>= C_floor`.
- **No new independent parameter**: `C_floor = (1 + a) / 2` is derived algebraically from the EXISTING local threshold `a = normal_compatibility_min_alignment = 0.85`, giving `C_floor = 0.925` (the exact concentration two normals at the alignment floor produce together). Verified analytically and against `torch.linalg.eigvalsh`.
- Deterministic sequential Kruskal-style union-find (edges processed by descending normal alignment, ties by ascending index) — inherently sequential because per-root state evolves; measured at 4,015,325 edges / 1.2M surfels in ~76s pure Python (fast enough, no vectorized approximation needed).
- Coverage preserved via a 3-role system: STRUCTURAL_CORE_MEMBER (real merged region members), OWNERSHIP_PROPAGATED_MEMBER (a solitary surfel attached to exactly ONE neighbouring structural region via its best local edge — never updates that region's M_R, never bridges two regions, fully vectorized one-hop assignment), ISOLATED_FALLBACK_MEMBER (no reachable structural region — own singleton subset, never dropped).

**Real measurement on the Worklog 96 checkpoint** (`output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`, same local candidate graph for both arms):
- Largest subset fraction: **74.70% -> 21.20%**.
- Subset count: 58,646 -> 104,548. Region-coherence rejected merges: 553,357.
- The former 894,378-surfel giant subset decomposed into 31,564 final subsets (largest descendant = 253,853 = 28.4% of the original giant = exactly the new largest subset).
- Worklog 96's 40,410 singleton surfels ALL remain isolated fallback in the new partition too (expected: same local graph means they never had an accepted edge to propagate through).
- Coverage identity holds: 1,197,331 assigned, 0 unassigned/multiply-owned.
- Visual check: the remaining giant subset (21.2%) now visually looks like one genuine flat surface (the ground), not a chained artifact; the hedge/background that used to be fused with it is now fragmented into many small coherent subsets.

Review export: `output/osn_gs_region_coherent_surfel_partition/` (6 views: ORIGINAL_SCENE, WL96_PAIRWISE_CC_PARTITION, REGION_COHERENT_PARTITION, REGION_ORIENTATION_DISPERSION_VIEW, OWNERSHIP_ROLE_VIEW, ANTI_CHAINING_BOUNDARY_VIEW).

15 new focused tests (drift-chain splits, bounded-curvature sheet stays one region, sign invariance, determinism, anti-chaining cannot be bypassed via ownership-only surfels, etc.), all pass.

Also corrected a factual chronology error in Worklog 96 (§5-A implied the CUDA-extension build was reconfirmed by §8's full regression, but that regression predated the build — added a dated correction rather than silently rewriting).

**No architecture decision made.** Result is a large, real improvement but not full resolution (21.2% giant subset remains, real though). See [[project_2dgs_coverage_first_surfel_partition]] for the Worklog 96 baseline this extends.
