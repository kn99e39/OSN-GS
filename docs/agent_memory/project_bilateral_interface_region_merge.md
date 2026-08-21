---
name: project-bilateral-interface-region-merge
description: "Worklog 100 replaced WL99's min()-combined edge residual with region-conditioned + bilateral (AND, not min) per-edge certificate; real-scene largest subset dropped 53.86%->22.91%, and all 5 WL99 patio-to-hedge lineage merges confirmed rejected under the new rule; no architecture decision"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-21T03:07:33.375Z
---

Worklog 100 (2026-08-21, `arch/2dgs-coverage-first-surface`): the user hypothesized [[project_interface_coherent_region_merge]] (WL99)'s residual percolation (largest subset 53.86%) came from reusing WL98's `min(r_i->j, r_j->i)` (one-side-permissive) residual in a REGION-MERGE context, a much stronger claim than WL98's original per-edge-cut use. Kept everything else from WL99 fixed (positional-gated WL97 init, candidate graph, support/extent floors, residual threshold formula, majority=0.5) and replaced only the per-edge certificate:

1. **Region-conditioned local shape operators** (`osn_gs/surface/torch_bilateral_interface_region_merge.py::_fit_region_conditioned_shape_operators`): each boundary surfel's `S_i` is refit EVERY ROUND using only its kNN neighbors that currently belong to its OWN region (cross-region neighbors get zero weight, not down-weighted). Fewer than 2 same-region neighbors -> UNSUPPORTED (structural minimum for a 2x2 fit, not tunable), and unsupported directions never count as smooth.
2. **Bilateral certificate**: `r_A->B` (A's own model predicting toward B) and `r_B->A` (B's own model predicting toward A) are combined via AND, not min -- both must independently pass the threshold for an edge to be `bilateral_smooth`.

**Real bug found and fixed**: originally recomputed the residual threshold (median+3*MAD) every round from only that round's cross-region edges, mirroring WL99's per-round style. A synthetic 90-degree zigzag crease fixture exposed a real degenerate case: a perfectly uniform crease has ZERO residual variance, so median+3*MAD collapses to the residual value itself and NOTHING is ever classified as an outlier no matter how large in absolute terms (100% of a real crease was misclassified as bilaterally smooth). Fixed by computing the threshold ONCE (like WL98/99's own precedent), from the INITIAL round's region-conditioned residuals over ALL spatial edges (not just cross-region ones, so same-region near-zero residuals give the median+MAD statistic real contrast to detect against), then reusing it for every round.

**Zero new free parameters** -- every threshold reused verbatim from WL99 (`interface_smooth_majority_fraction=0.5` untouched).

**Synthetic fixtures (14 new tests): all pass**, including a hand-constructed deterministic one-sided-interface case (6 nodes, exact arithmetic: r_A->B=1.2 fails, r_B->A=0.0 passes -- confirmed rejected) and boundary-contamination exclusion. One disclosed limitation found: very small/poorly-populated regions (e.g. a 16-surfel WL97 fragment) can have residuals at a numerically distinct but still-tiny scale from large regions' near-exact fits, and if the two populations are near evenly split, median+MAD (a ~50%-contamination-breakdown-prone statistic) can misclassify -- worked around via test fixture geometry choice (n_theta=90 instead of 60), not by touching any threshold; real-scene data is expected to have much larger, more natural residual/noise scales.

**Real-scene measurement (same 1,190,469-surfel checkpoint as WL97/98/99): clearly positive.** A=WL99: initial 114,420 -> final 108,848, largest **53.86%**, 5,572 merges, 6,051/1,855,041 interfaces accepted. B=this module: initial 114,420 -> final 112,768, largest **22.91%** (close to the init's own ~20.62%), only 1,652 merges, 1,742/1,763,096 interfaces accepted.

**Lineage trace (directive section 5, `scripts/devtools/worklog99_lineage_trace.py`)**: found the exact 5-merge WL99 provenance chain connecting a patio-side seed to a hedge-side seed (via BFS on WL99's own merge graph), replayed WL99's FULL 5,572-merge sequence (not just the chain -- an early bug replaying only the chain badly undercounted a root region's true membership, giving false "no surviving edges" for 4/5 steps) to reconstruct the exact membership state at each merge, then re-evaluated with the new bilateral evidence. **All 5 lineage merges are rejected** under the new certificate; 2 of 5 are explicitly one-sided (0.242 vs 0.788, 0.192 vs 0.615 smooth fraction each direction). Confirmed the two seeds land in different final subsets in the real B run (distinct `f_dc` colors).

Also fixed a real prose typo in WL99's own doc (`assigned == unassigned == 0` -> `assigned == total_surfels`; the code was always correct).

Review export: `output/osn_gs_bilateral_interface_region_merge/` (5 views + preview_png/). 14 new focused tests, full regression 1158 passed 1 skipped (+14 from WL99's 1144). **No architecture decision made** -- result is clearly positive but table-curve-recovery attribution (merge mechanism vs. WL97-positional-gate init) remains unresolved, same open question as WL99. See [[project_interface_coherent_region_merge]] and [[project_discontinuity_first_surfel_partition]] for the preceding stages.
