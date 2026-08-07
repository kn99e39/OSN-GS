---
name: project_single_chart_parameterization_validity
description: "worklog 69 - corrected worklog 67/68's patch-count arithmetic error (22 not 21); all 22 patches fail single-chart UV validity, but the dominant cause is boundary-loop size mismatch, not multi-sheet folding; 0 partitions applied (fail-closed)"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T05:28:20.282Z
---

worklog 69: first corrected an arithmetic reporting error in worklog 67/68 -- both said "21 total patches" when the actual per-condition sum (5+11+4+2) is 22. Fixed in-place in worklog 67/68 text, Master doc, and memory (correct fractions: worklog 67 "21/22 extrapolative", worklog 68 "20/22 (90.9%) overfitting").

Then tested whether [[project_dense_evidence_fit_capacity_calibration]]'s (worklog 68) overfitting finding is explained by regions not actually being valid single NURBS charts. New `osn_gs/surface/torch_single_chart_uv_validity.py`: UV duplicate/near-collision, 3D-kNN vs UV-kNN neighborhood preservation (Jaccard), accepted-edge UV crossing (reuses existing `_segments_intersect`), local fold via 3D-normal sign agreement between UV-adjacent Delaunay triangles (NOT via UV winding, which is trivially always consistent), boundary-polygon containment of interior evidence, UV area distortion, and parallel-sheet suspicion (gap in normal-axis projection).

Result: all 22/22 patches fail (`partition_materialization_required`, 0 `uv_valid`). But `neighborhood_preservation` stayed healthy for 19/22 -- arguing AGAINST true multi-sheet/fold as the dominant cause. The overwhelming, 22/22 signal was `interior_outside_boundary` (usually 90-100% of interior evidence outside the boundary polygon) -- explained structurally: boundary loops are only 3-4 representative points (a near-degenerate tiny polygon) while region-owned full evidence spans hundreds-to-thousands of points over the region's real spatial extent, so a tiny polygon can never contain it regardless of true validity. This is a NEW, distinct structural finding from worklog 67/68's own conclusions.

Density subsampling (25%/50%/100%) found raw and dense-NN-normalized error move together proportionally -- reconfirms worklog 68's `metric_density_dependent`=0 finding.

Partition repair: attempted only where parallel-sheet was suspected AND the region's own existing accepted-edge graph already showed <=5% cross-cluster edges (i.e. topology nearly already separated them). 0/22 qualified -- every patch's existing topology has >5% cross-edges, so no safe partition boundary was derivable without inventing one (forbidden). All 22 stayed fail-closed, exactly as instructed when evidence is insufficient.

Found+fixed a real bug in the new module during validation: `parallel_sheet_suspicion`'s gap detector had no minimum-cluster-size requirement, so a single routine ADC-training outlier Gaussian (this whole project's sessions are full of these) created a spurious huge gap ratio (39-2088x). Fixed by requiring both sides of a candidate split to hold >=10% of points; ratios dropped to a more plausible 7-25x (still above the 3.0 threshold, so still flagged -- reported honestly, not tuned away).

`surface_self_intersection` explicitly reported as `"not_checked"` in every record, per instruction.

13 new focused tests passed. Full pytest NOT run per task instruction (same as worklog 66-68).

**Why:** answers whether worklog 68's overfitting pattern is a topology/parameterization problem (fixable via partitioning) vs a fitting-resolution problem worklog 68 already ruled out raising -- it's neither cleanly; the real structural issue found is boundary-loop-vs-evidence-extent scale mismatch, a new open problem.

**How to apply:** when reporting patch counts from worklog 67/68, use 22 (not 21). If future work addresses the boundary-loop-too-small finding, do it as boundary geometry becoming evidence-aware, NOT by changing region formation/topology (both were explicitly protected this round and worklog 67).
