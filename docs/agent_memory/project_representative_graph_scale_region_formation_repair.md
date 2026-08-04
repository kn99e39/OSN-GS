---
name: project_representative_graph_scale_region_formation_repair
description: "Worklog 33 (docs/worklogs/33_representative_graph_scale_invariance_and_region_formation_repair.md) — proved worklog 32's rejected graph-scale candidates WERE rigid-transform invariant; the failure was selection-perturbation contamination in end-to-end tests; shipped G1 (representative kNN spacing), real 3k/5k/10k region_count recovered 0->64-85, boundary_failure_stage advanced A->C (candidate/region formation solved, closed-loop boundary linking is now the sole remaining blocker)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3697c6bf-838e-4135-bfc1-38e17fb7cfc0
  modified: 2026-07-31T09:03:19.282Z
---

Follows [[project_contextual_reliability_manifold_affinity_scale_repair]] (worklog 32), which rejected 4 RepresentativeGraphScale candidates because they broke end-to-end rigid-rotation invariance tests. This round separated two previously-conflated questions via a "frozen representative replay": (1) is the graph-scale ESTIMATOR itself invariant, holding the representative SET fixed? (2) does representative SELECTION return a different subset under rotation (already documented, accepted, out-of-scope non-invariance of the axis-aligned voxel grid), and how much does that alone perturb topology?

**Answer: the estimator was never the problem.** With representatives frozen (no re-selection), G1 (each representative's own median distance to its k=8 nearest OTHER representatives, pure function of position, `torch.cdist`+`topk`) was **exactly** rigid-rotation/translation/uniform-scale invariant on real 3k/5k/10k checkpoints — zero relation mismatches. Separately measured: representative selection itself only has 23-26% stable-ID overlap under a moderate rotation on the small test fixtures that were failing — a severe, PRE-EXISTING perturbation source, unrelated to graph-scale correctness, that worklog 32's end-to-end tests were conflating with estimator invariance.

**Shipped**: G1 wired into `_construct_canonical_with_full_evidence` (both `candidate_scale` and `residual_scale` roles in `build_manifold_affinity_graph`/`_compute_pair_metrics`, which gained these two independent optional params, `footprint_overlap` still uses Gaussian's own `equivalent_tangent_scale` unconditionally). Real-checkpoint same_surface edge counts went from 11/10/6 (3k/5k/10k) to 2125/2151/1908 — 190x+. Production result: **region_count 0 → 75/85/64**, `boundary_failure_stage` advanced from `A_candidate_generation_failed` all the way to `C_component_admission_failed` (the last of 3 stages) on all three snapshots. Materialization still fails (0 closed boundary loops) — that's the new, sole remaining bottleneck, in `torch_directed_boundary_ordering.py`/boundary-linking territory, explicitly out of scope across three consecutive worklogs (30/31/32/33) by standing instruction.

**Test-contract change (sanctioned, not silent)**: the two end-to-end invariance tests that re-run full selection (`test_density_preserving_representative_selection.py`, `test_full_cloud_continuation_shell.py`) were EXPLICITLY relaxed from exact `region_count` equality to a topology-stability bound (both >0, within 5x of each other) — documented in the test docstrings as measuring selection-perturbation robustness ("Test B"), not estimator invariance. A NEW test file (`tests/test_representative_graph_scale.py`) formalizes the frozen-representative exact-invariance check ("Test A") that G1 passes.

**LocalEvidenceScale re-judged**: kept in production (option B, provisional) — its remaining flaws (3k regression, normal_consensus increase) don't block region formation anymore, since `_seed_core_components`'s core-edge eligibility only checks `intrinsic_class`, not the contextual-dependent `reliability_class`.

**Why:** User explicitly demanded separating "is the estimator invariant" from "does selection perturb topology" via a frozen-representative harness, rather than re-trying graph-scale candidates against the same conflated test. This single diagnostic move overturned worklog 32's core conclusion.

**How to apply:** Do NOT re-litigate G1's invariance — it's proven exactly invariant on a frozen representative set. The NEXT bottleneck is closed-loop boundary linking (`boundary_component_closed_count=0` despite real candidates/components now existing) — do not touch boundary linking policy without a new, separately-scoped round. Positive control: `box_face` now unifies into 1 region (was fragmented into 2), `cylinder` forms exactly 3 regions (side+2 caps, correct topology). Sphere's 8-region over-fragmentation is a separate, already-disclosed issue (worklog 125), untouched.

Full detail: `docs/worklogs/33_representative_graph_scale_invariance_and_region_formation_repair.md`.
