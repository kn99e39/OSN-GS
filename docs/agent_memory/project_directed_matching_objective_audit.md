---
name: directed-matching-objective-audit
description: "Worklog 52 — audited whether the directed one-in/one-out Hungarian matching wrongly rejects two specific edges (worklog 51's finding); found no solver defect, just genuine topological infeasibility or a correctly-preferred stronger non-cyclic assignment"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-04T01:50:28.692Z
---

Worklog 52 (docs/worklogs/52_directed_matching_objective_audit.md) audited [[project_raw_full_cloud_boundary_evidence_audit]] (worklog 51)'s claim that region 52's `666904↔1086120` and region 56's `1110285↔278207` edges (3k checkpoint, cap 2048) are rejected by Hungarian matching competition despite having both raw and representative evidence.

**Correction to worklog 51:** both edges are actually IN the final matching (region 52: `1086120→666904` score 3.1172, the region's highest; region 56: `278207→1110285` score 2.0504). Worklog 51's "first_gate" trace only looked at each endpoint's single nearest candidate, not full one-in/one-out feasibility — it misattributed the fragmentation to these two edges specifically.

**Method:** new `scripts/devtools/trace_directed_matching_objective_audit.py` reuses production `_compatible_directed_edges`/`_max_weight_one_in_one_out_matching` unchanged, dumps every edge + score component, AND does an exhaustive permutation search (regions are 4-6 candidates, cheap and exact) for the best-scoring feasible cycle (length≥3) to compare against the matching's actual choice.

**Findings:**
- Region 52 (1020950, 1085315, 1086120, 666904): only 4 compatible edges total; 1020950 and 666904 have ZERO outgoing edges each — a cycle is combinatorially impossible regardless of scoring. `best_feasible_cycle_score = None`.
- Region 56 (1039800, 1110285, 278207, 819956): a feasible 3-cycle exists (819956→278207→1110285→819956, score 6.4619) but the matching's actual chosen assignment scores 7.9408 — a 1.4789 (23%) margin, not a tie. The two strong edges (1039800↔819956 mutual pair, scores 3.18/2.71) that the matching keeps are genuinely stronger evidence than the weaker edge (819956→278207, score 1.99) a cycle would require.
- Cross-checked 4 fragmented 10k regions (104, 6, 11, 52): 3 have zero feasible cycles, 1 (region 52) has a feasible cycle scoring 15% below the matching's actual choice. Same pattern, not scene-specific.
- Score formula (`forward/distance + tan_align + normal_align + outward_align - lateral/max_lateral`) audited for unit/sign bugs: all terms are dimensionless [0,1] ratios, penalty term correctly subtracted — no defect found. No genuine ties exist in any checked case, so the task's permitted "lexicographic tie-break toward closed cycles on equal evidence" doesn't even apply (there's no tie to break).

**Conclusion:** no Hungarian solver/objective defect — rejected the hypothesis. Production code unchanged; full pytest 720 passed unchanged (trivially, nothing changed).

**How to apply:** this is the fourth consecutive worklog (49/50/51/52... actually 48/49/50/51/52, five) to audit a different stage of the pipeline (no_gap classification, representative selection, single-radius suppression, raw-cloud reduction, directed matching) and find either a real-but-inconsequential defect or no defect at all — the closed-loop bottleneck on 3k/10k is conclusively NOT a pipeline transfer/classification/matching defect at any audited stage. It is candidate evidence density/topology itself (too few candidates per region, or candidates whose only closure route requires accepting demonstrably weaker evidence). Do not re-audit these five stages again without new information; if asked to keep chasing closed-loop count, the next round should look at candidate GENERATION density (why regions only get 3-6 candidates at all), not transfer/classification/ordering correctness.
