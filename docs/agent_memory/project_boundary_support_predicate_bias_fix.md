---
name: project_boundary_support_predicate_bias_fix
description: "worklog 77 - found and FIXED a real predicate defect (angular-gap discretization bias: straight boundary's true empty sector is exactly pi, so the raw estimate converges to pi from below). Synthetic recall 0.44->1.00 with precision unchanged at 1.000; first-ever valid closed loop with 0.000 interior-outside. But real closed loops stay 0 and 71.7% of remaining gaps are genuine evidence absence."
metadata:
  type: project
---

worklog 77 audited the dense boundary-support predicate after [[project_normal_source_architecture_decision]] (75) removed the normal source and [[project_boundary_support_spacing_scale_decision]] (76) removed the connectivity scale.

**Real defect found and fixed.** `extract_dense_boundary_support` measures the empty angular sector between two flanking neighbour RAYS, but neighbours are point samples of a continuous surface, so the measurement understates the true sector by ~the local angular sampling resolution. A STRAIGHT boundary's true empty sector is exactly pi, so the raw estimate converges to pi FROM BELOW and `gap >= pi` fails in the limit of perfect sampling — an asymptotically biased estimator, not a strict threshold. Measured on box_face: true-boundary gap/pi median 0.9989 with 25 of 32 points within 1% of pi, while interior sits at 0.25pi (cleanly separated, so discriminative power was always fine — only the estimate was biased).

Fix: threshold stays pi; subtract the point's OWN median non-maximal gap (its local angular sampling resolution). No new constant, uses only that point's own evidence, and the correction VANISHES as density rises (recovers exactly `gap >= pi`).

**Proof it is a bias correction, not a threshold relaxation:** precision stayed exactly 1.000 on every fixture, the closed sphere still yields 0 candidates, and the new tests fail ONLY on the straight-boundary cases before the fix while all fail-closed tests pass both before and after (verified by git-stash A/B).

Synthetic (ground truth derived from fixture construction): recall box_face 0.438->1.000, cylinder side 0.250->1.000, caps 0.44/0.50->0.75 (cap labels are themselves approximate). Consecutive-missed-run length along the perimeter 4-5 -> 0 — that run structure, not the aggregate count, is what a local certificate cannot bridge. With the UNCHANGED connectivity path, box_face then produced the **first valid closed loop in worklogs 69-77 with interior_outside_boundary 0.000**, and cylinder side correctly recovered its two separate rings without merging them.

Real baseline_compatible@2900 (7 regions, 3526 points): candidates 785->940 (+19.7%), degenerate frames 0, branch 0, crossings 0, but **closed loops still 0**. Continuity attribution: of 940 accepted-candidate nearest-neighbour paths, only 266 (28.3%) contain rejected observed points; **674 (71.7%) contain no observed evidence at all**.

**Architectural answer: both, but dominated by genuine evidence absence.** The predicate had a real, fixable defect — and after fixing it, the same predicate demonstrably materializes a complete dense perimeter and a valid loop when observation is actually continuous (synthetic), while real data still yields 0 loops. That synthetic/real contrast attributes the remaining failure to the evidence, not the predicate.

Measured but deliberately NOT actioned: 40-66% of gap-rejected real points have >=1 near-normal neighbour whose tangent projection carries no usable azimuth (`atan2(0,0)=0` can fabricate support). Synthetic fixtures show 0 of these, so there is no ground truth to validate such a change against — making it would be indistinguishable from tuning toward a desired result.

**How to apply:** the boundary-construction axis is now exhausted (69-77). Next candidates are making the OBSERVATION denser near perimeters, or a non-boundary-loop representation — not another orientation/scale/boundary-algorithm variant. Revisit the near-normal contamination only when a ground-truth-checkable fixture exhibits it.
