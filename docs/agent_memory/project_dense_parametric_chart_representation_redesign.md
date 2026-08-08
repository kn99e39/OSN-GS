---
name: project_dense_parametric_chart_representation_redesign
description: "worklog 80 - redesigned the parametric chart representation so sparse topology supplies only order+typed provenance while dense observed boundary-support supplies geometry. Coverage 0/5 -> 4/5 passing, but valid_supported still 0: regions are not single regular charts (evidence UV fold 21-36%) and topology proves no split. Verdict (3)."
metadata:
  type: project
---

worklog 80 acted on [[project_constructor_chart_domain_coverage_verdict]] (worklog 79) by redesigning the representation rather than running more diagnostics.

New `osn_gs/surface/torch_dense_parametric_chart_support.py` separates two roles that were previously conflated:
- **Topology role** (sparse accepted representative cycle): supplies only the perimeter's cyclic ORDER and each arc's typed frontier PROVENANCE. Never the chart's geometric extent.
- **Geometry role** (region-owned dense boundary-support candidates from worklog 77's corrected predicate, unmodified): supplies the actual boundary. Every chart vertex is an observed Gaussian.

Justified by measurement: dense candidate extent / owned evidence extent = **0.966-1.020**, versus **0.148-0.667** for the representatives they replace. Each candidate is assigned to its nearest sparse ARC (topology constrains membership and inherits that arc's typed kind), ordered monotonically within the arc, then concatenated in cyclic order — representatives are NOT vertices of the result. No dense support => fail closed, never a sparse-polygon fallback. Arcs with no candidate are recorded `evidence_backed=False`. Multiple charts only when the 2-core splits into >=2 disjoint components.

**Self-caught implementation defect**: the arc binning derived its bin count from the SPARSE CHORD length, reintroducing the very scale mismatch the module removes (region 6 kept only 16 of 218 candidates, coverage 72.7%). Fixed to derive from the candidates' own projected span: region 6 -> 88 vertices/56.1%, region 0 -> 6->17 vertices, 59.1%->31.2%.

**7-region before/after (baseline_compatible@2900)**: chart-domain coverage passing **0/5 -> 4/5**; chart vertices 3,3,3,3,3 -> **17,84,92,20,88**; evidence outside domain **82.6-99.6% -> 29.0-56.1%**; Jacobian near-degenerate 0 throughout; fitted-surface folding <=0.45%. But all 4 passing charts are `extrapolative` (held-out p95 5.49-19.92) and **valid_supported is still 0**.

**The cause is now isolated**: parameterizing each region's OWN evidence by PCA-UV shows **21-36% of UV-adjacent triangles with disagreeing 3D normals** (neighborhood preservation 0.53-0.70) — the region is not a single regular chart. worklog 69 saw this too but it was confounded with the extent problem; removing the extent problem isolates it. Multi-chart was checked and is NOT provable: `independent_chart_components` is 1 for regions 0/1/2/3/4/6 and 0 for region 5, so the existing topology proves no split anywhere, and splitting without evidence is forbidden.

**VERDICT (3): the existing region/evidence representation can supply evidence-backed chart domains but NOT valid SINGLE chart domains.** The redesign itself is viable and worth keeping — it genuinely resolved worklog 79's structural defect — but it does not yield usable real charts; the failure merely moved from "chart cannot contain its evidence" (before parameterization) to "evidence is not a single regular chart" (during parameterization).

**How to apply:** the remaining bottleneck is BOTH that regions are not single charts AND that the accepted topology lacks the resolution to prove how to split them. Address them together — define the chart unit at region formation, or raise topology resolution to evidence density. Keep worklog 80's topology/geometry separation; redefine the chart unit on top of it. Do not reopen orientation (75), scale (76), predicate (77), or chart geometry (79/80) without new evidence.
