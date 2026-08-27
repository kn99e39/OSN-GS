---
name: project-evidence-bounded-projective-tsdf
description: WL127 evidence-bounded projective TSDF replaces topology/boundary-first Visible Surface construction; verdict B (viable premise, insufficient fixed TSDF)
metadata:
  type: project
---

Worklog 127 (branch `arch/2dgs-coverage-first-surface`) tested a completely different Visible
Surface CONSTRUCTION premise: canonical renderer median-depth observations fused directly into an
evidence-bounded projective TSDF, masked zero level-set, no historical topology/KNN/region/boundary/
chart used to decide where surface exists.

**Verdict: B — IMPLICIT VISIBLE SURFACE GEOMETRY IS VIABLE, BUT THE CURRENT FIXED PROJECTIVE TSDF IS
INSUFFICIENT AS THE CANONICAL CONSTRUCTION.** Not A, because "no material unsupported-gap
fabrication" was NOT positively established on the real scene (11.90% of sampled mesh points are
>2h from any median event; 20.83% of triangles stand on single-view support). Not C, because the
candidate clearly works. Not D, because every stage ran at full 161-view scale and reproduced twice.

Key measurements: h = 0.012105485424 (global median renderer footprint, no sweep), mu = 3h,
authoritative voxels 76,720,314, mesh 28.7M verts / 45.1M faces / 582,646 components,
renderer-evidence coverage 89.84% within h (exhaustive over all 43,817,760 events),
ray-hit coverage 99.88%. Baseline arm A replayed the historical path unmodified and reproduced
WL107/109 topology (559,989 / 535,910 / 0.36771) and WL119's 14,900 charts exactly;
A/B: coverage 58.47% vs 89.84%, unsupported bridges 26.42% vs 11.90%, hedge 11.53% vs 87.52%.
Occlusion audit: mesh independently reproduces 98.78% of Candidate B's OCCLUDED verdicts, and
98.62% of the reverse disagreements are explained by real reconstructed surface, not invention.

**Why the fixed TSDF is insufficient**: one global h cannot serve footprints spanning 0.05h..242h.
Far-field enumeration never closes (residual ~0.014%/round, confined to depth 176-1412 world units),
and mu = 3h makes opposite-face bands overlap on thin structure (S6 reconstructs 3x nominal width).

NURBS handoff verdict: **PROMISING** (4/4 spatial-ROI crops fit with the existing fitter,
residual median 0.13h-1.07h, 100% finite normals). Independent of the primary verdict.

Nothing was deleted. See [[project_worklog127_tsdf_code_layout]] for the code family.
