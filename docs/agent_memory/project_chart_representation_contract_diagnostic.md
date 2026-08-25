---
name: project_chart_representation_contract_diagnostic
description: "OSN-GS worklog 113 -- diagnostic-only replay identifying WHY one camera-observed blob fails as one rectangular NURBS chart; 4 causes (A/B/C/D) each assigned to different symptoms, no architecture decision"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T07:35:38.208Z
---

Worklog 113 (branch `arch/2dgs-coverage-first-surface`): froze [[project_renderer_native_pixel_surface_nurbs]] (WL112) exactly -- same WL107/109 topology, WL111 blob construction, WL112 pixel-surface geometry, fixed 8x4/degree-2 NURBS -- and diagnosed WHY "one camera-observed connected blob == one rectangular tensor-product NURBS chart" fails, without tuning or implementing chart subdivision.

**Corrected support accounting**: of 155,457 representative-bearing components, only 1.2% (1,857) ever produced a >=32-pixel blob in ANY view, and ALL 1,857 were covered (100%). Only 11 components had >=32 representatives yet zero >=32-pixel blob (genuinely different "thinly-scattered observation" failure, but negligible in count).

**Zero-coverage attribution (clean result)**: ALL 153,600 zero-coverage components (100%) are `NO_VIEW_BLOB_REACHES_32_PIXEL_SAMPLES` -- the current fit loop has no failure/skip branch once a blob passes the pixel threshold, so "sufficient support but fitting still fails" was never observed. Classification A (SUPPORT_LIMITED) is the sole, exhaustive cause of zero-coverage.

**Domain shape (typical chart, not just outliers)**: fitted-chart bbox occupancy median = 0.5 (half of every chart's bounding rectangle has zero observed pixels), 43.5% of charts have >=1 enclosed hole. Charts with holes have ~2.6x higher median residual than hole-free charts (0.0078 vs 0.0030) -- Classification B (rectangular-domain/hole mismatch) has REAL effect on typical fit quality, not just tails.

**Fixed 8x4 capacity**: 51.4% of charts reach full design-matrix rank (well-determined). Counter-intuitively, full-rank charts have HIGHER median residual (0.0067) than rank-deficient ones (0.0031) -- interpreted as rank-deficient charts being tiny near-interpolating fits, while full-rank charts have enough real geometric complexity to resist a perfect fit. Sample-count/residual correlation at full rank = 0.545 (moderate). Classification C (capacity failure) applies narrowly, concentrated in a handful of giant components, not generally.

**Extreme-outlier provenance traced exactly (the key finding)**: residual_max (1517) and overlap_position_discrepancy_max (1514) come from TWO DISJOINT MECHANISMS. 12/15 worst-residual charts are all the same giant patio component (id=0, the known 36.77%-of-surface component) or table_top's big component (id=1): huge pixel counts (4万-21万), huge hole counts (780-3029), always full rank -- genuine B+C combination limited to a few giant components. All 15 worst-overlap charts are tiny (38-276 pixels) components (table_side_curved 19/40/26/38, patio 180/324); several show depth_std 10-104 WITHIN one tiny chart (e.g. depth ranging 8.76-1723 in a 271-pixel chart) -- clean, isolated evidence of Classification D (renderer median-depth numerical instability), unrelated to holes or capacity.

**Region attribution**: table_side_curved (62.2%) and hedge (54.8%, lowest) are almost purely Classification A (fragmented observation, holes rare in their charts). table_top (75.5%) and patio (79.9%, highest coverage) carry the hole-heavy giant charts responsible for most extreme-tail pathology -- coverage and fit-stability are independent axes.

**No architecture decision** (directive explicitly forbade one this batch). Minimal next-step evidence noted but not implemented: A-dominated regions need no NURBS change at all (observation itself is insufficient); B needs a non-rectangular/trimmed chart *unit*, not just resolution; C is narrow enough to warrant subdividing only the few giant components rather than a general resolution increase; D is a renderer-depth stability issue independent of chart design.

12 new focused tests (pure numpy/scipy: bbox/occupancy/hole accounting on hand-built masks, rank/conditioning on synthetic planar/degenerate charts, distribution/quantile-binning utilities), all passing. No CUDA/production code touched. One real bug caught and fixed during the smoke test (`bbox_area` referenced but not extracted from the refactored `blob_domain_shape` helper) before the full run. Real-scene script `scripts/devtools/chart_representation_contract_diagnostic.py`, report at `output/113_osn_gs_chart_contract_diagnostic/chart_representation_contract_diagnostic_report.json`, 10 named review exports. Full 161-view topology replay matched WL107/109/111/112 exactly (36.77%/45.02%), and `valid_chart_count=14900` matched WL112 exactly, confirming this batch's new diagnostics did not perturb the frozen pipeline.
