---
name: project_nonrepresentative_evidence_attribution
description: "OSN-GS worklog 110 -- attribution (not attachment) of forward-accepted, never-representative surfels relative to the WL107/109 canonical topology"
metadata: 
  node_type: memory
  type: project
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-24T07:40:38.779Z
---

Worklog 110 (branch `arch/2dgs-coverage-first-surface`): froze [[project_renderer_native_topology_gate_closure]]'s canonical topology unmodified and attributed the role of the 395,676 real-scene surfels that are forward-accepted but never a median-surface representative (WL109's cross-tab). Extended the diagnostic-only CUDA sibling with bounded per-pixel accepted-contributor slots (K=16, `out_contrib_ids`/`out_contrib_post_median`/`out_contrib_count`) reusing the kernel's own existing `T` read at median-crossing to classify PRE_MEDIAN/POST_MEDIAN with no new threshold, and streamed contributor<->representative-component co-support pairs across all 161 views without ever materializing a pixel x surfel matrix.

**Central finding: severe truncation.** 97.4% (42,660,905 / 43,817,760) of all pixel-view slots exceeded the K=16 cap. Truncation always drops farther (more post-median-likely, more multi-component-likely) contributors first, since slots fill in depth order — so it biases the measurement toward the simpler/single-component/pre-median interpretation, never the other way.

**Real-scene result despite that bias**: only 26.2% (103,776) of accepted non-representatives co-support exactly one canonical component; 48.0% (189,927) co-support 2+ (median 4, max 609); 63.3% show POST_MEDIAN evidence (30.4% POST_MEDIAN-only) vs 43.8% PRE_MEDIAN. Table cleanest (36.3% no association), hedge/background most layered (POST-series 64.7%, multi-component 54.4%). table<->patio and largest-component<->hedge cross-touch = 0/30 sampled.

**Architecture decision: AMBIGUOUS/LAYERED SUPPORT** (not IDENTIFIABLE SUPPORT). Non-representative renderer evidence must NOT be treated as a single, attachable Visible Surface Support population -- a future architecture must preserve the layered/ambiguous evidence rather than forcing component ownership. No attachment happened this batch (explicitly forbidden). Also corrects [[project_renderer_native_topology_gate_closure]]'s framing: Trust is not responsible for patio/hedge semantic separation -- that belongs to future surface decomposition/NURBS representation, confirmed explicitly this batch.

25 new focused tests (14 CUDA render-invariance/pre-post/count in `test_surfel_representative_diagnostics.py`, 11 pure-logic in new `test_nonrepresentative_evidence_attribution.py`), all passing. Full regression not rerun (canonical training/topology untouched, per directive). New pure-logic module `osn_gs/surface/torch_nonrepresentative_evidence_attribution.py`. Real-scene script `scripts/devtools/nonrepresentative_evidence_attribution.py`, report at `output/osn_gs_nonrepresentative_evidence_attribution/nonrepresentative_evidence_attribution_report.json`, 10 named PLY/PPM/PNG review exports with Korean per-view READMEs.
