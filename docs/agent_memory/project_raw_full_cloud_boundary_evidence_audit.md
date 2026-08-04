---
name: raw-full-cloud-boundary-evidence-audit
description: "Worklog 51 — audited whether raw full-cloud (pre-representative-reduction) boundary chains exist that representative selection loses; found no clean case, rejected the boundary-anchor sidecar, and located a separate out-of-scope ordering defect instead"
metadata: 
  node_type: memory
  type: project
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-03T12:14:45.146Z
---

Worklog 51 (docs/worklogs/51_raw_full_cloud_boundary_evidence_audit.md) audited, per user directive, whether 3k/10k's RAW full cloud (before representative reduction) contains coherent physical-boundary chains across representative-level open-chain gaps that representative reduction is losing — and if so, to implement a region-level "boundary-anchor sidecar" (raw stable IDs + typed evidence, used only for boundary compatibility/ordering/materialization, never region formation or interior fitting).

**Method:** new `scripts/devtools/trace_raw_full_cloud_boundary_evidence.py` samples 5 points along the straight line between two representative-level chain endpoints, anchors each to the nearest RAW full-cloud Gaussian, and runs the SAME production same-mode + largest-circular-gap algorithm directly on raw Gaussians (own frame, other raw Gaussians in the same representative-assigned region as support) — a true ground-truth measurement with zero representative reduction involved.

**Findings across 5 gaps (3k regions 60×2, 52, 56, 77):**
- 3 gaps: raw cloud itself shows no coherent chain — one is genuinely interior (gap 2-28°, well below the 24° threshold, at all 5 samples), one is patchy (real 102° gap at one sample, literally NO raw anchor at two adjacent samples, borderline elsewhere), one has zero raw data anywhere along a 9.79-unit path (not actually adjacent boundary segments at all).
- 2 gaps (region 52: 666904↔1086120, region 56: 1110285↔278207, both very short ~0.12-0.13 unit gaps): raw cloud shows a STRONG, CONSISTENT edge signature (56-226°, well-supported 33-70 Gaussians) at all 5 samples. But re-checking the original representative-level trace showed these endpoint pairs were ALREADY mutually `first_gate: compatible` at the representative level too — the representatives exist there and pass geometric compatibility; the reason they stay open-chain is the Hungarian one-in-one-out matching not selecting that edge (competition with other candidates), not missing evidence at any stage.

**Conclusion:** rejected the boundary-anchor-sidecar hypothesis — no case matched "raw chain exists, lost specifically at representative reduction" (the sidecar's exact remit). The 3 no-chain cases fail closed as directed ("no raw chain → leave unsupported, no forced closure"). The 2 strong-signal cases are a genuine but DIFFERENT defect (directed-ordering/Hungarian-matching candidate competition) that this task explicitly forbade touching (Hungarian solver out of scope) — a sidecar wouldn't even help since the representative-level candidate already exists and is already flagged compatible. No production code changed this round; full pytest 720 passed unchanged (trivially, since nothing changed).

**How to apply:** if a future round is authorized to touch directed ordering / the Hungarian matching step, region 52 (666904↔1086120) and region 56 (1110285↔278207) on the 3k checkpoint (cap 2048) are concrete, reproducible starting points — both representative-level AND raw-full-cloud evidence already confirm a real edge there; the sole blocker is matching-step candidate competition, not evidence. Don't re-attempt a boundary-anchor sidecar for these — it addresses a different failure mode (reduction loss) that isn't what's happening here.
