---
name: reference-osn-gs-docs
description: "Which OSN-GS docs to read at session start, in order, and which older text not to trust"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9f58b1e8-0abf-4c0b-a3b4-3b8396c6006c
---

Read in this order at the start of a session that needs project context:

1. `docs/architecture.md` — what OSN-GS is and the intended one-way data flow (visible Gaussians → NURBS → infer occluded surface → generate Gaussians there). Corrected 2026-07-15; see [[project-osn-gs-direction]].
2. `docs/worklogs/19_nurbs_direction_correction.md` — **read before any older worklog.** States the direction rule (visible Gaussians are never moved by the NURBS) and explicitly takes precedence over worklogs 01–18, which were written under the opposite "NURBS is the source of truth" premise and are intentionally left uncorrected as historical records.
3. `TODO.md` — the current work queue. Its own rule (line 1): delete an item once its goal is confirmed. Remaining items are secondary quality-gap candidates plus a NURBS-construction stabilization roadmap.
4. `docs/README.md` — the primary multi-agent handoff/worklog log. The dated sections at the top are the recent change history (skim the newest ones).
5. `nurbs_constructor_benchmark/README.md` — the eval tool. Scores the generated NURBS against ground truth on three separate concerns (fitting accuracy / surface support / patch topology) and emits a GT NURBS for renderer overlay. Use it to measure any constructor change before/after.
6. `docs/nurbs_construction.md` — full Gaussian→NURBS pipeline with equations and a function→file:line map. Detailed; read on demand and verify against code since line numbers drift.
7. `docs/voxel_role.md` — what the one-time voxel bootstrap decides downstream.
8. `Agent.md` — environment/workflow rules: Windows specifics, the Korean `.md` UTF-8 encoding rules (there was a past mojibake incident), and the multi-agent handoff/worklog conventions.

**How to apply:** another agent (Codex) edits this repo concurrently and leaves notes in `docs/worklogs/`, so always trust the actual file contents over any remembered state — re-read before editing. Before non-trivial changes to `osn_gs/surface/*`, `osn_gs/core/torch_pipeline.py`, or the losses, skim items 1–2 first; the direction rule is the constraint most easily violated by accident. Related: [[project-notebook-cli-parity]], [[project-baseline-comparison]].
