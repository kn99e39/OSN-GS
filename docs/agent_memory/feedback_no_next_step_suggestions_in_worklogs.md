---
name: feedback_no_next_step_suggestions_in_worklogs
description: "don't include \"다음 자연스러운 후속\" / next-step suggestions inside docs/worklogs/*.md files"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c91e18fb-6002-40ed-b911-d218589c420a
  modified: 2026-08-07T04:46:11.488Z
---

The user explicitly asked to stop including next-step/follow-up suggestions inside worklog files (docs/worklogs/*.md) going forward. Prior worklogs (e.g. 63-67) ended with a "다음 자연스러운 후속은 ..." section proposing what to investigate next; this pattern should no longer appear in worklog content.

**Why:** not stated explicitly beyond the instruction itself — worklogs should record what was done and found, not editorialize about future direction.

**How to apply:** when writing a new docs/worklogs/*.md file, end with the completion-criteria/results/tests sections only — omit any "다음 후속"/"미착수"/"next step" section. If forward-looking pointers are still useful, they belong in `docs/Urgent_Work/OSN_GS_Urgent_Work_Master.md`'s own addendum/known-issues sections (already the canonical place for open-direction tracking), not in the worklog itself. This applies from the point the instruction was given onward — no need to retroactively edit past worklogs.
