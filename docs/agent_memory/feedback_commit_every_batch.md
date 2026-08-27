---
name: feedback_commit_every_batch
description: Commit at the end of every work batch without being asked
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 24cb901d-62f4-4192-8bb7-bb0b66edd28f
  modified: 2026-08-27T06:14:00.940Z
---

Commit at the end of **every** batch of work — do not wait to be asked, and do not
finish a worklog batch leaving the tree dirty.

**Why**: the user runs long multi-batch architecture sessions and reviews the
repository between them. Uncommitted work from batch N makes batch N+1's
`git status` ambiguous (which changes belong to which worklog?) and forced an
explicit "전부 커밋해" instruction twice, after worklog 120 and again after
worklog 122 had accumulated two uncommitted batches.

**How to apply**:
- Commit once per completed batch, after the tests and the real-scene replay pass.
- Bundle concurrent agents' changes into the same commit — see
  [[feedback_commit_scope]].
- Stay on the current work branch (never the default branch); see
  [[project_branch_rule]].
- The worklog's "Exact Branch / Commit" section needs the batch's own SHA, which
  only exists after the commit — record it in a small follow-up commit rather
  than amending.
- Still confirm before anything outward-facing (push, PR); committing locally is
  what this instruction covers.
