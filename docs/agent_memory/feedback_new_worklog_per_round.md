---
name: feedback-new-worklog-per-round
description: "Don't keep appending follow-up sections to one worklog across rounds; write a new worklog file for each new round of work"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3ab75852-24f1-41ab-9e47-206246eb025c
  modified: 2026-07-31T06:32:03.777Z
---

When a Gate/task gets a follow-up round of work (e.g. "fix these two more contract
defects", another hardening pass, another correction round), write a **new**
`docs/worklogs/NN_*.md` file for that round instead of appending another section
to the existing worklog that already covered the earlier round.

**Why:** the user explicitly corrected this (2026-07-26) after I had appended two
follow-up sections directly onto worklog 88 (`2_uncertain_gaussian_append_adapter_foundation.md`
as of the 2026-07-31 worklog renumbering — originally `88_uncertain_gaussian_append_adapter_foundation.md`)
across two rounds of Gate hardening — first the transaction/contract verification
pass, then the receipt/ledger contract fix pass. They said to stop updating the
existing worklog and keep writing new ones going forward.

**How to apply:** this reverses the pattern I'd been following for Phase D/E/F
gate-approval status-line edits (those are fine — a short status-line/table update
on an already-approved worklog is not the same as appending a new multi-section
report). The line to draw: if the new work produces its own "변경 파일 / 결과 /
회귀" report content, it gets its own new worklog number, cross-referencing the
prior worklog by number/link rather than growing it. Only trivial status
corrections (e.g. "Gate X 승인 완료" line edits already covered elsewhere) stay
as small edits to existing docs. Check `ls docs/worklogs/` for the current
highest number before assigning the next one, since other concurrent
agents/sessions may have already added worklogs past the last one I saw.

**Exception (2026-07-27):** this default is overridden the moment the user
explicitly names a specific worklog to append to for the current round (e.g.
"이번 작업은 동일 Ownership Foundation Gate 보완이므로 Worklog 96에 결과를
추가하라" — happened for worklog 96 (now `3_occluded_chart_ownership_foundation.md`
as of the 2026-07-31 renumbering)'s lifecycle-invariant follow-up, while the
Ownership Foundation Gate itself was still open/unapproved). Explicit current
instruction always wins over this saved default; don't insist on a new
worklog number just because the general rule says so once the user has
directly said otherwise for that task. The two aren't really in conflict:
"new worklog per round" is what to do absent other direction; an explicit
"add it to worklog N" for a still-open Gate is that other direction.

**Exception revoked (2026-07-27, same day, boundary-first isolated topology
rebuild thread — see [[project_boundary_first_isolated_topology_rebuild]]).**
That session's opening instructions said to append the exporter-review-layer
work to worklog 109 (that number no longer exists as a file after the 2026-07-31
worklog renumbering/pruning pass — it was one of the older worklogs discarded
from the working directory, recoverable only via git history) rather than create a new one; after I did that, the user
immediately said "다음부턴 그냥 새 worklog 만들어" (from now on just make a
new worklog). So for this specific thread going forward, the general default
applies again: each new round of work gets its own new `docs/worklogs/NN_*.md`
file, not another appended section on 109 or any other prior worklog. Treat
any future "append to worklog N" instruction as a one-round-only exception
again unless the user repeats it.
