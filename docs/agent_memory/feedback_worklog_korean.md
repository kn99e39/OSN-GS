---
name: feedback-worklog-korean
description: "OSN-GS docs/worklogs/*.md entries must be written in Korean, matching the existing worklog corpus"
metadata:
  type: feedback
  originSessionId: ba7ab28f-0918-40fb-8c00-9cc74bf417f1
  modified: 2026-07-21T03:53:06.110Z
---

Write new `docs/worklogs/*.md` entries in Korean, not English.

**Why:** on 2026-07-21 I wrote worklog 39 in English (matching the language of the most recent worklogs I had read, e.g. 31/37/38, which happened to be English). The user corrected this explicitly: "worklog는 이전 worklog들처럼 한국어로 작성하도록 하렴" ("write worklogs in Korean like the previous ones"). I rewrote it in Korean and continued writing subsequent worklogs (40) in Korean without being asked again. This also matches [[reference_osn_gs_docs]]'s note that `Agent.md` documents Korean `.md` encoding conventions for this project.

**How to apply:** default to Korean for any new worklog entry in this repo, regardless of what language nearby existing worklogs happen to be in. Technical terms/code identifiers can stay in English inline (as the existing Korean worklogs already do), but prose/analysis/headers should be Korean. This is specific to `docs/worklogs/` — other docs (code comments, `OSN_GS_*.md` plan docs, memory) follow normal project conventions, not necessarily Korean.
