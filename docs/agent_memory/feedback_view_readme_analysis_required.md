---
name: feedback-view-readme-analysis-required
description: every intermediate-output view README must include an analysis/evaluation section grounded in the batch's own report numbers, not just color legend
metadata:
  type: feedback
---

Starting from Worklog 127, every view export folder's `README.md` must include a
`## 분석 및 평가` (Analysis and Evaluation) section, in addition to the existing
`## 색상 의미` (color meaning) and `## 이 이미지가 보여주는 것` (what this shows) sections.

**Why:** the user asked that each visualization's README also carry an assessment of what the
image actually demonstrates, not just describe its color coding. A color legend alone does not
tell a reviewer whether the numbers behind the picture are good, bad, or ambiguous.

**How to apply:**
- Ground every claim in that specific view's own measured numbers from the batch's JSON report —
  never a vibe-based read of the picture. Cite the report section/field the number came from
  (e.g. `renderer_evidence_reproduction.all_events`) so it's traceable.
- Cross-reference other views in the same batch where the same phenomenon shows up from a
  different angle (e.g. a depth-error tail explained by an occlusion-disagreement view).
- Keep the interpretation guards the worklog itself imposes (e.g. "same component ≠ correct
  physical continuity") — the analysis section must not contradict them.
- Write it in the script that generates the README (same pattern as
  [[feedback_view_readme_required]]), not as a manual post-step, once the pattern is established
  for a given devtools driver. For WL127 it was added retroactively to the already-published views.
- This applies to ALL future worklog exports going forward, not just WL127.
