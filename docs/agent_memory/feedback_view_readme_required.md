---
name: feedback_view_readme_required
description: Every export view folder needs its own Korean README.md — this is required, not optional
metadata:
  type: feedback
---

Every `output/<batch>/<VIEW_NAME>/` folder must contain a **Korean `README.md`**
explaining that view's colour coding and what the image actually shows, with the
batch's own measured numbers in it.

**Why**: this is already written down in
[[reference_output_folder_conventions]] (`docs/output_folder_conventions.md`),
but worklogs 120, 121 and 122 all shipped exports without it and the user had to
ask again. The PNG alone is not reviewable — the reader cannot tell what green
vs orange means, or whether a pattern is the finding or an artifact, without the
per-view text.

**Format** (established by worklog 103, follow it):
```
# <VIEW_NAME>

## 색상 의미
- **<색>** (`r, g, b`): <그 색이 뜻하는 것>
...

## 이 이미지가 보여주는 것
<이 배치의 실측 수치를 넣은 설명. 해석 주의사항이 있으면 여기에.>

---
체크포인트: `output/arch_.../checkpoint.pt` (N surfel, 161 train camera)
전체 리포트: `../<report>.json` · Worklog: [`docs/worklogs/<n>_<name>.md`](../../../../docs/worklogs/<n>_<name>.md)
```

**How to apply**: write the README in the devtools script itself, next to the PLY
and PPM, so it can never be forgotten again — not as a manual post-step. Include
any interpretation guard the worklog states (e.g. "midpoint OBSERVED is not
surface continuity"), because the export is exactly where a reader is most likely
to over-read a picture.

Related: [[feedback_png_preview_in_output]] (PNGs go in one shared
`preview_png/` folder per batch), [[feedback_include_subset_visualization]].
