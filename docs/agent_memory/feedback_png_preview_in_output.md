---
name: feedback-png-preview-in-output
description: Always convert render.ppm review exports to PNG and place them inside the output/ folder (not scratchpad) for the user to browse
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 06f8c1f6-8e00-47ed-9b87-f3ca26aeaf84
  modified: 2026-08-25T07:39:59.089Z
---

The user confirmed (2026-08-20, after Worklog 96/97's review exports) that they want this as the standing workflow: whenever a batch produces `render.ppm` review-export images, also convert them to PNG and place the PNGs inside the relevant `output/<...>/` directory itself (e.g. `output/<export_dir>/preview_png/<VIEW_NAME>.png`), not just in the session scratchpad. They first objected when the PNGs were only in scratchpad ("보기 힘든 곳에 있잖아" — that's a hard-to-reach place), asked for them to be put in the output folder instead, and then explicitly said "다음에도 이런 식으로 png로 시각화해서 보여줘" (do this PNG-visualization the same way next time too).

**Why:** `render.ppm` isn't viewable in most tools/IDEs directly; the user wants a quick way to browse results without loading them into WebRenderer. Scratchpad is session-temporary and not a place the user naturally looks.

**How to apply:** whenever a devtools export script writes `render.ppm` files (coverage-first partition exports, region-coherent partition exports, or any future review-export batch), after running it, also: 1) convert each view's `render.ppm` to a thumbnailed PNG (e.g. via PIL, `im.thumbnail((900,900))`), 2) save ALL of a batch's view PNGs together into ONE shared `preview_png/` folder at that batch's own top-level `output/0NN_osn_gs_.../` directory, named `<VIEW_NAME>.png` — **never** a separate `preview_png/` subfolder inside each individual view's own folder (that per-view scattering crept in during worklogs ~109-113 and was corrected back during worklog 113; see [[reference_output_folder_conventions]] for the exact layout), 3) mention the PNG paths inside `output/` in the final report rather than only scratchpad paths. Do this proactively as part of finishing a visual review-export batch, without waiting to be asked each time.
