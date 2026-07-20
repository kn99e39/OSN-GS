# 29. Renderer SH Streaming

Date: 2026-07-20

## Work performed

- Extended both OSN-GS snapshot producers (`TorchOSNGSTrainer` and the legacy
  `train_with_ws_runtime.py` hook) to emit the model's active SH degree and
  coefficient-major RGB SH data.
- Added `shDegree` and flattened `shCoefficients` to packed WebSocket payloads.
  The coefficient count per Gaussian is `(shDegree + 1)^2`; tensor layout is
  `[gaussian, coefficient, RGB]`.
- Updated the renderer protocol documentation. The WebGPU renderer consumes
  these fields in Gaussian Composition and retains DC-color compatibility for
  older snapshots that omit them.

## Result

Live OSN-GS snapshots can preserve the same active SH coefficients used by the
CUDA rasterizer instead of reducing all streamed Gaussians to DC RGB. This
enables view-dependent color in the WebGPU Composition path once the browser
receives an updated producer payload.

## Evaluation

- Python AST parsing passed for both producer files.
- Renderer-side packed-snapshot SH decoding is covered by
  `3DGS_Renderer/tests/gaussian_stream_sh_smoke_test.js`.
- No end-to-end WebSocket/browser render was run in this environment because a
  browser target was unavailable.

## Remaining risks

Degree-3 JSON snapshots carry 48 float values per Gaussian for SH alone, versus
three RGB values in the previous stream. This increases serialization, network,
and browser upload cost substantially. Use `stream_every` and
`stream_max_gaussians` to control cadence and size; consider a binary snapshot
format before enabling frequent full-resolution degree-3 streaming on large
scenes.
