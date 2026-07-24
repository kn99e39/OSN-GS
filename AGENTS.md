# Agent Notes

This project is being edited through Codex on Windows. Keep the workflow simple
and avoid spending time fighting the shell.

## File Reading

- Prefer `cmd` commands for quick reads:
  - `type path\to\file.py`
  - `dir /b path\to\folder`
  - `findstr /s /n /i "pattern" path\*.py`
- If `powershell` exits immediately or returns `-1073741502`, switch to `cmd`.
- If inline Python or quoting breaks, write a temporary script under `C:\tmp`
  and run it with `.venv\Scripts\python.exe`.
- Do not assume a failed shell command means the code is broken. Separate tool
  failure from project failure.


## Windows Remote-SSH Sandbox Limits

On this Windows Remote-SSH machine, the restricted sandbox cannot reliably launch PowerShell, Git, or Python modules that load native DLLs such as `ctypes`, `torch`, and CUDA/build tooling. These failures often appear as exit code `-1073741502`, which corresponds to Windows `0xC0000142` / DLL initialization failure.

Use the restricted sandbox only for reading files and simple `cmd` built-in commands, such as `type`, `dir`, and basic text inspection.

Run with `require_escalated` whenever a command needs any of the following:

- PowerShell
- Git
- Python native extensions or DLL-backed imports, including `ctypes`, `torch`, CUDA, and build tooling
- DLL-backed Windows executables such as `where.exe`, `tasklist.exe`, compiler probes, or similar process/environment inspection commands
- Any command that previously failed with `-1073741502`

Do not spend time retrying those commands inside the restricted sandbox. Treat the failure as an environment limitation and either rerun once with `require_escalated` or use a smaller static/file-based check.
## Korean Markdown Encoding Rules

- 한글이 포함된 `.md` 파일은 반드시 UTF-8 또는 UTF-8 with BOM을 보존해서 다룬다.
- 기존 `.md` 파일을 수정할 때는 `read_text(..., errors="ignore")`를 절대 사용하지 않는다. 디코딩 실패 바이트가 조용히 삭제되어 원문 복원이 불가능해질 수 있다.
- Windows 콘솔 출력은 `cp949`일 수 있으므로, 한글 파일 내용을 검증 목적으로 그대로 `print()`하지 않는다. 필요한 경우 `unicode_escape` preview, byte length, 한글 문자 수 같은 간접 검증을 사용한다.
- 문서에 로그를 덧붙일 때도 파일 전체를 임의 인코딩으로 다시 쓰지 않는다. 먼저 기존 인코딩을 판별하고, 가능하면 `apply_patch`로 필요한 부분만 수정한다.
- Git에 남아 있는 원문을 확인할 때는 `git show <rev>:path`의 바이트를 기준으로 비교한다. 깨진 현재 HEAD를 원본으로 착각하지 않는다.
- 이번 `docs/architecture.md` 손상 원인은 깨진 인코딩 상태의 문서를 UTF-8 텍스트로 다시 저장하고, 일부 문자가 `?` 또는 잘못된 CJK 문자로 굳어진 것이다. 정상 원문은 `7a96999:docs/architecture.md`에서 복원했다.
## Editing

- Prefer `apply_patch` for edits.
- If `apply_patch` is blocked by the Windows sandbox wrapper, use a direct
  workspace write command with escalation and keep the change tightly scoped.
- Avoid editing external reference projects. If code is needed from an external
  project, vendor it into OSN-GS and make runtime paths point only to OSN-GS.

## Verification

- Use `python -B` or import-check scripts when `py_compile` fails because it
  cannot write `__pycache__`.
- Useful checks:

```powershell
.venv\Scripts\python.exe train.py --help
```

```powershell
.venv\Scripts\python.exe -B C:\tmp\osn_gs_import_check.py
```

- For notebook edits, validate JSON after modification:

```powershell
.venv\Scripts\python.exe C:\tmp\check_notebook_json.py
```

## Current Rendering Structure

- First-class torch renderer:
  - `osn_gs/render/gaussian_rasterizer.py`
- Vendored CUDA rasterizer source:
  - `osn_gs/render/vendor/diff_gaussian_rasterization`
- Avoid reintroducing `adapter`-style render APIs unless there is a clear
  compatibility reason.

## Vendored Baseline Scene-Split Logic (2026-07-22)

- `osn_gs/data/vendor/graphdeco_scene_split.py` is a verbatim port (license
  header preserved) of two small, self-contained pieces of the original
  Graphdeco `gaussian-splatting/` baseline: its held-out test-camera
  selection (`readColmapSceneInfo`'s `llffhold` branch) and its resolution
  auto-downscale decision (`loadCam`'s `>1600px` rule). Exists so the
  OSN-GS vs. baseline 3DGS quality A/B (`TODO.md`'s top section) can use the
  IDENTICAL train/test camera split and effective training resolution on
  both sides -- verified bit-for-bit against upstream's own function output
  on the real `DATASET/` scene (185 images: same 24 held out, same
  `(1600, 1036)` resolution at scale `3.241875`).
- `osn_gs/data/colmap_scene.py::load_colmap_scene_with_eval_split` is the
  OSN-GS-side loader that uses it, returning a train-only `TorchScene` plus
  the held-out test cameras/images kept separate (never sampled during
  training). `load_colmap_scene` (the original, no-split loader) is
  unchanged and still used by the normal training entry points.
- Per the vendoring rule above: `gaussian-splatting/` itself is never
  imported or modified at runtime -- only these two ported functions, living
  under OSN-GS's own tree.

## Communication

- If a command fails for environment reasons, say that clearly and move to a
  smaller verification step.
- When the user provides a traceback or notebook output, treat it as the source
  of truth. Do not infer a different cell or command without checking.
- 2026-07-23: When the user gives a conditional instruction ahead of time
  ("once training finishes, do your remaining work and then shut the system
  down"), execute it automatically the moment the condition is met. Do not
  stop to re-confirm minor follow-up details the user already settled (e.g.
  via an earlier clarifying question they already answered) — that just
  delays execution for no benefit. Re-confirm only if something materially
  new or risky comes up that the original instruction did not cover. Getting
  this wrong once left the user's machine running all night waiting on a
  question that had already been answered.

## Ongoing Context Log

- 2026-07-01: User requested that whenever the environment, project situation, or task direction changes, the relevant `.md` files should be updated with that context instead of relying only on chat history.
- 2026-07-01: NURBS is an intermediate representation, not a replacement final output. Training should keep Gaussian primitives as the main output while preserving visible NURBS reconstruction data for later visualization tools.
- 2026-07-01: The Colab training notebook should pass NURBS-related configuration alongside OSN-GS training/Gaussian primitive output handling so downstream visualization can consume both Gaussian and NURBS artifacts.


- 2026-07-01: WebRenderer PLY compatibility request. Renderer requires Graphdeco-style Gaussian fields `x`, `y`, `z`, `f_dc_0..2`, raw `opacity`, optional raw log `scale_0..2`, and `rot_0..3`. OSN-GS has corresponding primitives in `TorchGaussianModel`, so `save_ply` should emit those names instead of debug-only RGB/`scale_x` fields.
- 2026-07-01: `colab_train_3dgs.ipynb` output download cell must remain Colab/local compatible. Use `google.colab.files.download` only when `IS_COLAB` is true; local notebook runs should create the output zip in the project root (`GS_ROOT` or `NOTEBOOK_ROOT`) and show/print that path.
- 2026-07-01: Notebook output packaging now includes NURBS visualization data. OSN-GS output inspection creates `visualization_manifest.json` under `MODEL_ROOT`, pairing each `point_cloud.ply` with its sibling `nurbs_surface.json` so external tools can load Gaussian primitives and the visible NURBS intermediate together.
- 2026-07-02: Added `visible_surface_resolution_scale` so Stage 1 visible NURBS control-grid density can be increased from the notebook Train cell without changing the base U/V parameters. Final resolution is computed from `visible_surface_resolution_u/v * scale`.
- 2026-07-02: High `visible_surface_resolution_scale` can increase NURBS fitting memory. Added `visible_surface_fit_device` and `visible_surface_fit_chunk_size` so fitting can run on CPU and process the uv grid in chunks while keeping the final NURBS intermediate available for visualization.
- 2026-07-02: Notebook Train cell should show live RAM/VRAM usage, current training iteration, and a bounded train.py output tail so long runs do not leave the user waiting without feedback.
- 2026-07-02: Implemented basic 3DGS-style Adaptive Density Control for OSN-GS. ADC now accumulates viewspace gradients/radii, clones or splits certain Gaussians, prunes low-opacity/oversized certain Gaussians, and is wired to `densify_until_iter`, `densification_interval`, and `densify_grad_threshold`. Uncertain-to-certain promotion is explicitly disabled; uncertain cleanup may prune only.
- 2026-07-02: OSN-GS saved iteration output folders now use plain numeric names such as `1000` and `10000` instead of `iteration_001000`. Notebook output inspection sorts numeric iteration folders and still treats `final` as the latest consolidated output.

## 2026-07-06 Training throughput note

- Notebook training defaults no longer save full outputs at iteration 1; explicit save iterations are treated as exact output checkpoints.
- Image staging and visible-surface NURBS fitting default to the selected training device, with chunked fitting kept configurable for VRAM control.
- ADC growth is capped by default in the notebook so density control cannot accidentally exhaust VRAM during short experiments.

## 2026-07-06 Runtime NURBS Chunk Sizing

- Visible-surface NURBS fitting now treats chunk size `0` as auto mode.
- Auto mode samples available CUDA VRAM once at pipeline initialization, fixes the chosen chunk size for the run, and logs the selected value.

## 2026-07-06 Automatic Image Placement

- Training image storage now supports `auto`, which loads images on CPU first, estimates the full stack size, and moves the stack to CUDA only when current free VRAM can safely hold it.
- If the image stack exceeds the runtime VRAM budget, images remain on CPU while CUDA still handles Gaussian tensors, rasterization, NURBS fitting, and training math.

## 2026-07-06 Per-View Image Staging

- Training images now remain as CPU-staged per-view tensors instead of one full stacked tensor.
- Each iteration samples the required view batch and transfers only that small batch to the training device, matching the original 3DGS memory pattern more closely.

## 2026-07-06 ADC Gradient Fallback

- Adaptive Density Control now falls back to Gaussian xyz gradients when the CUDA rasterizer does not populate screen-space point gradients.
- ADC passes always log tracked gradient statistics, even when clone/split/prune counts are zero, so disabled or ineffective density control is visible in training output.

## 2026-07-06 Streaming NURBS Snapshots

- OSN-GS training can now stream packed Gaussian snapshots over WebSocket directly from `train.py`.
- Streamed snapshots can include the visible NURBS intermediate as `nurbs_surface`; the payload is sent when the surface is first available or rebuilt.
- Notebook OSN-GS training exposes streaming knobs and can disable slow PLY/NURBS/checkpoint file output when using the renderer stream.

## 2026-07-06 Covariance Initialization Pipeline

- OSN-GS now initializes Gaussian covariance scale from chunked nearest-neighbor point spacing, following the original 3DGS scale+rotation covariance convention without requiring `simple-knn`.
- Notebook and CLI controls expose covariance initialization mode, KNN chunk sizing, min scale, max scene-scale ratio, and scale multiplier.



## Multi-Agent Handoff Rules

- The user is working with multiple agents, including Codex and Claude. Keep `docs/README.md` updated as the primary follow-up/worklog file whenever implementation direction, important defaults, or known risks change.
- Keep `docs/architecture.md` focused on framework-level design decisions. Keep `AGENTS.md` focused on environment, workflow, and agent-operation rules.
- When changing notebook training behavior, record the user-visible knobs and their intended semantics in `docs/README.md`.
- Do not rely on chat-only memory for decisions such as "NURBS/Voxel must stay strongly integrated" or "uncertain-to-certain promotion is forbidden".
- 2026-07-24: `docs/agent_memory/` is an in-repo mirror of Claude Code's persistent auto-memory (user-preference/feedback/project-state notes accumulated across Claude sessions on this project), kept there specifically so Codex and other agents can read it too. See `docs/agent_memory/README.md` for the sync convention. Claude keeps this mirror in sync whenever it updates its own memory; other agents should treat it as read-only project history, not as instructions.

## 2026-07-10 Legacy Prototype Framework Removed

- The original numpy-only prototype framework (`osn_gs/core/{framework,pipeline,state,trainer}.py`, non-`torch_*` files under `osn_gs/gaussian`, `osn_gs/surface`, `osn_gs/losses`, `osn_gs/optim`, `osn_gs/data/{cameras,scene_loader}.py`, `osn_gs/render/prototype_renderer.py`, and `osn_gs/utils/{checkpoint,geometry,logging,typing}.py`) has been deleted. It was self-documented as a smoke-test/algorithm-sketch scaffold, had zero dependents in the active torch training path, and its only consumers (`scripts/train_osn_gs.py`, `tests/test_framework_smoke.py`) were already broken.
- Every `osn_gs/**/__init__.py` now exports only the real `torch_*` symbols. If you need something from the deleted prototype, look at the equivalent `torch_*` module instead of restoring the old file.
- Two already-executed one-off migration scripts were also removed: `scripts/devtools/finalize_dataset_cleanup.py` and `scripts/devtools/patch_dataset_and_remove_synthetic.py`.
- `osn_gs/gaussian/torch_model.py` had mojibake-corrupted Korean comments (pre-existing since the file's first commit, not recoverable from git history); they were rewritten in clean English.

## Incremental Worklog Rule

- For substantial multi-part work, create `docs/worklogs/` if needed.
- After each completed implementation area, add a short Markdown report containing: work performed, result, evaluation, and remaining risks.
- Write worklogs in Korean. Keep headings, status, conclusions, decisions, metrics interpretation, and follow-up risks in Korean; technical identifiers, commands, paths, and literal API/CLI names may remain in English.
- Keep these reports concise and link the final status from `docs/README.md` so Codex and Claude can continue from the same evidence.

