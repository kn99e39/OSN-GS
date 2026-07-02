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
- Prototype numpy renderer:
  - `osn_gs/render/prototype_renderer.py`
- Vendored CUDA rasterizer source:
  - `osn_gs/render/vendor/diff_gaussian_rasterization`
- Avoid reintroducing `adapter`-style render APIs unless there is a clear
  compatibility reason.

## Communication

- If a command fails for environment reasons, say that clearly and move to a
  smaller verification step.
- When the user provides a traceback or notebook output, treat it as the source
  of truth. Do not infer a different cell or command without checking.

## Ongoing Context Log

- 2026-07-01: User requested that whenever the environment, project situation, or task direction changes, the relevant `.md` files should be updated with that context instead of relying only on chat history.
- 2026-07-01: NURBS is an intermediate representation, not a replacement final output. Training should keep Gaussian primitives as the main output while preserving visible NURBS reconstruction data for later visualization tools.
- 2026-07-01: The Colab training notebook should pass NURBS-related configuration alongside OSN-GS training/Gaussian primitive output handling so downstream visualization can consume both Gaussian and NURBS artifacts.


- 2026-07-01: WebRenderer PLY compatibility request. Renderer requires Graphdeco-style Gaussian fields `x`, `y`, `z`, `f_dc_0..2`, raw `opacity`, optional raw log `scale_0..2`, and `rot_0..3`. OSN-GS has corresponding primitives in `TorchGaussianModel`, so `save_ply` should emit those names instead of debug-only RGB/`scale_x` fields.
- 2026-07-01: `colab_train_3dgs.ipynb` output download cell must remain Colab/local compatible. Use `google.colab.files.download` only when `IS_COLAB` is true; local notebook runs should create the output zip in the project root (`GS_ROOT` or `NOTEBOOK_ROOT`) and show/print that path.
- 2026-07-01: Notebook output packaging now includes NURBS visualization data. OSN-GS output inspection creates `visualization_manifest.json` under `MODEL_ROOT`, pairing each `point_cloud.ply` with its sibling `nurbs_surface.json` so external tools can load Gaussian primitives and the visible NURBS intermediate together.
- 2026-07-02: Added `visible_surface_resolution_scale` so Stage 1 visible NURBS control-grid density can be increased from the notebook Train cell without changing the base U/V parameters. Final resolution is computed from `visible_surface_resolution_u/v * scale`.
