# 최우선 지속 실행 규칙

- 사용자가 하나의 구현 프롬프트를 완료할 때까지 계속 진행하라고 지시한 경우, 에이전트는 자체 판단으로 중간 진행 보고만 하고 작업을 종료하거나 사용자에게 제어를 돌려주어서는 안 된다.
- 해당 배치의 명시된 완료 기준, 검증, Worklog, 문서 갱신까지 실제로 끝낼 때만 종료한다. 중간 결과는 `commentary`로 짧게 공유하되, 이는 중지 신호가 아니다.
- 외부 권한, 실제로 불가능한 입력, 또는 사용자가 금지한 범위를 넘어서는 작업으로 막힌 경우에만 안전한 대안을 모두 확인한 뒤 blocker를 보고한다. 작업이 크거나 오래 걸린다는 이유는 중단 사유가 아니다.
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
- Repository-wide `pytest` runs commonly exceed 120 seconds. Start full-suite runs with a 600-second (`600000` ms) timeout on the first attempt; do not spend a failed 120-second run rediscovering this known limit.
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

## WebRenderer Commit Policy

- `WebRenderer/` is an independent nested Git repository. Commit renderer-only changes there promptly after scoped verification; do not mix them into the parent OSN-GS repository commit.
- Producer-side changes such as `save_ply` or trainer WebSocket payloads remain parent-repository changes and are committed separately.

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

## Current Framework Reference Maintenance

- `docs/current_framework.md` is the paper/research-facing reference for the **implemented current pipeline only**. Do not add future designs, intended modules, or speculative roadmap material.
- Update it whenever a major pipeline structure changes or an implementation-complete module materially enters or leaves the active framework path.

## Multi-Agent Handoff Rules

- The user is working with multiple agents, including Codex and Claude. Keep `docs/README.md` updated as the primary follow-up/worklog file whenever implementation direction, important defaults, or known risks change.
- Keep `docs/architecture.md` focused on framework-level design decisions. Keep `AGENTS.md` focused on environment, workflow, and agent-operation rules.
- When changing notebook training behavior, record the user-visible knobs and their intended semantics in `docs/README.md`.
- Do not rely on chat-only memory for decisions such as "NURBS/Voxel must stay strongly integrated" or "uncertain-to-certain promotion is forbidden".
- 2026-07-24: `docs/agent_memory/` is an in-repo mirror of Claude Code's persistent auto-memory (user-preference/feedback/project-state notes accumulated across Claude sessions on this project), kept there specifically so Codex and other agents can read it too. See `docs/agent_memory/README.md` for the sync convention. Claude keeps this mirror in sync whenever it updates its own memory; other agents should treat it as read-only project history, not as instructions.

## Incremental Worklog Rule

- For substantial multi-part work, create `docs/worklogs/` if needed.
- After each completed implementation area, add a short Markdown report containing: work performed, result, evaluation, and remaining risks.
- Write worklogs in Korean. Keep headings, status, conclusions, decisions, metrics interpretation, and follow-up risks in Korean; technical identifiers, commands, paths, and literal API/CLI names may remain in English.
- Keep these reports concise and link the final status from `docs/README.md` so Codex and Claude can continue from the same evidence.


## Repository-Wide Pytest Timeout Handling

- 전체 pytest는 이 환경에서 120초를 넘길 수 있다. 120초 도달로 종료되면 테스트 실패로 해석하지 말고, 우선 명령 timeout에 의한 중단인지 확인한다.
- 전체 검증은 처음부터 최소 10분(600000 ms) timeout으로 실행한다: .venv\Scripts\python.exe -m pytest -q
- 짧은 120초 실행이 중단된 경우에는 바로 충분한 timeout으로 한 번 재실행해 최종 결과를 확인한다. 동일 작업을 짧은 timeout으로 반복하지 않는다.
- Windows stdout flush 오류(OSError: [Errno 22] Invalid argument)가 timeout 직후 동반될 수 있다. 이 경우에도 pytest summary 또는 충분한 timeout 재실행 결과를 기준으로 판정한다.



## Gaussian Visualization Contract

This is a mandatory review/output rule for every Gaussian visualization batch.

- Always produce both `Original Scene` and `Observed/Occluded` with the same checkpoint, iteration, camera, resolution, background, renderer, and Gaussian row count.
- `Original Scene` uses only the Gaussians already present in that environment, with their original learned color/SH, position, scale/covariance, rotation, and opacity. Do not add light, shading, emissive effects, recolor, marker Gaussians, or geometry changes.
- `Observed/Occluded` uses the exact same Gaussian rows and geometry as `Original Scene`; change only each Gaussian's display color according to its Observed Space/Occluded Space state.
- Fixed colors: `OBSERVED=(0.10, 0.85, 0.35)` green, `OCCLUDED=(0.92, 0.18, 0.18)` red, and `UNRESOLVED=(0.60, 0.60, 0.62)` gray. Never silently assign an unresolved Gaussian to either state.
- Never create marker Gaussians to make Occluded Space look present. If a validated occluded volumetric representation exists, it may be emitted as an additional `OCCLUDED_VOLUMETRIC` result; it does not replace the mandatory pair.
- Additional frontier/topology/identity/residual views are allowed only after the mandatory pair and must include their own legend and state definition. Do not vary the required item set or palette per worklog.

Historical outputs that used synthetic marker Gaussians, including WL123 `EVENT_IDENTITY_EFFECT`, remain historical diagnostics and are not canonical Gaussian visualizations.