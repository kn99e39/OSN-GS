# 30. Notebook Train MSVC Environment

Date: 2026-07-16

## Cause

The stored Train-cell log stopped at iteration 0: the OSN-GS CUDA rasterizer preflight could not find cl.exe. The CUDA extension setup cell activated vcvars64.bat only in its temporary build child, while Train started a separate train.py subprocess with no MSVC PATH, INCLUDE, or LIB.

## Fix

The Train cell now writes a temporary cmd launcher, calls vcvars64.bat, captures its environment with set, and merges it into the train.py subprocess environment. It probes where cl before launch and reports a targeted error if activation is incomplete. The temporary cmd pattern matches the successful CUDA extension-build cell.

## Verification

Notebook JSON and Train-cell Python syntax compile. The new helper was executed against the local VS Build Tools installation and found cl.exe at the x64 MSVC path with non-empty INCLUDE and LIB.

## Next Run

Re-run the CUDA extension setup cell if the kernel was restarted, then run Train. It should print Train MSVC environment before the training command.
