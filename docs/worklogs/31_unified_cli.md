# 31. Unified PowerShell CLI

Date: 2026-07-16

## Work

Added standard packaging metadata and the osn-gs console script. The top-level help exposes train, benchmark, inspect-surface, and stream-server. Each command delegates to the existing entry point rather than copying its implementation, so command-specific help and behavior remain authoritative.

## Use

Activate the repository virtual environment, then run osn-gs --help. Refresh the local editable installation after pulling changes with .venv\Scripts\python.exe -m pip install -e . --no-deps.

## Verification

Verified top-level help plus train and benchmark command help from the installed console script.
