"""Unified command-line entry point for OSN-GS tools."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _delegate_main(module_name: str, argv: list[str]) -> int:
    module = __import__(module_name, fromlist=["main"])
    old_argv = sys.argv
    try:
        sys.argv = [f"osn-gs {module_name.rsplit('.', 1)[-1]}", *argv]
        result = module.main()
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def _delegate_script(relative_path: str, argv: list[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [f"osn-gs {relative_path}", *argv]
        runpy.run_path(str(_ROOT / relative_path), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osn-gs",
        description="Unified OSN-GS command line.",
        epilog="Use 'osn-gs <command> --help' for command-specific options.",
    )
    parser.add_argument("--version", action="version", version="OSN-GS development")
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.add_parser("train", help="Train OSN-GS from a COLMAP scene.")
    subparsers.add_parser("benchmark", help="Run the synthetic NURBS constructor benchmark.")
    subparsers.add_parser("inspect-surface", help="Inspect initial voxel/NURBS surface without training.")
    subparsers.add_parser("stream-server", help="Run the loopback WebSocket snapshot server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    command, remainder = argv[0], argv[1:]
    if command == "train":
        return _delegate_script("train.py", remainder)
    if command == "benchmark":
        return _delegate_main("nurbs_constructor_benchmark.runner", remainder)
    if command == "inspect-surface":
        return _delegate_script("scripts/devtools/inspect_visible_surface.py", remainder)
    if command == "stream-server":
        return _delegate_script("osn_gs/interop/trainer_ws_server.py", remainder)
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
