#!/usr/bin/env python3
"""Small log helper CLI for common local workflows.

Examples:
    python tools/logs.py latest
    python tools/logs.py latest -n 20 --pattern "*.log"
    python tools/logs.py tail --lines 100
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LogFile:
    path: Path
    mtime: float
    size: int


def discover_log_files(log_dir: Path, pattern: str | None = None) -> list[LogFile]:
    if not log_dir.exists() or not log_dir.is_dir():
        return []

    if pattern:
        paths = [p for p in log_dir.rglob(pattern) if p.is_file()]
    else:
        paths = [p for p in log_dir.rglob("*") if p.is_file()]

    files = [LogFile(path=p, mtime=p.stat().st_mtime, size=p.stat().st_size) for p in paths]
    files.sort(key=lambda item: item.mtime, reverse=True)
    return files


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def command_latest(args: argparse.Namespace) -> int:
    log_dir = Path(args.dir)
    files = discover_log_files(log_dir, args.pattern)

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: log directory does not exist: {log_dir}", file=sys.stderr)
        return 2

    if not files:
        print(f"No files found in {log_dir}")
        return 0

    top = files[: args.count]
    for item in top:
        timestamp = datetime.fromtimestamp(item.mtime).strftime("%Y-%m-%d %H:%M:%S")
        rendered_path = str(item.path.resolve()) if args.absolute else str(item.path)
        print(f"{timestamp}  {format_bytes(item.size):>8}  {rendered_path}")
    return 0


def command_tail(args: argparse.Namespace) -> int:
    log_dir = Path(args.dir)
    files = discover_log_files(log_dir, args.pattern)

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: log directory does not exist: {log_dir}", file=sys.stderr)
        return 2

    if not files:
        print(f"No files found in {log_dir}")
        return 0

    latest_file = files[0].path
    print(f"==> {latest_file} <==")

    # Read once and slice from the end to keep the implementation simple.
    lines = latest_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.lines :]:
        print(line)
    return 0


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--dir",
        default="logs",
        help="log directory to inspect (default: logs)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default=None,
        help='glob pattern to filter logs, e.g. "*.log"',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper commands for project log files")

    subparsers = parser.add_subparsers(dest="command", required=True)

    latest = subparsers.add_parser("latest", help="show newest log files")
    add_common_options(latest)
    latest.add_argument("-n", "--count", type=int, default=10, help="files to print")
    latest.add_argument(
        "--absolute",
        action="store_true",
        help="print absolute file paths",
    )
    latest.set_defaults(handler=command_latest)

    tail = subparsers.add_parser("tail", help="print tail of newest log file")
    add_common_options(tail)
    tail.add_argument("--lines", type=int, default=50, help="lines to print")
    tail.set_defaults(handler=command_tail)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
