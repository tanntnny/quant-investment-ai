#!/usr/bin/env python3
"""Small log helper CLI for common local workflows.

Examples:
    python tools/logs.py latest
    python tools/logs.py latest -n 20 --pattern "*.log"
    python tools/logs.py tail --lines 100
    python tools/logs.py monitor --lines 50 --interval 10
    python tools/logs.py monitor --include-name eval-icnale
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LogFile:
    path: Path
    mtime: float
    size: int


@dataclass
class TailFollower:
    path: Path
    process: subprocess.Popen[str]
    reader_thread: threading.Thread


def discover_log_files(
    log_dir: Path,
    pattern: str | None = None,
    include_name: str | None = None,
) -> list[LogFile]:
    if not log_dir.exists() or not log_dir.is_dir():
        return []

    if pattern:
        paths = [p for p in log_dir.rglob(pattern) if p.is_file()]
    else:
        paths = [p for p in log_dir.rglob("*") if p.is_file()]

    if include_name:
        paths = [p for p in paths if include_name in p.name]

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


def command_monitor(args: argparse.Namespace) -> int:
    log_dir = Path(args.dir)

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: log directory does not exist: {log_dir}", file=sys.stderr)
        return 2

    if args.interval <= 0:
        print("Error: --interval must be greater than 0", file=sys.stderr)
        return 2

    if args.lines <= 0:
        print("Error: --lines must be greater than 0", file=sys.stderr)
        return 2

    def latest_matching_path(pattern: str) -> Path | None:
        files = discover_log_files(log_dir, pattern, args.include_name)
        if not files:
            return None
        return files[0].path

    def stream_tail_output(process: subprocess.Popen[str], label: str) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            print(f"[{label}] {line}", end="")

    def start_follower(path: Path, label: str) -> TailFollower:
        process = subprocess.Popen(
            ["tail", "-n", str(args.lines), "-F", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        reader_thread = threading.Thread(
            target=stream_tail_output,
            args=(process, label),
            daemon=True,
        )
        reader_thread.start()
        return TailFollower(path=path, process=process, reader_thread=reader_thread)

    def stop_follower(follower: TailFollower) -> None:
        if follower.process.poll() is None:
            follower.process.terminate()
            try:
                follower.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                follower.process.kill()
                follower.process.wait(timeout=2)
        follower.reader_thread.join(timeout=1)

    followers: dict[str, TailFollower | None] = {"out": None, "err": None}
    last_missing: dict[str, bool] = {"out": False, "err": False}

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"monitoring {log_dir} (auto-switch every {args.interval:g}s)"
    )

    try:
        while True:
            for key, pattern, label in (("out", "*.out", "OUT"), ("err", "*.err", "ERR")):
                latest = latest_matching_path(pattern)
                current = followers[key]

                if latest is None:
                    if not last_missing[key]:
                        if args.include_name:
                            print(
                                f"[{label}] no {pattern} files containing "
                                f"{args.include_name!r} found in {log_dir}"
                            )
                        else:
                            print(f"[{label}] no {pattern} files found in {log_dir}")
                        last_missing[key] = True
                    continue

                last_missing[key] = False

                if current is None or current.path != latest:
                    if current is not None:
                        print(f"[{label}] switching to {latest}")
                        stop_follower(current)
                    else:
                        print(f"[{label}] following {latest}")
                    followers[key] = start_follower(latest, label)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        for follower in followers.values():
            if follower is not None:
                stop_follower(follower)
        print("Stopped monitor.")
    return 0


def add_common_options(parser: argparse.ArgumentParser, include_pattern: bool = True) -> None:
    parser.add_argument(
        "-d",
        "--dir",
        default="logs",
        help="log directory to inspect (default: logs)",
    )
    if include_pattern:
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

    monitor = subparsers.add_parser(
        "monitor",
        help="loop print newest .out and .err files",
    )
    add_common_options(monitor, include_pattern=False)
    monitor.add_argument(
        "--include-name",
        default=None,
        help="only follow log files whose filename contains this text",
    )
    monitor.add_argument("--lines", type=int, default=50, help="lines to print")
    monitor.add_argument(
        "--interval",
        type=float,
        default=10,
        help="seconds between refreshes (default: 10)",
    )
    monitor.set_defaults(handler=command_monitor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
