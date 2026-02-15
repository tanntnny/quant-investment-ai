#!/usr/bin/env python3
"""Project helper command entrypoint.

Examples:
    python tools/dev.py logs latest
    python tools/dev.py logs tail --lines 100
"""

from __future__ import annotations

import argparse

import logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Developer helper commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    logs_parser = subparsers.add_parser("logs", help="log helper commands")
    logs_parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for logs tool")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "logs":
        return logs.main(args.args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
