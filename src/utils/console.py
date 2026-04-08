from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.table import Table


_CONSOLE = Console()


def get_console() -> Console:
    return _CONSOLE


def announce(message: str, *, style: str = "cyan") -> None:
    _CONSOLE.print(message, style=style)


def render_kv_table(
    title: str,
    data: Mapping[str, Any],
    *,
    key_header: str = "Field",
    value_header: str = "Value",
) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column(key_header, style="bold cyan", no_wrap=True)
    table.add_column(value_header, style="white")

    for key, value in data.items():
        table.add_row(str(key), _format_value(value))

    _CONSOLE.print(table)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value is None:
        return "-"
    return str(value)
