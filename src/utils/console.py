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


def render_records_table(
    title: str,
    rows: list[Mapping[str, Any]],
    *,
    columns: list[tuple[str, str]] | None = None,
) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")

    resolved_columns = columns
    if resolved_columns is None:
        keys = list(rows[0].keys()) if rows else []
        resolved_columns = [(key, key) for key in keys]

    for _, header in resolved_columns:
        table.add_column(header, style="white")

    for row in rows:
        table.add_row(*[_format_value(row.get(key)) for key, _ in resolved_columns])

    _CONSOLE.print(table)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value is None:
        return "-"
    return str(value)
