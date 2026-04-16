from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

from src.utils.console import announce, render_kv_table


DEFAULT_ENV_FILE = ".env"
DEFAULT_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"

logger = logging.getLogger(__name__)


class AlphaVantageError(RuntimeError):
    pass


def _load_dotenv(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _format_multi_value(value: str | Iterable[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value

    items = [item.strip() for item in value if item and item.strip()]
    return ",".join(items) if items else None


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(params)
    if "apikey" in redacted:
        redacted["apikey"] = "***"
    return redacted


@dataclass(slots=True)
class AlphaVantage:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    timeout_seconds: int = 30
    api_calls_per_minute: int | None = None
    total_request_count: int = 0
    total_wait_seconds: float = 0.0
    _last_request_started_at: float | None = None

    @classmethod
    def from_env(
        cls,
        env_path: str | Path = DEFAULT_ENV_FILE,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        api_calls_per_minute: int | None = None,
    ) -> "AlphaVantage":
        env_path = Path(env_path)
        env_values = _load_dotenv(env_path)
        api_key = os.environ.get(api_key_env) or env_values.get(api_key_env)

        if not api_key:
            raise AlphaVantageError(
                f"Missing {api_key_env}. Set it in the environment or in {env_path}."
            )

        return cls(api_key=api_key, api_calls_per_minute=api_calls_per_minute)

    def get_news_sentiment(
        self,
        *,
        tickers: str | Iterable[str] | None = None,
        topics: str | Iterable[str] | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        sort: str = "LATEST",
        limit: int | None = 50,
    ) -> dict[str, Any]:
        normalized_sort = sort.upper()
        if normalized_sort not in {"LATEST", "EARLIEST", "RELEVANCE"}:
            raise ValueError("sort must be one of: LATEST, EARLIEST, RELEVANCE")
        if limit is not None and not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

        params: dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "sort": normalized_sort,
        }

        optional_params = {
            "tickers": _format_multi_value(tickers),
            "topics": _format_multi_value(topics),
            "time_from": time_from,
            "time_to": time_to,
            "limit": limit,
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})

        return self._get(params)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        self._wait_for_cooldown()
        url = f"{self.base_url}?{urlencode(params)}"
        safe_params = _redact_params(params)
        logger.info("Calling Alpha Vantage endpoint with params=%s", safe_params)
        render_kv_table("Alpha Vantage Request", safe_params, key_header="Param", value_header="Value")
        with urlopen(url, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.total_request_count += 1

        if not isinstance(payload, dict):
            raise AlphaVantageError("Unexpected Alpha Vantage response format.")

        error_message = payload.get("Error Message") or payload.get("Information") or payload.get(
            "Note"
        )
        if error_message:
            raise AlphaVantageError(
                f"{error_message} request_params={safe_params}"
            )

        logger.info(
            "Alpha Vantage request completed successfully. total_request_count=%s",
            self.total_request_count,
        )
        render_kv_table("Alpha Vantage Usage", self.get_usage_stats())
        return payload

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "api_calls_per_minute": self.api_calls_per_minute,
            "total_request_count": self.total_request_count,
            "total_wait_seconds": round(self.total_wait_seconds, 3),
        }

    def _wait_for_cooldown(self) -> None:
        if not self.api_calls_per_minute or self.api_calls_per_minute <= 0:
            self._last_request_started_at = time.monotonic()
            return

        min_interval_seconds = 60.0 / self.api_calls_per_minute
        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            remaining = min_interval_seconds - elapsed
            if remaining > 0:
                message = (
                    "Alpha Vantage cooldown active. "
                    f"Waiting {remaining:.2f}s to respect "
                    f"{self.api_calls_per_minute} call(s)/minute."
                )
                logger.info(message)
                announce(message, style="yellow")
                time.sleep(remaining)
                self.total_wait_seconds += remaining

        self._last_request_started_at = time.monotonic()
