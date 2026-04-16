from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.alphavantage import _load_dotenv


class FmpError(RuntimeError):
    pass


@dataclass
class FinancialModelingPrep:
    api_key: str | None = None
    env_file: str | Path = ".env"
    api_key_env: str = "FMP_API_KEY"
    base_url: str = "https://financialmodelingprep.com/stable"
    api_calls_per_minute: int | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    _request_timestamps: list[float] = field(default_factory=list, init=False)
    _api_calls_executed: int = field(default=0, init=False)

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | Path = ".env",
        api_key_env: str = "FMP_API_KEY",
        api_calls_per_minute: int | None = None,
        base_url: str = "https://financialmodelingprep.com/stable",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> "FinancialModelingPrep":
        env_values = _load_dotenv(env_file)
        api_key = os.environ.get(api_key_env) or env_values.get(api_key_env)
        return cls(
            api_key=api_key,
            env_file=env_file,
            api_key_env=api_key_env,
            base_url=base_url.rstrip("/"),
            api_calls_per_minute=api_calls_per_minute,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def get_economic_indicator(
        self,
        *,
        name: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"name": name}
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        return self._get_json("/economic-indicators", params=params)

    def get_technical_indicator(
        self,
        *,
        symbol: str,
        indicator: str,
        period_length: int,
        timeframe: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "periodLength": period_length,
            "timeframe": timeframe,
        }
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        return self._get_json(f"/technical-indicators/{indicator}", params=params)

    def get_usage_stats(self) -> dict[str, int | None]:
        return {
            "fmp_api_calls_executed": self._api_calls_executed,
            "fmp_configured_api_calls_per_minute": self.api_calls_per_minute,
        }

    def _get_json(self, path: str, *, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.api_key:
            raise FmpError(
                f"Missing FMP API key. Set {self.api_key_env} in the environment or {self.env_file}."
            )
        request_params = {**params, "apikey": self.api_key}
        query = urllib.parse.urlencode(request_params)
        url = f"{self.base_url}{path}?{query}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
                    self._api_calls_executed += 1
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # pragma: no cover - network error shape varies by platform.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise FmpError(f"FMP request failed for {path}: {exc}") from exc

            if isinstance(payload, dict) and payload.get("Error Message"):
                raise FmpError(str(payload["Error Message"]))
            if isinstance(payload, dict) and payload.get("error"):
                raise FmpError(str(payload["error"]))
            if not isinstance(payload, list):
                raise FmpError(f"Unexpected FMP response for {path}: {type(payload).__name__}")
            return payload

        raise FmpError(f"FMP request failed for {path}: {last_error}")

    def _throttle(self) -> None:
        if not self.api_calls_per_minute or self.api_calls_per_minute <= 0:
            return
        now = time.monotonic()
        window_start = now - 60.0
        self._request_timestamps = [
            timestamp for timestamp in self._request_timestamps if timestamp >= window_start
        ]
        if len(self._request_timestamps) >= self.api_calls_per_minute:
            sleep_seconds = 60.0 - (now - self._request_timestamps[0])
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        self._request_timestamps.append(time.monotonic())
