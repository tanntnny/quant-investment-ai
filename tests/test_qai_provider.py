from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.providers.base import RawTickerData
from src.data.providers.simfin_provider import (
    SimFinProvider,
    pick_available_ticker,
    ticker_candidates,
)


class _FakeSimFin:
    def __init__(self) -> None:
        self.calls = 0
        idx = pd.MultiIndex.from_tuples(
            [("BRK.B", pd.Timestamp("2024-03-31"))], names=["Ticker", "Report Date"]
        )
        self._income = pd.DataFrame(
            [{"Revenue": 10.0, "Net Income": 2.0, "Shares (Diluted)": 1.0}],
            index=idx,
        )
        self._balance = pd.DataFrame(
            [{"Total Assets": 20.0, "Total Liabilities": 5.0, "Total Equity": 15.0}],
            index=idx,
        )
        self._cashflow = pd.DataFrame(
            [{"Net Cash from Operating Activities": 3.0}],
            index=idx,
        )
        price_idx = pd.MultiIndex.from_tuples(
            [("BRK.B", pd.Timestamp("2024-03-31"))], names=["Ticker", "Date"]
        )
        self._prices = pd.DataFrame([{"Close": 100.0}], index=price_idx)
        self._companies = pd.DataFrame(
            [{"Company Name": "Berkshire Hathaway"}], index=pd.Index(["BRK.B"], name="Ticker")
        )

    def load_income(self, **kwargs):
        self.calls += 1
        return self._income

    def load_balance(self, **kwargs):
        return self._balance

    def load_cashflow(self, **kwargs):
        return self._cashflow

    def load_shareprices(self, **kwargs):
        return self._prices

    def load_companies(self, **kwargs):
        return self._companies


def test_ticker_candidates_and_alias_match() -> None:
    assert ticker_candidates("BRK-B") == ["BRK-B", "BRK.B"]
    index = pd.MultiIndex.from_tuples([("BRK.B", "2024-03-31")])
    assert pick_available_ticker("BRK-B", index) == "BRK.B"


def test_provider_uses_cache_after_first_fetch(tmp_path: Path, monkeypatch) -> None:
    provider = SimFinProvider(
        api_key=None,
        env_file=tmp_path / ".env",
        api_key_env="SIMFIN_API_KEY",
        raw_data_dir=tmp_path,
        cache_template="simfin_raw_{ticker}.pkl",
        historical_price_years=10,
        market="us",
        fundamentals_variant="quarterly",
        prices_variant="daily",
    )
    fake_simfin = _FakeSimFin()
    monkeypatch.setattr(provider, "_configure_simfin", lambda: fake_simfin)

    first = provider.fetch_raw_data("BRK-B")
    second = provider.fetch_raw_data("BRK-B")

    assert isinstance(first, RawTickerData)
    assert first.info["Company Name"] == "Berkshire Hathaway"
    assert second.info["Company Name"] == "Berkshire Hathaway"
    assert fake_simfin.calls == 1


def test_provider_reads_api_key_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SIMFIN_API_KEY=test-key\n")
    provider = SimFinProvider(
        api_key=None,
        env_file=env_file,
        api_key_env="SIMFIN_API_KEY",
        raw_data_dir=tmp_path,
        cache_template="simfin_raw_{ticker}.pkl",
        historical_price_years=10,
        market="us",
        fundamentals_variant="quarterly",
        prices_variant="daily",
    )

    assert provider._resolve_api_key() == "test-key"
