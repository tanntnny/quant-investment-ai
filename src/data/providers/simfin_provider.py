from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.providers.base import RawTickerData
from src.utils.alphavantage import _load_dotenv
from src.utils.console import announce, render_kv_table

logger = logging.getLogger(__name__)


def _announce(message: str) -> None:
    logger.info(message)
    announce(message)


def ticker_candidates(ticker: str) -> list[str]:
    candidates = [ticker]
    if "-" in ticker:
        candidates.append(ticker.replace("-", "."))
    if "." in ticker:
        candidates.append(ticker.replace(".", "-"))
    return list(dict.fromkeys(candidates))


def pick_available_ticker(requested_ticker: str, index_obj: pd.Index) -> str | None:
    if isinstance(index_obj, pd.MultiIndex):
        available: Iterable[str] = index_obj.get_level_values(0)
    else:
        available = index_obj
    available_set = set(available)
    for candidate in ticker_candidates(requested_ticker):
        if candidate in available_set:
            return candidate
    return None


class SimFinProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        env_file: str | Path = ".env",
        api_key_env: str = "SIMFIN_API_KEY",
        raw_data_dir: Path,
        cache_template: str,
        historical_price_years: int,
        market: str,
        fundamentals_variant: str,
        prices_variant: str,
    ) -> None:
        self.api_key = api_key
        self.env_file = Path(env_file)
        self.api_key_env = api_key_env
        self.raw_data_dir = raw_data_dir
        self.cache_template = cache_template
        self.historical_price_years = historical_price_years
        self.market = market
        self.fundamentals_variant = fundamentals_variant
        self.prices_variant = prices_variant

    def _cache_path(self, ticker: str) -> Path:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_data_dir / self.cache_template.format(ticker=ticker)

    @staticmethod
    def _is_stale(cached: object) -> bool:
        if not isinstance(cached, RawTickerData):
            return True
        return cached.income.empty or cached.balance.empty

    def _resolve_api_key(self) -> str:
        env_values = _load_dotenv(self.env_file)
        return os.environ.get(self.api_key_env) or env_values.get(self.api_key_env) or self.api_key or "free"

    def _configure_simfin(self) -> object:
        try:
            import simfin as sf
        except ImportError as exc:
            raise RuntimeError(
                "SimFin is not installed. Install project dependencies before running prepare_data."
            ) from exc

        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*date_parser.*"
        )
        logging.getLogger("simfin").setLevel(logging.ERROR)
        sf.set_api_key(self._resolve_api_key())
        sf.set_data_dir(self.raw_data_dir)
        return sf

    def fetch_raw_data(self, ticker: str, force_refresh: bool = False) -> RawTickerData:
        cache_path = self._cache_path(ticker)
        _announce(
            f"[SimFin] Loading raw data for ticker={ticker} force_refresh={force_refresh} "
            f"cache_path={cache_path}"
        )
        if not force_refresh and cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if not self._is_stale(cached):
                _announce(f"[SimFin] Loaded cached raw data for ticker={ticker}")
                render_kv_table(
                    "SimFin Cache Hit",
                    {
                        "Ticker": ticker,
                        "Cache path": cache_path,
                    },
                )
                return cached
            _announce(f"[SimFin] Refreshing stale raw data for ticker={ticker}")

        sf = self._configure_simfin()
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=self.historical_price_years)

        income_df = sf.load_income(variant=self.fundamentals_variant, market=self.market)
        balance_df = sf.load_balance(
            variant=self.fundamentals_variant, market=self.market
        )
        cashflow_df = sf.load_cashflow(
            variant=self.fundamentals_variant, market=self.market
        )
        companies_df = sf.load_companies(market=self.market)
        price_df = sf.load_shareprices(variant=self.prices_variant, market=self.market)

        income_ticker = pick_available_ticker(ticker, income_df.index)
        balance_ticker = pick_available_ticker(ticker, balance_df.index)
        cashflow_ticker = pick_available_ticker(ticker, cashflow_df.index)
        price_ticker = pick_available_ticker(ticker, price_df.index)
        info_ticker = pick_available_ticker(ticker, companies_df.index)

        ticker_income = income_df.loc[income_ticker] if income_ticker else pd.DataFrame()
        ticker_balance = (
            balance_df.loc[balance_ticker] if balance_ticker else pd.DataFrame()
        )
        ticker_cashflow = (
            cashflow_df.loc[cashflow_ticker] if cashflow_ticker else pd.DataFrame()
        )
        ticker_price = price_df.loc[price_ticker] if price_ticker else pd.DataFrame()
        if not ticker_price.empty:
            ticker_price = ticker_price.loc[start_date:end_date]

        info = companies_df.loc[info_ticker].to_dict() if info_ticker else {}
        raw_data = RawTickerData(
            income=ticker_income,
            balance=ticker_balance,
            cashflow=ticker_cashflow,
            info=info,
            history=ticker_price,
        )
        with cache_path.open("wb") as handle:
            pickle.dump(raw_data, handle)
        _announce(
            f"[SimFin] Fetched and cached raw data for ticker={ticker} "
            f"history_rows={len(ticker_price)} income_rows={len(ticker_income)} "
            f"balance_rows={len(ticker_balance)} cashflow_rows={len(ticker_cashflow)}"
        )
        render_kv_table(
            "SimFin Fetch Result",
            {
                "Ticker": ticker,
                "History rows": len(ticker_price),
                "Income rows": len(ticker_income),
                "Balance rows": len(ticker_balance),
                "Cashflow rows": len(ticker_cashflow),
                "Cache path": cache_path,
            },
        )
        return raw_data
