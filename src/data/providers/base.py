from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import pandas as pd


@dataclass
class RawTickerData:
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    info: Dict[str, Any]
    history: pd.DataFrame


class FinancialDataProvider(Protocol):
    def fetch_raw_data(self, ticker: str, force_refresh: bool = False) -> RawTickerData:
        ...
