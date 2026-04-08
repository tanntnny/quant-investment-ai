from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class NormalizedResult:
    tensor: np.ndarray
    dates: list[pd.Timestamp]
    tickers: list[str]
    features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "dates": self.dates,
            "tickers": self.tickers,
            "features": self.features,
        }


class QaiNormalizer:
    per_ticker_metrics = [
        "gross_margin",
        "net_margin",
        "roa",
        "asset_turnover",
        "current_ratio",
        "debt_to_equity",
    ]

    def __init__(self) -> None:
        self.global_scaler = RobustScaler()
        self.ticker_scalers: dict[str, RobustScaler] = {}
        self.numeric_features: list[str] | None = None
        self.sorted_metrics: list[str] | None = None
        self.global_medians: dict[str, float] = {}
        self.active_global: list[str] = []
        self.active_ticker: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> NormalizedResult:
        df_clean = df.copy()
        df_clean["report_date"] = pd.to_datetime(df_clean["report_date"])
        self.numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        tickers = sorted(df_clean["ticker"].unique())
        self.active_ticker = [m for m in self.numeric_features if m in self.per_ticker_metrics]
        self.active_global = [m for m in self.numeric_features if m not in self.per_ticker_metrics]

        for col in self.numeric_features:
            if df_clean[col].isna().all():
                df_clean[col] = 0.0
            else:
                self.global_medians[col] = float(df_clean[col].median())
                df_clean[col] = df_clean[col].fillna(self.global_medians[col])

        if self.active_global:
            df_clean[self.active_global] = self.global_scaler.fit_transform(
                df_clean[self.active_global]
            )

        if self.active_ticker:
            for ticker in tickers:
                scaler = RobustScaler()
                idx = df_clean["ticker"] == ticker
                scaled = scaler.fit_transform(df_clean.loc[idx, self.active_ticker])
                df_clean.loc[idx, self.active_ticker] = np.clip(scaled, -5, 5)
                self.ticker_scalers[ticker] = scaler

        q_start = df_clean["report_date"].min().to_period("Q").to_timestamp("Q")
        q_end = df_clean["report_date"].max().to_period("Q").to_timestamp("Q")
        quarter_ends = pd.date_range(start=q_start, end=q_end, freq="QE")

        aligned_chunks = []
        for ticker in tickers:
            t_df = (
                df_clean[df_clean["ticker"] == ticker]
                .sort_values("report_date")
                .set_index("report_date")
            )
            t_aligned = t_df[self.numeric_features].reindex(quarter_ends, method="ffill")
            t_aligned["ticker"] = ticker
            t_aligned.index.name = "report_date"
            aligned_chunks.append(t_aligned.reset_index())

        df_aligned = pd.concat(aligned_chunks, ignore_index=True)
        df_wide = df_aligned.pivot(index="report_date", columns="ticker", values=self.numeric_features)
        df_wide = df_wide.fillna(0)
        df_wide.columns = df_wide.columns.swaplevel(0, 1)
        df_wide = df_wide.sort_index(axis=1)

        dates = df_wide.index.tolist()
        final_tickers = list(df_wide.columns.levels[0])
        self.sorted_metrics = list(df_wide.columns.levels[1])
        tensor = df_wide.to_numpy().reshape(
            (len(dates), len(final_tickers), len(self.sorted_metrics))
        )
        return NormalizedResult(
            tensor=tensor,
            dates=dates,
            tickers=final_tickers,
            features=self.sorted_metrics,
        )
