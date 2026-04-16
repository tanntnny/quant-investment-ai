from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import features as fe
from src.data.providers.base import FinancialDataProvider, RawTickerData

FINAL_FEATURE_COLUMNS = [
    "piotroski_fscore",
    "beneish_mscore",
    "ohlson_oscore",
    "altman_zscore",
    "gross_margin",
    "net_margin",
    "roa",
    "asset_turnover",
    "current_ratio",
    "debt_to_equity",
    "fcf_yield",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "rd_intensity",
    "capex_to_sales",
    "pe_ratio",
    "ps_ratio",
    "ev_ebitda",
    "buyback_yield",
]

logger = logging.getLogger(__name__)


def _announce(message: str) -> None:
    logger.info(message)
    print(message)


@dataclass
class DatasetBuildConfig:
    ttm_window: int
    zscore_window: int
    zscore_min_periods: int
    winsorize_lower_quantile: float
    winsorize_upper_quantile: float
    market_cap_min: float


def build_fundamental_dataset(
    tickers: Sequence[str],
    provider: FinancialDataProvider,
    config: DatasetBuildConfig,
    force_refresh: bool = False,
) -> pd.DataFrame:
    _announce(
        f"[DatasetBuilder] Building fundamental dataset for {len(tickers)} ticker(s). "
        f"force_refresh={force_refresh}"
    )
    all_raw_records: list[dict[str, object]] = []
    raw_by_ticker: dict[str, RawTickerData] = {}

    for ticker in tickers:
        _announce(f"[DatasetBuilder] Fetching source data for ticker={ticker}")
        raw = provider.fetch_raw_data(ticker, force_refresh=force_refresh)
        raw_by_ticker[ticker] = raw
        inc_df, bal_df, cf_df, history_df = (
            raw.income,
            raw.balance,
            raw.cashflow,
            raw.history,
        )
        if inc_df is None or bal_df is None or inc_df.empty or bal_df.empty:
            _announce(
                f"[DatasetBuilder] Skipping ticker={ticker} because income/balance data is missing."
            )
            continue

        price_col = next((c for c in history_df.columns if "Close" in c), None)
        prices = history_df[price_col] if price_col else None

        for curr_idx in inc_df.index.unique():
            q_inc = inc_df.loc[[curr_idx]]
            q_bal = bal_df.loc[[curr_idx]] if curr_idx in bal_df.index else pd.DataFrame()
            q_cf = cf_df.loc[[curr_idx]] if curr_idx in cf_df.index else pd.DataFrame()
            if q_bal.empty:
                continue

            price_at_time = prices.asof(curr_idx) if prices is not None else np.nan
            shares = fe.get_value(q_inc, ["Shares (Diluted)", "Shares (Basic)"])
            mkt_cap = price_at_time * shares

            change_ppe = fe.get_value(q_cf, ["Change in Fixed Assets & Intangibles"], default=0)
            capex = -change_ppe if not np.isnan(change_ppe) else np.nan
            op_cf = fe.get_value(q_cf, ["Net Cash from Operating Activities"], default=np.nan)
            if np.isnan(capex):
                capex = 0
            fcf = op_cf - capex if not np.isnan(op_cf) else np.nan

            all_raw_records.append(
                {
                    "ticker": ticker,
                    "report_date": curr_idx,
                    "mkt_cap": mkt_cap,
                    "rev": fe.get_value(q_inc, ["Revenue"]),
                    "ni": fe.get_value(q_inc, ["Net Income"]),
                    "gp": fe.get_value(q_inc, ["Gross Profit"]),
                    "op_inc": fe.get_value(q_inc, ["Operating Income (Loss)"]),
                    "capex": capex if not np.isnan(capex) else 0,
                    "rnd": fe.get_value(q_inc, ["Research & Development"], default=0),
                    "fcf": fcf if not np.isnan(fcf) else 0,
                    "div_paid": fe.get_value(q_cf, ["Dividends Paid"], default=0),
                    "repurchase": fe.get_value(
                        q_cf, ["Cash from (Repurchase of) Equity"], default=0
                    ),
                    "assets": fe.get_value(q_bal, ["Total Assets"]),
                    "liabilities": fe.get_value(q_bal, ["Total Liabilities"]),
                    "curr_assets": fe.get_value(q_bal, ["Total Current Assets"]),
                    "curr_liab": fe.get_value(q_bal, ["Total Current Liabilities"]),
                    "equity": fe.get_value(q_bal, ["Total Equity"]),
                    "ebit": fe.get_value(q_inc, ["Operating Income (Loss)"]),
                    "dep_amort": fe.get_value(
                        q_inc, ["Depreciation & Amortization"], default=0
                    ),
                }
            )
        _announce(
            f"[DatasetBuilder] Extracted raw quarterly records for ticker={ticker} "
            f"records_added={sum(1 for record in all_raw_records if record['ticker'] == ticker)}"
        )

    df_raw = pd.DataFrame(all_raw_records)
    if df_raw.empty:
        _announce("[DatasetBuilder] No raw records available after source fetch.")
        return pd.DataFrame(columns=["ticker", "report_date", *FINAL_FEATURE_COLUMNS])

    for col in ["capex", "rnd", "fcf", "div_paid", "repurchase", "dep_amort"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(0)

    processed_chunks: list[pd.DataFrame] = []
    for ticker, sub in df_raw.groupby("ticker"):
        _announce(
            f"[DatasetBuilder] Computing derived features for ticker={ticker} input_rows={len(sub)}"
        )
        sub = sub.sort_values("report_date").copy()

        flow_cols = ["rev", "ni", "gp", "op_inc", "capex", "rnd", "fcf", "div_paid", "repurchase"]
        for col in flow_cols:
            sub[f"ttm_{col}"] = sub[col].rolling(
                window=config.ttm_window, min_periods=1
            ).sum()

        stock_cols = ["assets", "liabilities", "curr_assets", "curr_liab", "equity"]
        for col in stock_cols:
            sub[f"ttm_{col}"] = sub[col].rolling(
                window=config.ttm_window, min_periods=1
            ).mean()

        sub["mkt_cap"] = np.where(sub["mkt_cap"] < config.market_cap_min, np.nan, sub["mkt_cap"])
        sub["revenue_growth_yoy"] = sub["ttm_rev"].pct_change(periods=1)
        sub["eps_growth_yoy"] = sub["ttm_ni"].pct_change(periods=1)
        sub["gross_margin"] = np.where(sub["ttm_rev"] != 0, sub["ttm_gp"] / sub["ttm_rev"], np.nan)
        sub["net_margin"] = np.where(sub["ttm_rev"] != 0, sub["ttm_ni"] / sub["ttm_rev"], np.nan)
        sub["roa"] = np.where(sub["ttm_assets"] != 0, sub["ttm_ni"] / sub["ttm_assets"], np.nan)
        sub["asset_turnover"] = np.where(
            sub["ttm_assets"] != 0, sub["ttm_rev"] / sub["ttm_assets"], np.nan
        )
        sub["current_ratio"] = np.where(
            sub["ttm_curr_liab"] != 0, sub["ttm_curr_assets"] / sub["ttm_curr_liab"], np.nan
        )
        sub["debt_to_equity"] = np.where(
            sub["ttm_equity"] != 0, sub["ttm_liabilities"] / sub["ttm_equity"], np.nan
        )
        sub["fcf_yield"] = np.where(sub["mkt_cap"] != 0, sub["ttm_fcf"] / sub["mkt_cap"], np.nan)
        sub["rd_intensity"] = np.where(sub["ttm_rev"] != 0, sub["ttm_rnd"] / sub["ttm_rev"], np.nan)
        sub["capex_to_sales"] = np.where(
            sub["ttm_rev"] != 0, sub["ttm_capex"] / sub["ttm_rev"], np.nan
        )
        sub["pe_ratio"] = np.where(sub["ttm_ni"] != 0, sub["mkt_cap"] / sub["ttm_ni"], np.nan)
        sub["ps_ratio"] = np.where(sub["ttm_rev"] != 0, sub["mkt_cap"] / sub["ttm_rev"], np.nan)
        ebitda = sub["ebit"] + sub["dep_amort"]
        sub["ev_ebitda"] = np.where(
            ebitda != 0, (sub["mkt_cap"] + sub["ttm_liabilities"]) / ebitda, np.nan
        )
        sub["buyback_yield"] = np.where(
            sub["mkt_cap"] != 0,
            (sub["ttm_repurchase"].abs() + sub["ttm_div_paid"].abs()) / sub["mkt_cap"],
            np.nan,
        )
        sub["piotroski_fscore"] = np.nan
        sub["beneish_mscore"] = np.nan
        sub["ohlson_oscore"] = np.nan

        raw = raw_by_ticker[ticker]
        inc_df = raw.income.copy()
        bal_df = raw.balance.copy()
        cf_df = raw.cashflow.copy()
        if not inc_df.empty:
            inc_df.index = pd.to_datetime(inc_df.index)
        if not bal_df.empty:
            bal_df.index = pd.to_datetime(bal_df.index)
        if not cf_df.empty:
            cf_df.index = pd.to_datetime(cf_df.index)

        dates_list = sorted(pd.to_datetime(sub["report_date"].unique()))
        for idx, row in sub.iterrows():
            curr_date = pd.to_datetime(row["report_date"])
            curr_pos = dates_list.index(curr_date) if curr_date in dates_list else None
            if curr_pos is None or curr_pos == 0:
                continue
            prev_date = dates_list[curr_pos - 1]
            q_inc = inc_df.loc[[curr_date]] if curr_date in inc_df.index else pd.DataFrame()
            q_bal = bal_df.loc[[curr_date]] if curr_date in bal_df.index else pd.DataFrame()
            q_cf = cf_df.loc[[curr_date]] if curr_date in cf_df.index else pd.DataFrame()
            p_inc = inc_df.loc[[prev_date]] if prev_date in inc_df.index else pd.DataFrame()
            p_bal = bal_df.loc[[prev_date]] if prev_date in bal_df.index else pd.DataFrame()

            if not q_inc.empty and not q_bal.empty and not p_inc.empty and not p_bal.empty:
                sub.loc[idx, "piotroski_fscore"] = fe.calculate_piotroski_fscore(
                    q_inc, q_bal, q_cf, p_inc, p_bal
                )
                sub.loc[idx, "beneish_mscore"] = fe.calculate_beneish_mscore(
                    q_inc, q_bal, p_inc, p_bal
                )
                sub.loc[idx, "ohlson_oscore"] = fe.calculate_ohlson_oscore(
                    q_inc, q_bal, q_cf, p_inc
                )

        sub["altman_zscore"] = (
            (1.2 * (sub["ttm_curr_assets"] - sub["ttm_curr_liab"]) / sub["ttm_assets"])
            + (1.4 * sub["ttm_ni"] / sub["ttm_assets"])
        )
        processed_chunks.append(sub)

    df_final = pd.concat(processed_chunks)
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    winsorise_cols = [
        "fcf_yield",
        "buyback_yield",
        "pe_ratio",
        "ps_ratio",
        "ev_ebitda",
        "beneish_mscore",
        "altman_zscore",
        "gross_margin",
        "net_margin",
        "roa",
        "asset_turnover",
        "current_ratio",
        "debt_to_equity",
        "revenue_growth_yoy",
        "eps_growth_yoy",
    ]
    for col in winsorise_cols:
        if col in df_final.columns:
            lo = df_final[col].quantile(config.winsorize_lower_quantile)
            hi = df_final[col].quantile(config.winsorize_upper_quantile)
            df_final[col] = df_final[col].clip(lo, hi)

    def rolling_zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(config.zscore_window, min_periods=config.zscore_min_periods)
        return (series - roll.mean()) / roll.std().replace(0, np.nan)

    def historical_percentile(series: pd.Series) -> pd.Series:
        return series.expanding(min_periods=1).apply(
            lambda values: (values < values.iloc[-1]).sum() / len(values)
        )

    for _, mask in df_final.groupby("ticker").groups.items():
        for col in ["revenue_growth_yoy", "eps_growth_yoy", "rd_intensity", "capex_to_sales"]:
            df_final.loc[mask, col] = rolling_zscore(df_final.loc[mask, col])
        for col in ["pe_ratio", "ps_ratio"]:
            df_final.loc[mask, col] = historical_percentile(df_final.loc[mask, col])

    _announce(
        f"[DatasetBuilder] Completed dataset build. output_rows={len(df_final)} "
        f"output_tickers={df_final['ticker'].nunique() if not df_final.empty else 0}"
    )
    return df_final[["ticker", "report_date", *FINAL_FEATURE_COLUMNS]].reset_index(drop=True)
