from __future__ import annotations

import math

import numpy as np
import pandas as pd


def get_value(df: pd.DataFrame | pd.Series | None, col_names, default=np.nan):
    if df is None or (hasattr(df, "empty") and df.empty):
        return default
    candidates = col_names if isinstance(col_names, list) else [col_names]
    target = df.iloc[0] if isinstance(df, pd.DataFrame) else df
    for col in candidates:
        if col in target.index:
            value = target[col]
            if not pd.isna(value):
                return value
    return default


def calculate_piotroski_fscore(
    inc_df: pd.DataFrame,
    bal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    prev_inc_df: pd.DataFrame,
    prev_bal_df: pd.DataFrame,
) -> float:
    if inc_df.empty or bal_df.empty:
        return np.nan
    try:
        score = 0
        net_income = get_value(inc_df, ["Net Income"])
        total_assets = get_value(bal_df, ["Total Assets"])
        roa = net_income / total_assets if total_assets > 0 else np.nan

        if roa > 0:
            score += 1
        operating_cf = get_value(cf_df, ["Net Cash from Operating Activities"])
        if operating_cf > 0:
            score += 1

        if not prev_inc_df.empty and not prev_bal_df.empty:
            prev_roa = get_value(prev_inc_df, ["Net Income"]) / get_value(
                prev_bal_df, ["Total Assets"]
            )
            if roa > prev_roa:
                score += 1

        if operating_cf > net_income:
            score += 1

        total_liab = get_value(bal_df, ["Total Liabilities"])
        debt_ratio = total_liab / total_assets if total_assets > 0 else np.nan
        if not prev_bal_df.empty:
            prev_ratio = get_value(prev_bal_df, ["Total Liabilities"]) / get_value(
                prev_bal_df, ["Total Assets"]
            )
            if debt_ratio < prev_ratio:
                score += 1

        curr_ratio = get_value(bal_df, ["Total Current Assets"]) / get_value(
            bal_df, ["Total Current Liabilities"]
        )
        if not prev_bal_df.empty:
            prev_curr = get_value(prev_bal_df, ["Total Current Assets"]) / get_value(
                prev_bal_df, ["Total Current Liabilities"]
            )
            if curr_ratio > prev_curr:
                score += 1

        gross_margin = get_value(inc_df, ["Gross Profit"]) / get_value(
            inc_df, ["Revenue"]
        )
        if not prev_inc_df.empty:
            prev_margin = get_value(prev_inc_df, ["Gross Profit"]) / get_value(
                prev_inc_df, ["Revenue"]
            )
            if gross_margin > prev_margin:
                score += 1

        asset_turnover = get_value(inc_df, ["Revenue"]) / total_assets
        if not prev_inc_df.empty and not prev_bal_df.empty:
            prev_turnover = get_value(prev_inc_df, ["Revenue"]) / get_value(
                prev_bal_df, ["Total Assets"]
            )
            if asset_turnover > prev_turnover:
                score += 1

        return float(score)
    except Exception:
        return np.nan


def calculate_beneish_mscore(
    inc_df: pd.DataFrame,
    bal_df: pd.DataFrame,
    prev_inc_df: pd.DataFrame,
    prev_bal_df: pd.DataFrame,
) -> float:
    if inc_df.empty or prev_inc_df.empty or bal_df.empty or prev_bal_df.empty:
        return np.nan
    try:
        rev = get_value(inc_df, ["Revenue"])
        rec = get_value(bal_df, ["Accounts & Notes Receivable"], 0)
        cogs = get_value(inc_df, ["Cost of Revenue"])
        ca = get_value(bal_df, ["Total Current Assets"])
        ta = get_value(bal_df, ["Total Assets"])
        ppe = get_value(bal_df, ["Property, Plant & Equipment, Net"], 0)
        dep = get_value(inc_df, ["Depreciation & Amortization"], 0)
        sga = get_value(inc_df, ["Selling, General & Administrative"], 0)
        liab = get_value(bal_df, ["Total Liabilities"])
        ni = get_value(inc_df, ["Net Income"])

        p_rev = get_value(prev_inc_df, ["Revenue"])
        p_rec = get_value(prev_bal_df, ["Accounts & Notes Receivable"], 0)
        p_cogs = get_value(prev_inc_df, ["Cost of Revenue"])
        p_ca = get_value(prev_bal_df, ["Total Current Assets"])
        p_ta = get_value(prev_bal_df, ["Total Assets"])
        p_ppe = get_value(prev_bal_df, ["Property, Plant & Equipment, Net"], 0)
        p_dep = get_value(prev_inc_df, ["Depreciation & Amortization"], 0)
        p_sga = get_value(prev_inc_df, ["Selling, General & Administrative"], 0)
        p_liab = get_value(prev_bal_df, ["Total Liabilities"])

        if p_rev == 0 or rev == 0 or ta == 0 or p_ta == 0:
            return np.nan

        dsri = (rec / rev) / (p_rec / p_rev) if p_rec > 0 else 1
        gmi = ((p_rev - p_cogs) / p_rev) / ((rev - cogs) / rev) if (rev - cogs) > 0 else 1
        aqi = (1 - (ca + ppe) / ta) / (1 - (p_ca + p_ppe) / p_ta) if p_ta > 0 else 1
        sgi = rev / p_rev
        dep_rate = dep / (ppe + dep) if (ppe + dep) > 0 else 0
        p_dep_rate = p_dep / (p_ppe + p_dep) if (p_ppe + p_dep) > 0 else 0
        depi = p_dep_rate / dep_rate if dep_rate > 0 else 1
        sgai = (sga / rev) / (p_sga / p_rev) if p_sga > 0 else 1
        lvgi = (liab / ta) / (p_liab / p_ta) if p_liab > 0 else 1
        tata = (ni - get_value(inc_df, ["Net Income"], ni)) / ta

        dsri = np.clip(dsri, 0.01, 10)
        gmi = np.clip(gmi, 0.01, 10)
        aqi = np.clip(aqi, 0.01, 10)
        sgi = np.clip(sgi, 0.01, 10)
        depi = np.clip(depi, 0.01, 10)
        sgai = np.clip(sgai, 0.01, 10)
        lvgi = np.clip(lvgi, 0.01, 10)

        score = (
            -4.84
            + (0.920 * dsri)
            + (0.528 * gmi)
            + (0.404 * aqi)
            + (0.892 * sgi)
            + (0.115 * depi)
            - (0.172 * sgai)
            + (4.679 * tata)
            - (0.327 * lvgi)
        )
        return float(score)
    except Exception:
        return np.nan


def calculate_ohlson_oscore(
    inc_df: pd.DataFrame,
    bal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    prev_inc_df: pd.DataFrame,
) -> float:
    try:
        ta = get_value(bal_df, ["Total Assets"])
        tl = get_value(bal_df, ["Total Liabilities"])
        wc = get_value(bal_df, ["Total Current Assets"]) - get_value(
            bal_df, ["Total Current Liabilities"]
        )
        cl = get_value(bal_df, ["Total Current Liabilities"])
        ca = get_value(bal_df, ["Total Current Assets"])
        ni = get_value(inc_df, ["Net Income"])
        p_ni = get_value(prev_inc_df, ["Net Income"])
        cfo = get_value(cf_df, ["Net Cash from Operating Activities"])
        if ta <= 0:
            return np.nan
        x1 = math.log(ta)
        x2 = tl / ta
        x3 = wc / ta
        x4 = cl / ca if ca > 0 else 0
        x5 = 1 if tl > ta else 0
        x6 = ni / ta
        x7 = cfo / tl if tl > 0 else 0
        x8 = 1 if (ni < 0 and p_ni < 0) else 0
        x9 = (ni - p_ni) / (abs(ni) + abs(p_ni)) if (abs(ni) + abs(p_ni)) > 0 else 0
        o_score = (
            -1.32
            - 0.407 * x1
            + 6.03 * x2
            - 1.43 * x3
            + 0.0757 * x4
            - 1.72 * x5
            - 2.37 * x6
            - 1.83 * x7
            + 0.285 * x8
            - 0.521 * x9
        )
        return float(math.exp(o_score) / (1 + math.exp(o_score)))
    except Exception:
        return np.nan
