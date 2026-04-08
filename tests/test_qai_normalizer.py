from __future__ import annotations

import pandas as pd

from src.data.normalization import QaiNormalizer


def test_qai_normalizer_returns_expected_tensor_shape() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": "2024-03-31",
                "gross_margin": 0.40,
                "net_margin": 0.10,
                "roa": 0.05,
                "asset_turnover": 0.8,
                "current_ratio": 1.2,
                "debt_to_equity": 0.4,
                "piotroski_fscore": 7.0,
                "beneish_mscore": -2.0,
                "ohlson_oscore": 0.2,
                "altman_zscore": 3.0,
                "fcf_yield": 0.03,
                "revenue_growth_yoy": 0.1,
                "eps_growth_yoy": 0.1,
                "rd_intensity": 0.05,
                "capex_to_sales": 0.03,
                "pe_ratio": 0.6,
                "ps_ratio": 0.4,
                "ev_ebitda": 12.0,
                "buyback_yield": 0.01,
            },
            {
                "ticker": "AAA",
                "report_date": "2024-06-30",
                "gross_margin": 0.42,
                "net_margin": 0.11,
                "roa": 0.06,
                "asset_turnover": 0.82,
                "current_ratio": 1.25,
                "debt_to_equity": 0.39,
                "piotroski_fscore": 8.0,
                "beneish_mscore": -1.9,
                "ohlson_oscore": 0.19,
                "altman_zscore": 3.1,
                "fcf_yield": 0.04,
                "revenue_growth_yoy": 0.12,
                "eps_growth_yoy": 0.11,
                "rd_intensity": 0.06,
                "capex_to_sales": 0.04,
                "pe_ratio": 0.7,
                "ps_ratio": 0.45,
                "ev_ebitda": 11.5,
                "buyback_yield": 0.015,
            },
            {
                "ticker": "BBB",
                "report_date": "2024-03-31",
                "gross_margin": 0.30,
                "net_margin": 0.08,
                "roa": 0.04,
                "asset_turnover": 0.7,
                "current_ratio": 1.1,
                "debt_to_equity": 0.5,
                "piotroski_fscore": 6.0,
                "beneish_mscore": -2.1,
                "ohlson_oscore": 0.25,
                "altman_zscore": 2.8,
                "fcf_yield": 0.02,
                "revenue_growth_yoy": 0.08,
                "eps_growth_yoy": 0.09,
                "rd_intensity": 0.04,
                "capex_to_sales": 0.02,
                "pe_ratio": 0.55,
                "ps_ratio": 0.35,
                "ev_ebitda": 13.0,
                "buyback_yield": 0.005,
            },
            {
                "ticker": "BBB",
                "report_date": "2024-06-30",
                "gross_margin": 0.31,
                "net_margin": 0.07,
                "roa": 0.041,
                "asset_turnover": 0.72,
                "current_ratio": 1.15,
                "debt_to_equity": 0.51,
                "piotroski_fscore": 6.0,
                "beneish_mscore": -2.05,
                "ohlson_oscore": 0.24,
                "altman_zscore": 2.7,
                "fcf_yield": 0.022,
                "revenue_growth_yoy": 0.09,
                "eps_growth_yoy": 0.1,
                "rd_intensity": 0.041,
                "capex_to_sales": 0.025,
                "pe_ratio": 0.56,
                "ps_ratio": 0.36,
                "ev_ebitda": 12.8,
                "buyback_yield": 0.006,
            },
        ]
    )

    result = QaiNormalizer().fit_transform(df)

    assert result.tensor.shape == (2, 2, 19)
    assert result.tickers == ["AAA", "BBB"]
    assert len(result.features) == 19
