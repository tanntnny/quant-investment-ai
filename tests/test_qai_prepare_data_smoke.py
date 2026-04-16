from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.data.providers.base import RawTickerData
from src.preparers.qai_preparer import (
    QaiPreparer,
    _allocate_window_limits,
    _build_story_request_windows,
)


class FakeProvider:
    def fetch_raw_data(self, ticker: str, force_refresh: bool = False) -> RawTickerData:
        report_dates = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]
        index = pd.Index(report_dates, name="Report Date")
        price_index = pd.Index(
            [pd.Timestamp("2024-03-29"), pd.Timestamp("2024-06-28")], name="Date"
        )
        income = pd.DataFrame(
            [
                {
                    "Revenue": 100.0,
                    "Net Income": 10.0,
                    "Gross Profit": 40.0,
                    "Operating Income (Loss)": 15.0,
                    "Research & Development": 5.0,
                    "Shares (Diluted)": 10.0,
                    "Depreciation & Amortization": 2.0,
                },
                {
                    "Revenue": 120.0,
                    "Net Income": 14.0,
                    "Gross Profit": 50.0,
                    "Operating Income (Loss)": 18.0,
                    "Research & Development": 6.0,
                    "Shares (Diluted)": 10.0,
                    "Depreciation & Amortization": 2.5,
                },
            ],
            index=index,
        )
        balance = pd.DataFrame(
            [
                {
                    "Total Assets": 200.0,
                    "Total Liabilities": 80.0,
                    "Total Current Assets": 90.0,
                    "Total Current Liabilities": 40.0,
                    "Total Equity": 120.0,
                },
                {
                    "Total Assets": 210.0,
                    "Total Liabilities": 78.0,
                    "Total Current Assets": 95.0,
                    "Total Current Liabilities": 38.0,
                    "Total Equity": 132.0,
                },
            ],
            index=index,
        )
        cashflow = pd.DataFrame(
            [
                {
                    "Net Cash from Operating Activities": 20.0,
                    "Change in Fixed Assets & Intangibles": -8.0,
                    "Dividends Paid": -1.0,
                    "Cash from (Repurchase of) Equity": -2.0,
                },
                {
                    "Net Cash from Operating Activities": 21.0,
                    "Change in Fixed Assets & Intangibles": -9.0,
                    "Dividends Paid": -1.0,
                    "Cash from (Repurchase of) Equity": -2.5,
                },
            ],
            index=index,
        )
        history = pd.DataFrame(
            [{"Close": 120000.0}, {"Close": 130000.0}],
            index=price_index,
        )
        return RawTickerData(
            income=income,
            balance=balance,
            cashflow=cashflow,
            info={"ticker": ticker},
            history=history,
        )


class FakeStoryClient:
    def __init__(self) -> None:
        self.calls = []

    def get_news_sentiment(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "feed": [
                {
                    "title": "AAA earnings beat",
                    "summary": "Quarterly results beat expectations.",
                    "source": "MarketBeat",
                    "source_domain": "MarketBeat",
                    "category_within_source": "General",
                    "url": "https://example.com/aaa",
                    "time_published": "20240405T074158",
                    "overall_sentiment_score": 0.15,
                    "overall_sentiment_label": "Somewhat-Bullish",
                    "authors": ["Reporter"],
                    "topics": [{"topic": "earnings", "relevance_score": "0.9"}],
                    "banner_image": "https://example.com/banner.png",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAA",
                            "relevance_score": "1.000000",
                            "ticker_sentiment_score": "0.22",
                            "ticker_sentiment_label": "Somewhat-Bullish",
                        },
                        {
                            "ticker": "ZZZ",
                            "relevance_score": "0.300000",
                            "ticker_sentiment_score": "0.01",
                            "ticker_sentiment_label": "Neutral",
                        },
                    ],
                },
                {
                    "title": "AAA earnings beat",
                    "summary": "Quarterly results beat expectations.",
                    "source": "MarketBeat",
                    "source_domain": "MarketBeat",
                    "category_within_source": "General",
                    "url": "https://example.com/aaa",
                    "time_published": "20240405T074158",
                    "overall_sentiment_score": 0.15,
                    "overall_sentiment_label": "Somewhat-Bullish",
                    "authors": ["Reporter"],
                    "topics": [{"topic": "earnings", "relevance_score": "0.9"}],
                    "banner_image": "https://example.com/banner.png",
                    "ticker_sentiment": [
                        {
                            "ticker": "AAA",
                            "relevance_score": "1.000000",
                            "ticker_sentiment_score": "0.22",
                            "ticker_sentiment_label": "Somewhat-Bullish",
                        }
                    ],
                },
                {
                    "title": "BBB update",
                    "summary": "Analyst reaction to BBB.",
                    "source": "TradingView",
                    "source_domain": "TradingView",
                    "category_within_source": "General",
                    "url": "https://example.com/bbb",
                    "time_published": "20240610T121500",
                    "overall_sentiment_score": -0.05,
                    "overall_sentiment_label": "Neutral",
                    "authors": ["Analyst"],
                    "topics": [{"topic": "finance", "relevance_score": "0.7"}],
                    "banner_image": None,
                    "ticker_sentiment": [
                        {
                            "ticker": "BBB",
                            "relevance_score": "0.850000",
                            "ticker_sentiment_score": "-0.1",
                            "ticker_sentiment_label": "Neutral",
                        }
                    ],
                },
            ]
        }


class FakeFmpClient:
    def __init__(self) -> None:
        self.economic_calls = []
        self.technical_calls = []

    def get_economic_indicator(self, **kwargs):
        self.economic_calls.append(kwargs)
        return [
            {"date": "2024-03-31", "value": 1.0},
            {"date": "2024-06-30", "value": 2.0},
        ]

    def get_technical_indicator(self, **kwargs):
        self.technical_calls.append(kwargs)
        indicator = kwargs["indicator"]
        return [
            {
                "date": "2024-03-28 00:00:00",
                indicator: 10.0,
            },
            {
                "date": "2024-06-28 00:00:00",
                indicator: 20.0,
            },
        ]

    def get_usage_stats(self):
        return {"fmp_api_calls_executed": len(self.economic_calls) + len(self.technical_calls)}


def test_qai_preparer_writes_fundamental_and_story_outputs(tmp_path: Path) -> None:
    story_client = FakeStoryClient()
    preparer = QaiPreparer(provider=FakeProvider(), story_client=story_client)
    paths_cfg = OmegaConf.create(
        {
            "raw_data_dir": str(tmp_path / "raw"),
            "cleaned_data_dir": str(tmp_path / "cleaned"),
            "processed_data_dir": str(tmp_path / "processed"),
        }
    )
    data_cfg = OmegaConf.create(
        {
            "tickers": ["AAA", "BBB"],
            "time_from": "20240401T0000",
            "time_to": "20241231T2359",
            "provider": {
                "simfin": {"type": "simfin", "force_refresh": False},
                "alphavantage": {
                    "env_file": str(tmp_path / ".env"),
                    "api_key_env": "ALPHA_VANTAGE_API_KEY",
                    "topics": None,
                    "sort": "LATEST",
                    "limit": 50,
                    "divide_range_days": None,
                    "api_call_per_minute": 75,
                },
            },
            "dataset": {
                "ttm_window": 4,
                "zscore_window": 12,
                "zscore_min_periods": 1,
                "winsorize_lower_quantile": 0.01,
                "winsorize_upper_quantile": 0.99,
                "market_cap_min": 1000000.0,
            },
            "outputs": {
                "fundamental_filename": "fundamental.csv",
                "story_filename": "story.csv",
            },
        }
    )

    output_dir = preparer.prepare(data_cfg=data_cfg, paths_cfg=paths_cfg)

    assert output_dir == tmp_path / "cleaned"
    fundamental = pd.read_csv(output_dir / "fundamental.csv")
    story = pd.read_csv(output_dir / "story.csv")

    assert list(fundamental.columns[:2]) == ["ticker", "date"]
    assert "piotroski_fscore" in fundamental.columns
    assert set(fundamental["date"]) == {"2024-06-30"}
    assert set(fundamental["ticker"]) == {"AAA", "BBB"}

    assert list(story.columns[:2]) == ["ticker", "date"]
    assert len(story) == 3
    assert set(story["ticker"]) == {"AAA", "BBB", "ZZZ"}
    assert set(story["date"]) == {"2024-04-05", "2024-06-10"}
    assert "ticker_sentiment_score" in story.columns
    assert "authors_json" in story.columns
    assert len(story_client.calls) == 1
    assert story_client.calls[0]["tickers"] is None
    assert preparer.last_run_summary is not None
    assert preparer.last_run_summary["api_calls_planned"] == 1
    assert preparer.last_run_summary["api_calls_executed"] == 1
    assert preparer.last_run_summary["configured_api_calls_per_minute"] == 75
    assert preparer.last_run_summary["story_rows_deduplicated"] == 3


def test_qai_preparer_writes_fmp_economics_and_technical_outputs(tmp_path: Path) -> None:
    story_client = FakeStoryClient()
    fmp_client = FakeFmpClient()
    preparer = QaiPreparer(
        provider=FakeProvider(),
        story_client=story_client,
        fmp_client=fmp_client,
    )
    paths_cfg = OmegaConf.create(
        {
            "raw_data_dir": str(tmp_path / "raw"),
            "cleaned_data_dir": str(tmp_path / "cleaned"),
            "processed_data_dir": str(tmp_path / "processed"),
        }
    )
    data_cfg = OmegaConf.create(
        {
            "tickers": ["AAA", "BBB"],
            "time_from": "20240101T0000",
            "time_to": "20241231T2359",
            "provider": {
                "simfin": {"type": "simfin", "force_refresh": False},
                "alphavantage": {
                    "env_file": str(tmp_path / ".env"),
                    "api_key_env": "ALPHA_VANTAGE_API_KEY",
                    "topics": None,
                    "sort": "LATEST",
                    "limit": 50,
                    "divide_range_days": None,
                    "api_call_per_minute": 75,
                },
                "fmp": {
                    "economic_indicators": [{"name": "GDP", "column": "gdp"}],
                    "technical_indicators": [
                        {
                            "name": "rsi",
                            "periodLength": 14,
                            "timeframe": "1day",
                        }
                    ],
                    "allow_partial_fmp": False,
                },
            },
            "dataset": {
                "ttm_window": 4,
                "zscore_window": 12,
                "zscore_min_periods": 1,
                "winsorize_lower_quantile": 0.01,
                "winsorize_upper_quantile": 0.99,
                "market_cap_min": 1000000.0,
            },
            "outputs": {
                "fundamental_filename": "fundamental.csv",
                "story_filename": "story.csv",
                "economics_filename": "economics.csv",
                "technical_filename": "technical.csv",
            },
        }
    )

    output_dir = preparer.prepare(data_cfg=data_cfg, paths_cfg=paths_cfg)

    economics = pd.read_csv(output_dir / "economics.csv")
    technical = pd.read_csv(output_dir / "technical.csv")

    assert list(economics.columns[:2]) == ["date", "time_range"]
    assert "econ_gdp" in economics.columns
    assert set(economics["time_range"]) == {"q1y2024", "q2y2024"}
    assert list(technical.columns[:3]) == ["ticker", "date", "time_range"]
    assert "tech_rsi_14_1day" in technical.columns
    assert set(technical["ticker"]) == {"AAA", "BBB"}
    assert len(fmp_client.economic_calls) == 1
    assert len(fmp_client.technical_calls) == 2
    assert preparer.last_run_summary["economics_rows"] == 2
    assert preparer.last_run_summary["technical_rows"] == 4


def test_story_range_helpers_split_windows_and_budget() -> None:
    windows = _build_story_request_windows(
        time_from="20240101T0000",
        time_to="20241027T0000",
        divide_range_days=30,
    )
    limits = _allocate_window_limits(100, len(windows))

    assert len(windows) == 10
    assert limits == [10] * 10


def test_qai_preparer_splits_story_requests_across_windows(tmp_path: Path) -> None:
    story_client = FakeStoryClient()
    preparer = QaiPreparer(provider=FakeProvider(), story_client=story_client)
    paths_cfg = OmegaConf.create(
        {
            "raw_data_dir": str(tmp_path / "raw"),
            "cleaned_data_dir": str(tmp_path / "cleaned"),
            "processed_data_dir": str(tmp_path / "processed"),
        }
    )
    data_cfg = OmegaConf.create(
        {
            "tickers": ["AAA", "BBB"],
            "time_from": "20240101T0000",
            "time_to": "20240331T0000",
            "provider": {
                "simfin": {"type": "simfin", "force_refresh": False},
                "alphavantage": {
                    "env_file": str(tmp_path / ".env"),
                    "api_key_env": "ALPHA_VANTAGE_API_KEY",
                    "topics": None,
                    "sort": "LATEST",
                    "limit": 10,
                    "divide_range_days": 30,
                    "api_call_per_minute": 75,
                },
            },
            "dataset": {
                "ttm_window": 4,
                "zscore_window": 12,
                "zscore_min_periods": 1,
                "winsorize_lower_quantile": 0.01,
                "winsorize_upper_quantile": 0.99,
                "market_cap_min": 1000000.0,
            },
            "outputs": {
                "fundamental_filename": "fundamental.csv",
                "story_filename": "story.csv",
            },
        }
    )

    preparer.prepare(data_cfg=data_cfg, paths_cfg=paths_cfg)

    assert len(story_client.calls) == 3
    assert [call["limit"] for call in story_client.calls] == [4, 3, 3]
    assert all(call["tickers"] is None for call in story_client.calls)
    assert story_client.calls[0]["time_from"] == "20240101T0000"
    assert story_client.calls[-1]["time_to"] == "20240331T0000"
