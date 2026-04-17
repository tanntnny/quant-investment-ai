from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.preparers.qai_preprocess import (
    QaiPreprocessPreparer,
    _aggregate_story_frame,
    _compute_global_split_boundaries,
    _to_time_range,
)


def test_to_time_range_uses_expected_format() -> None:
    assert _to_time_range(pd.Timestamp("2026-03-31")) == "q1y2026"
    assert _to_time_range(pd.Timestamp("2020-08-15")) == "q3y2020"


def test_aggregate_story_frame_expands_topics_and_sentiment() -> None:
    story = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2024-03-15",
                "quarter_end_date": pd.Timestamp("2024-03-15"),
                "time_range": "q1y2024",
                "ticker_sentiment_score": 0.3,
                "overall_sentiment_score": 0.2,
                "ticker_relevance_score": 0.9,
                "ticker_sentiment_label": "Bullish",
                "topics_json": json.dumps(
                    [
                        {"topic": "earnings", "relevance_score": "0.8"},
                        {"topic": "finance", "relevance_score": "0.4"},
                    ]
                ),
            },
            {
                "ticker": "AAA",
                "date": "2024-03-25",
                "quarter_end_date": pd.Timestamp("2024-03-25"),
                "time_range": "q1y2024",
                "ticker_sentiment_score": -0.1,
                "overall_sentiment_score": -0.2,
                "ticker_relevance_score": 0.7,
                "ticker_sentiment_label": "Bearish",
                "topics_json": json.dumps(
                    [
                        {"topic": "earnings", "relevance_score": "0.6"},
                    ]
                ),
            },
        ]
    )

    aggregated = _aggregate_story_frame(story)

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["story_count"] == 2
    assert row["ticker_sentiment_score"] == pytest.approx(0.1)
    assert row["overall_sentiment_score"] == pytest.approx(0.0)
    assert row["bullish_count"] == 1.0
    assert row["bearish_count"] == 1.0
    assert row["ratio_of_bullish_bearish"] == pytest.approx(1.0)
    assert row["topic_relevance_earnings"] == pytest.approx(0.7)
    assert row["topic_relevance_finance"] == pytest.approx(0.2)


def test_compute_global_split_boundaries_uses_shared_dates() -> None:
    boundaries = _compute_global_split_boundaries(
        pd.to_datetime(
            [
                "2024-03-31",
                "2024-06-30",
                "2024-09-30",
                "2024-12-31",
                "2025-03-31",
            ]
        ),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert boundaries["train_end"] == pd.Timestamp("2024-09-30")
    assert boundaries["val_end"] == pd.Timestamp("2024-12-31")


def test_qai_preprocess_preparer_builds_merged_table_and_metadata(tmp_path: Path) -> None:
    fundamental = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-03-31", "gross_margin": 0.4, "roa": 0.1},
            {"ticker": "AAA", "date": "2024-06-30", "gross_margin": 0.42, "roa": 0.12},
            {"ticker": "AAA", "date": "2024-09-30", "gross_margin": 0.43, "roa": 0.14},
            {"ticker": "AAA", "date": "2024-12-31", "gross_margin": 0.45, "roa": 0.16},
            {"ticker": "BBB", "date": "2024-03-31", "gross_margin": 0.3, "roa": 0.08},
            {"ticker": "BBB", "date": "2024-06-30", "gross_margin": 0.31, "roa": 0.09},
            {"ticker": "BBB", "date": "2024-09-30", "gross_margin": 0.33, "roa": 0.1},
            {"ticker": "BBB", "date": "2024-12-31", "gross_margin": 0.35, "roa": 0.11},
        ]
    )
    story = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2024-03-15",
                "ticker_sentiment_score": 0.2,
                "overall_sentiment_score": 0.1,
                "ticker_relevance_score": 1.0,
                "ticker_sentiment_label": "Somewhat-Bullish",
                "topics_json": json.dumps(
                    [{"topic": "earnings", "relevance_score": "0.8"}]
                ),
            },
            {
                "ticker": "AAA",
                "date": "2024-09-10",
                "ticker_sentiment_score": -0.3,
                "overall_sentiment_score": -0.2,
                "ticker_relevance_score": 0.7,
                "ticker_sentiment_label": "Bearish",
                "topics_json": json.dumps(
                    [{"topic": "finance", "relevance_score": "0.6"}]
                ),
            },
        ]
    )
    economics = pd.DataFrame(
        [
            {"date": "2024-03-31", "time_range": "q1y2024", "econ_gdp": 1.0},
            {"date": "2024-06-30", "time_range": "q2y2024", "econ_gdp": 2.0},
            {"date": "2024-09-30", "time_range": "q3y2024", "econ_gdp": 3.0},
            {"date": "2024-12-31", "time_range": "q4y2024", "econ_gdp": 4.0},
        ]
    )
    technical = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": date,
                "time_range": time_range,
                "tech_rsi_14_1day": value,
            }
            for ticker in ("AAA", "BBB")
            for date, time_range, value in [
                ("2024-03-31", "q1y2024", 40.0),
                ("2024-06-30", "q2y2024", 45.0),
                ("2024-09-30", "q3y2024", 50.0),
                ("2024-12-31", "q4y2024", 55.0),
            ]
        ]
    )

    fundamental_path = tmp_path / "fundamental.csv"
    story_path = tmp_path / "story.csv"
    economics_path = tmp_path / "economics.csv"
    technical_path = tmp_path / "technical.csv"
    fundamental.to_csv(fundamental_path, index=False)
    story.to_csv(story_path, index=False)
    economics.to_csv(economics_path, index=False)
    technical.to_csv(technical_path, index=False)

    preparer = QaiPreprocessPreparer()
    paths_cfg = type(
        "PathsCfg",
        (),
        {
            "cleaned_data_dir": str(tmp_path / "cleaned"),
            "preprocessed_data_dir": str(tmp_path / "preprocessed"),
        },
    )()
    data_cfg = OmegaConf.create(
        {
            "inputs": {
                "fundamental_path": str(fundamental_path),
                "story_path": str(story_path),
                "economics_path": str(economics_path),
                "technical_path": str(technical_path),
            },
            "outputs": {
                "preprocessed_filename": "qai_preprocessed.csv",
                "train_filename": "train.csv",
                "val_filename": "val.csv",
                "test_filename": "test.csv",
                "metadata_filename": "qai_preprocess_metadata.json",
            },
            "ticker_column": "ticker",
            "date_column": "date",
            "sequence_length": 2,
            "time_horizon": 1,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "fill_story_missing_with_zero": True,
            "fill_fmp_missing_with_zero": False,
            "min_sequence_points": None,
        }
    )

    output_dir = preparer.prepare(data_cfg=data_cfg, paths_cfg=paths_cfg)

    assert output_dir == tmp_path / "preprocessed"
    merged = pd.read_csv(output_dir / "qai_preprocessed.csv")
    train = pd.read_csv(output_dir / "train.csv")
    val = pd.read_csv(output_dir / "val.csv")
    test = pd.read_csv(output_dir / "test.csv")
    metadata = json.loads((output_dir / "qai_preprocess_metadata.json").read_text())

    assert metadata["split_boundaries"] == {
        "train_end": "2024-06-30",
        "val_end": "2024-09-30",
    }
    assert metadata["imputation_strategy"] == "train_split_median"
    assert metadata["standardization_strategy"] == "train_split_zscore"
    assert "topic_relevance_earnings" in metadata["topic_columns"]
    assert "topic_relevance_finance" in metadata["topic_columns"]
    assert metadata["economics_columns"] == ["econ_gdp"]
    assert metadata["technical_columns"] == ["tech_rsi_14_1day"]
    assert "econ_gdp" in merged.columns
    assert "tech_rsi_14_1day" in merged.columns
    assert "quarter_end_date_story" not in merged.columns
    assert "quarter_end_date_economics" not in merged.columns
    assert "quarter_end_date_technical" not in merged.columns
    assert "quarter_end_date_story" not in metadata["feature_columns"]
    aaa_q2 = merged.loc[(merged["ticker"] == "AAA") & (merged["time_range"] == "q2y2024")].iloc[0]
    assert aaa_q2["story_count"] == pytest.approx(-0.5773502691896258)
    assert aaa_q2["ticker_sentiment_score"] == pytest.approx(-0.5773502691896258)
    assert set(merged["split"]) == {"train", "val", "test"}
    assert merged["window_ready"].sum() == 6
    assert set(train["split"]) == {"train"}
    assert set(val["split"]) == {"val"}
    assert set(test["split"]) == {"test"}
    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2
    assert train["gross_margin"].mean() == pytest.approx(0.0)
    assert train["gross_margin"].std(ddof=0) == pytest.approx(1.0)
    assert metadata["split_output_paths"]["train"].endswith("train.csv")
