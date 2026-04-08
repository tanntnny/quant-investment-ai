from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd


EPSILON = 1e-8


@dataclass(slots=True)
class QaiWindowSample:
    ticker: str
    time_range: str
    window_end_date: pd.Timestamp
    features: list[list[float]]
    target: float


@dataclass(slots=True)
class QaiPreprocessor:
    fundamental_path: str = "data/processed/fundamental.csv"
    story_path: str = "data/processed/story.csv"
    batch_size: int = 32
    num_workers: int = 0
    sequence_length: int = 4
    time_horizon: int = 1
    time_horizontal: int | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_mode: str = "global_date"
    ticker_column: str = "ticker"
    date_column: str = "date"
    fill_story_missing_with_zero: bool = True
    min_sequence_points: int | None = None
    target_column: str | None = None
    target_kind: str = "price"

    merged_frame: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    feature_frame: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    feature_columns: list[str] = field(init=False, default_factory=list)
    topic_columns: list[str] = field(init=False, default_factory=list)
    split_boundaries: dict[str, str | None] = field(init=False, default_factory=dict)
    samples_by_split: dict[str, list[QaiWindowSample]] = field(
        init=False,
        default_factory=dict,
    )

    def setup(self) -> None:
        _ = self.num_workers
        effective_horizon = self.time_horizontal or self.time_horizon
        min_sequence_points = self.min_sequence_points or self.sequence_length

        _validate_split_ratios(self.train_ratio, self.val_ratio, self.test_ratio)

        fundamental = _load_fundamental_frame(
            Path(self.fundamental_path),
            ticker_column=self.ticker_column,
            date_column=self.date_column,
        )
        story = _load_story_frame(
            Path(self.story_path),
            ticker_column=self.ticker_column,
            date_column=self.date_column,
        )
        story_agg = _aggregate_story_frame(story)

        merged = fundamental.merge(
            story_agg,
            how="left",
            on=["ticker", "time_range"],
            suffixes=("", "_story"),
        )
        merged = merged.sort_values(["ticker", "quarter_end_date"]).reset_index(drop=True)

        story_numeric_columns = [
            column
            for column in story_agg.columns
            if column not in {"ticker", "time_range", "quarter_end_date"}
        ]
        if self.fill_story_missing_with_zero:
            for column in story_numeric_columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(0.0)

        feature_columns = _resolve_feature_columns(
            merged,
            excluded_columns={
                "ticker",
                "time_range",
                "quarter_end_date",
                self.target_column,
            },
        )

        boundaries = _compute_global_split_boundaries(
            merged["quarter_end_date"],
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        samples_by_split = _build_window_samples(
            frame=merged,
            feature_columns=feature_columns,
            sequence_length=self.sequence_length,
            min_sequence_points=min_sequence_points,
            time_horizon=effective_horizon,
            boundaries=boundaries,
        )

        self.merged_frame = merged
        self.feature_frame = merged.loc[:, ["ticker", "time_range", "quarter_end_date", *feature_columns]]
        self.feature_columns = feature_columns
        self.topic_columns = [column for column in feature_columns if column.startswith("topic_relevance_")]
        self.split_boundaries = {
            key: value.strftime("%Y-%m-%d") if value is not None else None
            for key, value in boundaries.items()
        }
        self.samples_by_split = samples_by_split

    def train_dataloader(self) -> Iterator[list[tuple[list[list[float]], float]]]:
        yield from self._iterate_split("train")

    def val_dataloader(self) -> Iterator[list[tuple[list[list[float]], float]]]:
        yield from self._iterate_split("val")

    def test_dataloader(self) -> Iterator[list[tuple[list[list[float]], float]]]:
        yield from self._iterate_split("test")

    def get_split_samples(self, split: str) -> list[QaiWindowSample]:
        return list(self.samples_by_split.get(split, []))

    def _iterate_split(self, split: str) -> Iterator[list[tuple[list[list[float]], float]]]:
        samples = self.samples_by_split.get(split, [])
        for start in range(0, len(samples), self.batch_size):
            batch = samples[start : start + self.batch_size]
            yield [(sample.features, sample.target) for sample in batch]


def _load_fundamental_frame(
    path: Path,
    *,
    ticker_column: str,
    date_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={ticker_column: "ticker", date_column: "date"})
    frame["ticker"] = frame["ticker"].astype(str)
    frame["quarter_end_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "quarter_end_date"]).copy()
    frame["time_range"] = frame["quarter_end_date"].map(_to_time_range)

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"ticker", "date", "time_range", "quarter_end_date"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregated = (
        frame.groupby(["ticker", "time_range"], as_index=False)
        .agg(
            quarter_end_date=("quarter_end_date", "max"),
            **{column: (column, "mean") for column in numeric_columns},
        )
        .sort_values(["ticker", "quarter_end_date"])
        .reset_index(drop=True)
    )
    return aggregated


def _load_story_frame(
    path: Path,
    *,
    ticker_column: str,
    date_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={ticker_column: "ticker", date_column: "date"})
    frame["ticker"] = frame["ticker"].astype(str)
    frame["quarter_end_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "quarter_end_date"]).copy()
    frame["time_range"] = frame["quarter_end_date"].map(_to_time_range)
    return frame


def _aggregate_story_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "time_range", "quarter_end_date"])

    enriched = frame.copy()
    numeric_columns = [
        "ticker_sentiment_score",
        "overall_sentiment_score",
        "ticker_relevance_score",
    ]
    for column in numeric_columns:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    label_series = enriched.get("ticker_sentiment_label", pd.Series(index=enriched.index, dtype="object"))
    enriched["bullish_count"] = label_series.fillna("").map(_is_bullish_label).astype(float)
    enriched["bearish_count"] = label_series.fillna("").map(_is_bearish_label).astype(float)
    enriched["neutral_count"] = (
        (~(enriched["bullish_count"].astype(bool) | enriched["bearish_count"].astype(bool)))
    ).astype(float)

    topic_frame = _expand_topics(enriched.get("topics_json", pd.Series(index=enriched.index, dtype="object")))
    enriched = pd.concat([enriched, topic_frame], axis=1)

    topic_columns = list(topic_frame.columns)
    grouped = (
        enriched.groupby(["ticker", "time_range"], as_index=False)
        .agg(
            quarter_end_date=("quarter_end_date", "max"),
            story_count=("ticker", "size"),
            ticker_sentiment_score=("ticker_sentiment_score", "mean"),
            overall_sentiment_score=("overall_sentiment_score", "mean"),
            ticker_relevance_score=("ticker_relevance_score", "mean"),
            bullish_count=("bullish_count", "sum"),
            bearish_count=("bearish_count", "sum"),
            neutral_count=("neutral_count", "sum"),
            **{column: (column, "mean") for column in topic_columns},
        )
        .sort_values(["ticker", "quarter_end_date"])
        .reset_index(drop=True)
    )
    grouped["ratio_of_bullish_bearish"] = grouped["bullish_count"] / (
        grouped["bearish_count"] + EPSILON
    )
    ordered_columns = [
        "ticker",
        "time_range",
        "quarter_end_date",
        "story_count",
        "ticker_sentiment_score",
        "overall_sentiment_score",
        "ticker_relevance_score",
        "bullish_count",
        "bearish_count",
        "neutral_count",
        "ratio_of_bullish_bearish",
        *sorted(topic_columns),
    ]
    return grouped.loc[:, ordered_columns]


def _expand_topics(topics_series: pd.Series) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for raw_topics in topics_series.fillna("[]"):
        topic_values: dict[str, float] = {}
        try:
            parsed = json.loads(raw_topics)
        except (TypeError, json.JSONDecodeError):
            parsed = []

        if not isinstance(parsed, list):
            parsed = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            topic = item.get("topic")
            if not topic:
                continue
            column = f"topic_relevance_{_slugify_topic(topic)}"
            score = pd.to_numeric(item.get("relevance_score"), errors="coerce")
            topic_values[column] = float(score) if pd.notna(score) else 0.0
        records.append(topic_values)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).fillna(0.0)


def _resolve_feature_columns(frame: pd.DataFrame, excluded_columns: set[str | None]) -> list[str]:
    excluded = {column for column in excluded_columns if column is not None}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _compute_global_split_boundaries(
    dates: Sequence[pd.Timestamp],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, pd.Timestamp | None]:
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    unique_dates = sorted(pd.Series(dates).dropna().unique().tolist())
    if not unique_dates:
        return {"train_end": None, "val_end": None}

    total_dates = len(unique_dates)
    train_count = max(1, int(math.floor(total_dates * train_ratio)))
    val_count = max(1, int(math.floor(total_dates * val_ratio))) if total_dates > 2 else 0
    assigned = train_count + val_count
    if assigned >= total_dates:
        val_count = max(0, total_dates - train_count - 1)
        assigned = train_count + val_count

    if assigned >= total_dates:
        train_count = max(1, total_dates - 1)
        val_count = 0

    train_end = unique_dates[train_count - 1]
    val_end = None
    if val_count > 0:
        val_end = unique_dates[train_count + val_count - 1]

    return {"train_end": train_end, "val_end": val_end}


def _build_window_samples(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
    min_sequence_points: int,
    time_horizon: int,
    boundaries: dict[str, pd.Timestamp | None],
) -> dict[str, list[QaiWindowSample]]:
    if time_horizon <= 0:
        raise ValueError("time_horizon must be a positive integer")

    samples_by_split: dict[str, list[QaiWindowSample]] = {"train": [], "val": [], "test": []}
    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("quarter_end_date").reset_index(drop=True)
        if len(ticker_frame) < max(min_sequence_points, sequence_length):
            continue

        for end_idx in range(sequence_length - 1, len(ticker_frame)):
            start_idx = end_idx - sequence_length + 1
            window = ticker_frame.iloc[start_idx : end_idx + 1]
            window_end_date = pd.Timestamp(window.iloc[-1]["quarter_end_date"])
            split = _assign_split(window_end_date, boundaries)
            if split is None:
                continue

            target = 0.0
            if time_horizon > 0 and end_idx + time_horizon < len(ticker_frame):
                target = 0.0

            features = window.loc[:, feature_columns].fillna(0.0).astype(float).values.tolist()
            samples_by_split[split].append(
                QaiWindowSample(
                    ticker=str(ticker),
                    time_range=str(window.iloc[-1]["time_range"]),
                    window_end_date=window_end_date,
                    features=features,
                    target=target,
                )
            )
    return samples_by_split


def _assign_split(
    window_end_date: pd.Timestamp,
    boundaries: dict[str, pd.Timestamp | None],
) -> str | None:
    train_end = boundaries.get("train_end")
    val_end = boundaries.get("val_end")

    if train_end is None:
        return None
    if window_end_date <= train_end:
        return "train"
    if val_end is not None and window_end_date <= val_end:
        return "val"
    return "test"


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")


def _to_time_range(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    return f"q{timestamp.quarter}y{timestamp.year}"


def _is_bullish_label(value: str) -> bool:
    return "bullish" in value.lower()


def _is_bearish_label(value: str) -> bool:
    return "bearish" in value.lower()


def _slugify_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
