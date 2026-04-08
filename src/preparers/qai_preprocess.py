from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from src.utils.console import announce, render_kv_table

logger = logging.getLogger(__name__)

EPSILON = 1e-8


def _announce(message: str) -> None:
    logger.info(message)
    announce(message)


class QaiPreprocessPreparer:
    def __init__(self) -> None:
        self.last_run_summary: dict[str, Any] | None = None

    def prepare(self, data_cfg, paths_cfg) -> Path:
        preprocessed_dir = Path(getattr(paths_cfg, "preprocessed_data_dir", "data/preprocessed"))
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        effective_horizon = int(data_cfg.get("time_horizontal") or data_cfg.get("time_horizon", 1))
        sequence_length = int(data_cfg.get("sequence_length", 4))
        min_sequence_points = int(data_cfg.get("min_sequence_points") or sequence_length)
        _validate_split_ratios(
            float(data_cfg.train_ratio),
            float(data_cfg.val_ratio),
            float(data_cfg.test_ratio),
        )

        cleaned_dir = Path(getattr(paths_cfg, "cleaned_data_dir", "data/cleaned"))
        fundamental_path = Path(data_cfg.inputs.get("fundamental_path", cleaned_dir / "fundamental.csv"))
        story_path = Path(data_cfg.inputs.get("story_path", cleaned_dir / "story.csv"))

        _announce(
            "[QAI Preprocess] Starting preprocessing "
            f"fundamental={fundamental_path} story={story_path} output_dir={preprocessed_dir}"
        )

        fundamental = _load_fundamental_frame(
            fundamental_path,
            ticker_column=data_cfg.get("ticker_column", "ticker"),
            date_column=data_cfg.get("date_column", "date"),
        )
        story = _load_story_frame(
            story_path,
            ticker_column=data_cfg.get("ticker_column", "ticker"),
            date_column=data_cfg.get("date_column", "date"),
        )
        aggregated_story = _aggregate_story_frame(story)
        merged = fundamental.merge(
            aggregated_story,
            how="left",
            on=["ticker", "time_range"],
            suffixes=("", "_story"),
        )
        merged = merged.sort_values(["ticker", "quarter_end_date"]).reset_index(drop=True)

        story_numeric_columns = [
            column
            for column in aggregated_story.columns
            if column not in {"ticker", "time_range", "quarter_end_date"}
        ]
        if bool(data_cfg.get("fill_story_missing_with_zero", True)):
            for column in story_numeric_columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(0.0)

        boundaries = _compute_global_split_boundaries(
            merged["quarter_end_date"],
            train_ratio=float(data_cfg.train_ratio),
            val_ratio=float(data_cfg.val_ratio),
            test_ratio=float(data_cfg.test_ratio),
        )
        merged = _annotate_time_series_state(
            merged,
            sequence_length=sequence_length,
            min_sequence_points=min_sequence_points,
            time_horizon=effective_horizon,
            boundaries=boundaries,
        )

        feature_columns = _resolve_feature_columns(
            merged,
            excluded_columns={
                "ticker",
                "time_range",
                "quarter_end_date",
                "split",
                "ticker_history_index",
                "window_ready",
                "horizon_target_ready",
            },
        )
        imputation_values = _fit_imputation_values(merged, feature_columns)
        merged = _apply_imputation(merged, feature_columns, imputation_values)
        scaling_stats = _fit_standardization_stats(merged, feature_columns)
        merged = _apply_standardization(merged, feature_columns, scaling_stats)

        output_path = preprocessed_dir / data_cfg.outputs.preprocessed_filename
        merged.to_csv(output_path, index=False)

        split_outputs = {
            "train": preprocessed_dir / data_cfg.outputs.train_filename,
            "val": preprocessed_dir / data_cfg.outputs.val_filename,
            "test": preprocessed_dir / data_cfg.outputs.test_filename,
        }
        for split_name, split_path in split_outputs.items():
            split_frame = merged.loc[merged["split"] == split_name].reset_index(drop=True)
            split_frame.to_csv(split_path, index=False)

        topic_columns = [column for column in feature_columns if column.startswith("topic_relevance_")]

        metadata = {
            "fundamental_input_path": str(fundamental_path),
            "story_input_path": str(story_path),
            "preprocessed_output_path": str(output_path),
            "rows": int(len(merged)),
            "tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
            "time_ranges": int(merged["time_range"].nunique()) if not merged.empty else 0,
            "sequence_length": sequence_length,
            "time_horizon": effective_horizon,
            "min_sequence_points": min_sequence_points,
            "train_ratio": float(data_cfg.train_ratio),
            "val_ratio": float(data_cfg.val_ratio),
            "test_ratio": float(data_cfg.test_ratio),
            "split_boundaries": {
                key: value.strftime("%Y-%m-%d") if value is not None else None
                for key, value in boundaries.items()
            },
            "feature_columns": feature_columns,
            "topic_columns": topic_columns,
            "splits": merged["split"].value_counts(dropna=False).to_dict(),
            "imputation_strategy": "train_split_median",
            "standardization_strategy": "train_split_zscore",
            "imputation_values": imputation_values,
            "scaling_stats": scaling_stats,
            "split_output_paths": {key: str(value) for key, value in split_outputs.items()},
        }
        metadata_path = preprocessed_dir / data_cfg.outputs.metadata_filename
        metadata_path.write_text(json.dumps(metadata, indent=2))

        self.last_run_summary = metadata
        render_kv_table("QAI Preprocess Summary", metadata)
        _announce(f"[QAI Preprocess] Saved preprocessed dataset to {output_path}")
        _announce(f"[QAI Preprocess] Saved metadata to {metadata_path}")
        return preprocessed_dir


def _load_fundamental_frame(path: Path, *, ticker_column: str, date_column: str) -> pd.DataFrame:
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
    return (
        frame.groupby(["ticker", "time_range"], as_index=False)
        .agg(
            quarter_end_date=("quarter_end_date", "max"),
            **{column: (column, "mean") for column in numeric_columns},
        )
        .sort_values(["ticker", "quarter_end_date"])
        .reset_index(drop=True)
    )


def _load_story_frame(path: Path, *, ticker_column: str, date_column: str) -> pd.DataFrame:
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
    for column in [
        "ticker_sentiment_score",
        "overall_sentiment_score",
        "ticker_relevance_score",
    ]:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    labels = enriched.get("ticker_sentiment_label", pd.Series(index=enriched.index, dtype="object"))
    enriched["bullish_count"] = labels.fillna("").map(_is_bullish_label).astype(float)
    enriched["bearish_count"] = labels.fillna("").map(_is_bearish_label).astype(float)
    enriched["neutral_count"] = (
        ~(enriched["bullish_count"].astype(bool) | enriched["bearish_count"].astype(bool))
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
    return grouped


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
    return pd.DataFrame.from_records(records).fillna(0.0) if records else pd.DataFrame()


def _annotate_time_series_state(
    frame: pd.DataFrame,
    *,
    sequence_length: int,
    min_sequence_points: int,
    time_horizon: int,
    boundaries: dict[str, pd.Timestamp | None],
) -> pd.DataFrame:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be a positive integer")

    annotated_groups: list[pd.DataFrame] = []
    for _, ticker_frame in frame.groupby("ticker", sort=False):
        ticker_frame = ticker_frame.sort_values("quarter_end_date").reset_index(drop=True).copy()
        ticker_frame["ticker_history_index"] = range(len(ticker_frame))
        ticker_frame["window_ready"] = ticker_frame["ticker_history_index"] + 1 >= max(
            sequence_length, min_sequence_points
        )
        ticker_frame["horizon_target_ready"] = (
            ticker_frame["ticker_history_index"] + time_horizon < len(ticker_frame)
        )
        annotated_groups.append(ticker_frame)

    annotated = pd.concat(annotated_groups, ignore_index=True) if annotated_groups else frame.copy()
    annotated["split"] = annotated["quarter_end_date"].map(lambda value: _assign_split(value, boundaries))
    return annotated


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

    return {
        "train_end": unique_dates[train_count - 1],
        "val_end": unique_dates[train_count + val_count - 1] if val_count > 0 else None,
    }


def _assign_split(
    quarter_end_date: pd.Timestamp,
    boundaries: dict[str, pd.Timestamp | None],
) -> str | None:
    train_end = boundaries.get("train_end")
    val_end = boundaries.get("val_end")
    if train_end is None:
        return None
    if quarter_end_date <= train_end:
        return "train"
    if val_end is not None and quarter_end_date <= val_end:
        return "val"
    return "test"


def _resolve_feature_columns(frame: pd.DataFrame, excluded_columns: set[str | None]) -> list[str]:
    excluded = {column for column in excluded_columns if column is not None}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _fit_imputation_values(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, float]:
    train_frame = frame.loc[frame["split"] == "train", list(feature_columns)]
    reference_frame = train_frame if not train_frame.empty else frame.loc[:, list(feature_columns)]
    values: dict[str, float] = {}
    for column in feature_columns:
        series = pd.to_numeric(reference_frame[column], errors="coerce")
        if series.dropna().empty:
            values[column] = 0.0
        else:
            values[column] = float(series.median())
    return values


def _apply_imputation(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    imputation_values: dict[str, float],
) -> pd.DataFrame:
    transformed = frame.copy()
    for column in feature_columns:
        transformed[column] = pd.to_numeric(transformed[column], errors="coerce").fillna(
            imputation_values[column]
        )
    return transformed


def _fit_standardization_stats(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, dict[str, float]]:
    train_frame = frame.loc[frame["split"] == "train", list(feature_columns)]
    reference_frame = train_frame if not train_frame.empty else frame.loc[:, list(feature_columns)]
    stats: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        values = pd.to_numeric(reference_frame[column], errors="coerce")
        mean = float(values.mean()) if not values.dropna().empty else 0.0
        std = float(values.std(ddof=0)) if not values.dropna().empty else 1.0
        if np.isclose(std, 0.0):
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def _apply_standardization(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    scaling_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    transformed = frame.copy()
    for column in feature_columns:
        mean = scaling_stats[column]["mean"]
        std = scaling_stats[column]["std"]
        transformed[column] = (pd.to_numeric(transformed[column], errors="coerce") - mean) / std
    return transformed


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")


def _to_time_range(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    return f"q{timestamp.quarter}y{timestamp.year}"


def _slugify_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_bullish_label(value: str) -> bool:
    return "bullish" in value.lower()


def _is_bearish_label(value: str) -> bool:
    return "bearish" in value.lower()
