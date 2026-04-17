from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.datamodules.qai_datamodule import (
    DEFAULT_EXCLUDE_COLUMNS,
    DEFAULT_HORIZONS,
    DEFAULT_METADATA_COLUMNS,
    DEFAULT_TARGET_PRICE_COLUMN,
    DEFAULT_TARGET_FREQUENCY,
    _align_target_prices,
    _attach_future_targets,
    _load_price_history,
    _load_split_frame,
    _resolve_feature_columns,
)


class QaiPortfolioDataset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


@dataclass
class QaiPortfolioDataModule:
    train_path: str = "data/preprocessed/train.csv"
    val_path: str = "data/preprocessed/val.csv"
    test_path: str = "data/preprocessed/test.csv"
    price_history_path: str = "data/raw/us-shareprices-daily.csv"
    sequence_length: int = 4
    feature_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    horizons: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    target_horizon: int | None = None
    target_frequency: str = DEFAULT_TARGET_FREQUENCY
    price_field: str = DEFAULT_TARGET_PRICE_COLUMN
    flat_return_threshold: float = 0.02
    batch_size: int = 8
    num_workers: int = 0

    feature_dim: int = field(init=False, default=0)
    min_sequence_length: int = field(init=False, default=0)
    max_sequence_length: int = field(init=False, default=0)
    max_tickers_per_sample: int = field(init=False, default=0)
    split_frames: dict[str, pd.DataFrame] = field(init=False, default_factory=dict)
    combined_frame: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    samples_by_split: dict[str, list[dict[str, Any]]] = field(init=False, default_factory=dict)
    sequence_lengths_by_split: dict[str, list[int]] = field(init=False, default_factory=dict)
    ticker_counts_by_split: dict[str, list[int]] = field(init=False, default_factory=dict)
    feature_names: list[str] = field(init=False, default_factory=list)
    target_match_summary: dict[str, dict[str, int | float | str]] = field(
        init=False, default_factory=dict
    )
    sample_quality_by_split: dict[str, dict[str, int | float | str]] = field(
        init=False, default_factory=dict
    )

    def setup(self) -> None:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for QaiPortfolioDataModule") from exc

        split_frames = {
            "train": _load_split_frame(
                self.train_path,
                "train",
                target_frequency=self.target_frequency,
            ),
            "val": _load_split_frame(
                self.val_path,
                "val",
                target_frequency=self.target_frequency,
            ),
            "test": _load_split_frame(
                self.test_path,
                "test",
                target_frequency=self.target_frequency,
            ),
        }
        combined = (
            pd.concat(split_frames.values(), ignore_index=True)
            .sort_values(["ticker", "quarter_end_date"])
            .reset_index(drop=True)
        )

        price_history = _filter_price_history_for_portfolio_frame(
            _load_price_history(self.price_history_path, price_field=self.price_field),
            frame=combined,
            horizons=self.horizons,
            target_frequency=self.target_frequency,
        )
        quarter_prices = _align_target_prices(
            price_history,
            price_field=self.price_field,
            target_frequency=self.target_frequency,
        )
        targets = _attach_future_targets(
            quarter_prices,
            horizons=self.horizons,
            flat_return_threshold=self.flat_return_threshold,
            target_frequency=self.target_frequency,
        )
        combined = combined.merge(
            targets,
            how="left",
            on=["ticker", "quarter_end_date", "time_range"],
        )

        feature_names = _resolve_feature_columns(
            combined,
            feature_columns=self.feature_columns,
            exclude_columns=(self.exclude_columns or []) + list(DEFAULT_EXCLUDE_COLUMNS),
        )
        target_horizon = self.target_horizon or max(self.horizons)
        future_price_column = f"future_price_t{target_horizon}"
        future_quarter_end_date_column = f"future_quarter_end_date_t{target_horizon}"
        target_match_summary = _summarize_portfolio_target_matches(
            frame=combined,
            future_price_column=future_price_column,
            future_quarter_end_date_column=future_quarter_end_date_column,
        )
        samples_by_split = _build_portfolio_samples_by_split(
            frame=combined,
            feature_columns=feature_names,
            sequence_length=self.sequence_length,
            target_horizon=target_horizon,
        )
        sample_quality_by_split = _summarize_portfolio_sample_quality(samples_by_split)

        self._torch = torch
        self._dataloader_cls = DataLoader
        self.target_horizon = target_horizon
        self.split_frames = split_frames
        self.combined_frame = combined
        self.samples_by_split = samples_by_split
        self.target_match_summary = target_match_summary
        self.sample_quality_by_split = sample_quality_by_split
        self.sequence_lengths_by_split = {
            split: [int(length) for sample in samples for length in sample["sequence_lengths"]]
            for split, samples in samples_by_split.items()
        }
        self.ticker_counts_by_split = {
            split: [int(sample["ticker_count"]) for sample in samples]
            for split, samples in samples_by_split.items()
        }
        self.feature_names = feature_names
        self.feature_dim = len(feature_names)
        self.min_sequence_length = self.sequence_length
        all_sequence_lengths = [
            sequence_length
            for split_lengths in self.sequence_lengths_by_split.values()
            for sequence_length in split_lengths
        ]
        self.max_sequence_length = max(all_sequence_lengths, default=self.sequence_length)
        all_ticker_counts = [
            ticker_count
            for split_counts in self.ticker_counts_by_split.values()
            for ticker_count in split_counts
        ]
        self.max_tickers_per_sample = max(all_ticker_counts, default=0)

    def train_dataloader(self):
        return self._make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader("test", shuffle=False)

    def _make_dataloader(self, split: str, *, shuffle: bool):
        dataset = QaiPortfolioDataset(self.samples_by_split.get(split, []))
        return self._dataloader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch = self._torch
        batch_size = len(batch)
        max_tickers = max((int(sample["ticker_count"]) for sample in batch), default=0)
        max_seq_len = max(
            (int(max(sample["sequence_lengths"])) for sample in batch if sample["sequence_lengths"]),
            default=self.sequence_length,
        )
        feature_dim = batch[0]["features"].size(-1) if batch else self.feature_dim

        features = torch.zeros(batch_size, max_tickers, max_seq_len, feature_dim, dtype=torch.float32)
        history_attention_mask = torch.zeros(batch_size, max_tickers, max_seq_len, dtype=torch.bool)
        ticker_attention_mask = torch.zeros(batch_size, max_tickers, dtype=torch.bool)
        current_prices = torch.zeros(batch_size, max_tickers, dtype=torch.float32)
        future_prices = torch.zeros(batch_size, max_tickers, dtype=torch.float32)

        sequence_lengths: list[list[int]] = []
        ticker_lists: list[list[str]] = []
        time_ranges: list[str] = []
        quarter_end_dates: list[pd.Timestamp] = []
        future_quarter_end_dates: list[pd.Timestamp] = []

        for batch_idx, sample in enumerate(batch):
            ticker_count = int(sample["ticker_count"])
            ticker_attention_mask[batch_idx, :ticker_count] = True
            current_prices[batch_idx, :ticker_count] = sample["current_prices"]
            future_prices[batch_idx, :ticker_count] = sample["future_prices"]
            sequence_lengths.append([int(value) for value in sample["sequence_lengths"]])
            ticker_lists.append(list(sample["tickers"]))
            time_ranges.append(str(sample["time_range"]))
            quarter_end_dates.append(sample["quarter_end_date"])
            future_quarter_end_dates.append(sample["future_quarter_end_date"])

            for ticker_idx in range(ticker_count):
                sample_length = int(sample["sequence_lengths"][ticker_idx])
                features[batch_idx, ticker_idx, :sample_length] = sample["features"][ticker_idx, :sample_length]
                history_attention_mask[batch_idx, ticker_idx, :sample_length] = True

        return {
            "features": features,
            "history_attention_mask": history_attention_mask,
            "ticker_attention_mask": ticker_attention_mask,
            "current_prices": current_prices,
            "future_prices": future_prices,
            "tickers": ticker_lists,
            "time_range": time_ranges,
            "quarter_end_date": quarter_end_dates,
            "future_quarter_end_date": future_quarter_end_dates,
            "sequence_lengths": sequence_lengths,
            "ticker_count": torch.tensor([len(tickers) for tickers in ticker_lists], dtype=torch.long),
        }


def _filter_price_history_for_portfolio_frame(
    price_history: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    horizons: Sequence[int],
    target_frequency: str,
) -> pd.DataFrame:
    if price_history.empty or frame.empty:
        return price_history

    tickers = set(frame["ticker"].astype(str).unique())
    min_date = frame["quarter_end_date"].min()
    max_date = frame["quarter_end_date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return price_history.loc[price_history["Ticker"].astype(str).isin(tickers)].copy()

    max_horizon = max((int(horizon) for horizon in horizons), default=0)
    if target_frequency.lower() in {"d", "day", "daily"}:
        max_target_date = max_date + pd.DateOffset(days=max_horizon)
    else:
        max_target_date = max_date + pd.offsets.QuarterEnd(max_horizon)

    return price_history.loc[
        price_history["Ticker"].astype(str).isin(tickers)
        & (price_history["Date"] >= min_date)
        & (price_history["Date"] <= max_target_date)
    ].copy()


def _build_portfolio_samples_by_split(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
    target_horizon: int,
) -> dict[str, list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for QAI portfolio sample generation") from exc

    samples_by_split = {"train": [], "val": [], "test": []}
    future_price_column = f"future_price_t{target_horizon}"
    future_quarter_end_date_column = f"future_quarter_end_date_t{target_horizon}"

    combined_by_ticker = {}
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        sorted_ticker_frame = ticker_frame.sort_values("quarter_end_date").reset_index(drop=True)
        combined_by_ticker[str(ticker)] = {
            "dates": sorted_ticker_frame["quarter_end_date"],
            "features": sorted_ticker_frame.loc[:, feature_columns].to_numpy(dtype="float32"),
        }

    for split, split_frame in frame.groupby("split", sort=False):
        valid_rows = split_frame.loc[
            split_frame["window_ready"].fillna(False)
            & split_frame["horizon_target_ready"].fillna(False)
            & split_frame[future_price_column].notna()
            & split_frame[future_quarter_end_date_column].notna()
            & split_frame["quarter_price"].notna()
        ].copy()

        for (time_range, quarter_end_date), group in valid_rows.groupby(
            ["time_range", "quarter_end_date"], sort=True
        ):
            ticker_records: list[dict[str, Any]] = []
            for row in group.sort_values("ticker").itertuples(index=False):
                ticker_history = combined_by_ticker[str(row.ticker)]
                row_position = int(
                    ticker_history["dates"].searchsorted(row.quarter_end_date, side="right") - 1
                )
                history_start = row_position - sequence_length + 1
                if history_start < 0:
                    continue
                feature_array = ticker_history["features"][history_start : row_position + 1]
                ticker_records.append(
                    {
                        "ticker": str(row.ticker),
                        "sequence_length": len(feature_array),
                        "features": torch.tensor(feature_array, dtype=torch.float32),
                        "current_price": float(row.quarter_price),
                        "future_price": float(getattr(row, future_price_column)),
                        "future_quarter_end_date": pd.Timestamp(
                            getattr(row, future_quarter_end_date_column)
                        ),
                    }
                )

            if not ticker_records:
                continue

            max_seq_len = max(record["sequence_length"] for record in ticker_records)
            ticker_count = len(ticker_records)
            features = torch.zeros(ticker_count, max_seq_len, len(feature_columns), dtype=torch.float32)
            sequence_lengths: list[int] = []
            tickers: list[str] = []
            current_prices: list[float] = []
            future_prices: list[float] = []
            future_quarter_end_dates: list[pd.Timestamp] = []

            for ticker_idx, record in enumerate(ticker_records):
                seq_len = int(record["sequence_length"])
                features[ticker_idx, :seq_len] = record["features"]
                sequence_lengths.append(seq_len)
                tickers.append(record["ticker"])
                current_prices.append(record["current_price"])
                future_prices.append(record["future_price"])
                future_quarter_end_dates.append(record["future_quarter_end_date"])

            unique_future_dates = sorted({pd.Timestamp(value) for value in future_quarter_end_dates})
            if len(unique_future_dates) != 1:
                raise ValueError(
                    "Portfolio sample contains mixed future target dates; expected a single t+horizon date."
                )

            samples_by_split[str(split)].append(
                {
                    "features": features,
                    "sequence_lengths": sequence_lengths,
                    "tickers": tickers,
                    "ticker_count": ticker_count,
                    "current_prices": torch.tensor(current_prices, dtype=torch.float32),
                    "future_prices": torch.tensor(future_prices, dtype=torch.float32),
                    "time_range": str(time_range),
                    "quarter_end_date": pd.Timestamp(quarter_end_date),
                    "future_quarter_end_date": unique_future_dates[0],
                }
            )

    return samples_by_split


def _summarize_portfolio_target_matches(
    *,
    frame: pd.DataFrame,
    future_price_column: str,
    future_quarter_end_date_column: str,
) -> dict[str, dict[str, int | float | str]]:
    summary: dict[str, dict[str, int | float | str]] = {}
    for split in ("train", "val", "test"):
        split_frame = frame.loc[frame["split"] == split]
        if split_frame.empty:
            summary[split] = {
                "rows": 0,
                "window_ready_rows": 0,
                "horizon_ready_rows": 0,
                "target_price_rows": 0,
                "valid_target_rows": 0,
                "target_match_rate": 0.0,
                "valid_target_rate": 0.0,
            }
            continue

        target_ready = (
            split_frame[future_price_column].notna()
            & split_frame[future_quarter_end_date_column].notna()
            & split_frame["quarter_price"].notna()
        )
        valid = (
            split_frame["window_ready"].fillna(False)
            & split_frame["horizon_target_ready"].fillna(False)
            & target_ready
        )
        row_count = int(len(split_frame))
        target_count = int(target_ready.sum())
        valid_count = int(valid.sum())
        summary[split] = {
            "rows": row_count,
            "window_ready_rows": int(split_frame["window_ready"].fillna(False).sum()),
            "horizon_ready_rows": int(split_frame["horizon_target_ready"].fillna(False).sum()),
            "target_price_rows": target_count,
            "valid_target_rows": valid_count,
            "target_match_rate": target_count / max(row_count, 1),
            "valid_target_rate": valid_count / max(row_count, 1),
        }
    return summary


def _summarize_portfolio_sample_quality(
    samples_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int | float | str]]:
    summary: dict[str, dict[str, int | float | str]] = {}
    for split in ("train", "val", "test"):
        samples = samples_by_split.get(split, [])
        ticker_counts = [int(sample["ticker_count"]) for sample in samples]
        growth_values: list[float] = []
        for sample in samples:
            current_prices = sample["current_prices"]
            future_prices = sample["future_prices"]
            growth = future_prices / current_prices.clamp_min(1e-8)
            growth_values.extend(float(value) for value in growth.tolist())

        if not growth_values:
            summary[split] = {
                "samples": int(len(samples)),
                "asset_targets": 0,
                "avg_tickers_per_sample": 0.0,
                "min_asset_growth": "n/a",
                "max_asset_growth": "n/a",
                "mean_asset_growth": "n/a",
                "std_asset_growth": "n/a",
            }
            continue

        growth_series = pd.Series(growth_values, dtype="float64")
        summary[split] = {
            "samples": int(len(samples)),
            "asset_targets": int(len(growth_values)),
            "avg_tickers_per_sample": float(sum(ticker_counts) / max(len(ticker_counts), 1)),
            "min_asset_growth": float(growth_series.min()),
            "max_asset_growth": float(growth_series.max()),
            "mean_asset_growth": float(growth_series.mean()),
            "std_asset_growth": float(growth_series.std(ddof=0)),
        }
    return summary
