from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_HORIZONS = (1, 2, 3, 4)
DEFAULT_EXCLUDE_COLUMNS = {
    "split",
    "window_ready",
    "horizon_target_ready",
    "ticker_history_index",
    "quarter_price",
}
DEFAULT_METADATA_COLUMNS = {"ticker", "time_range", "quarter_end_date"}
DEFAULT_TARGET_PRICE_COLUMN = "Adj. Close"


def _load_split_frame(path: str | Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["split"] = split_name
    frame["quarter_end_date"] = pd.to_datetime(frame["quarter_end_date"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "quarter_end_date"]).copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["time_range"] = frame["time_range"].astype(str)
    return frame


def _load_price_history(
    path: str | Path,
    *,
    price_field: str = DEFAULT_TARGET_PRICE_COLUMN,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    if price_field not in frame.columns:
        raise ValueError(f"Price field '{price_field}' not found in {path}")

    frame["Ticker"] = frame["Ticker"].astype(str)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Ticker", "Date", price_field]).copy()
    frame[price_field] = pd.to_numeric(frame[price_field], errors="coerce")
    frame = frame.dropna(subset=[price_field]).copy()
    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return frame


def _align_quarter_close_prices(
    price_history: pd.DataFrame,
    *,
    price_field: str = DEFAULT_TARGET_PRICE_COLUMN,
) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(
            columns=["ticker", "quarter_end_date", "quarter_price", "time_range"]
        )

    aligned = price_history.copy()
    aligned["ticker"] = aligned["Ticker"].astype(str)
    aligned["quarter_end_date"] = aligned["Date"].dt.to_period("Q").dt.end_time.dt.normalize()
    aligned = (
        aligned.groupby(["ticker", "quarter_end_date"], as_index=False)
        .tail(1)
        .loc[:, ["ticker", "quarter_end_date", price_field]]
        .rename(columns={price_field: "quarter_price"})
        .sort_values(["ticker", "quarter_end_date"])
        .reset_index(drop=True)
    )
    aligned["time_range"] = aligned["quarter_end_date"].map(_to_time_range)
    return aligned


def _attach_future_targets(
    quarter_prices: pd.DataFrame,
    *,
    horizons: Sequence[int],
    flat_return_threshold: float,
) -> pd.DataFrame:
    if quarter_prices.empty:
        return quarter_prices.copy()

    enriched = (
        quarter_prices.sort_values(["ticker", "quarter_end_date"])
        .reset_index(drop=True)
        .copy()
    )
    grouped_prices = enriched.groupby("ticker", sort=False)["quarter_price"]

    for horizon in horizons:
        future_prices = grouped_prices.shift(-horizon)
        returns = (future_prices - enriched["quarter_price"]) / enriched["quarter_price"]
        enriched[f"future_price_t{horizon}"] = future_prices
        enriched[f"future_return_t{horizon}"] = returns
        enriched[f"class_target_t{horizon}"] = returns.map(
            lambda value: _classify_return(value, flat_return_threshold)
        )

    return enriched


def _classify_return(value: float | None, threshold: float) -> int | None:
    if value is None or pd.isna(value):
        return None
    if value < -threshold:
        return 0
    if value > threshold:
        return 2
    return 1


def _to_time_range(timestamp: pd.Timestamp) -> str:
    quarter = timestamp.quarter
    return f"q{quarter}y{timestamp.year}"


def _resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None,
    exclude_columns: Iterable[str],
) -> list[str]:
    if feature_columns:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Configured feature columns not found: {missing}")
        return list(feature_columns)

    excluded = set(exclude_columns) | DEFAULT_METADATA_COLUMNS
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        and not column.startswith("future_price_t")
        and not column.startswith("future_return_t")
        and not column.startswith("class_target_t")
    ]


class QaiSequenceDataset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


@dataclass
class QaiDataModule:
    train_path: str = "data/preprocessed/train.csv"
    val_path: str = "data/preprocessed/val.csv"
    test_path: str = "data/preprocessed/test.csv"
    price_history_path: str = "data/raw/us-shareprices-daily.csv"
    sequence_length: int = 4
    feature_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    horizons: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    price_field: str = DEFAULT_TARGET_PRICE_COLUMN
    flat_return_threshold: float = 0.02
    batch_size: int = 32
    num_workers: int = 0

    feature_dim: int = field(init=False, default=0)
    min_sequence_length: int = field(init=False, default=0)
    max_sequence_length: int = field(init=False, default=0)
    split_frames: dict[str, pd.DataFrame] = field(init=False, default_factory=dict)
    combined_frame: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    samples_by_split: dict[str, list[dict[str, Any]]] = field(init=False, default_factory=dict)
    sequence_lengths_by_split: dict[str, list[int]] = field(init=False, default_factory=dict)
    feature_names: list[str] = field(init=False, default_factory=list)

    def setup(self) -> None:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for QaiDataModule") from exc

        split_frames = {
            "train": _load_split_frame(self.train_path, "train"),
            "val": _load_split_frame(self.val_path, "val"),
            "test": _load_split_frame(self.test_path, "test"),
        }
        combined = (
            pd.concat(split_frames.values(), ignore_index=True)
            .sort_values(["ticker", "quarter_end_date"])
            .reset_index(drop=True)
        )

        price_history = _load_price_history(self.price_history_path, price_field=self.price_field)
        quarter_prices = _align_quarter_close_prices(price_history, price_field=self.price_field)
        targets = _attach_future_targets(
            quarter_prices,
            horizons=self.horizons,
            flat_return_threshold=self.flat_return_threshold,
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

        samples_by_split = _build_samples_by_split(
            frame=combined,
            feature_columns=feature_names,
            sequence_length=self.sequence_length,
            horizons=self.horizons,
        )

        self._torch = torch
        self._dataloader_cls = DataLoader
        self.split_frames = split_frames
        self.combined_frame = combined
        self.samples_by_split = samples_by_split
        self.sequence_lengths_by_split = {
            split: [int(sample["sequence_length"]) for sample in samples]
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

    def train_dataloader(self):
        return self._make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader("test", shuffle=False)

    def _make_dataloader(self, split: str, *, shuffle: bool):
        dataset = QaiSequenceDataset(self.samples_by_split.get(split, []))
        return self._dataloader_cls(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch = self._torch
        sequence_lengths = torch.tensor(
            [int(sample["sequence_length"]) for sample in batch],
            dtype=torch.long,
        )
        max_length = int(sequence_lengths.max().item()) if len(batch) > 0 else 0
        feature_dim = batch[0]["features"].size(-1) if batch else self.feature_dim
        padded_features = torch.zeros(
            len(batch),
            max_length,
            feature_dim,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)

        for idx, sample in enumerate(batch):
            sample_length = int(sample["sequence_length"])
            padded_features[idx, :sample_length] = sample["features"]
            attention_mask[idx, :sample_length] = True

        return {
            "features": padded_features,
            "attention_mask": attention_mask,
            "sequence_lengths": sequence_lengths,
            "price_targets": torch.stack(
                [sample["price_targets"] for sample in batch], dim=0
            ),
            "class_targets": torch.stack(
                [sample["class_targets"] for sample in batch], dim=0
            ),
            "ticker": [sample["ticker"] for sample in batch],
            "time_range": [sample["time_range"] for sample in batch],
            "quarter_end_date": [sample["quarter_end_date"] for sample in batch],
        }


def _build_samples_by_split(
    *,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
    horizons: Sequence[int],
) -> dict[str, list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for QAI sample generation") from exc

    samples_by_split = {"train": [], "val": [], "test": []}
    required_price_columns = [f"future_price_t{h}" for h in horizons]
    required_class_columns = [f"class_target_t{h}" for h in horizons]

    for _, ticker_frame in frame.groupby("ticker", sort=False):
        ticker_frame = ticker_frame.sort_values("quarter_end_date").reset_index(drop=True).copy()
        for end_idx in range(len(ticker_frame)):
            row = ticker_frame.iloc[end_idx]
            if not bool(row.get("window_ready", False)):
                continue
            if end_idx + 1 < sequence_length:
                continue
            if not bool(row.get("horizon_target_ready", True)):
                continue
            if row[required_price_columns].isna().any():
                continue
            if row[required_class_columns].isna().any():
                continue

            window = ticker_frame.iloc[: end_idx + 1]
            features = torch.tensor(
                window.loc[:, feature_columns].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            price_targets = torch.tensor(
                row[required_price_columns].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            class_targets = torch.tensor(
                row[required_class_columns].to_numpy(dtype="int64"),
                dtype=torch.long,
            )
            split = str(row["split"])
            samples_by_split[split].append(
                {
                    "features": features,
                    "price_targets": price_targets,
                    "class_targets": class_targets,
                    "ticker": str(row["ticker"]),
                    "time_range": str(row["time_range"]),
                    "quarter_end_date": row["quarter_end_date"],
                    "sequence_length": len(window),
                }
            )

    return samples_by_split
