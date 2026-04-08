from __future__ import annotations

from pathlib import Path

from src.datamodules.qai_datamodule import (
    QaiDataModule,
    _align_quarter_close_prices,
    _attach_future_targets,
    _load_price_history,
)
from tests.qai_fixtures import build_qai_fixture


def test_price_history_parsing_and_quarter_alignment(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    history = _load_price_history(paths["prices"])
    aligned = _align_quarter_close_prices(history)

    assert history["Date"].dtype.kind == "M"
    assert aligned["quarter_price"].tolist()[:4] == [100.0, 101.0, 98.0, 100.0]


def test_future_target_generation_and_class_boundaries(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    aligned = _align_quarter_close_prices(_load_price_history(paths["prices"]))
    targets = _attach_future_targets(aligned, horizons=[1, 2, 3, 4], flat_return_threshold=0.02)

    first = targets.iloc[0]
    assert first["future_price_t1"] == 101.0
    assert first["future_price_t2"] == 98.0
    assert first["class_target_t1"] == 1
    assert first["class_target_t2"] == 1
    assert first["class_target_t3"] == 1
    assert first["class_target_t4"] == 1


def test_qai_datamodule_builds_sequences_and_filters_missing_horizons(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    datamodule = QaiDataModule(
        train_path=str(paths["train"]),
        val_path=str(paths["val"]),
        test_path=str(paths["test"]),
        price_history_path=str(paths["prices"]),
        sequence_length=4,
        batch_size=2,
    )

    datamodule.setup()

    assert datamodule.feature_dim == 2
    assert len(datamodule.samples_by_split["train"]) == 1
    assert len(datamodule.samples_by_split["val"]) == 1
    assert len(datamodule.samples_by_split["test"]) == 0

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["features"].shape == (1, 4, 2)
    assert batch["price_targets"].shape == (1, 4)
    assert batch["class_targets"].shape == (1, 4)
    assert batch["features"].dtype.is_floating_point
    assert str(batch["class_targets"].dtype) == "torch.int64"
