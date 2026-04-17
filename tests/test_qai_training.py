from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.callbacks.example import ExampleCallbacks
from src.datamodules.qai_datamodule import QaiDataModule
from src.datamodules.qai_portfolio_datamodule import QaiPortfolioDataModule
from src.logger.example import ExampleLogger
from src.losses.qai_multitask import QaiMultiTaskLoss
from src.losses.qai_portfolio_growth import QaiPortfolioGrowthLoss
from src.metrics.qai_multitask import QaiMultiTaskMetrics
from src.metrics.qai_portfolio import QaiPortfolioMetrics
from src.models.qai_attention import QaiAttentionModel
from src.models.qai_portfolio_attention import QaiPortfolioAttentionModel
from src.pipelines.training_preflight import TrainingPreflightError, run_training_preflight
from src.trainers.pytorch_trainer import PytorchTrainer
from tests.qai_fixtures import build_qai_fixture


def test_qai_multitask_loss_smoke() -> None:
    loss_fn = QaiMultiTaskLoss()
    outputs = {
        "price_preds": torch.tensor([[100.0, 101.0, 102.0, 103.0]], dtype=torch.float32),
        "class_logits": torch.randn(1, 4, 3),
    }
    batch = {
        "price_targets": torch.tensor([[101.0, 100.0, 102.0, 104.0]], dtype=torch.float32),
        "class_targets": torch.tensor([[1, 0, 2, 1]], dtype=torch.long),
    }

    losses = loss_fn(outputs, batch)

    assert set(losses) == {"loss", "regression_loss", "classification_loss"}
    assert float(losses["loss"]) > 0.0


def test_qai_multitask_loss_applies_normalized_horizon_weights() -> None:
    loss_fn = QaiMultiTaskLoss(
        classification_weight=0.0,
        horizon_weights=[1.0, 3.0],
    )
    outputs = {
        "price_preds": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "class_logits": torch.randn(1, 2, 3),
    }
    batch = {
        "price_targets": torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        "class_targets": torch.tensor([[1, 2]], dtype=torch.long),
    }

    losses = loss_fn(outputs, batch)

    assert losses["regression_loss"].item() == pytest.approx(7.0)
    assert losses["loss"].item() == pytest.approx(7.0)


def test_pytorch_trainer_qai_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
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

    model = QaiAttentionModel(
        input_dim=datamodule.feature_dim,
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
    )
    trainer = PytorchTrainer(max_epochs=1, max_steps=1)
    metrics = trainer.fit(
        datamodule=datamodule,
        model=model,
        loss_fn=QaiMultiTaskLoss(),
        metric_fn=QaiMultiTaskMetrics(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        scheduler=None,
        callbacks=ExampleCallbacks(),
        logger=ExampleLogger(),
    )

    assert metrics["train_steps"] == 1
    assert "train_loss" in metrics
    assert "val_loss" in metrics


def test_training_preflight_rejects_reserved_feature_column(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    datamodule = QaiPortfolioDataModule(
        train_path=str(paths["train"]),
        val_path=str(paths["val"]),
        test_path=str(paths["test"]),
        price_history_path=str(paths["prices"]),
        sequence_length=4,
        batch_size=2,
        target_horizon=1,
        horizons=[1],
    )
    datamodule.setup()
    datamodule.feature_names.append("quarter_end_date_story")
    model = QaiPortfolioAttentionModel(
        input_dim=datamodule.feature_dim,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
    )

    with pytest.raises(TrainingPreflightError, match="helper columns"):
        run_training_preflight(
            datamodule=datamodule,
            model=model,
            trainer=PytorchTrainer(max_epochs=1, max_steps=1),
            loss_fn=QaiPortfolioGrowthLoss(),
            metric_fn=QaiPortfolioMetrics(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        )


def test_training_preflight_rejects_nan_features(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    datamodule = QaiPortfolioDataModule(
        train_path=str(paths["train"]),
        val_path=str(paths["val"]),
        test_path=str(paths["test"]),
        price_history_path=str(paths["prices"]),
        sequence_length=4,
        batch_size=2,
        target_horizon=1,
        horizons=[1],
    )
    datamodule.setup()
    datamodule.samples_by_split["train"][0]["features"][0, 0, 0] = float("nan")
    model = QaiPortfolioAttentionModel(
        input_dim=datamodule.feature_dim,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
    )

    with pytest.raises(TrainingPreflightError, match="NaN or infinite") as exc_info:
        run_training_preflight(
            datamodule=datamodule,
            model=model,
            trainer=PytorchTrainer(max_epochs=1, max_steps=1),
            loss_fn=QaiPortfolioGrowthLoss(),
            metric_fn=QaiPortfolioMetrics(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        )

    message = str(exc_info.value)
    assert "shape=" in message
    assert "nonfinite_count=1" in message
    assert "first_nonfinite_indices=" in message


def test_training_preflight_reports_portfolio_target_diagnostics(tmp_path: Path) -> None:
    paths = build_qai_fixture(tmp_path)
    datamodule = QaiPortfolioDataModule(
        train_path=str(paths["train"]),
        val_path=str(paths["val"]),
        test_path=str(paths["test"]),
        price_history_path=str(paths["prices"]),
        sequence_length=4,
        batch_size=2,
        target_frequency="daily",
        target_horizon=1,
        horizons=[1],
    )
    datamodule.setup()
    model = QaiPortfolioAttentionModel(
        input_dim=datamodule.feature_dim,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
    )

    with pytest.raises(TrainingPreflightError, match="Portfolio target diagnostics") as exc_info:
        run_training_preflight(
            datamodule=datamodule,
            model=model,
            trainer=PytorchTrainer(max_epochs=1, max_steps=1),
            loss_fn=QaiPortfolioGrowthLoss(),
            metric_fn=QaiPortfolioMetrics(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        )

    message = str(exc_info.value)
    assert "target_frequency=daily" in message
    assert "train_target_price_rows=0" in message


def test_training_preflight_allows_masked_portfolio_scores() -> None:
    class DummyDataModule:
        feature_dim = 2
        feature_names = ["feature_a", "feature_b"]

        def __init__(self) -> None:
            self.samples_by_split = {
                "train": [
                    {
                        "features": torch.ones(2, 4, 2),
                        "sequence_lengths": [4, 4],
                        "ticker_count": 2,
                        "current_prices": torch.tensor([100.0, 200.0]),
                        "future_prices": torch.tensor([101.0, 202.0]),
                    }
                ],
            }

        def train_dataloader(self):
            batch = {
                "features": torch.ones(1, 3, 4, 2),
                "history_attention_mask": torch.tensor(
                    [[
                        [True, True, True, True],
                        [True, True, True, True],
                        [False, False, False, False],
                    ]]
                ),
                "ticker_attention_mask": torch.tensor([[True, True, False]]),
                "current_prices": torch.tensor([[100.0, 200.0, 0.0]]),
                "future_prices": torch.tensor([[101.0, 202.0, 0.0]]),
            }
            return iter([batch])

    datamodule = DummyDataModule()
    model = QaiPortfolioAttentionModel(
        input_dim=datamodule.feature_dim,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        sequence_length=4,
    )

    summary = run_training_preflight(
        datamodule=datamodule,
        model=model,
        trainer=PytorchTrainer(max_epochs=1, max_steps=1),
        loss_fn=QaiPortfolioGrowthLoss(),
        metric_fn=QaiPortfolioMetrics(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    )

    assert summary["status"] == "passed"
