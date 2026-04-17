from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.callbacks.example import ExampleCallbacks
from src.datamodules.qai_datamodule import QaiDataModule
from src.datamodules.qai_portfolio_datamodule import QaiPortfolioDataModule
from src.logger.example import ExampleLogger
from src.losses.qai_multitask import QaiMultiTaskLoss
from src.metrics.qai_multitask import QaiMultiTaskMetrics
from src.metrics.qai_portfolio import QaiPortfolioMetrics
from src.models.qai_attention import QaiAttentionModel
from src.models import qai_portfolio_lightgbm as qai_portfolio_lightgbm_module
from src.models.qai_portfolio_lightgbm import QaiPortfolioLightGBMModel
from src.pipelines.training_preflight import TrainingPreflightError, run_training_preflight
from src.trainers.lightgbm_portfolio_trainer import LightGBMPortfolioTrainer
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


def test_lightgbm_portfolio_trainer_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lightgbm")
    monkeypatch.chdir(tmp_path)
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

    trainer = LightGBMPortfolioTrainer()
    metrics = trainer.fit(
        datamodule=datamodule,
        model=QaiPortfolioLightGBMModel(n_estimators=2, min_child_samples=1),
        loss_fn=None,
        metric_fn=QaiPortfolioMetrics(),
        optimizer=None,
        scheduler=None,
        callbacks=ExampleCallbacks(),
        logger=ExampleLogger(),
    )

    assert metrics["epoch"] == 1
    assert "train_portfolio_growth" in metrics
    assert (tmp_path / "artifacts" / "model.pkl").exists()


def test_lightgbm_portfolio_trainer_falls_back_without_lightgbm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
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

    original_find_spec = qai_portfolio_lightgbm_module.importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(
        qai_portfolio_lightgbm_module.importlib.util,
        "find_spec",
        fake_find_spec,
    )

    trainer = LightGBMPortfolioTrainer()
    model = QaiPortfolioLightGBMModel(n_estimators=2, min_child_samples=1)
    metrics = trainer.fit(
        datamodule=datamodule,
        model=model,
        loss_fn=None,
        metric_fn=QaiPortfolioMetrics(),
        optimizer=None,
        scheduler=None,
        callbacks=ExampleCallbacks(),
        logger=ExampleLogger(),
    )

    assert model._backend == "sklearn"
    assert metrics["epoch"] == 1
    assert "train_portfolio_growth" in metrics
    assert (tmp_path / "artifacts" / "model.pkl").exists()


def test_lightgbm_portfolio_model_accepts_input_dim() -> None:
    model = QaiPortfolioLightGBMModel(input_dim=8, n_estimators=2, min_child_samples=1)

    assert model.input_dim == 8


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

    with pytest.raises(TrainingPreflightError, match="helper columns"):
        run_training_preflight(
            datamodule=datamodule,
            model=QaiPortfolioLightGBMModel(n_estimators=2, min_child_samples=1),
            trainer=LightGBMPortfolioTrainer(),
            loss_fn=None,
            metric_fn=QaiPortfolioMetrics(),
            optimizer=None,
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

    with pytest.raises(TrainingPreflightError, match="NaN or infinite"):
        run_training_preflight(
            datamodule=datamodule,
            model=QaiPortfolioLightGBMModel(n_estimators=2, min_child_samples=1),
            trainer=LightGBMPortfolioTrainer(),
            loss_fn=None,
            metric_fn=QaiPortfolioMetrics(),
            optimizer=None,
        )
