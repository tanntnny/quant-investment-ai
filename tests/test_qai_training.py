from __future__ import annotations

from pathlib import Path

import torch

from src.callbacks.example import ExampleCallbacks
from src.datamodules.qai_datamodule import QaiDataModule
from src.logger.example import ExampleLogger
from src.losses.qai_multitask import QaiMultiTaskLoss
from src.metrics.qai_multitask import QaiMultiTaskMetrics
from src.models.qai_attention import QaiAttentionModel
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


def test_pytorch_trainer_qai_smoke(tmp_path: Path) -> None:
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
    assert "loss" in metrics
    assert "val_loss" in metrics
