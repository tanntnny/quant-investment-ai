from __future__ import annotations

from pathlib import Path

from src.utils.console import announce, render_kv_table
from src.utils.io import save_json
from src.utils.logging import ensure_dir


class PytorchTrainer:
    def __init__(
        self,
        max_epochs: int = 1,
        max_steps: int = -1,
        precision: int = 32,
        accelerator: str = "cpu",
        gradient_accumulation_steps: int = 1,
        val_check_interval: float = 1.0,
    ) -> None:
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.precision = precision
        self.accelerator = accelerator
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.val_check_interval = val_check_interval

    def fit(
        self,
        datamodule,
        model,
        loss_fn,
        metric_fn,
        optimizer,
        scheduler,
        callbacks,
        logger,
    ) -> dict:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for PytorchTrainer") from exc

        device = _resolve_device(torch, self.accelerator)
        render_kv_table(
            "Pytorch Trainer Start",
            {
                "max_epochs": self.max_epochs,
                "max_steps": self.max_steps,
                "precision": self.precision,
                "accelerator": self.accelerator,
                "device": str(device),
                "grad_accum_steps": self.gradient_accumulation_steps,
                "val_check_interval": self.val_check_interval,
            },
        )
        model.to(device)
        if hasattr(datamodule, "setup") and not getattr(datamodule, "samples_by_split", None):
            announce("Trainer calling datamodule.setup()", style="cyan")
            datamodule.setup()
        announce("Trainer starting train loop", style="green")
        callbacks.on_train_start()

        train_aggregates: dict[str, float] = {}
        val_aggregates: dict[str, float] = {}
        train_steps = 0
        val_steps = 0

        model.train()
        for _ in range(self.max_epochs):
            for batch in datamodule.train_dataloader():
                optimizer.zero_grad(set_to_none=True)
                batch = _move_to_device(batch, device)
                features, targets = _split_batch(batch)
                outputs = model(features)
                loss_result = loss_fn(outputs, targets)
                loss = _resolve_loss_tensor(loss_result)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    metric_values = metric_fn(outputs, targets)

                _accumulate_metrics(
                    train_aggregates,
                    {"loss": loss.detach(), **_as_metric_dict(loss_result), **_as_metric_dict(metric_values)},
                )
                train_steps += 1

                if 0 < self.max_steps <= train_steps:
                    break
            if 0 < self.max_steps <= train_steps:
                break

        if hasattr(datamodule, "val_dataloader"):
            model.eval()
            with torch.no_grad():
                for batch in datamodule.val_dataloader():
                    batch = _move_to_device(batch, device)
                    features, targets = _split_batch(batch)
                    outputs = model(features)
                    loss_result = loss_fn(outputs, targets)
                    metric_values = metric_fn(outputs, targets)
                    _accumulate_metrics(
                        val_aggregates,
                        {
                            "loss": _resolve_loss_tensor(loss_result).detach(),
                            **_as_metric_dict(loss_result),
                            **_as_metric_dict(metric_values),
                        },
                    )
                    val_steps += 1

        callbacks.on_train_end()

        metrics = {"train_steps": train_steps, **_average_metrics(train_aggregates, train_steps)}
        if val_steps > 0:
            metrics["val_steps"] = val_steps
            metrics.update(
                {
                    f"val_{key}": value
                    for key, value in _average_metrics(val_aggregates, val_steps).items()
                }
            )
        logger.log_metrics(metrics)
        render_kv_table("Pytorch Trainer Metrics", metrics)

        run_dir = Path.cwd()
        ensure_dir(run_dir / "artifacts")
        ensure_dir(run_dir / "figures")
        ensure_dir(run_dir / "tables")
        save_json(metrics, run_dir / "metrics.json")
        return metrics


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(item, device) for item in batch)
    return batch.to(device) if hasattr(batch, "to") else batch


def _split_batch(batch):
    if isinstance(batch, dict):
        return batch["features"], batch
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch format: {type(batch)!r}")


def _resolve_loss_tensor(loss_result):
    if isinstance(loss_result, dict):
        return loss_result["loss"]
    return loss_result


def _as_metric_dict(value):
    if isinstance(value, dict):
        return {
            key: item
            for key, item in value.items()
            if key != "loss"
        }
    return {"metric": value}


def _accumulate_metrics(aggregates: dict[str, float], values: dict[str, object]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        aggregates[key] = aggregates.get(key, 0.0) + float(value)


def _average_metrics(aggregates: dict[str, float], steps: int) -> dict[str, float]:
    return {key: value / max(steps, 1) for key, value in aggregates.items()}


def _resolve_device(torch_module, accelerator: str):
    normalized = (accelerator or "cpu").strip().lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if normalized == "gpu":
        normalized = "cuda"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("Trainer requested CUDA, but torch.cuda.is_available() is false")
    return torch_module.device(normalized)
