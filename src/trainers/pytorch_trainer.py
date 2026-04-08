from __future__ import annotations

import csv
from pathlib import Path

from src.utils.console import announce, render_kv_table, render_records_table
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

        epoch_history: list[dict[str, float | int]] = []
        total_train_steps = 0
        stop_training = False
        artifacts_dir = Path.cwd() / "artifacts"
        ensure_dir(artifacts_dir)
        best_val_loss = float("inf")
        best_checkpoint_summary: dict[str, float | int | None] | None = None

        for epoch in range(1, self.max_epochs + 1):
            train_aggregates: dict[str, float] = {}
            val_aggregates: dict[str, float] = {}
            train_steps = 0
            val_steps = 0

            model.train()
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
                total_train_steps += 1

                if 0 < self.max_steps <= total_train_steps:
                    stop_training = True
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

            epoch_metrics = {
                "epoch": epoch,
                "train_steps": train_steps,
                **_prefix_metrics(_average_metrics(train_aggregates, train_steps), "train_"),
            }
            if val_steps > 0:
                epoch_metrics["val_steps"] = val_steps
                epoch_metrics.update(
                    _prefix_metrics(_average_metrics(val_aggregates, val_steps), "val_")
                )
            epoch_history.append(epoch_metrics)
            logger.log_metrics(epoch_metrics)
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_metrics=epoch_metrics,
                path=artifacts_dir / "last_checkpoint.pt",
            )

            current_val_loss = epoch_metrics.get("val_loss")
            if isinstance(current_val_loss, (int, float)) and current_val_loss < best_val_loss:
                best_val_loss = float(current_val_loss)
                best_checkpoint_summary = {
                    "epoch": epoch_metrics.get("epoch"),
                    "val_loss": current_val_loss,
                }
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_metrics=epoch_metrics,
                    path=artifacts_dir / "best_checkpoint.pt",
                )
                save_json(epoch_metrics, artifacts_dir / "best_metrics.json")

            if stop_training:
                break

        callbacks.on_train_end()

        final_metrics = dict(epoch_history[-1]) if epoch_history else {"epoch": 0}
        render_records_table(
            "Pytorch Epoch Summary",
            _build_epoch_summary_rows(epoch_history),
            columns=[
                ("epoch", "epoch"),
                ("train_loss", "train/loss"),
                ("val_loss", "val/loss"),
                ("train_price_mae", "train/price_mae"),
                ("val_price_mae", "val/price_mae"),
                ("train_class_accuracy", "train/accuracy"),
                ("val_class_accuracy", "val/accuracy"),
            ],
        )
        render_kv_table("Pytorch Trainer Metrics", final_metrics)

        run_dir = Path.cwd()
        ensure_dir(run_dir / "artifacts")
        ensure_dir(run_dir / "figures")
        ensure_dir(run_dir / "tables")
        save_json(epoch_history, run_dir / "epoch_metrics.json")
        _save_epoch_metrics_csv(epoch_history, run_dir / "tables" / "epoch_metrics.csv")
        save_json(final_metrics, run_dir / "metrics.json")
        if best_checkpoint_summary is not None:
            save_json(best_checkpoint_summary, run_dir / "artifacts" / "best_checkpoint_summary.json")
        return final_metrics


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


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _build_epoch_summary_rows(epoch_history: list[dict[str, float | int]]) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for epoch_metrics in epoch_history:
        rows.append(
            {
                "epoch": epoch_metrics.get("epoch"),
                "train_loss": epoch_metrics.get("train_loss"),
                "val_loss": epoch_metrics.get("val_loss"),
                "train_price_mae": epoch_metrics.get("train_price_mae"),
                "val_price_mae": epoch_metrics.get("val_price_mae"),
                "train_class_accuracy": epoch_metrics.get("train_class_accuracy"),
                "val_class_accuracy": epoch_metrics.get("val_class_accuracy"),
            }
        )
    return rows


def _save_epoch_metrics_csv(
    epoch_history: list[dict[str, float | int]],
    path: Path,
) -> None:
    if not epoch_history:
        return

    fieldnames = sorted({key for row in epoch_history for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in epoch_history:
            writer.writerow(row)


def _save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    epoch_metrics: dict[str, float | int],
    path: Path,
) -> None:
    import torch

    checkpoint = {
        "epoch": epoch_metrics.get("epoch"),
        "metrics": epoch_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _resolve_device(torch_module, accelerator: str):
    normalized = (accelerator or "cpu").strip().lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if normalized == "gpu":
        normalized = "cuda"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("Trainer requested CUDA, but torch.cuda.is_available() is false")
    return torch_module.device(normalized)
