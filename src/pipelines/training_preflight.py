from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.trainers.pytorch_trainer import _forward_model, _resolve_loss_tensor, _split_batch


class TrainingPreflightError(RuntimeError):
    """Raised when a training run cannot produce a valid model artifact."""


RESERVED_FEATURE_COLUMNS = {
    "quarter_end_date_story",
    "quarter_end_date_economics",
    "quarter_end_date_technical",
}


def run_training_preflight(
    *,
    datamodule,
    model,
    trainer,
    loss_fn,
    metric_fn,
    optimizer,
) -> dict[str, Any]:
    if hasattr(datamodule, "setup") and not getattr(datamodule, "samples_by_split", None):
        datamodule.setup()

    feature_names = list(getattr(datamodule, "feature_names", []) or [])
    _assert_valid_feature_schema(feature_names)
    _assert_positive_int("feature_dim", getattr(datamodule, "feature_dim", 0))

    samples_by_split = getattr(datamodule, "samples_by_split", {}) or {}
    train_samples = list(samples_by_split.get("train", []))
    if not train_samples:
        raise TrainingPreflightError("Training split has no samples; refusing to start training.")

    sample_counts = {split: len(list(samples_by_split.get(split, []))) for split in ("train", "val", "test")}
    for split, samples in samples_by_split.items():
        _validate_samples(split, list(samples))

    if getattr(model, "training_backend", None) == "lightgbm":
        _validate_lightgbm_training_rows(model, train_samples)
    elif getattr(trainer, "requires_optimizer", True):
        _validate_trainable_model(model, optimizer)
        _validate_first_torch_batch(datamodule, model, loss_fn, metric_fn)

    return {
        "status": "passed",
        "model": model.__class__.__name__,
        "trainer": trainer.__class__.__name__,
        "feature_dim": int(getattr(datamodule, "feature_dim", 0)),
        "reserved_feature_columns": "none",
        "train_samples": sample_counts.get("train", 0),
        "val_samples": sample_counts.get("val", 0),
        "test_samples": sample_counts.get("test", 0),
    }


def _assert_valid_feature_schema(feature_names: Sequence[str]) -> None:
    if not feature_names:
        raise TrainingPreflightError("Datamodule resolved zero feature columns.")

    reserved = sorted(column for column in feature_names if column in RESERVED_FEATURE_COLUMNS)
    generated_date_helpers = sorted(
        column for column in feature_names if column.startswith("quarter_end_date_")
    )
    bad_columns = sorted(set(reserved + generated_date_helpers))
    if bad_columns:
        raise TrainingPreflightError(
            "Datamodule feature schema includes merge/date helper columns that must not be model "
            f"features: {bad_columns}"
        )


def _assert_positive_int(name: str, value: Any) -> None:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingPreflightError(f"{name} must be a positive integer; got {value!r}.") from exc
    if numeric_value <= 0:
        raise TrainingPreflightError(f"{name} must be positive; got {numeric_value}.")


def _validate_samples(split: str, samples: list[dict[str, Any]]) -> None:
    for sample_idx, sample in enumerate(samples):
        context = f"{split} sample {sample_idx}"
        _assert_finite_value(sample.get("features"), f"{context}.features")

        if "sequence_length" in sample:
            _assert_positive_int(f"{context}.sequence_length", sample["sequence_length"])
        if "sequence_lengths" in sample:
            sequence_lengths = list(sample["sequence_lengths"])
            if not sequence_lengths:
                raise TrainingPreflightError(f"{context}.sequence_lengths is empty.")
            for ticker_idx, sequence_length in enumerate(sequence_lengths):
                _assert_positive_int(
                    f"{context}.sequence_lengths[{ticker_idx}]",
                    sequence_length,
                )

        if "ticker_count" in sample:
            _assert_positive_int(f"{context}.ticker_count", sample["ticker_count"])
        if "price_targets" in sample:
            _assert_finite_value(sample["price_targets"], f"{context}.price_targets")
        if "class_targets" in sample:
            _assert_finite_value(sample["class_targets"], f"{context}.class_targets")
        if "current_prices" in sample:
            _assert_positive_finite_tensor(sample["current_prices"], f"{context}.current_prices")
        if "future_prices" in sample:
            _assert_positive_finite_tensor(sample["future_prices"], f"{context}.future_prices")


def _validate_lightgbm_training_rows(model, train_samples: list[dict[str, Any]]) -> None:
    samples_to_xy = getattr(model, "_samples_to_xy", None)
    if not callable(samples_to_xy):
        raise TrainingPreflightError("LightGBM model does not expose _samples_to_xy for validation.")
    x_train, y_train = samples_to_xy(train_samples)
    if getattr(x_train, "size", 0) == 0 or getattr(y_train, "size", 0) == 0:
        raise TrainingPreflightError("LightGBM preflight produced zero training rows.")
    _assert_finite_value(x_train, "lightgbm.x_train")
    _assert_finite_value(y_train, "lightgbm.y_train")


def _validate_trainable_model(model, optimizer) -> None:
    parameters_fn = getattr(model, "parameters", None)
    if not callable(parameters_fn):
        raise TrainingPreflightError(f"{model.__class__.__name__} does not expose parameters().")
    trainable_params = sum(
        parameter.numel() for parameter in parameters_fn() if getattr(parameter, "requires_grad", False)
    )
    if trainable_params <= 0:
        raise TrainingPreflightError(
            f"{model.__class__.__name__} has zero trainable parameters; optimizer would be empty."
        )
    if optimizer is None:
        raise TrainingPreflightError("Trainer requires an optimizer but optimizer is None.")


def _validate_first_torch_batch(datamodule, model, loss_fn, metric_fn) -> None:
    try:
        import torch
    except ImportError as exc:
        raise TrainingPreflightError("PyTorch is required for neural training preflight.") from exc

    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader), None)
    if batch is None:
        raise TrainingPreflightError("Training dataloader yielded no batches.")

    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        features, targets = _split_batch(batch)
        _assert_finite_value(features, "first_train_batch.features")
        outputs = _forward_model(model, features, batch)
        _assert_finite_value(outputs, "first_train_batch.outputs")
        loss_result = loss_fn(outputs, targets)
        loss = _resolve_loss_tensor(loss_result)
        _assert_finite_value(loss, "first_train_batch.loss")
        metric_values = metric_fn(outputs, targets)
        _assert_finite_value(metric_values, "first_train_batch.metrics")
    if model_was_training:
        model.train()


def _assert_positive_finite_tensor(value: Any, name: str) -> None:
    _assert_finite_value(value, name)
    try:
        import torch
    except ImportError as exc:
        raise TrainingPreflightError("PyTorch is required for tensor validation.") from exc

    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(value)
    if tensor.numel() == 0:
        raise TrainingPreflightError(f"{name} is empty.")
    if bool((tensor <= 0).any().item()):
        raise TrainingPreflightError(f"{name} contains non-positive values.")


def _assert_finite_value(value: Any, name: str) -> None:
    if value is None:
        raise TrainingPreflightError(f"{name} is None.")

    if hasattr(value, "detach"):
        finite_mask = value.detach().isfinite()
        if not bool(finite_mask.all().item()):
            raise TrainingPreflightError(f"{name} contains NaN or infinite values.")
        return

    if hasattr(value, "to_numpy"):
        _assert_finite_value(value.to_numpy(), name)
        return

    if hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy as np

        array = np.asarray(value)
        if array.size == 0:
            raise TrainingPreflightError(f"{name} is empty.")
        if not np.isfinite(array).all():
            raise TrainingPreflightError(f"{name} contains NaN or infinite values.")
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_value(item, f"{name}.{key}")
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_value(item, f"{name}[{index}]")
        return

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise TrainingPreflightError(f"{name} contains NaN or infinite values.")
        return

