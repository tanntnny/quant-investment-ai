from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.pipelines.training_preflight import (
    TrainingPreflightError,
    run_training_preflight,
)
from src.utils.console import announce, render_kv_table, render_records_table
from src.utils.io import save_json
from src.utils.logging import ensure_dir
from src.utils.runtime import get_hydra_output_dir


def run(cfg: DictConfig) -> None:
    announce("Starting QAI training pipeline", style="green")
    render_kv_table(
        "QAI Train Pipeline",
        {
            "mode": cfg.mode.name,
            "data": cfg.data._target_ if "_target_" in cfg.data else str(cfg.data),
            "model": cfg.model._target_ if "_target_" in cfg.model else str(cfg.model),
            "trainer": cfg.trainer._target_ if "_target_" in cfg.trainer else str(cfg.trainer),
        },
    )

    datamodule = instantiate(cfg.data)
    if hasattr(datamodule, "setup"):
        announce("Setting up datamodule", style="cyan")
        datamodule.setup()

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if (
        isinstance(model_cfg, dict)
        and model_cfg.get("input_dim") in {None, "auto"}
        and hasattr(datamodule, "feature_dim")
    ):
        model_cfg["input_dim"] = datamodule.feature_dim
    if (
        isinstance(model_cfg, dict)
        and model_cfg.get("sequence_length") in {None, "auto"}
        and hasattr(datamodule, "max_sequence_length")
    ):
        model_cfg["sequence_length"] = datamodule.max_sequence_length
    model = instantiate(model_cfg)
    loss_fn = instantiate(cfg.loss)
    metric_fn = instantiate(cfg.metrics)
    trainer = instantiate(cfg.trainer)
    optimizer = None
    if getattr(trainer, "requires_optimizer", True):
        try:
            optimizer = instantiate(cfg.optimizer, params=model.parameters())
        except TypeError:
            optimizer = instantiate(cfg.optimizer)
    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg is not None and optimizer is not None:
        try:
            scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
        except TypeError:
            scheduler = instantiate(scheduler_cfg)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)

    try:
        preflight_summary = run_training_preflight(
            datamodule=datamodule,
            model=model,
            trainer=trainer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            optimizer=optimizer,
        )
    except TrainingPreflightError as exc:
        announce("QAI training preflight failed", style="bold red")
        render_kv_table(
            "QAI Training Preflight Failure",
            {
                "error": exc,
                "model": model.__class__.__name__,
                "trainer": trainer.__class__.__name__,
                "feature_dim": getattr(datamodule, "feature_dim", "-"),
            },
        )
        raise
    render_kv_table("QAI Training Preflight", preflight_summary)

    run_dir = Path.cwd()
    output_run_dir = get_hydra_output_dir()
    artifact_run_dirs = [run_dir]
    if output_run_dir is not None and output_run_dir != run_dir:
        artifact_run_dirs.append(output_run_dir)
    for artifact_run_dir in artifact_run_dirs:
        ensure_dir(artifact_run_dir / "artifacts")
        ensure_dir(artifact_run_dir / "tables")

    data_summary_rows = _build_data_summary_rows(datamodule)
    render_records_table(
        "QAI Data Summary",
        data_summary_rows,
        columns=[
            ("split", "split"),
            ("raw_rows", "raw_rows"),
            ("samples", "samples"),
            ("tickers", "tickers"),
            ("avg_seq_len", "avg_seq_len"),
        ],
    )

    training_hyperparams = _build_training_hyperparams(
        cfg=cfg,
        datamodule=datamodule,
        model_cfg=model_cfg,
    )
    render_kv_table("QAI Training Hyperparameters", training_hyperparams)

    model_structure_rows = _build_model_structure_rows(model)
    render_kv_table(
        "QAI Model Summary",
        {
            "model": model.__class__.__name__,
            "total_params": _count_parameters(model),
            "trainable_params": _count_parameters(model, trainable_only=True),
        },
    )
    render_records_table(
        "QAI Model Structure",
        model_structure_rows,
        columns=[
            ("module", "module"),
            ("type", "type"),
            ("parameters", "parameters"),
        ],
    )

    training_setup_payload = _build_training_setup_payload(cfg=cfg, model_cfg=model_cfg)
    for artifact_run_dir in artifact_run_dirs:
        save_json(data_summary_rows, artifact_run_dir / "artifacts" / "data_summary.json")
        save_json(training_hyperparams, artifact_run_dir / "artifacts" / "training_hyperparameters.json")
        save_json(model_structure_rows, artifact_run_dir / "artifacts" / "model_structure.json")
        save_json(
            training_setup_payload,
            artifact_run_dir / "artifacts" / "training_setup.json",
        )
        save_json(preflight_summary, artifact_run_dir / "artifacts" / "training_preflight.json")
        (artifact_run_dir / "artifacts" / "model_structure.txt").write_text(f"{model}\n")

    render_kv_table(
        "QAI Trainer Ready",
        {
            "datamodule": datamodule.__class__.__name__,
            "model": model.__class__.__name__,
            "callbacks": callbacks.__class__.__name__,
            "logger": logger.__class__.__name__,
        },
    )
    announce("Entering trainer.fit", style="green")
    trainer.fit(
        datamodule=datamodule,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        logger=logger,
    )


def _build_data_summary_rows(datamodule) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | float | str]] = []
    for split in ("train", "val", "test"):
        split_frame = getattr(datamodule, "split_frames", {}).get(split)
        samples = getattr(datamodule, "samples_by_split", {}).get(split, [])
        sequence_lengths = getattr(datamodule, "sequence_lengths_by_split", {}).get(split, [])
        rows.append(
            {
                "split": split,
                "raw_rows": len(split_frame) if split_frame is not None else 0,
                "samples": len(samples),
                "tickers": split_frame["ticker"].nunique() if split_frame is not None else 0,
                "avg_seq_len": _average_sequence_length(sequence_lengths),
            }
        )
    return rows


def _build_training_hyperparams(
    *,
    cfg: DictConfig,
    datamodule,
    model_cfg: dict,
) -> dict[str, object]:
    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)

    return {
        "experiment": cfg.get("name", "train"),
        "batch_size": getattr(datamodule, "batch_size", "-"),
        "min_sequence_length": getattr(datamodule, "min_sequence_length", "-"),
        "max_sequence_length": getattr(datamodule, "max_sequence_length", "-"),
        "max_tickers_per_sample": getattr(datamodule, "max_tickers_per_sample", "-"),
        "target_horizon": getattr(datamodule, "target_horizon", "-"),
        "feature_dim": getattr(datamodule, "feature_dim", "-"),
        "max_epochs": trainer_cfg.get("max_epochs"),
        "max_steps": trainer_cfg.get("max_steps"),
        "accelerator": trainer_cfg.get("accelerator"),
        "precision": trainer_cfg.get("precision"),
        "optimizer": _target_name(optimizer_cfg),
        "learning_rate": optimizer_cfg.get("lr"),
        "weight_decay": optimizer_cfg.get("weight_decay"),
        "loss": _target_name(loss_cfg),
        "regression_loss": loss_cfg.get("regression_loss"),
        "use_log_growth": loss_cfg.get("use_log_growth"),
        "risk_penalty": loss_cfg.get("risk_penalty"),
        "concentration_penalty": loss_cfg.get("concentration_penalty"),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "num_heads": model_cfg.get("num_heads"),
        "num_layers": model_cfg.get("num_layers"),
        "dropout": model_cfg.get("dropout"),
        "forecast_horizons": model_cfg.get("forecast_horizons"),
    }


def _build_model_structure_rows(model) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    named_children = getattr(model, "named_children", None)
    if not callable(named_children):
        return rows
    for module_name, module in named_children():
        rows.append(
            {
                "module": module_name,
                "type": module.__class__.__name__,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
            }
        )
    return rows


def _build_training_setup_payload(*, cfg: DictConfig, model_cfg: dict) -> dict[str, object]:
    return {
        "name": cfg.get("name", "train"),
        "mode": OmegaConf.to_container(cfg.mode, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "model": model_cfg,
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
        "metrics": OmegaConf.to_container(cfg.metrics, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True)
        if cfg.get("scheduler") is not None
        else None,
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
    }


def _count_parameters(model, *, trainable_only: bool = False) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(
        parameter.numel()
        for parameter in parameters()
        if not trainable_only or parameter.requires_grad
    )


def _target_name(config: dict[str, object]) -> str:
    return str(config.get("_target_", "-")).rsplit(".", maxsplit=1)[-1]


def _average_sequence_length(sequence_lengths: list[int]) -> float:
    if not sequence_lengths:
        return 0.0
    return sum(sequence_lengths) / len(sequence_lengths)
