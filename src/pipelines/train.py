from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.console import announce, render_kv_table


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
    model = instantiate(model_cfg)
    loss_fn = instantiate(cfg.loss)
    metric_fn = instantiate(cfg.metrics)
    try:
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
    except TypeError:
        optimizer = instantiate(cfg.optimizer)
    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg is not None:
        try:
            scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
        except TypeError:
            scheduler = instantiate(scheduler_cfg)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer)

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
