from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def run(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    if hasattr(datamodule, "setup"):
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
