from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.pipelines.training_preflight import run_training_preflight  # noqa: E402
from src.trainers.pytorch_trainer import (
    _forward_model,
    _move_to_device,
    _resolve_loss_tensor,
    _split_batch,
)  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import ensure_dir  # noqa: E402


def _resolve_auto_model_fields(model_cfg: dict[str, Any], datamodule) -> dict[str, Any]:
    resolved = dict(model_cfg)
    if resolved.get("input_dim") in {None, "auto"} and hasattr(datamodule, "feature_dim"):
        resolved["input_dim"] = datamodule.feature_dim
    if (
        resolved.get("sequence_length") in {None, "auto"}
        and hasattr(datamodule, "max_sequence_length")
    ):
        resolved["sequence_length"] = datamodule.max_sequence_length
    return resolved


def _metric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}

    metrics: dict[str, float] = {}
    for key, item in value.items():
        if key == "loss":
            continue
        if hasattr(item, "detach"):
            item = item.detach().cpu()
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, (int, float)):
            metrics[key] = float(item)
    return metrics


def _select_batch(dataloader, batch_index: int) -> dict[str, Any]:
    for index, batch in enumerate(dataloader):
        if index == batch_index:
            return batch
    raise RuntimeError(f"Training dataloader did not yield batch_index={batch_index}.")


def _run_overfit_loop(
    *,
    cfg: DictConfig,
    datamodule,
    model,
    loss_fn,
    metric_fn,
    optimizer,
    scheduler,
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if str(cfg.trainer.get("accelerator", "cpu")) in {"auto", "cuda", "gpu"}
        and torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    steps = int(cfg.overfit.steps)
    log_every_n_steps = max(int(cfg.overfit.log_every_n_steps), 1)
    batch_index = int(cfg.overfit.batch_index)
    batch = _move_to_device(_select_batch(datamodule.train_dataloader(), batch_index), device)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    initial_loss: float | None = None
    final_loss = float("inf")

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        features, targets = _split_batch(batch)
        outputs = _forward_model(model, features, batch)
        loss_result = loss_fn(outputs, targets)
        loss = _resolve_loss_tensor(loss_result)
        if not bool(loss.detach().isfinite().all().item()):
            raise RuntimeError(f"Non-finite overfit loss at step={step}.")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            metric_values = metric_fn(outputs, targets)

        loss_value = float(loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        best_loss = min(best_loss, loss_value)

        if step == 1 or step == steps or step % log_every_n_steps == 0:
            row: dict[str, float | int] = {
                "step": step,
                "loss": loss_value,
                **_metric_dict(loss_result),
                **_metric_dict(metric_values),
            }
            history.append(row)
            print(
                f"step={step} loss={loss_value:.8f} "
                f"best_loss={best_loss:.8f}"
            )

    initial_loss = float(initial_loss if initial_loss is not None else final_loss)
    best_loss_reduction = initial_loss - best_loss
    final_loss_reduction = initial_loss - final_loss
    min_loss_reduction = float(cfg.overfit.min_loss_reduction)
    passed = best_loss_reduction >= min_loss_reduction
    summary = {
        "status": "passed" if passed else "failed",
        "steps": steps,
        "batch_index": batch_index,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "final_loss_reduction": final_loss_reduction,
        "best_loss_reduction": best_loss_reduction,
        "min_loss_reduction": min_loss_reduction,
        "history": history,
    }
    if bool(cfg.overfit.fail_on_no_improvement) and not passed:
        raise RuntimeError(
            "Portfolio overfit test did not improve enough: "
            f"initial_loss={initial_loss:.8f}, best_loss={best_loss:.8f}, "
            f"min_loss_reduction={min_loss_reduction:.8f}."
        )
    return summary


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.get("seed", 123)))

    datamodule = instantiate(cfg.data)
    datamodule.setup()

    model_cfg = _resolve_auto_model_fields(
        OmegaConf.to_container(cfg.model, resolve=True),
        datamodule,
    )
    model = instantiate(model_cfg)
    loss_fn = instantiate(cfg.loss)
    metric_fn = instantiate(cfg.metrics)
    trainer = instantiate(cfg.trainer)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg is not None:
        scheduler = instantiate(scheduler_cfg, optimizer=optimizer)

    preflight_summary = run_training_preflight(
        datamodule=datamodule,
        model=model,
        trainer=trainer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
    )
    summary = _run_overfit_loop(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    payload = {
        "name": cfg.get("name", "qai_portfolio_overfit"),
        "model": model_cfg,
        "preflight": preflight_summary,
        "overfit": summary,
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_targets = [output_dir]
    artifact_dir = os.environ.get("RUN_ARTIFACTS_DIR")
    if artifact_dir:
        output_targets.append(Path(artifact_dir))

    for target_dir in output_targets:
        ensure_dir(target_dir / "artifacts")
        save_json(payload, target_dir / "artifacts" / "portfolio_overfit_report.json")
        save_json(summary, target_dir / "metrics.json")


if __name__ == "__main__":
    main()
