from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.console import announce, render_kv_table
from src.utils.io import save_json
from src.utils.logging import ensure_dir
from src.utils.runtime import get_hydra_output_dir


class LightGBMPortfolioTrainer:
    requires_optimizer = False

    def fit(
        self,
        datamodule,
        model,
        loss_fn=None,
        metric_fn=None,
        optimizer=None,
        scheduler=None,
        callbacks=None,
        logger=None,
    ) -> dict:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to evaluate LightGBM portfolio batches.") from exc

        if hasattr(datamodule, "setup") and not getattr(datamodule, "samples_by_split", None):
            announce("Trainer calling datamodule.setup()", style="cyan")
            datamodule.setup()
        if callbacks is not None:
            callbacks.on_train_start()

        train_samples = list(datamodule.samples_by_split.get("train", []))
        model.fit(train_samples)

        metrics: dict[str, float | int] = {
            "epoch": 1,
            "train_steps": len(train_samples),
        }
        for split in ("train", "val", "test"):
            split_samples = list(datamodule.samples_by_split.get(split, []))
            if not split_samples:
                continue
            split_metrics = _evaluate_samples(
                samples=split_samples,
                model=model,
                metric_fn=metric_fn,
                torch=torch,
            )
            metrics[f"{split}_steps"] = len(split_samples)
            metrics.update({f"{split}_{key}": value for key, value in split_metrics.items()})

        if callbacks is not None:
            callbacks.on_train_end()
        if logger is not None:
            logger.log_metrics(metrics)

        _write_outputs(metrics)
        render_kv_table("LightGBM Portfolio Trainer Metrics", metrics)
        return metrics


def _evaluate_samples(*, samples, model, metric_fn, torch) -> dict[str, float]:
    weights_by_sample = model.predict_sample_weights(samples)
    if not weights_by_sample:
        return {}

    max_tickers = max(int(sample["ticker_count"]) for sample in samples)
    weights = torch.zeros(len(samples), max_tickers, dtype=torch.float32)
    current_prices = torch.zeros(len(samples), max_tickers, dtype=torch.float32)
    future_prices = torch.zeros(len(samples), max_tickers, dtype=torch.float32)
    ticker_attention_mask = torch.zeros(len(samples), max_tickers, dtype=torch.bool)

    for sample_idx, (sample, sample_weights) in enumerate(zip(samples, weights_by_sample)):
        ticker_count = int(sample["ticker_count"])
        weights[sample_idx, :ticker_count] = torch.tensor(
            np.asarray(sample_weights, dtype=np.float32),
            dtype=torch.float32,
        )
        current_prices[sample_idx, :ticker_count] = sample["current_prices"][:ticker_count]
        future_prices[sample_idx, :ticker_count] = sample["future_prices"][:ticker_count]
        ticker_attention_mask[sample_idx, :ticker_count] = True

    outputs = {"weights": weights}
    batch = {
        "current_prices": current_prices,
        "future_prices": future_prices,
        "ticker_attention_mask": ticker_attention_mask,
    }
    if metric_fn is None:
        return {
            "portfolio_growth": float(
                torch.sum(weights * future_prices, dim=-1).div(
                    torch.sum(weights * current_prices, dim=-1).clamp_min(1e-8)
                ).mean()
            )
        }
    return {
        key: float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
        for key, value in metric_fn(outputs, batch).items()
    }


def _write_outputs(metrics: dict[str, float | int]) -> None:
    root_run_dir = Path.cwd()
    output_run_dir = get_hydra_output_dir()
    run_dirs = [root_run_dir]
    if output_run_dir is not None and output_run_dir != root_run_dir:
        run_dirs.append(output_run_dir)
    for run_dir in run_dirs:
        ensure_dir(run_dir / "artifacts")
        ensure_dir(run_dir / "tables")
        save_json(metrics, run_dir / "metrics.json")
        save_json([metrics], run_dir / "epoch_metrics.json")
        epoch_csv = run_dir / "tables" / "epoch_metrics.csv"
        epoch_csv.write_text(
            ",".join(metrics.keys()) + "\n" + ",".join(str(value) for value in metrics.values()) + "\n"
        )
