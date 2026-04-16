from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datamodules.qai_datamodule import (
    _align_quarter_close_prices,
    _attach_future_targets,
    _load_price_history,
    _load_split_frame,
)
from src.utils.console import announce, render_kv_table, render_records_table
from src.utils.io import save_json
from src.utils.logging import ensure_dir


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for QaiPortfolioMultirunEvaluator") from exc


class QaiPortfolioMultirunEvaluator:
    def __init__(
        self,
        multirun_root: str = "saves/run_artifacts/train_qai_portfolio",
        evaluation_scopes: list[str] | None = None,
        checkpoint_filename: str = "artifacts/best_checkpoint.pt",
        top_strategy_count: int = 5,
        top_holdings_per_quarter: int = 10,
        device: str = "cpu",
        write_outputs: bool = True,
    ) -> None:
        self.multirun_root = multirun_root
        self.evaluation_scopes = evaluation_scopes or ["val", "all"]
        self.checkpoint_filename = checkpoint_filename
        self.top_strategy_count = top_strategy_count
        self.top_holdings_per_quarter = top_holdings_per_quarter
        self.device = device
        self.write_outputs = write_outputs

    def evaluate(self, datamodule, model, metric_fn, run_dir: Path) -> dict:
        _ = datamodule, model, metric_fn
        multirun_root = self._resolve_multirun_root(self.multirun_root)
        run_dirs = self._discover_run_dirs(multirun_root)
        device = self._resolve_device(self.device)
        eval_output_dir = run_dir / "eval_outputs"
        if self.write_outputs:
            ensure_dir(eval_output_dir)

        render_kv_table(
            "Portfolio Multirun Evaluation",
            {
                "multirun_root": str(multirun_root),
                "run_count": len(run_dirs),
                "evaluation_scopes": ", ".join(self.evaluation_scopes),
                "device": str(device),
                "write_outputs": self.write_outputs,
            },
        )

        raw_counts_df, target_counts_df, sample_counts_df = self._build_validation_diagnostics()
        render_records_table(
            "Validation Raw Counts",
            raw_counts_df.to_dict(orient="records"),
        )
        render_records_table(
            "Validation Target Availability",
            target_counts_df.to_dict(orient="records"),
        )
        render_records_table(
            "Portfolio Samples by Split",
            sample_counts_df.to_dict(orient="records"),
        )

        val_results = self._summarize_scope(run_dirs, "val", device)
        top_run_ids = val_results["run_summary"].head(self.top_strategy_count)["run_id"].tolist()
        render_records_table(
            "Validation Top Strategies",
            val_results["run_summary"].head(self.top_strategy_count).to_dict(orient="records"),
        )

        metrics_payload: dict[str, Any] = {
            "multirun_root": str(multirun_root),
            "run_eval_output_dir": str(eval_output_dir),
            "top_runs_from_val": top_run_ids,
            "validation_diagnostics": {
                "raw_row_groups": raw_counts_df.to_dict(orient="records"),
                "target_availability": target_counts_df.to_dict(orient="records"),
                "portfolio_sample_counts": sample_counts_df.to_dict(orient="records"),
            },
            "scopes": {},
        }

        heatmap_path = multirun_root / "portfolio_eval_heatmaps_val.png"
        run_heatmap_path = eval_output_dir / "portfolio_eval_heatmaps_val.png"
        self._plot_validation_heatmaps(val_results["run_summary"], heatmap_path)
        self._plot_validation_heatmaps(val_results["run_summary"], run_heatmap_path)

        for scope in self.evaluation_scopes:
            scope_results = val_results if scope == "val" else self._summarize_scope(run_dirs, scope, device)
            output_paths = self._make_output_paths(multirun_root, scope)
            run_output_paths = self._make_output_paths(eval_output_dir, scope)
            top_tables = self._build_top_strategy_tables(
                scope_results["weights"],
                scope_results["sample_summary"],
                val_results["run_summary"],
                top_run_ids,
            )
            nav_frames = self._build_nav_frames(scope_results["sample_summary"])
            self._plot_scope_growth(scope, nav_frames, top_run_ids, output_paths["growth_plot"])
            self._plot_scope_growth(scope, nav_frames, top_run_ids, run_output_paths["growth_plot"])

            render_records_table(
                f"{scope.upper()} Run Summary",
                scope_results["run_summary"].to_dict(orient="records"),
            )
            render_records_table(
                f"{scope.upper()} Top-{self.top_strategy_count} Growth",
                top_tables["growth"].to_dict(orient="records"),
            )

            if self.write_outputs:
                scope_results["run_summary"].to_csv(output_paths["run_summary"], index=False)
                scope_results["weights"].to_csv(output_paths["sample_weights"], index=False)
                top_tables["growth"].to_csv(output_paths["growth_by_quarter"], index=False)
                top_tables["allocations"].to_csv(output_paths["top5_allocations"], index=False)
                top_tables["prices"].to_csv(output_paths["top_holdings_prices"], index=False)
                scope_results["run_summary"].to_csv(run_output_paths["run_summary"], index=False)
                scope_results["weights"].to_csv(run_output_paths["sample_weights"], index=False)
                top_tables["growth"].to_csv(run_output_paths["growth_by_quarter"], index=False)
                top_tables["allocations"].to_csv(run_output_paths["top5_allocations"], index=False)
                top_tables["prices"].to_csv(run_output_paths["top_holdings_prices"], index=False)

            metrics_payload["scopes"][scope] = {
                "run_count": int(len(scope_results["run_summary"])),
                "sample_count": int(len(scope_results["sample_summary"])),
                "dates": sorted(
                    {
                        pd.Timestamp(value).strftime("%Y-%m-%d")
                        for value in scope_results["sample_summary"]["quarter_end_date"].dropna()
                    }
                ),
                "top_strategies": scope_results["run_summary"]
                .head(self.top_strategy_count)
                .to_dict(orient="records"),
                "benchmark_terminal_nav": float(nav_frames["benchmark"]["rebased_nav"].iloc[-1]),
                "outputs": {
                    "multirun_root": {key: str(value) for key, value in output_paths.items()},
                    "run_dir": {key: str(value) for key, value in run_output_paths.items()},
                },
            }

        diagnostics_csv_path = multirun_root / "portfolio_eval_val_diagnostics.csv"
        run_diagnostics_csv_path = eval_output_dir / "portfolio_eval_val_diagnostics.csv"
        summary_json_path = multirun_root / "portfolio_eval_summary.json"
        run_summary_json_path = eval_output_dir / "portfolio_eval_summary.json"
        if self.write_outputs:
            target_counts_df.to_csv(diagnostics_csv_path, index=False)
            target_counts_df.to_csv(run_diagnostics_csv_path, index=False)
            save_json(metrics_payload, summary_json_path)
            save_json(metrics_payload, run_summary_json_path)

        ensure_dir(run_dir)
        save_json(metrics_payload, run_dir / "metrics_eval.json")
        return metrics_payload

    def _resolve_device(self, configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if configured_device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(configured_device)

    def _resolve_multirun_root(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (self._repo_root() / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Multirun path does not exist: {candidate}")
        if (candidate / "multirun.yaml").exists():
            return candidate
        if candidate.name.isdigit() and (candidate.parent / "multirun.yaml").exists():
            return candidate.parent
        raise FileNotFoundError(
            f"Could not resolve a multirun root from {candidate}; expected multirun.yaml in that folder or its parent."
        )

    def _repo_root(self) -> Path:
        current = Path.cwd().resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return candidate
        raise FileNotFoundError("Could not locate project root.")

    def _discover_run_dirs(self, multirun_root: Path) -> list[Path]:
        return sorted(
            [path for path in multirun_root.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda item: int(item.name),
        )

    def _load_yaml(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text())

    def _load_json(self, path: Path) -> dict | list | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _resolve_repo_path(self, path_value: str | Path | None) -> str | None:
        if path_value in {None, "", "null"}:
            return path_value
        path = Path(path_value)
        return str(path if path.is_absolute() else (self._repo_root() / path).resolve())

    def _build_datamodule(self, run_cfg: dict):
        data_cfg_dict = dict(run_cfg["data"])
        for key in ["train_path", "val_path", "test_path", "price_history_path"]:
            if key in data_cfg_dict:
                data_cfg_dict[key] = self._resolve_repo_path(data_cfg_dict[key])
        datamodule = instantiate(OmegaConf.create(data_cfg_dict))
        datamodule.setup()
        return datamodule

    def _build_model(self, run_cfg: dict, datamodule):
        model_cfg = dict(run_cfg["model"])
        if model_cfg.get("input_dim") in {None, "auto"}:
            model_cfg["input_dim"] = datamodule.feature_dim
        if model_cfg.get("sequence_length") in {None, "auto"}:
            model_cfg["sequence_length"] = datamodule.max_sequence_length
        return instantiate(OmegaConf.create(model_cfg))

    def _dataloaders_for_scope(self, datamodule, evaluation_scope: str):
        if evaluation_scope == "val":
            return [("val", datamodule.val_dataloader())]
        if evaluation_scope == "all":
            return [
                ("train", datamodule.train_dataloader()),
                ("val", datamodule.val_dataloader()),
                ("test", datamodule.test_dataloader()),
            ]
        raise ValueError(f"Unsupported evaluation_scope: {evaluation_scope}")

    def _move_to_device(self, value, device: torch.device):
        if isinstance(value, dict):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._move_to_device(item, device) for item in value)
        return value.to(device) if hasattr(value, "to") else value

    def _evaluate_run(
        self,
        run_dir: Path,
        evaluation_scope: str,
        device: torch.device,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        run_cfg = self._load_yaml(run_dir / "config_resolved.yaml")
        final_metrics = self._load_json(run_dir / "metrics.json") or {}
        best_metrics = self._load_json(run_dir / "artifacts" / "best_metrics.json") or {}
        best_checkpoint_summary = (
            self._load_json(run_dir / "artifacts" / "best_checkpoint_summary.json") or {}
        )

        datamodule = self._build_datamodule(run_cfg)
        model = self._build_model(run_cfg, datamodule)
        checkpoint_path = run_dir / self.checkpoint_filename
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        sample_rows: list[dict[str, Any]] = []
        sample_summary_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for source_split, dataloader in self._dataloaders_for_scope(datamodule, evaluation_scope):
                for batch in dataloader:
                    batch = self._move_to_device(batch, device)
                    outputs = model(
                        batch["features"],
                        history_attention_mask=batch.get("history_attention_mask"),
                        ticker_attention_mask=batch.get("ticker_attention_mask"),
                    )
                    weights = outputs["weights"].detach().cpu()
                    current_prices = batch["current_prices"].detach().cpu()
                    future_prices = batch["future_prices"].detach().cpu()
                    ticker_attention_mask = batch["ticker_attention_mask"].detach().cpu().bool()

                    for sample_idx in range(weights.size(0)):
                        valid_mask = ticker_attention_mask[sample_idx]
                        ticker_count = int(valid_mask.sum().item())
                        if ticker_count == 0:
                            continue

                        valid_weights = weights[sample_idx][valid_mask].numpy().astype(float)
                        valid_weights = valid_weights / max(valid_weights.sum(), 1e-8)
                        valid_current = current_prices[sample_idx][valid_mask].numpy().astype(float)
                        valid_future = future_prices[sample_idx][valid_mask].numpy().astype(float)
                        equal_weights = np.full(ticker_count, 1.0 / ticker_count, dtype=float)
                        tickers = list(batch["tickers"][sample_idx])[:ticker_count]
                        quarter_end_date = pd.Timestamp(batch["quarter_end_date"][sample_idx]).normalize()
                        future_quarter_end_date = pd.Timestamp(
                            batch["future_quarter_end_date"][sample_idx]
                        ).normalize()
                        time_range = str(batch["time_range"][sample_idx])

                        weighted_current = float(np.sum(valid_weights * valid_current))
                        weighted_future = float(np.sum(valid_weights * valid_future))
                        benchmark_current = float(np.sum(equal_weights * valid_current))
                        benchmark_future = float(np.sum(equal_weights * valid_future))
                        portfolio_growth = weighted_future / max(weighted_current, 1e-8)
                        benchmark_growth = benchmark_future / max(benchmark_current, 1e-8)
                        growth_alpha = portfolio_growth - benchmark_growth
                        max_weight = float(valid_weights.max())
                        effective_holdings = float(1.0 / max(np.square(valid_weights).sum(), 1e-8))
                        sample_key = f"{source_split}__{time_range}__{quarter_end_date.date()}"

                        sample_summary_rows.append(
                            {
                                "evaluation_scope": evaluation_scope,
                                "source_split": source_split,
                                "run_id": int(run_dir.name),
                                "sample_key": sample_key,
                                "time_range": time_range,
                                "quarter_end_date": quarter_end_date,
                                "future_quarter_end_date": future_quarter_end_date,
                                "risk_penalty": float(run_cfg["loss"]["risk_penalty"]),
                                "concentration_penalty": float(run_cfg["loss"]["concentration_penalty"]),
                                "ticker_count": ticker_count,
                                "portfolio_growth": portfolio_growth,
                                "benchmark_growth": benchmark_growth,
                                "growth_alpha": growth_alpha,
                                "max_weight": max_weight,
                                "effective_holdings": effective_holdings,
                            }
                        )

                        ticker_returns = valid_future / np.maximum(valid_current, 1e-8) - 1.0
                        for ticker, weight, equal_weight, current_price, future_price, ticker_return in zip(
                            tickers,
                            valid_weights,
                            equal_weights,
                            valid_current,
                            valid_future,
                            ticker_returns,
                        ):
                            sample_rows.append(
                                {
                                    "evaluation_scope": evaluation_scope,
                                    "source_split": source_split,
                                    "run_id": int(run_dir.name),
                                    "risk_penalty": float(run_cfg["loss"]["risk_penalty"]),
                                    "concentration_penalty": float(run_cfg["loss"]["concentration_penalty"]),
                                    "sample_key": sample_key,
                                    "time_range": time_range,
                                    "quarter_end_date": quarter_end_date,
                                    "future_quarter_end_date": future_quarter_end_date,
                                    "ticker": str(ticker),
                                    "weight": float(weight),
                                    "equal_weight": float(equal_weight),
                                    "current_price": float(current_price),
                                    "future_price": float(future_price),
                                    "ticker_return": float(ticker_return),
                                    "portfolio_growth": portfolio_growth,
                                    "benchmark_growth": benchmark_growth,
                                    "growth_alpha": growth_alpha,
                                    "max_weight": max_weight,
                                    "effective_holdings": effective_holdings,
                                }
                            )

        metadata = {
            "evaluation_scope": evaluation_scope,
            "run_id": int(run_dir.name),
            "risk_penalty": float(run_cfg["loss"]["risk_penalty"]),
            "concentration_penalty": float(run_cfg["loss"]["concentration_penalty"]),
            "best_epoch": best_metrics.get("epoch")
            or best_checkpoint_summary.get("epoch")
            or checkpoint.get("epoch"),
            "final_epoch": final_metrics.get("epoch"),
            "saved_best_val_growth_alpha": best_metrics.get("val_growth_alpha"),
            "saved_best_val_effective_holdings": best_metrics.get("val_effective_holdings"),
            "saved_final_val_growth_alpha": final_metrics.get("val_growth_alpha"),
            "saved_final_val_effective_holdings": final_metrics.get("val_effective_holdings"),
        }
        return pd.DataFrame(sample_rows), pd.DataFrame(sample_summary_rows), metadata

    def _summarize_scope(
        self,
        run_dirs: list[Path],
        evaluation_scope: str,
        device: torch.device,
    ) -> dict[str, pd.DataFrame]:
        all_sample_frames = []
        all_sample_summary_frames = []
        run_summary_rows = []

        for run_dir in run_dirs:
            sample_df, sample_summary_df, metadata = self._evaluate_run(run_dir, evaluation_scope, device)
            all_sample_frames.append(sample_df)
            all_sample_summary_frames.append(sample_summary_df)

            sample_summary_df = sample_summary_df.sort_values(
                ["quarter_end_date", "source_split"]
            ).reset_index(drop=True)
            cumulative_growth = (
                float(sample_summary_df["portfolio_growth"].cumprod().iloc[-1])
                if not sample_summary_df.empty
                else float("nan")
            )
            cumulative_benchmark_growth = (
                float(sample_summary_df["benchmark_growth"].cumprod().iloc[-1])
                if not sample_summary_df.empty
                else float("nan")
            )
            run_summary_rows.append(
                {
                    **metadata,
                    "strategy_label": (
                        f"run {metadata['run_id']} | "
                        f"risk={metadata['risk_penalty']} | "
                        f"conc={metadata['concentration_penalty']}"
                    ),
                    "sample_count": int(len(sample_summary_df)),
                    "portfolio_growth_mean": float(sample_summary_df["portfolio_growth"].mean()),
                    "benchmark_growth_mean": float(sample_summary_df["benchmark_growth"].mean()),
                    "growth_alpha_mean": float(sample_summary_df["growth_alpha"].mean()),
                    "max_weight_mean": float(sample_summary_df["max_weight"].mean()),
                    "effective_holdings_mean": float(sample_summary_df["effective_holdings"].mean()),
                    "ticker_count_mean": float(sample_summary_df["ticker_count"].mean()),
                    "cumulative_growth": cumulative_growth,
                    "benchmark_cumulative_growth": cumulative_benchmark_growth,
                    "cumulative_alpha": cumulative_growth - cumulative_benchmark_growth,
                }
            )

        return {
            "weights": pd.concat(all_sample_frames, ignore_index=True)
            .sort_values(["run_id", "quarter_end_date", "source_split", "ticker"])
            .reset_index(drop=True),
            "sample_summary": pd.concat(all_sample_summary_frames, ignore_index=True)
            .sort_values(["run_id", "quarter_end_date", "source_split"])
            .reset_index(drop=True),
            "run_summary": pd.DataFrame(run_summary_rows)
            .sort_values("growth_alpha_mean", ascending=False)
            .reset_index(drop=True),
        }

    def _build_validation_diagnostics(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        repo_root = self._repo_root()
        val_df = pd.read_csv(repo_root / "data/preprocessed/val.csv")
        val_df["quarter_end_date"] = pd.to_datetime(val_df["quarter_end_date"])
        raw_counts_df = (
            val_df.groupby(["time_range", "quarter_end_date"], as_index=False)
            .agg(ticker_count=("ticker", "size"))
            .sort_values("quarter_end_date")
        )

        val_split = _load_split_frame(repo_root / "data/preprocessed/val.csv", "val")
        price_history = _load_price_history(
            repo_root / "data/raw/us-shareprices-daily.csv",
            price_field="Adj. Close",
        )
        quarter_prices = _align_quarter_close_prices(price_history, price_field="Adj. Close")
        targets = _attach_future_targets(
            quarter_prices,
            horizons=[1],
            flat_return_threshold=0.02,
        )
        merged = val_split.merge(targets, how="left", on=["ticker", "quarter_end_date", "time_range"])
        target_counts_df = (
            merged.groupby(["time_range", "quarter_end_date"], as_index=False)
            .agg(
                tickers=("ticker", "size"),
                quarter_price_nonnull=("quarter_price", lambda s: int(s.notna().sum())),
                future_price_t1_nonnull=("future_price_t1", lambda s: int(s.notna().sum())),
            )
            .sort_values("quarter_end_date")
        )

        datamodule = instantiate(
            OmegaConf.create(
                {
                    "_target_": "src.datamodules.qai_portfolio_datamodule.QaiPortfolioDataModule",
                    "train_path": str(repo_root / "data/preprocessed/train.csv"),
                    "val_path": str(repo_root / "data/preprocessed/val.csv"),
                    "test_path": str(repo_root / "data/preprocessed/test.csv"),
                    "price_history_path": str(repo_root / "data/raw/us-shareprices-daily.csv"),
                    "sequence_length": 4,
                    "feature_columns": None,
                    "exclude_columns": [
                        "split",
                        "window_ready",
                        "horizon_target_ready",
                        "ticker_history_index",
                    ],
                    "horizons": [1],
                    "target_horizon": 1,
                    "price_field": "Adj. Close",
                    "flat_return_threshold": 0.02,
                    "batch_size": 8,
                    "num_workers": 0,
                }
            )
        )
        datamodule.setup()
        sample_counts_df = pd.DataFrame(
            [
                {
                    "split": split,
                    "portfolio_samples": len(datamodule.samples_by_split[split]),
                    "sample_dates": ", ".join(
                        sorted(
                            {
                                pd.Timestamp(sample["quarter_end_date"]).strftime("%Y-%m-%d")
                                for sample in datamodule.samples_by_split[split]
                            }
                        )
                    ),
                }
                for split in ["train", "val", "test"]
            ]
        )
        return raw_counts_df, target_counts_df, sample_counts_df

    def _make_output_paths(self, multirun_root: Path, scope: str) -> dict[str, Path]:
        return {
            "run_summary": multirun_root / f"portfolio_eval_run_summary_{scope}.csv",
            "sample_weights": multirun_root / f"portfolio_eval_sample_weights_{scope}.csv",
            "growth_by_quarter": multirun_root / f"portfolio_eval_top5_growth_by_quarter_{scope}.csv",
            "top5_allocations": multirun_root / f"portfolio_eval_top5_allocations_{scope}.csv",
            "top_holdings_prices": multirun_root
            / f"portfolio_eval_top5_top_holdings_prices_{scope}.csv",
            "growth_plot": multirun_root / f"portfolio_eval_growth_{scope}.png",
        }

    def _build_top_strategy_tables(
        self,
        weights_df: pd.DataFrame,
        sample_summary_df: pd.DataFrame,
        val_run_summary_df: pd.DataFrame,
        top_run_ids: list[int],
    ) -> dict[str, pd.DataFrame]:
        strategy_labels = val_run_summary_df[["run_id", "strategy_label"]]
        top_growth_df = (
            sample_summary_df.loc[sample_summary_df["run_id"].isin(top_run_ids)]
            .merge(strategy_labels, on="run_id", how="left")
            .sort_values(["run_id", "quarter_end_date", "source_split"])
            .reset_index(drop=True)
        )
        top_allocations_df = (
            weights_df.loc[weights_df["run_id"].isin(top_run_ids)]
            .merge(strategy_labels, on="run_id", how="left")
            .sort_values(
                ["run_id", "quarter_end_date", "source_split", "weight", "ticker"],
                ascending=[True, True, True, False, True],
            )
            .reset_index(drop=True)
        )
        top_prices_df = (
            top_allocations_df.groupby(["run_id", "quarter_end_date", "source_split"], group_keys=False)
            .head(self.top_holdings_per_quarter)
            .reset_index(drop=True)
        )
        return {
            "growth": top_growth_df,
            "allocations": top_allocations_df,
            "prices": top_prices_df,
        }

    def _build_nav_frames(self, sample_summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        benchmark_curve_df = (
            sample_summary_df.groupby(
                ["evaluation_scope", "source_split", "quarter_end_date", "time_range"],
                as_index=False,
            )["benchmark_growth"]
            .mean()
            .sort_values(["quarter_end_date", "source_split"])
        )
        base_date = pd.Timestamp(benchmark_curve_df["quarter_end_date"].min()) - pd.offsets.QuarterEnd(1)
        benchmark_curve_df["strategy_label"] = "equal weight benchmark"
        benchmark_nav_df = self._build_rebased_curve(
            benchmark_curve_df,
            value_column="benchmark_growth",
            group_columns=["strategy_label"],
            label_column="strategy_label",
            base_date=base_date,
        )

        strategy_curve_df = (
            sample_summary_df.groupby(
                ["run_id", "risk_penalty", "concentration_penalty", "source_split", "quarter_end_date", "time_range"],
                as_index=False,
            )["portfolio_growth"]
            .mean()
            .sort_values(["run_id", "quarter_end_date", "source_split"])
        )
        strategy_curve_df["strategy_label"] = strategy_curve_df.apply(
            lambda row: (
                f"run {int(row['run_id'])} | "
                f"risk={row['risk_penalty']} | "
                f"conc={row['concentration_penalty']}"
            ),
            axis=1,
        )
        strategy_nav_df = self._build_rebased_curve(
            strategy_curve_df,
            value_column="portfolio_growth",
            group_columns=["run_id", "risk_penalty", "concentration_penalty"],
            label_column="strategy_label",
            base_date=base_date,
        )
        return {"strategy": strategy_nav_df, "benchmark": benchmark_nav_df, "base_date": base_date}

    def _build_rebased_curve(
        self,
        curve_df: pd.DataFrame,
        value_column: str,
        group_columns: list[str],
        label_column: str,
        base_date: pd.Timestamp,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for _, group in curve_df.groupby(group_columns, sort=False):
            group = group.sort_values(["quarter_end_date", "source_split"]).reset_index(drop=True)
            label_value = group[label_column].iloc[0]
            base_row = {column: group[column].iloc[0] for column in group_columns}
            base_row[label_column] = label_value
            base_row["quarter_end_date"] = base_date
            base_row["source_split"] = "base"
            base_row["rebased_nav"] = 1.0
            rows.append(base_row)

            nav = 1.0
            for row in group.itertuples(index=False):
                nav *= float(getattr(row, value_column))
                row_dict = row._asdict()
                row_dict["rebased_nav"] = nav
                row_dict[label_column] = label_value
                rows.append(row_dict)
        return pd.DataFrame(rows)

    def _plot_validation_heatmaps(self, run_summary_df: pd.DataFrame, output_path: Path) -> None:
        alpha_heatmap = (
            run_summary_df.pivot(
                index="risk_penalty",
                columns="concentration_penalty",
                values="growth_alpha_mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        holdings_heatmap = (
            run_summary_df.pivot(
                index="risk_penalty",
                columns="concentration_penalty",
                values="effective_holdings_mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, matrix, title in [
            (axes[0], alpha_heatmap, "Validation Growth Alpha"),
            (axes[1], holdings_heatmap, "Validation Effective Holdings"),
        ]:
            image = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("concentration_penalty")
            ax.set_ylabel("risk_penalty")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels([str(column) for column in matrix.columns])
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels([str(index) for index in matrix.index])
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{matrix.iloc[row_idx, col_idx]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if self.write_outputs:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_scope_growth(
        self,
        scope: str,
        nav_frames: dict[str, pd.DataFrame],
        top_run_ids: list[int],
        output_path: Path,
    ) -> None:
        strategy_nav_df = nav_frames["strategy"]
        benchmark_nav_df = nav_frames["benchmark"]
        base_date = nav_frames["base_date"]

        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        actual_dates = sorted(
            pd.to_datetime(strategy_nav_df["quarter_end_date"].dropna().unique()).tolist()
        )
        tick_dates = [base_date] + actual_dates
        tick_positions, tick_labels = self._quarter_tick_labels(tick_dates)

        for run_id, group in strategy_nav_df.groupby("run_id"):
            label = group["strategy_label"].iloc[0]
            axes[0].plot(
                group["quarter_end_date"],
                group["rebased_nav"],
                marker="o",
                linewidth=1.4,
                alpha=0.85,
                label=label,
            )
        axes[0].plot(
            benchmark_nav_df["quarter_end_date"],
            benchmark_nav_df["rebased_nav"],
            color="black",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            label="equal weight benchmark",
        )
        axes[0].set_title(f"Rebased Cumulative NAV: {scope.upper()} Replay")
        axes[0].set_ylabel("NAV")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

        top_strategy_nav_df = strategy_nav_df.loc[strategy_nav_df["run_id"].isin(top_run_ids)]
        for run_id, group in top_strategy_nav_df.groupby("run_id"):
            label = group["strategy_label"].iloc[0]
            axes[1].plot(
                group["quarter_end_date"],
                group["rebased_nav"],
                marker="o",
                linewidth=2,
                label=label,
            )
        axes[1].plot(
            benchmark_nav_df["quarter_end_date"],
            benchmark_nav_df["rebased_nav"],
            color="black",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            label="equal weight benchmark",
        )
        axes[1].set_title(f"Top {self.top_strategy_count} Rebased NAV: {scope.upper()} Replay")
        axes[1].set_ylabel("NAV")
        axes[1].set_xlabel("Quarter End Date")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

        for ax in axes:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        plt.tight_layout()
        if self.write_outputs:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _quarter_tick_labels(self, dates: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[str]]:
        ordered = sorted(pd.to_datetime(pd.Series(dates).dropna().unique()).tolist())
        return ordered, [pd.Timestamp(value).strftime("%Y-%m-%d") for value in ordered]
