from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.evaluators.qai_portfolio_multirun_evaluator import QaiPortfolioMultirunEvaluator
from src.models.qai_portfolio_attention import QaiPortfolioAttentionModel
from tests.qai_fixtures import build_qai_fixture


def _write_indicator_fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = build_qai_fixture(tmp_path)
    indicator_frames = {
        "train": pd.DataFrame(
            [
                {"econ_gdp": 1.0, "econ_cpi": 2.0, "tech_rsi_14_1day": 3.0, "tech_ema_20_1day": 4.0},
                {"econ_gdp": 1.1, "econ_cpi": 2.1, "tech_rsi_14_1day": 3.1, "tech_ema_20_1day": 4.1},
                {"econ_gdp": 1.2, "econ_cpi": 2.2, "tech_rsi_14_1day": 3.2, "tech_ema_20_1day": 4.2},
                {"econ_gdp": 1.3, "econ_cpi": 2.3, "tech_rsi_14_1day": 3.3, "tech_ema_20_1day": 4.3},
            ]
        ),
        "val": pd.DataFrame(
            [
                {"econ_gdp": 1.4, "econ_cpi": 2.4, "tech_rsi_14_1day": 3.4, "tech_ema_20_1day": 4.4}
            ]
        ),
        "test": pd.DataFrame(
            [
                {"econ_gdp": 1.5, "econ_cpi": 2.5, "tech_rsi_14_1day": 3.5, "tech_ema_20_1day": 4.5},
                {"econ_gdp": 1.6, "econ_cpi": 2.6, "tech_rsi_14_1day": 3.6, "tech_ema_20_1day": 4.6},
                {"econ_gdp": 1.7, "econ_cpi": 2.7, "tech_rsi_14_1day": 3.7, "tech_ema_20_1day": 4.7},
                {"econ_gdp": 1.8, "econ_cpi": 2.8, "tech_rsi_14_1day": 3.8, "tech_ema_20_1day": 4.8},
                {"econ_gdp": 1.9, "econ_cpi": 2.9, "tech_rsi_14_1day": 3.9, "tech_ema_20_1day": 4.9},
            ]
        ),
    }
    for split_name, frame in indicator_frames.items():
        split_path = paths[split_name]
        base_frame = pd.read_csv(split_path)
        enriched = pd.concat([base_frame.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        enriched.to_csv(split_path, index=False)
    return paths


def _write_run_dir(root: Path, paths: dict[str, Path]) -> Path:
    run_dir = root / "0"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (root / "multirun.yaml").write_text("sweep:\n  dir: multirun\n")

    run_cfg = {
        "name": "portfolio_attention_run",
        "portfolio_model": "attention",
        "data": {
            "_target_": "src.datamodules.qai_portfolio_datamodule.QaiPortfolioDataModule",
            "train_path": str(paths["train"]),
            "val_path": str(paths["val"]),
            "test_path": str(paths["test"]),
            "price_history_path": str(paths["prices"]),
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
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "_target_": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
            "input_dim": "auto",
            "hidden_dim": 4,
            "num_heads": 1,
            "num_layers": 1,
            "dropout": 0.0,
            "sequence_length": "auto",
        },
        "loss": {"risk_penalty": 1.0, "concentration_penalty": 10.0},
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps({"epoch": 1, "val_growth_alpha": 0.1}))
    (artifacts_dir / "best_metrics.json").write_text(
        json.dumps({"epoch": 1, "val_growth_alpha": 0.1, "val_effective_holdings": 2.5})
    )
    (artifacts_dir / "best_checkpoint_summary.json").write_text(json.dumps({"epoch": 1}))

    train_frame = pd.read_csv(paths["train"])
    feature_columns = [
        column
        for column in train_frame.columns
        if column
        not in {
            "ticker",
            "time_range",
            "quarter_end_date",
            "split",
            "window_ready",
            "horizon_target_ready",
            "ticker_history_index",
        }
        and pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    data_module = QaiPortfolioAttentionModel(
        input_dim=max(len(feature_columns), 1),
        hidden_dim=4,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        sequence_length=4,
    )
    checkpoint_payload = {"model_state_dict": data_module.state_dict(), "epoch": 1}
    torch.save(checkpoint_payload, artifacts_dir / "best_checkpoint.pt")
    torch.save(checkpoint_payload, artifacts_dir / "last_checkpoint.pt")
    return run_dir


def test_portfolio_multirun_evaluator_reports_indicator_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_indicator_fixture(tmp_path / "data")
    multirun_root = tmp_path / "multirun"
    run_dir = _write_run_dir(multirun_root, paths)

    evaluator = QaiPortfolioMultirunEvaluator(
        multirun_root=str(multirun_root),
        evaluation_scopes=["val"],
        write_outputs=True,
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self: (
            pd.DataFrame([{"time_range": "q1y2020", "quarter_end_date": "2020-03-31", "ticker_count": 1}]),
            pd.DataFrame(
                [
                    {
                        "time_range": "q1y2020",
                        "quarter_end_date": "2020-03-31",
                        "tickers": 1,
                        "quarter_price_nonnull": 1,
                        "future_price_t1_nonnull": 1,
                    }
                ]
            ),
            pd.DataFrame([{"split": "train", "portfolio_samples": 1, "sample_dates": "2020-03-31"}]),
        ),
    )

    metrics = evaluator.evaluate(
        datamodule=None,
        model=None,
        metric_fn=None,
        run_dir=tmp_path / "eval_outputs",
    )

    run_summary = pd.read_csv(multirun_root / "portfolio_eval_run_summary_val.csv")

    assert run_summary.loc[0, "model_family"] == "attention"
    assert run_summary.loc[0, "economics_feature_count"] == 2
    assert run_summary.loc[0, "technical_feature_count"] == 2
    assert run_summary.loc[0, "indicator_feature_count"] == 4
    assert "model=attention" in run_summary.loc[0, "strategy_label"]
    assert metrics["scopes"]["val"]["indicator_feature_profile"]["model_family"] == "attention"
    assert metrics["scopes"]["val"]["indicator_feature_profile"]["indicator_feature_count"] == 4


def test_portfolio_multirun_evaluator_builds_attention_model_with_auto_dims(
    tmp_path: Path,
) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    paths = build_qai_fixture(tmp_path / "data")
    evaluator = QaiPortfolioMultirunEvaluator()
    datamodule = evaluator._build_datamodule(
        {
            "data": {
                "_target_": "src.datamodules.qai_portfolio_datamodule.QaiPortfolioDataModule",
                "train_path": str(paths["train"]),
                "val_path": str(paths["val"]),
                "test_path": str(paths["test"]),
                "price_history_path": str(paths["prices"]),
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
                "batch_size": 2,
                "num_workers": 0,
            }
        }
    )

    model = evaluator._build_model(
        {
            "seed": 777,
            "model": {
                "_target_": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "input_dim": "auto",
                "hidden_dim": 4,
                "num_heads": 1,
                "num_layers": 1,
                "dropout": 0.0,
                "sequence_length": "auto",
            },
        },
        datamodule,
    )

    assert model.input_projection.in_features == datamodule.feature_dim
    assert model.position_embedding.shape[1] == datamodule.max_sequence_length


def test_portfolio_multirun_evaluator_uses_last_checkpoint_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    paths = build_qai_fixture(tmp_path / "data")
    multirun_root = tmp_path / "multirun"
    run_dir = _write_run_dir(multirun_root, paths)
    best_checkpoint = run_dir / "artifacts" / "best_checkpoint.pt"
    if best_checkpoint.exists():
        best_checkpoint.unlink()
    assert (run_dir / "artifacts" / "last_checkpoint.pt").exists()

    evaluator = QaiPortfolioMultirunEvaluator(
        multirun_root=str(multirun_root),
        evaluation_scopes=["val"],
        write_outputs=False,
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self: (
            pd.DataFrame([{"time_range": "q1y2020", "quarter_end_date": "2020-03-31", "ticker_count": 1}]),
            pd.DataFrame(
                [
                    {
                        "time_range": "q1y2020",
                        "quarter_end_date": "2020-03-31",
                        "tickers": 1,
                        "quarter_price_nonnull": 1,
                        "future_price_t1_nonnull": 1,
                    }
                ]
            ),
            pd.DataFrame([{"split": "train", "portfolio_samples": 1, "sample_dates": "2020-03-31"}]),
        ),
    )

    metrics = evaluator.evaluate(
        datamodule=None,
        model=None,
        metric_fn=None,
        run_dir=tmp_path / "eval_outputs",
    )

    assert metrics["scopes"]["val"]["run_count"] == 1


def test_portfolio_multirun_evaluator_heatmap_handles_duplicate_penalty_rows(tmp_path: Path) -> None:
    evaluator = QaiPortfolioMultirunEvaluator(write_outputs=True)
    output_path = tmp_path / "heatmap.png"
    run_summary_df = pd.DataFrame(
        [
            {
                "model_family": "attention",
                "risk_penalty": 1.0,
                "concentration_penalty": 1.0,
                "growth_alpha_mean": 0.1,
                "effective_holdings_mean": 2.0,
            },
            {
                "model_family": "attention",
                "risk_penalty": 1.0,
                "concentration_penalty": 1.0,
                "growth_alpha_mean": 0.2,
                "effective_holdings_mean": 3.0,
            },
            {
                "model_family": "bilstm",
                "risk_penalty": 1.0,
                "concentration_penalty": 1.0,
                "growth_alpha_mean": 0.3,
                "effective_holdings_mean": 4.0,
            },
            {
                "model_family": "bilstm",
                "risk_penalty": 10.0,
                "concentration_penalty": 10.0,
                "growth_alpha_mean": 0.4,
                "effective_holdings_mean": 5.0,
            },
        ]
    )

    evaluator._plot_validation_heatmaps(run_summary_df, output_path)

    assert output_path.exists()
