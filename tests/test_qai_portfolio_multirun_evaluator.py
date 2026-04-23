from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from src.evaluators.qai_portfolio_multirun_evaluator import QaiPortfolioMultirunEvaluator
from src.models.qai_portfolio_attention import QaiPortfolioAttentionModel
from src.models.qai_portfolio_bilstm import QaiPortfolioBiLSTMModel
from tests.qai_fixtures import build_qai_fixture


def _mock_validation_diagnostics():
    return (
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
    )


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


def _write_run_dir(
    root: Path,
    paths: dict[str, Path],
    *,
    run_id: int = 0,
    portfolio_model: str = "attention",
    val_growth_alpha: float = 0.1,
    risk_penalty: float = 1.0,
    concentration_penalty: float = 10.0,
) -> Path:
    run_dir = root / str(run_id)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (root / "multirun.yaml").write_text("sweep:\n  dir: multirun\n")

    is_attention = portfolio_model == "attention"
    model_target = (
        "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel"
        if is_attention
        else "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel"
    )

    run_cfg = {
        "name": f"portfolio_{portfolio_model}_run",
        "portfolio_model": portfolio_model,
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
            "_target_": model_target,
            "input_dim": "auto",
            "hidden_dim": 4 if is_attention else 36,
            "num_heads": 1 if is_attention else None,
            "num_layers": 1 if is_attention else 3,
            "dropout": 0.0,
            "sequence_length": "auto",
        },
        "loss": {"risk_penalty": risk_penalty, "concentration_penalty": concentration_penalty},
    }
    if not is_attention:
        run_cfg["model"].pop("num_heads", None)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps({"epoch": 1, "val_growth_alpha": val_growth_alpha}))
    (artifacts_dir / "best_metrics.json").write_text(
        json.dumps({"epoch": 1, "val_growth_alpha": val_growth_alpha, "val_effective_holdings": 2.5})
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
    if is_attention:
        checkpoint_model = QaiPortfolioAttentionModel(
            input_dim=max(len(feature_columns), 1),
            hidden_dim=4,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            sequence_length=4,
        )
    else:
        checkpoint_model = QaiPortfolioBiLSTMModel(
            input_dim=max(len(feature_columns), 1),
            hidden_dim=36,
            num_layers=3,
            dropout=0.0,
            sequence_length=4,
        )
    checkpoint_payload = {"model_state_dict": checkpoint_model.state_dict(), "epoch": 1}
    torch.save(checkpoint_payload, artifacts_dir / "best_checkpoint.pt")
    torch.save(checkpoint_payload, artifacts_dir / "last_checkpoint.pt")
    return run_dir


def _build_ensemble_scope_results() -> dict[str, pd.DataFrame]:
    run_summary = pd.DataFrame(
        [
            {
                "run_id": 10,
                "model_family": "attention",
                "strategy_label": "run 10 | model=attention | risk=0.0 | conc=0.0",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
                "growth_alpha_mean": 0.20,
                "effective_holdings_mean": 1.8,
                "portfolio_growth_mean": 1.10,
                "benchmark_growth_mean": 1.02,
            },
            {
                "run_id": 11,
                "model_family": "attention",
                "strategy_label": "run 11 | model=attention | risk=0.0 | conc=0.1",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.1,
                "growth_alpha_mean": 0.05,
                "effective_holdings_mean": 1.9,
                "portfolio_growth_mean": 1.04,
                "benchmark_growth_mean": 1.02,
            },
            {
                "run_id": 20,
                "model_family": "bilstm",
                "strategy_label": "run 20 | model=bilstm | risk=0.1 | conc=0.0",
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
                "growth_alpha_mean": 0.30,
                "effective_holdings_mean": 1.7,
                "portfolio_growth_mean": 1.12,
                "benchmark_growth_mean": 1.02,
            },
        ]
    )

    weights = pd.DataFrame(
        [
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 10,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "AAA",
                "weight": 0.7,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 120.0,
                "ticker_return": 0.2,
                "portfolio_growth": 1.13,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.08,
                "max_weight": 0.7,
                "effective_holdings": 1.72,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 10,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "BBB",
                "weight": 0.3,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 90.0,
                "ticker_return": -0.1,
                "portfolio_growth": 1.13,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.08,
                "max_weight": 0.7,
                "effective_holdings": 1.72,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 11,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "AAA",
                "weight": 0.55,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 120.0,
                "ticker_return": 0.2,
                "portfolio_growth": 1.07,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.02,
                "max_weight": 0.55,
                "effective_holdings": 1.98,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.1,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 11,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "BBB",
                "weight": 0.45,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 90.0,
                "ticker_return": -0.1,
                "portfolio_growth": 1.07,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.02,
                "max_weight": 0.55,
                "effective_holdings": 1.98,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "risk_penalty": 0.0,
                "concentration_penalty": 0.1,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 20,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "AAA",
                "weight": 0.2,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 120.0,
                "ticker_return": 0.2,
                "portfolio_growth": 1.0,
                "benchmark_growth": 1.05,
                "growth_alpha": -0.05,
                "max_weight": 0.8,
                "effective_holdings": 1.47,
                "model_family": "bilstm",
                "model_target": "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel",
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 20,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "ticker": "BBB",
                "weight": 0.8,
                "equal_weight": 0.5,
                "current_price": 100.0,
                "future_price": 90.0,
                "ticker_return": -0.1,
                "portfolio_growth": 1.0,
                "benchmark_growth": 1.05,
                "growth_alpha": -0.05,
                "max_weight": 0.8,
                "effective_holdings": 1.47,
                "model_family": "bilstm",
                "model_target": "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel",
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
            },
        ]
    )

    sample_summary = pd.DataFrame(
        [
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 10,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "ticker_count": 2,
                "portfolio_growth": 1.13,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.08,
                "max_weight": 0.7,
                "effective_holdings": 1.72,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 11,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "risk_penalty": 0.0,
                "concentration_penalty": 0.1,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "ticker_count": 2,
                "portfolio_growth": 1.07,
                "benchmark_growth": 1.05,
                "growth_alpha": 0.02,
                "max_weight": 0.55,
                "effective_holdings": 1.98,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 20,
                "sample_key": "val__q1y2020__2020-03-31",
                "time_range": "q1y2020",
                "quarter_end_date": pd.Timestamp("2020-03-31"),
                "future_quarter_end_date": pd.Timestamp("2020-06-30"),
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
                "model_family": "bilstm",
                "model_target": "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel",
                "ticker_count": 2,
                "portfolio_growth": 1.0,
                "benchmark_growth": 1.05,
                "growth_alpha": -0.05,
                "max_weight": 0.8,
                "effective_holdings": 1.47,
            },
        ]
    )
    return {
        "weights": weights,
        "sample_summary": sample_summary,
        "run_summary": run_summary.sort_values("growth_alpha_mean", ascending=False).reset_index(drop=True),
        "skipped_runs": pd.DataFrame(),
    }


def test_portfolio_multirun_evaluator_reports_indicator_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_indicator_fixture(tmp_path / "data")
    multirun_root = tmp_path / "multirun"
    run_dir = _write_run_dir(multirun_root, paths)

    evaluator = QaiPortfolioMultirunEvaluator(
        multirun_root=str(multirun_root),
        evaluation_scopes=["val"],
        write_outputs=True,
        build_best_family_ensemble=False,
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self, run_cfg=None: _mock_validation_diagnostics(),
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
        build_best_family_ensemble=False,
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self, run_cfg=None: _mock_validation_diagnostics(),
    )

    metrics = evaluator.evaluate(
        datamodule=None,
        model=None,
        metric_fn=None,
        run_dir=tmp_path / "eval_outputs",
    )

    assert metrics["scopes"]["val"]["run_count"] == 1


def test_portfolio_multirun_evaluator_skips_runs_with_no_portfolio_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "multirun" / "0"
    run_dir.mkdir(parents=True)
    evaluator = QaiPortfolioMultirunEvaluator(write_outputs=False)

    def fake_evaluate_run(self, run_dir: Path, evaluation_scope: str, device: torch.device):
        metadata = {
            "evaluation_scope": evaluation_scope,
            "run_id": int(run_dir.name),
            "risk_penalty": 1.0,
            "concentration_penalty": 10.0,
            "model_family": "attention",
            "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
        }
        return (
            pd.DataFrame(columns=QaiPortfolioMultirunEvaluator.WEIGHTS_COLUMNS),
            pd.DataFrame(columns=QaiPortfolioMultirunEvaluator.SAMPLE_SUMMARY_COLUMNS),
            metadata,
        )

    monkeypatch.setattr(QaiPortfolioMultirunEvaluator, "_evaluate_run", fake_evaluate_run)

    results = evaluator._summarize_scope([run_dir], "val", torch.device("cpu"))

    assert results["run_summary"].empty
    assert results["sample_summary"].empty
    assert results["skipped_runs"].loc[0, "run_id"] == 0
    assert "No portfolio samples" in results["skipped_runs"].loc[0, "reason"]


def test_portfolio_multirun_evaluator_overlays_eval_data_frequency() -> None:
    evaluator = QaiPortfolioMultirunEvaluator(write_outputs=False)
    eval_datamodule = SimpleNamespace(
        train_path="data/preprocessed/train.csv",
        val_path="data/preprocessed/val.csv",
        test_path="data/preprocessed/test.csv",
        price_history_path="data/raw/us-shareprices-daily.csv",
        horizons=[1],
        target_horizon=1,
        target_frequency="daily",
        price_field="Adj. Close",
        flat_return_threshold=0.02,
        sequence_length=16,
    )
    evaluator._runtime_data_overrides = evaluator._data_overrides_from_datamodule(eval_datamodule)

    merged = evaluator._with_runtime_data_overrides(
        {
            "data": {
                "target_frequency": "quarterly",
                "sequence_length": 4,
                "price_history_path": "old_prices.csv",
            }
        }
    )

    assert merged["data"]["target_frequency"] == "daily"
    assert merged["data"]["price_history_path"] == "data/raw/us-shareprices-daily.csv"
    assert merged["data"]["sequence_length"] == 4


def test_portfolio_multirun_evaluator_recovers_model_from_checkpoint_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_indicator_fixture(tmp_path / "data")
    multirun_root = tmp_path / "multirun"
    run_dir = _write_run_dir(multirun_root, paths)
    artifacts_dir = run_dir / "artifacts"
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

    bilstm_checkpoint_model = QaiPortfolioBiLSTMModel(
        input_dim=max(len(feature_columns), 1),
        hidden_dim=36,
        num_layers=3,
        dropout=0.0,
        sequence_length=4,
    )
    checkpoint_payload = {"model_state_dict": bilstm_checkpoint_model.state_dict(), "epoch": 1}
    torch.save(checkpoint_payload, artifacts_dir / "best_checkpoint.pt")
    torch.save(checkpoint_payload, artifacts_dir / "last_checkpoint.pt")

    evaluator = QaiPortfolioMultirunEvaluator(
        multirun_root=str(multirun_root),
        evaluation_scopes=["val"],
        write_outputs=False,
        build_best_family_ensemble=False,
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self, run_cfg=None: _mock_validation_diagnostics(),
    )

    metrics = evaluator.evaluate(
        datamodule=None,
        model=None,
        metric_fn=None,
        run_dir=tmp_path / "eval_outputs",
    )

    assert metrics["scopes"]["val"]["run_count"] == 1
    assert metrics["scopes"]["val"]["top_strategies"][0]["model_family"] == "bilstm"
    assert (
        metrics["scopes"]["val"]["top_strategies"][0]["model_target"]
        == "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel"
    )


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


def test_portfolio_multirun_evaluator_builds_best_family_ensemble_from_top_family_runs() -> None:
    evaluator = QaiPortfolioMultirunEvaluator(write_outputs=False)
    scope_results = _build_ensemble_scope_results()

    ensemble = evaluator._build_best_family_ensemble(
        scope_results=scope_results,
        selection_run_summary_df=scope_results["run_summary"],
    )

    assert ensemble["summary"]["attention_run_id"] == 10
    assert ensemble["summary"]["bilstm_run_id"] == 20
    allocations = ensemble["allocations"]
    assert allocations["weight"].sum() == pytest.approx(1.0)
    aaa_weight = allocations.loc[allocations["ticker"] == "AAA", "weight"].iloc[0]
    bbb_weight = allocations.loc[allocations["ticker"] == "BBB", "weight"].iloc[0]
    assert aaa_weight == pytest.approx(0.45)
    assert bbb_weight == pytest.approx(0.55)
    assert ensemble["summary"]["portfolio_growth_mean"] == pytest.approx(1.035)
    assert ensemble["summary"]["benchmark_growth_mean"] == pytest.approx(1.05)
    assert ensemble["summary"]["growth_alpha_mean"] == pytest.approx(-0.015)


def test_portfolio_multirun_evaluator_writes_best_family_ensemble_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multirun_root = tmp_path / "multirun"
    multirun_root.mkdir(parents=True, exist_ok=True)
    (multirun_root / "multirun.yaml").write_text("sweep:\n  dir: multirun\n")
    scope_results = _build_ensemble_scope_results()
    evaluator = QaiPortfolioMultirunEvaluator(
        multirun_root=str(multirun_root),
        evaluation_scopes=["val"],
        write_outputs=True,
    )

    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_build_validation_diagnostics",
        lambda self, run_cfg=None: _mock_validation_diagnostics(),
    )
    monkeypatch.setattr(
        QaiPortfolioMultirunEvaluator,
        "_summarize_scope",
        lambda self, run_dirs, evaluation_scope, device: scope_results,
    )

    metrics = evaluator.evaluate(
        datamodule=None,
        model=None,
        metric_fn=None,
        run_dir=tmp_path / "eval_outputs",
    )

    ensemble_metrics = metrics["scopes"]["val"]["best_family_ensemble"]
    assert ensemble_metrics["attention_run_id"] == 10
    assert ensemble_metrics["bilstm_run_id"] == 20
    assert (
        multirun_root / "portfolio_eval_best_family_ensemble_growth_by_quarter_val.csv"
    ).exists()
    assert (
        multirun_root / "portfolio_eval_best_family_ensemble_allocations_val.csv"
    ).exists()
    assert (tmp_path / "eval_outputs" / "eval_outputs" / "portfolio_eval_growth_val.png").exists()

    summary_path = tmp_path / "eval_outputs" / "metrics_eval.json"
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["scopes"]["val"]["best_family_ensemble"]["growth_alpha_mean"] == pytest.approx(
        -0.015
    )


def test_portfolio_multirun_evaluator_requires_both_ensemble_families() -> None:
    evaluator = QaiPortfolioMultirunEvaluator(write_outputs=False)
    scope_results = _build_ensemble_scope_results()
    missing_bilstm_summary = scope_results["run_summary"].loc[
        scope_results["run_summary"]["model_family"] != "bilstm"
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="missing model_family=bilstm"):
        evaluator._build_best_family_ensemble(
            scope_results=scope_results,
            selection_run_summary_df=missing_bilstm_summary,
        )
