from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluators.qai_portfolio_saved_eval_ensemble import (
    build_saved_eval_ensemble_result,
)


def _build_archived_scope_frames() -> dict[str, pd.DataFrame]:
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
                "cumulative_growth": 1.10,
                "benchmark_cumulative_growth": 1.02,
                "cumulative_alpha": 0.08,
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
                "cumulative_growth": 1.12,
                "benchmark_cumulative_growth": 1.02,
                "cumulative_alpha": 0.10,
            },
        ]
    )
    weights = pd.DataFrame(
        [
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 10,
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "feature_dim": 2,
                "economics_feature_count": 0,
                "technical_feature_count": 0,
                "indicator_feature_count": 0,
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
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 10,
                "risk_penalty": 0.0,
                "concentration_penalty": 0.0,
                "model_family": "attention",
                "model_target": "src.models.qai_portfolio_attention.QaiPortfolioAttentionModel",
                "feature_dim": 2,
                "economics_feature_count": 0,
                "technical_feature_count": 0,
                "indicator_feature_count": 0,
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
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 20,
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
                "model_family": "bilstm",
                "model_target": "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel",
                "feature_dim": 2,
                "economics_feature_count": 0,
                "technical_feature_count": 0,
                "indicator_feature_count": 0,
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
                "portfolio_growth": 1.00,
                "benchmark_growth": 1.05,
                "growth_alpha": -0.05,
                "max_weight": 0.8,
                "effective_holdings": 1.47,
            },
            {
                "evaluation_scope": "val",
                "source_split": "val",
                "run_id": 20,
                "risk_penalty": 0.1,
                "concentration_penalty": 0.0,
                "model_family": "bilstm",
                "model_target": "src.models.qai_portfolio_bilstm.QaiPortfolioBiLSTMModel",
                "feature_dim": 2,
                "economics_feature_count": 0,
                "technical_feature_count": 0,
                "indicator_feature_count": 0,
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
                "portfolio_growth": 1.00,
                "benchmark_growth": 1.05,
                "growth_alpha": -0.05,
                "max_weight": 0.8,
                "effective_holdings": 1.47,
            },
        ]
    )
    return {"run_summary": run_summary, "weights": weights}


def test_build_saved_eval_ensemble_result_writes_self_contained_outputs(
    tmp_path: Path,
) -> None:
    scope_frames = _build_archived_scope_frames()
    archived_root = tmp_path / "archived_eval"
    eval_outputs_dir = archived_root / "eval_outputs"
    eval_outputs_dir.mkdir(parents=True, exist_ok=True)

    for scope in ["val", "all"]:
        run_summary = scope_frames["run_summary"].copy()
        weights = scope_frames["weights"].copy()
        weights["evaluation_scope"] = scope
        weights.to_csv(
            eval_outputs_dir / f"portfolio_eval_sample_weights_{scope}.csv",
            index=False,
        )
        run_summary.to_csv(
            eval_outputs_dir / f"portfolio_eval_run_summary_{scope}.csv",
            index=False,
        )
    pd.DataFrame([{"split": "val", "portfolio_samples": 1}]).to_csv(
        eval_outputs_dir / "portfolio_eval_val_diagnostics.csv",
        index=False,
    )

    result_root = tmp_path / "ensemble_model_result"
    metrics = build_saved_eval_ensemble_result(
        source_dir=archived_root,
        output_dir=result_root,
    )

    assert metrics["selected_components"]["attention"]["run_id"] == 10
    assert metrics["selected_components"]["bilstm"]["run_id"] == 20
    assert (result_root / "metrics_eval.json").exists()
    assert (result_root / "eval_outputs" / "portfolio_eval_summary.json").exists()
    assert (result_root / "eval_outputs" / "portfolio_eval_run_summary_val.csv").exists()
    assert (result_root / "eval_outputs" / "portfolio_eval_growth_val.png").exists()
    assert (result_root / "eval_outputs" / "portfolio_eval_val_diagnostics.csv").exists()

    allocations_df = pd.read_csv(
        result_root / "eval_outputs" / "portfolio_eval_sample_weights_val.csv"
    )
    aaa_weight = allocations_df.loc[allocations_df["ticker"] == "AAA", "weight"].iloc[0]
    bbb_weight = allocations_df.loc[allocations_df["ticker"] == "BBB", "weight"].iloc[0]
    assert aaa_weight == pytest.approx(0.45)
    assert bbb_weight == pytest.approx(0.55)

    growth_df = pd.read_csv(
        result_root / "eval_outputs" / "portfolio_eval_top5_growth_by_quarter_val.csv"
    )
    assert growth_df.loc[0, "strategy_label"] == "best attention + best bilstm (50/50)"

    summary_payload = json.loads(
        (result_root / "eval_outputs" / "portfolio_eval_summary.json").read_text()
    )
    assert summary_payload["scopes"]["val"]["run_count"] == 1
    assert summary_payload["scopes"]["val"]["chart_components"] == [
        {
            "family": "attention",
            "run_id": 10,
            "strategy_label": "best attention (run 10)",
        },
        {
            "family": "bilstm",
            "run_id": 20,
            "strategy_label": "best bilstm (run 20)",
        },
    ]
