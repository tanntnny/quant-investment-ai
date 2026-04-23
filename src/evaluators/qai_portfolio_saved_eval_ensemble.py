from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluators.qai_portfolio_multirun_evaluator import (
    QaiPortfolioMultirunEvaluator,
    plt,
)
from src.utils.io import save_json
from src.utils.logging import ensure_dir


DEFAULT_RESULT_NAME = "ensemble_model_result"


def build_saved_eval_ensemble_result(
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    top_holdings_per_quarter: int = 10,
) -> dict[str, Any]:
    eval_outputs_dir = _resolve_eval_outputs_dir(source_dir)
    source_root = eval_outputs_dir.parent
    result_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (source_root / DEFAULT_RESULT_NAME).resolve()
    )
    result_eval_outputs_dir = result_root / "eval_outputs"
    ensure_dir(result_eval_outputs_dir)

    evaluator = QaiPortfolioMultirunEvaluator(
        write_outputs=True,
        top_strategy_count=1,
        top_holdings_per_quarter=top_holdings_per_quarter,
    )

    selection_run_summary_df = _read_csv(
        eval_outputs_dir / "portfolio_eval_run_summary_val.csv",
        parse_dates=None,
    ).sort_values("growth_alpha_mean", ascending=False).reset_index(drop=True)
    selected_components = _select_best_family_rows(
        selection_run_summary_df,
        evaluator.ensemble_model_families,
    )

    metrics_payload: dict[str, Any] = {
        "source_eval_output_dir": str(eval_outputs_dir),
        "result_root": str(result_root),
        "result_name": result_root.name,
        "construction": "50/50 weights from best validation attention and best validation bilstm runs.",
        "selected_components": selected_components,
        "scopes": {},
    }

    source_diagnostics_path = eval_outputs_dir / "portfolio_eval_val_diagnostics.csv"
    if source_diagnostics_path.exists():
        shutil.copy2(
            source_diagnostics_path,
            result_eval_outputs_dir / source_diagnostics_path.name,
        )

    for scope in _available_scopes(eval_outputs_dir):
        weights_df = _read_csv(
            eval_outputs_dir / f"portfolio_eval_sample_weights_{scope}.csv",
            parse_dates=["quarter_end_date", "future_quarter_end_date"],
        )
        scope_results = {
            "weights": weights_df,
            "sample_summary": _derive_sample_summary(weights_df),
            "run_summary": _read_csv(
                eval_outputs_dir / f"portfolio_eval_run_summary_{scope}.csv",
                parse_dates=None,
            ),
            "skipped_runs": pd.DataFrame(),
        }

        ensemble = evaluator._build_best_family_ensemble(
            scope_results=scope_results,
            selection_run_summary_df=selection_run_summary_df,
        )
        benchmark_nav_df = _build_benchmark_nav(evaluator, ensemble["growth"])
        chart_components = _build_selected_component_navs(
            evaluator=evaluator,
            sample_summary_df=scope_results["sample_summary"],
            selected_components=selected_components,
            base_date=pd.Timestamp(benchmark_nav_df["quarter_end_date"].min()),
        )
        run_summary_df = _build_ensemble_run_summary(scope, ensemble)
        top_holdings_prices_df = (
            ensemble["allocations"]
            .groupby(["quarter_end_date", "source_split"], group_keys=False)
            .head(top_holdings_per_quarter)
            .reset_index(drop=True)
        )

        output_paths = {
            "run_summary": result_eval_outputs_dir / f"portfolio_eval_run_summary_{scope}.csv",
            "sample_weights": result_eval_outputs_dir / f"portfolio_eval_sample_weights_{scope}.csv",
            "growth_by_quarter": result_eval_outputs_dir
            / f"portfolio_eval_top5_growth_by_quarter_{scope}.csv",
            "top5_allocations": result_eval_outputs_dir
            / f"portfolio_eval_top5_allocations_{scope}.csv",
            "top_holdings_prices": result_eval_outputs_dir
            / f"portfolio_eval_top5_top_holdings_prices_{scope}.csv",
            "growth_plot": result_eval_outputs_dir / f"portfolio_eval_growth_{scope}.png",
        }

        run_summary_df.to_csv(output_paths["run_summary"], index=False)
        ensemble["allocations"].to_csv(output_paths["sample_weights"], index=False)
        ensemble["growth"].to_csv(output_paths["growth_by_quarter"], index=False)
        ensemble["allocations"].to_csv(output_paths["top5_allocations"], index=False)
        top_holdings_prices_df.to_csv(output_paths["top_holdings_prices"], index=False)
        _plot_ensemble_growth(
            evaluator=evaluator,
            scope=scope,
            benchmark_nav_df=benchmark_nav_df,
            chart_components=chart_components,
            ensemble_nav_df=ensemble["nav"],
            output_path=output_paths["growth_plot"],
        )

        metrics_payload["scopes"][scope] = {
            "run_count": 1,
            "sample_count": int(len(ensemble["growth"])),
            "dates": sorted(
                pd.to_datetime(ensemble["growth"]["quarter_end_date"])
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            ),
            "top_strategies": run_summary_df.to_dict(orient="records"),
            "benchmark_terminal_nav": float(benchmark_nav_df["rebased_nav"].iloc[-1]),
            "chart_components": [
                {
                    "family": item["family"],
                    "run_id": item["run_id"],
                    "strategy_label": item["strategy_label"],
                }
                for item in chart_components
            ],
            "outputs": {key: str(value) for key, value in output_paths.items()},
        }

    save_json(metrics_payload, result_root / "metrics_eval.json")
    save_json(metrics_payload, result_eval_outputs_dir / "portfolio_eval_summary.json")
    return metrics_payload


def _resolve_eval_outputs_dir(path_value: str | Path) -> Path:
    candidate = Path(path_value).resolve()
    if candidate.name == "eval_outputs":
        eval_outputs_dir = candidate
    else:
        eval_outputs_dir = candidate / "eval_outputs"
    if not eval_outputs_dir.exists():
        raise FileNotFoundError(
            f"Could not find eval_outputs directory under {candidate}."
        )
    return eval_outputs_dir


def _available_scopes(eval_outputs_dir: Path) -> list[str]:
    scopes: list[str] = []
    for scope in ["val", "all"]:
        if (eval_outputs_dir / f"portfolio_eval_sample_weights_{scope}.csv").exists():
            scopes.append(scope)
    if not scopes:
        raise FileNotFoundError(
            f"No portfolio_eval_sample_weights_<scope>.csv files found in {eval_outputs_dir}."
        )
    return scopes


def _read_csv(path: Path, parse_dates: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing archived evaluation file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def _select_best_family_rows(
    selection_run_summary_df: pd.DataFrame,
    families: list[str],
) -> dict[str, dict[str, Any]]:
    selected_components: dict[str, dict[str, Any]] = {}
    for family in families:
        family_df = selection_run_summary_df.loc[
            selection_run_summary_df["model_family"] == family
        ].reset_index(drop=True)
        if family_df.empty:
            raise ValueError(
                f"Could not find model_family={family} in archived validation run summary."
            )
        selected_components[family] = family_df.iloc[0].to_dict()
    return selected_components


def _derive_sample_summary(weights_df: pd.DataFrame) -> pd.DataFrame:
    if weights_df.empty:
        return pd.DataFrame(columns=QaiPortfolioMultirunEvaluator.SAMPLE_SUMMARY_COLUMNS)

    group_columns = [
        "evaluation_scope",
        "source_split",
        "run_id",
        "sample_key",
        "time_range",
        "quarter_end_date",
        "future_quarter_end_date",
    ]
    summary_df = (
        weights_df.groupby(group_columns, as_index=False, sort=True)
        .agg(
            risk_penalty=("risk_penalty", "first"),
            concentration_penalty=("concentration_penalty", "first"),
            model_family=("model_family", "first"),
            model_target=("model_target", "first"),
            ticker_count=("ticker", "size"),
            portfolio_growth=("portfolio_growth", "first"),
            benchmark_growth=("benchmark_growth", "first"),
            growth_alpha=("growth_alpha", "first"),
            max_weight=("max_weight", "first"),
            effective_holdings=("effective_holdings", "first"),
        )
        .loc[:, QaiPortfolioMultirunEvaluator.SAMPLE_SUMMARY_COLUMNS]
    )
    return summary_df.sort_values(
        ["run_id", "quarter_end_date", "source_split"]
    ).reset_index(drop=True)


def _build_benchmark_nav(
    evaluator: QaiPortfolioMultirunEvaluator,
    ensemble_growth_df: pd.DataFrame,
) -> pd.DataFrame:
    benchmark_curve_df = (
        ensemble_growth_df.groupby(
            ["evaluation_scope", "source_split", "quarter_end_date", "time_range"],
            as_index=False,
        )["benchmark_growth"]
        .mean()
        .sort_values(["quarter_end_date", "source_split"])
        .reset_index(drop=True)
    )
    base_date = (
        pd.Timestamp(benchmark_curve_df["quarter_end_date"].min())
        - pd.offsets.QuarterEnd(1)
    )
    benchmark_curve_df["strategy_label"] = "equal weight benchmark"
    return evaluator._build_rebased_curve(
        benchmark_curve_df,
        value_column="benchmark_growth",
        group_columns=["strategy_label"],
        label_column="strategy_label",
        base_date=base_date,
    )


def _build_selected_component_navs(
    evaluator: QaiPortfolioMultirunEvaluator,
    sample_summary_df: pd.DataFrame,
    selected_components: dict[str, dict[str, Any]],
    base_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    chart_components: list[dict[str, Any]] = []
    for family in evaluator.ensemble_model_families:
        component = selected_components[family]
        run_id = component["run_id"]
        component_growth_df = (
            sample_summary_df.loc[sample_summary_df["run_id"] == run_id]
            .copy()
            .sort_values(["quarter_end_date", "source_split"])
            .reset_index(drop=True)
        )
        if component_growth_df.empty:
            raise ValueError(
                f"Archived scope results do not contain selected {family} run_id={run_id}."
            )
        strategy_label = f"best {family} (run {run_id})"
        component_growth_df["strategy_label"] = strategy_label
        nav_df = evaluator._build_rebased_curve(
            component_growth_df,
            value_column="portfolio_growth",
            group_columns=["strategy_label"],
            label_column="strategy_label",
            base_date=base_date,
        )
        chart_components.append(
            {
                "family": family,
                "run_id": run_id,
                "strategy_label": strategy_label,
                "nav": nav_df,
            }
        )
    return chart_components


def _build_ensemble_run_summary(
    scope: str,
    ensemble: dict[str, Any],
) -> pd.DataFrame:
    summary = dict(ensemble["summary"])
    growth_df = ensemble["growth"]
    summary_row = {
        "evaluation_scope": scope,
        "run_id": "ensemble_best_attention_bilstm",
        "risk_penalty": "",
        "concentration_penalty": "",
        "model_family": "ensemble",
        "model_target": "archived_eval_saved_weights",
        "strategy_label": summary["strategy_label"],
        "sample_count": summary["sample_count"],
        "portfolio_growth_mean": summary["portfolio_growth_mean"],
        "benchmark_growth_mean": summary["benchmark_growth_mean"],
        "growth_alpha_mean": summary["growth_alpha_mean"],
        "max_weight_mean": summary["max_weight_mean"],
        "effective_holdings_mean": summary["effective_holdings_mean"],
        "ticker_count_mean": float(growth_df["ticker_count"].mean()),
        "cumulative_growth": summary["cumulative_growth"],
        "benchmark_cumulative_growth": summary["benchmark_cumulative_growth"],
        "cumulative_alpha": summary["cumulative_alpha"],
        "attention_run_id": summary["attention_run_id"],
        "bilstm_run_id": summary["bilstm_run_id"],
    }
    return pd.DataFrame([summary_row])


def _plot_ensemble_growth(
    evaluator: QaiPortfolioMultirunEvaluator,
    scope: str,
    benchmark_nav_df: pd.DataFrame,
    chart_components: list[dict[str, Any]],
    ensemble_nav_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(
        benchmark_nav_df["quarter_end_date"],
        benchmark_nav_df["rebased_nav"],
        color="black",
        linewidth=2.5,
        linestyle="--",
        marker="o",
        label="equal weight benchmark",
    )
    component_styles = {
        "attention": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0},
        "bilstm": {"color": "#2ca02c", "linestyle": ":", "linewidth": 2.0},
    }
    for item in chart_components:
        style = component_styles.get(
            item["family"],
            {"color": "#7f7f7f", "linestyle": "-", "linewidth": 2.0},
        )
        nav_df = item["nav"]
        ax.plot(
            nav_df["quarter_end_date"],
            nav_df["rebased_nav"],
            marker="o",
            label=item["strategy_label"],
            **style,
        )
    ax.plot(
        ensemble_nav_df["quarter_end_date"],
        ensemble_nav_df["rebased_nav"],
        color="#d62728",
        linewidth=2.5,
        linestyle="-.",
        marker="o",
        label=ensemble_nav_df["strategy_label"].iloc[0],
    )
    tick_dates = sorted(
        pd.to_datetime(
            pd.concat(
                [benchmark_nav_df["quarter_end_date"], ensemble_nav_df["quarter_end_date"]]
                + [item["nav"]["quarter_end_date"] for item in chart_components],
                ignore_index=True,
            ).dropna().unique()
        ).tolist()
    )
    tick_positions, tick_labels = evaluator._quarter_tick_labels(tick_dates)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_title(f"Rebased Cumulative NAV: {scope.upper()} Replay")
    ax.set_ylabel("NAV")
    ax.set_xlabel("Quarter End Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
