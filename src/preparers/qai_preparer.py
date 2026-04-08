from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from src.data.dataset_builder import DatasetBuildConfig, build_fundamental_dataset
from src.data.providers import create_provider
from src.utils.alphavantage import AlphaVantage, AlphaVantageError
from src.utils.console import announce, render_kv_table

logger = logging.getLogger(__name__)


STORY_COLUMNS = [
    "ticker",
    "date",
    "time_published",
    "title",
    "summary",
    "source",
    "source_domain",
    "category_within_source",
    "url",
    "overall_sentiment_score",
    "overall_sentiment_label",
    "ticker_relevance_score",
    "ticker_sentiment_score",
    "ticker_sentiment_label",
    "authors_json",
    "topics_json",
    "banner_image",
]


def _announce(message: str) -> None:
    logger.info(message)
    announce(message)


@dataclass(slots=True)
class StoryFetchSummary:
    window_count: int
    api_calls_planned: int
    api_calls_executed: int
    article_count_raw: int
    time_from: str | None
    time_to: str | None
    divide_range_days: int | None
    requested_limit: int | None
    configured_api_calls_per_minute: int | None


def _parse_api_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, format="%Y%m%dT%H%M", errors="coerce")


def _parse_published_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, format="%Y%m%dT%H%M%S", errors="coerce")


def _format_api_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%dT%H%M")


def _normalize_date_column(frame: pd.DataFrame, source_column: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized = normalized.rename(columns={source_column: "date"})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    ordered_columns = ["ticker", "date"] + [
        column for column in normalized.columns if column not in {"ticker", "date"}
    ]
    return normalized.loc[:, ordered_columns]


def _filter_fundamental_dates(
    frame: pd.DataFrame, time_from: str | None, time_to: str | None
) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["report_date"] = pd.to_datetime(filtered["report_date"], errors="coerce")
    start = _parse_api_timestamp(time_from)
    end = _parse_api_timestamp(time_to)
    if start is not None:
        filtered = filtered.loc[filtered["report_date"] >= start.normalize()]
    if end is not None:
        filtered = filtered.loc[filtered["report_date"] <= end.normalize()]
    return _normalize_date_column(filtered, "report_date")


def _build_story_rows(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in payload.get("feed", []):
        published_at = _parse_published_timestamp(article.get("time_published"))
        if published_at is None:
            continue
        for ticker_item in article.get("ticker_sentiment", []):
            ticker = ticker_item.get("ticker")
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "date": published_at.strftime("%Y-%m-%d"),
                    "time_published": article.get("time_published"),
                    "title": article.get("title"),
                    "summary": article.get("summary"),
                    "source": article.get("source"),
                    "source_domain": article.get("source_domain"),
                    "category_within_source": article.get("category_within_source"),
                    "url": article.get("url"),
                    "overall_sentiment_score": article.get("overall_sentiment_score"),
                    "overall_sentiment_label": article.get("overall_sentiment_label"),
                    "ticker_relevance_score": ticker_item.get("relevance_score"),
                    "ticker_sentiment_score": ticker_item.get("ticker_sentiment_score"),
                    "ticker_sentiment_label": ticker_item.get("ticker_sentiment_label"),
                    "authors_json": json.dumps(article.get("authors", [])),
                    "topics_json": json.dumps(article.get("topics", [])),
                    "banner_image": article.get("banner_image"),
                }
            )
    return pd.DataFrame(rows, columns=STORY_COLUMNS)


def _build_story_request_windows(
    time_from: str | None,
    time_to: str | None,
    divide_range_days: int | None,
) -> list[tuple[str | None, str | None]]:
    if not time_from or not time_to or not divide_range_days:
        return [(time_from, time_to)]

    start = _parse_api_timestamp(time_from)
    end = _parse_api_timestamp(time_to)
    if start is None or end is None or end < start or divide_range_days <= 0:
        return [(time_from, time_to)]

    windows: list[tuple[str, str]] = []
    current_start = start
    window_span = pd.Timedelta(days=divide_range_days)
    while current_start < end:
        next_boundary = min(current_start + window_span, end)
        current_end = next_boundary
        if next_boundary < end:
            current_end = next_boundary - pd.Timedelta(minutes=1)
        windows.append((_format_api_timestamp(current_start), _format_api_timestamp(current_end)))
        current_start = next_boundary

    if not windows:
        windows.append((_format_api_timestamp(start), _format_api_timestamp(end)))

    return windows


def _allocate_window_limits(total_limit: int | None, window_count: int) -> list[int | None]:
    if window_count <= 0:
        return []
    if total_limit is None:
        return [None] * window_count
    if window_count == 1:
        return [total_limit]

    base_limit = total_limit // window_count
    remainder = total_limit % window_count
    allocated = [base_limit] * window_count
    for index in range(remainder):
        allocated[index] += 1
    return allocated


def _fetch_story_payload(
    story_client: AlphaVantage,
    *,
    alphavantage_cfg,
    time_from: str | None,
    time_to: str | None,
) -> tuple[dict[str, Any], StoryFetchSummary]:
    windows = _build_story_request_windows(
        time_from=time_from,
        time_to=time_to,
        divide_range_days=alphavantage_cfg.get("divide_range_days"),
    )
    limits = _allocate_window_limits(alphavantage_cfg.limit, len(windows))

    merged_feed: list[dict[str, Any]] = []
    api_calls_executed = 0
    for (window_from, window_to), window_limit in zip(windows, limits):
        if window_limit == 0:
            continue
        _announce(
            "[QAI] Requesting Alpha Vantage stories "
            f"window={window_from}->{window_to} limit={window_limit}"
        )
        try:
            payload = story_client.get_news_sentiment(
                tickers=None,
                topics=alphavantage_cfg.get("topics"),
                time_from=window_from,
                time_to=window_to,
                sort=alphavantage_cfg.sort,
                limit=window_limit,
            )
        except AlphaVantageError as exc:
            partial_summary = StoryFetchSummary(
                window_count=len(windows),
                api_calls_planned=sum(1 for limit in limits if limit != 0),
                api_calls_executed=api_calls_executed,
                article_count_raw=len(merged_feed),
                time_from=time_from,
                time_to=time_to,
                divide_range_days=alphavantage_cfg.get("divide_range_days"),
                requested_limit=alphavantage_cfg.get("limit"),
                configured_api_calls_per_minute=alphavantage_cfg.get("api_call_per_minute"),
            )
            wrapped_error = AlphaVantageError(
                "Alpha Vantage NEWS_SENTIMENT failed "
                f"for unfiltered tickers window={window_from}->{window_to} "
                f"limit={window_limit}: {exc}"
            )
            wrapped_error.partial_story_payload = {"feed": list(merged_feed)}
            wrapped_error.partial_story_fetch_summary = partial_summary
            raise wrapped_error from exc
        api_calls_executed += 1
        window_feed = payload.get("feed", [])
        _announce(
            "[QAI] Alpha Vantage stories received "
            f"window={window_from}->{window_to} articles={len(window_feed)}"
        )
        merged_feed.extend(window_feed)

    summary = StoryFetchSummary(
        window_count=len(windows),
        api_calls_planned=sum(1 for limit in limits if limit != 0),
        api_calls_executed=api_calls_executed,
        article_count_raw=len(merged_feed),
        time_from=time_from,
        time_to=time_to,
        divide_range_days=alphavantage_cfg.get("divide_range_days"),
        requested_limit=alphavantage_cfg.get("limit"),
        configured_api_calls_per_minute=alphavantage_cfg.get("api_call_per_minute"),
    )
    return {"feed": merged_feed}, summary


def _deduplicate_story_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.sort_values(["time_published", "ticker", "url"], kind="stable")
        .drop_duplicates(subset=["ticker", "time_published", "url"], keep="first")
        .reset_index(drop=True)
    )


class QaiPreparer:
    def __init__(self, provider=None, story_client=None) -> None:
        self.provider = provider
        self.story_client = story_client
        self.last_run_summary: dict[str, Any] | None = None

    def _create_story_client(self, data_cfg) -> AlphaVantage:
        alpha_cfg = data_cfg.provider.alphavantage
        return AlphaVantage.from_env(
            env_path=alpha_cfg.env_file,
            api_key_env=alpha_cfg.api_key_env,
            api_calls_per_minute=alpha_cfg.get("api_call_per_minute"),
        )

    def prepare(self, data_cfg, paths_cfg) -> Path:
        processed_dir = Path(getattr(paths_cfg, "cleaned_data_dir", paths_cfg.processed_data_dir))
        processed_dir.mkdir(parents=True, exist_ok=True)
        tickers: Sequence[str] = list(data_cfg.tickers)
        _announce(
            f"[QAI] Starting data preparation. tickers={len(tickers)} "
            f"time_range={data_cfg.time_from}->{data_cfg.time_to} "
            f"processed_dir={processed_dir}"
        )

        provider = self.provider or create_provider(data_cfg.provider.simfin, paths_cfg.raw_data_dir)
        build_config = DatasetBuildConfig(
            ttm_window=data_cfg.dataset.ttm_window,
            zscore_window=data_cfg.dataset.zscore_window,
            zscore_min_periods=data_cfg.dataset.zscore_min_periods,
            winsorize_lower_quantile=data_cfg.dataset.winsorize_lower_quantile,
            winsorize_upper_quantile=data_cfg.dataset.winsorize_upper_quantile,
            market_cap_min=data_cfg.dataset.market_cap_min,
        )
        df_fundamental = build_fundamental_dataset(
            tickers=tickers,
            provider=provider,
            config=build_config,
            force_refresh=bool(data_cfg.provider.simfin.get("force_refresh", False)),
        )
        if df_fundamental.empty:
            raise RuntimeError("Pipeline returned an empty dataframe. Check data source and ticker coverage.")
        _announce(
            f"[QAI] Fundamental dataset built. raw_rows={len(df_fundamental)} "
            f"raw_tickers={df_fundamental['ticker'].nunique()}"
        )

        filtered_fundamental = _filter_fundamental_dates(
            frame=df_fundamental,
            time_from=data_cfg.time_from,
            time_to=data_cfg.time_to,
        )
        _announce(
            f"[QAI] Fundamental dataset filtered by date. rows={len(filtered_fundamental)} "
            f"tickers={filtered_fundamental['ticker'].nunique() if not filtered_fundamental.empty else 0}"
        )
        fundamental_path = processed_dir / data_cfg.outputs.fundamental_filename
        filtered_fundamental.to_csv(fundamental_path, index=False)
        _announce(f"[QAI] Saved fundamental dataset to {fundamental_path}")

        story_client = self.story_client or self._create_story_client(data_cfg)
        story_path = processed_dir / data_cfg.outputs.story_filename
        try:
            story_payload, story_fetch_summary = _fetch_story_payload(
                story_client,
                alphavantage_cfg=data_cfg.provider.alphavantage,
                time_from=data_cfg.time_from,
                time_to=data_cfg.time_to,
            )
        except AlphaVantageError as exc:
            partial_story_payload = getattr(exc, "partial_story_payload", None)
            partial_story_fetch_summary = getattr(exc, "partial_story_fetch_summary", None)
            if partial_story_payload is not None and partial_story_fetch_summary is not None:
                partial_story_frame_raw = _build_story_rows(partial_story_payload)
                partial_story_frame = _deduplicate_story_rows(partial_story_frame_raw)
                partial_story_frame.to_csv(story_path, index=False)
                _announce(
                    f"[QAI] Saved partial story dataset to {story_path} "
                    f"raw_rows={len(partial_story_frame_raw)} "
                    f"deduplicated_rows={len(partial_story_frame)} "
                    "before re-raising Alpha Vantage error"
                )
                self.last_run_summary = {
                    "tickers_requested": len(tickers),
                    "time_from": data_cfg.time_from,
                    "time_to": data_cfg.time_to,
                    "processed_dir": str(processed_dir),
                    "fundamental_rows_raw": int(len(df_fundamental)),
                    "fundamental_rows_filtered": int(len(filtered_fundamental)),
                    "fundamental_unique_tickers": int(filtered_fundamental["ticker"].nunique())
                    if not filtered_fundamental.empty
                    else 0,
                    "story_articles_raw": int(partial_story_fetch_summary.article_count_raw),
                    "story_rows_raw": int(len(partial_story_frame_raw)),
                    "story_rows_deduplicated": int(len(partial_story_frame)),
                    "story_unique_tickers": int(partial_story_frame["ticker"].nunique())
                    if not partial_story_frame.empty
                    else 0,
                    "story_unique_dates": int(partial_story_frame["date"].nunique())
                    if not partial_story_frame.empty
                    else 0,
                    **asdict(partial_story_fetch_summary),
                    "fundamental_output_path": str(fundamental_path),
                    "story_output_path": str(story_path),
                    "story_partial_save": True,
                }
                logger.info(
                    "[QAI] Partial run summary: %s",
                    json.dumps(self.last_run_summary, sort_keys=True),
                )
                render_kv_table("QAI Partial Run Summary", self.last_run_summary)
            raise
        story_frame_raw = _build_story_rows(story_payload)
        story_frame = _deduplicate_story_rows(story_frame_raw)
        story_frame.to_csv(story_path, index=False)
        _announce(
            f"[QAI] Saved story dataset to {story_path} "
            f"raw_rows={len(story_frame_raw)} deduplicated_rows={len(story_frame)}"
        )

        story_client_stats_getter = getattr(story_client, "get_usage_stats", None)
        story_client_stats = (
            story_client_stats_getter() if callable(story_client_stats_getter) else {}
        )
        self.last_run_summary = {
            "tickers_requested": len(tickers),
            "time_from": data_cfg.time_from,
            "time_to": data_cfg.time_to,
            "processed_dir": str(processed_dir),
            "fundamental_rows_raw": int(len(df_fundamental)),
            "fundamental_rows_filtered": int(len(filtered_fundamental)),
            "fundamental_unique_tickers": int(filtered_fundamental["ticker"].nunique())
            if not filtered_fundamental.empty
            else 0,
            "story_articles_raw": int(story_fetch_summary.article_count_raw),
            "story_rows_raw": int(len(story_frame_raw)),
            "story_rows_deduplicated": int(len(story_frame)),
            "story_unique_tickers": int(story_frame["ticker"].nunique())
            if not story_frame.empty
            else 0,
            "story_unique_dates": int(story_frame["date"].nunique()) if not story_frame.empty else 0,
            **asdict(story_fetch_summary),
            **story_client_stats,
            "fundamental_output_path": str(fundamental_path),
            "story_output_path": str(story_path),
        }
        logger.info("[QAI] Run summary: %s", json.dumps(self.last_run_summary, sort_keys=True))
        render_kv_table("QAI Run Summary", self.last_run_summary)

        return processed_dir
