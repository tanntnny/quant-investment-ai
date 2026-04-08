from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_split_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_price_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, sep=";")


def build_qai_fixture(tmp_path: Path) -> dict[str, Path]:
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    price_path = tmp_path / "prices.csv"

    all_rows = [
        {
            "ticker": "AAA",
            "time_range": f"q{((idx % 4) + 1)}y{2020 + (idx // 4)}",
            "quarter_end_date": date,
            "feature_a": float(idx + 1),
            "feature_b": float((idx + 1) * 10),
            "window_ready": idx >= 3,
            "horizon_target_ready": True,
            "ticker_history_index": idx,
            "split": split,
        }
        for idx, (date, split) in enumerate(
            [
                ("2020-03-31", "train"),
                ("2020-06-30", "train"),
                ("2020-09-30", "train"),
                ("2020-12-31", "train"),
                ("2021-03-31", "val"),
                ("2021-06-30", "test"),
                ("2021-09-30", "test"),
                ("2021-12-31", "test"),
                ("2022-03-31", "test"),
            ]
        )
    ]

    _write_split_csv(train_path, [row for row in all_rows if row["split"] == "train"])
    _write_split_csv(val_path, [row for row in all_rows if row["split"] == "val"])
    _write_split_csv(test_path, [row for row in all_rows if row["split"] == "test"])

    _write_price_csv(
        price_path,
        [
            {"Ticker": "AAA", "Date": "2020-03-30", "Adj. Close": 99.0},
            {"Ticker": "AAA", "Date": "2020-03-31", "Adj. Close": 100.0},
            {"Ticker": "AAA", "Date": "2020-06-29", "Adj. Close": 101.0},
            {"Ticker": "AAA", "Date": "2020-06-30", "Adj. Close": 101.0},
            {"Ticker": "AAA", "Date": "2020-09-29", "Adj. Close": 98.0},
            {"Ticker": "AAA", "Date": "2020-09-30", "Adj. Close": 98.0},
            {"Ticker": "AAA", "Date": "2020-12-30", "Adj. Close": 100.0},
            {"Ticker": "AAA", "Date": "2021-03-30", "Adj. Close": 100.0},
            {"Ticker": "AAA", "Date": "2021-06-29", "Adj. Close": 103.0},
            {"Ticker": "AAA", "Date": "2021-09-29", "Adj. Close": 95.0},
            {"Ticker": "AAA", "Date": "2021-12-30", "Adj. Close": 106.0},
            {"Ticker": "AAA", "Date": "2022-03-30", "Adj. Close": 104.0},
        ],
    )

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "prices": price_path,
    }
