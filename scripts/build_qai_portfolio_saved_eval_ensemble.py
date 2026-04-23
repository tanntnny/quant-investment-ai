from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluators.qai_portfolio_saved_eval_ensemble import (
    DEFAULT_RESULT_NAME,
    build_saved_eval_ensemble_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained 50/50 attention + BiLSTM ensemble portfolio "
            "evaluation from archived portfolio_eval CSV outputs."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Archived evaluation root or eval_outputs directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination result directory. Defaults to "
            f"<source_root>/{DEFAULT_RESULT_NAME}."
        ),
    )
    parser.add_argument(
        "--top-holdings-per-quarter",
        type=int,
        default=10,
        help="Rows to keep in portfolio_eval_top5_top_holdings_prices_<scope>.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_saved_eval_ensemble_result(
        source_dir=args.source,
        output_dir=args.output,
        top_holdings_per_quarter=args.top_holdings_per_quarter,
    )


if __name__ == "__main__":
    main()
