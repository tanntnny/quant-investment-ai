from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.alphavantage import AlphaVantage
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch market news and sentiment data from Alpha Vantage."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to the .env file containing ALPHA_VANTAGE_API_KEY.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL"],
        help="Ticker filters such as AAPL, CRYPTO:BTC, or FOREX:USD.",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        help="Topic filters such as technology or earnings.",
    )
    parser.add_argument(
        "--time-from",
        dest="time_from",
        help="Start timestamp in YYYYMMDDTHHMM format.",
    )
    parser.add_argument(
        "--time-to",
        dest="time_to",
        help="End timestamp in YYYYMMDDTHHMM format.",
    )
    parser.add_argument(
        "--sort",
        default="LATEST",
        choices=["LATEST", "EARLIEST", "RELEVANCE"],
        help="Sort order documented by Alpha Vantage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of articles to request, between 1 and 1000.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the raw API response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = AlphaVantage.from_env(env_path=args.env_file)
    response = client.get_news_sentiment(
        tickers=args.tickers,
        topics=args.topics,
        time_from=args.time_from,
        time_to=args.time_to,
        sort=args.sort,
        limit=args.limit,
    )

    feed = response.get("feed", [])
    print(f"Retrieved {len(feed)} articles.")

    for index, article in enumerate(feed[:5], start=1):
        title = article.get("title", "<missing title>")
        source = article.get("source", "<unknown source>")
        published_at = article.get("time_published", "<unknown time>")
        print(f"{index}. {title}")
        print(f"   source={source} published_at={published_at}")

    if args.output is not None:
        save_json(response, args.output)
        print(f"Saved raw response to {args.output}")


if __name__ == "__main__":
    main()
