from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_model import ModelConfig, analyze_us_swing  # noqa: E402
from stock_model.data import fetch_history, fetch_us_bundle  # noqa: E402


def read_watchlist() -> list[str]:
    path = PROJECT_ROOT / "watchlist.txt"
    if not path.exists():
        return ["MSFT", "AMZN", "GOOGL", "EOG", "NVDA", "JPM", "V", "RDW"]
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = line.split("#", 1)[0].strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:20]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def load_previous_report() -> dict[str, Any] | None:
    path = PROJECT_ROOT / "reports" / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def value_at(report: dict[str, Any], *path: str) -> Any:
    current: Any = report
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def compare_reports(
    previous: dict[str, Any] | None, current_reports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not previous:
        return [
            {
                "ticker": report["ticker"],
                "field": "Baseline",
                "previous": None,
                "current": report["status"],
            }
            for report in current_reports
        ]
    previous_by_ticker = {
        report["ticker"]: report for report in previous.get("reports", [])
    }
    fields = [
        ("Decision", ("status",)),
        ("Score", ("score",)),
        ("Directional view", ("directional_view", "bias")),
        ("Entry", ("trade_plan", "entry_trigger")),
        ("Stop", ("trade_plan", "stop")),
        ("Target 1", ("trade_plan", "target_1")),
        ("Earnings date", ("facts", "earnings_date")),
    ]
    changes = []
    for report in current_reports:
        old = previous_by_ticker.get(report["ticker"])
        if old is None:
            changes.append(
                {
                    "ticker": report["ticker"],
                    "field": "New ticker",
                    "previous": None,
                    "current": report["status"],
                }
            )
            continue
        for label, path in fields:
            before = value_at(old, *path)
            after = value_at(report, *path)
            if json_safe(before) != json_safe(after):
                changes.append(
                    {
                        "ticker": report["ticker"],
                        "field": label,
                        "previous": json_safe(before),
                        "current": json_safe(after),
                    }
                )
    return changes


def main() -> int:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    tickers = read_watchlist()
    previous_report = load_previous_report()
    config = ModelConfig()
    benchmark = fetch_history("SPY")
    market_date = pd.Timestamp(benchmark.index[-1]).date()
    reports = []
    errors: dict[str, str] = {}

    for ticker in tickers:
        try:
            bundle = fetch_us_bundle(ticker)
            result = analyze_us_swing(
                ticker,
                bundle.history,
                bundle.fundamentals,
                bundle.news,
                benchmark,
                config,
            )
            reports.append(result.to_dict())
        except Exception as exc:
            errors[ticker] = str(exc)

    priority = {"QUALIFIED": 0, "WATCH": 1, "REJECT FOR NOW": 2}
    reports.sort(key=lambda item: (priority.get(item["status"], 9), -item["score"]))
    summary = {
        "qualified": sum(item["status"] == "QUALIFIED" for item in reports),
        "watch": sum(item["status"] == "WATCH" for item in reports),
        "rejected": sum(item["status"] == "REJECT FOR NOW" for item in reports),
    }
    changes = compare_reports(previous_report, reports)
    payload = json_safe(
        {
            "generated_at": now_et.isoformat(),
            "market_date": market_date.isoformat(),
            "market_was_closed": market_date != now_et.date(),
            "watchlist": tickers,
            "summary": summary,
            "action": "NO TRADE" if summary["qualified"] == 0 else "REVIEW QUALIFIED SETUPS",
            "reports": reports,
            "errors": errors,
            "changes_from_previous_close": changes,
            "source_note": (
                "Automated report generated from the community yfinance adapter. Prices may be "
                "delayed and headlines are screened by rules; verify primary filings, earnings "
                "dates, and the executable Robinhood quote before acting."
            ),
        }
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (reports_dir / "latest.json").write_text(output, encoding="utf-8")
    (reports_dir / f"{market_date.isoformat()}.json").write_text(output, encoding="utf-8")
    print(
        f"Generated {len(reports)} report(s) for {market_date}; "
        f"{len(errors)} ticker error(s)."
    )
    return 0 if reports else 1


if __name__ == "__main__":
    raise SystemExit(main())
