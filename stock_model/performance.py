from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

HORIZONS = (1, 5, 10, 20)
ACTIONABLE = {"LONG", "STRONG LONG", "SHORT", "STRONG SHORT"}


def _direction(rec: str) -> str:
    return "LONG" if "LONG" in rec else "SHORT" if "SHORT" in rec else "NONE"


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append_signals(rows: list[dict[str, Any]], reports: list[dict[str, Any]], market_date: str) -> list[dict[str, Any]]:
    existing = {(r.get("ticker"), r.get("signal_date")) for r in rows}
    for report in reports:
        key = (report["ticker"], market_date)
        if key in existing:
            continue
        plan = report.get("trade_plan") or {}
        rec = report.get("recommendation", "NO TRADE")
        rows.append({
            "signal_id": f"{report['ticker']}-{market_date.replace('-', '')}",
            "signal_date": market_date,
            "ticker": report["ticker"],
            "recommendation": rec,
            "direction": _direction(rec),
            "confidence": report.get("confidence"),
            "signal_price": report.get("price"),
            "long_score": report.get("long_score"),
            "short_score": report.get("short_score"),
            "selected_score": max(float(report.get("long_score") or 0), float(report.get("short_score") or 0)),
            "strategy": report.get("strategy"),
            "entry": plan.get("entry_trigger"),
            "stop": plan.get("stop"),
            "target_1": plan.get("target_1"),
            "target_2": plan.get("target_2"),
            "target_3": plan.get("target_3"),
            "planned_risk": plan.get("planned_risk"),
            "rr_1": plan.get("reward_risk_1"),
            "outcomes": {},
        })
        existing.add(key)
    return rows


def grade_ledger(rows: list[dict[str, Any]], history_fetcher: Callable[[str], pd.DataFrame], benchmark: pd.DataFrame) -> list[dict[str, Any]]:
    spy = benchmark.copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    for row in rows:
        try:
            hist = history_fetcher(row["ticker"]).copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            start = pd.Timestamp(row["signal_date"])
            future = hist.loc[hist.index > start]
            spy_future = spy.loc[spy.index > start]
            p0 = float(row["signal_price"])
            direction = row.get("direction", "NONE")
            outcomes = row.setdefault("outcomes", {})
            for horizon in HORIZONS:
                key = f"d{horizon}"
                if key in outcomes or len(future) < horizon or len(spy_future) < horizon:
                    continue
                px = float(future.iloc[horizon - 1]["Close"])
                spy0_rows = spy.loc[spy.index <= start]
                if spy0_rows.empty:
                    continue
                spy0 = float(spy0_rows.iloc[-1]["Close"])
                spy_px = float(spy_future.iloc[horizon - 1]["Close"])
                raw = (px / p0 - 1.0) * 100
                directional = raw if direction == "LONG" else -raw if direction == "SHORT" else raw
                outcomes[key] = {
                    "close": round(px, 4),
                    "stock_return_pct": round(raw, 3),
                    "directional_return_pct": round(directional, 3) if direction != "NONE" else None,
                    "spy_return_pct": round((spy_px / spy0 - 1.0) * 100, 3),
                    "alpha_pct": round(directional - ((spy_px / spy0 - 1.0) * 100), 3) if direction == "LONG" else round(directional, 3) if direction == "SHORT" else None,
                    "win": bool(directional > 0) if direction != "NONE" else None,
                }
        except Exception:
            continue
    return rows


def performance_summary(rows: list[dict[str, Any]], horizon: int = 5) -> dict[str, Any]:
    key = f"d{horizon}"
    graded = [r for r in rows if r.get("recommendation") in ACTIONABLE and key in r.get("outcomes", {})]
    if not graded:
        return {"count": 0}
    returns = [float(r["outcomes"][key]["directional_return_pct"]) for r in graded]
    wins = [r for r in graded if r["outcomes"][key].get("win")]
    longs = [r for r in graded if r.get("direction") == "LONG"]
    shorts = [r for r in graded if r.get("direction") == "SHORT"]
    def wr(group):
        return round(100 * sum(bool(r["outcomes"][key].get("win")) for r in group) / len(group), 1) if group else None
    return {
        "count": len(graded), "win_rate": round(100 * len(wins) / len(graded), 1),
        "long_win_rate": wr(longs), "short_win_rate": wr(shorts),
        "avg_return_pct": round(sum(returns) / len(returns), 2),
        "avg_winner_pct": round(sum(x for x in returns if x > 0) / max(1, sum(x > 0 for x in returns)), 2),
        "avg_loser_pct": round(sum(x for x in returns if x <= 0) / max(1, sum(x <= 0 for x in returns)), 2),
    }
