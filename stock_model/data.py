from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class MarketBundle:
    ticker: str
    history: pd.DataFrame
    fundamentals: dict[str, Any]
    news: list[dict[str, Any]]
    source_note: str


def _yf():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return yf


def _clean_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        raise ValueError("No price history was returned.")
    data = history.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Market data is missing: {missing}")
    data = data[required].dropna().sort_index()
    if len(data) < 220:
        raise ValueError(f"Only {len(data)} observations were returned; 220 are required.")
    return data


def _first_earnings_date(calendar: Any) -> Any:
    if calendar is None:
        return None
    if isinstance(calendar, dict):
        value = calendar.get("Earnings Date") or calendar.get("EarningsDate")
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value
    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
        for label in ["Earnings Date", "EarningsDate"]:
            if label in calendar.index:
                value = calendar.loc[label].iloc[0]
                return value[0] if isinstance(value, (list, tuple)) and value else value
            if label in calendar.columns:
                return calendar[label].iloc[0]
    return None


def _normalize_news(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized = []
    for raw in items or []:
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = content.get("title") or raw.get("title") or "Untitled"
        provider = content.get("provider")
        publisher = (
            provider.get("displayName")
            if isinstance(provider, dict)
            else raw.get("publisher") or "Unknown"
        )
        published = content.get("pubDate") or raw.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
        canonical = content.get("canonicalUrl")
        click_through = content.get("clickThroughUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else None
        if not link and isinstance(click_through, dict):
            link = click_through.get("url")
        if not link:
            link = raw.get("link")
        normalized.append(
            {
                "title": str(title),
                "publisher": str(publisher),
                "published": published,
                "link": link,
                "summary": content.get("summary") or "",
            }
        )
    return normalized


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    yf = _yf()
    instrument = yf.Ticker(ticker.upper())
    history = instrument.history(period=period, interval="1d", auto_adjust=True)
    return _clean_history(history)


def fetch_us_bundle(ticker: str, period: str = "2y") -> MarketBundle:
    yf = _yf()
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker cannot be blank.")
    instrument = yf.Ticker(symbol)
    history = _clean_history(
        instrument.history(period=period, interval="1d", auto_adjust=True)
    )
    try:
        info = instrument.info or {}
    except Exception:
        info = {}
    try:
        calendar = instrument.calendar
    except Exception:
        calendar = None
    try:
        news = _normalize_news(instrument.news)
    except Exception:
        news = []

    fundamentals = {
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "profit_margin": info.get("profitMargins"),
        "free_cash_flow": info.get("freeCashflow"),
        "total_cash": info.get("totalCash"),
        "total_debt": info.get("totalDebt"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("trailingPegRatio") or info.get("pegRatio"),
        "earnings_date": _first_earnings_date(calendar),
        "currency": info.get("currency", "USD"),
        "website": info.get("website"),
        "business_summary": info.get("longBusinessSummary"),
        "short_percent_of_float": info.get("shortPercentOfFloat"),
        "short_ratio": info.get("shortRatio"),
        "shares_short": info.get("sharesShort"),
        "shares_short_prior_month": info.get("sharesShortPriorMonth"),
        "float_shares": info.get("floatShares"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "52_week_change": info.get("52WeekChange"),
        "beta": info.get("beta"),
        "average_volume": info.get("averageVolume"),
    }
    return MarketBundle(
        ticker=symbol,
        history=history,
        fundamentals=fundamentals,
        news=news,
        source_note=(
            "Price, company-profile, calendar, and headline data were retrieved through "
            "the community yfinance adapter. Confirm all actionable facts with primary "
            "filings and broker quotes."
        ),
    )
