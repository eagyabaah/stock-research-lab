from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import floor, isfinite
from typing import Any

import numpy as np
import pandas as pd

from .indicators import add_indicators, relative_strength_63
from .types import AnalysisResult, DirectionalView, GateResult, TradePlan


@dataclass(frozen=True)
class ModelConfig:
    portfolio_value: float = 1_000.0
    risk_per_trade_pct: float = 0.5
    risk_dollar_cap: float = 5.0
    max_position: float = 200.0
    max_total_open_risk: float = 20.0
    current_open_risk: float = 0.0
    current_drawdown_pct: float = 0.0
    halt_drawdown_pct: float = 10.0
    min_reward_risk: float = 2.0
    min_avg_dollar_volume: float = 50_000_000.0
    fractional_shares: bool = True
    allow_shorts: bool = True
    max_short_percent_float: float = 35.0
    max_short_ratio: float = 12.0

    @property
    def risk_budget(self) -> float:
        percentage_budget = self.portfolio_value * self.risk_per_trade_pct / 100
        remaining_open_risk = max(0.0, self.max_total_open_risk - self.current_open_risk)
        return max(0.0, min(percentage_budget, self.risk_dollar_cap, remaining_open_risk))


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _pct(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:.1f}%"


def assess_market_regime(benchmark: pd.DataFrame) -> GateResult:
    data = add_indicators(benchmark)
    row = data.iloc[-1]
    price = float(row["Close"])
    sma50 = float(row["SMA50"])
    sma200 = float(row["SMA200"])
    slope = float(row["SMA50_SLOPE20"])
    evidence = [
        f"Benchmark close {price:.2f}; SMA50 {sma50:.2f}; SMA200 {sma200:.2f}.",
        f"Twenty-session change in SMA50: {slope:+.2f}.",
    ]
    if price > sma200 and sma50 > sma200 and slope > 0:
        return GateResult("Market regime", "PASS", 0, 0, evidence)
    if price > sma200:
        evidence.append("Benchmark remains above SMA200, but intermediate confirmation is mixed.")
        return GateResult("Market regime", "CAUTION", 0, 0, evidence)
    evidence.append("Benchmark is below SMA200; this blocks new longs but does not automatically block shorts.")
    return GateResult("Market regime", "FAIL", 0, 0, evidence)


def _portfolio_risk_gate(config: ModelConfig) -> GateResult:
    remaining = max(0.0, config.max_total_open_risk - config.current_open_risk)
    evidence = [
        f"Current drawdown: {config.current_drawdown_pct:.1f}% of the portfolio.",
        f"Current open risk: ${config.current_open_risk:.2f}; remaining open-risk capacity: ${remaining:.2f}.",
        f"Effective risk budget for a new trade: ${config.risk_budget:.2f}.",
    ]
    if config.current_drawdown_pct >= config.halt_drawdown_pct:
        evidence.append("The drawdown halt has been reached; new trades are blocked in both directions.")
        return GateResult("Portfolio risk", "FAIL", 0, 0, evidence)
    if config.risk_budget <= 0:
        evidence.append("The maximum total open-risk allowance is already used.")
        return GateResult("Portfolio risk", "FAIL", 0, 0, evidence)
    if config.current_drawdown_pct >= 6:
        evidence.append("Drawdown breaker: only the strongest setups should be considered.")
        return GateResult("Portfolio risk", "CAUTION", 0, 0, evidence)
    return GateResult("Portfolio risk", "PASS", 0, 0, evidence)


def _technical_scores(
    price_data: pd.DataFrame, benchmark: pd.DataFrame | None
) -> tuple[GateResult, GateResult, str, str, dict[str, float]]:
    data = add_indicators(price_data)
    row = data.iloc[-1]
    previous = data.iloc[-2]
    close = float(row["Close"])
    atr = float(row["ATR14"])
    rsi = float(row["RSI14"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    sma200 = float(row["SMA200"])
    prior_high = float(row["PRIOR_HIGH20"])
    prior_low = float(row["PRIOR_LOW20"])
    volume_ratio = float(row["VOLUME_RATIO"])
    rs63 = relative_strength_63(data, benchmark)
    slope = float(row["SMA50_SLOPE20"])

    long_checks = [
        (close > sma50 > sma200, 8, "Price > SMA50 > SMA200"),
        (slope > 0, 4, "SMA50 is rising"),
        (abs(close - sma20) <= 1.5 * atr and close >= sma20 * 0.98, 5, "Price is near SMA20"),
        (45 <= rsi <= 68, 4, "RSI is in the controlled pullback zone"),
        (close > float(previous["Close"]), 3, "Latest close improved"),
        (rs63 is not None and rs63 > 0, 3, "63-day relative strength beats SPY"),
    ]
    breakout_checks = [
        (close >= prior_high, 9, "Close cleared prior 20-day high"),
        (volume_ratio >= 1.3, 6, "Volume is at least 1.3x its 20-day average"),
        (close > sma50 > sma200, 6, "Long and intermediate trends align"),
        (55 <= rsi <= 72, 4, "RSI confirms without being extreme"),
        (close - prior_high <= atr, 3, "Breakout is no more than one ATR extended"),
        (rs63 is not None and rs63 > 0, 2, "Relative strength beats SPY"),
    ]
    recent = data.iloc[-10:-1]
    recently_below = bool(((recent["Close"] < recent["SMA20"]) | (recent["Close"] < recent["SMA50"])).any())
    recovery_checks = [
        (close > sma200, 5, "Price remains above SMA200"),
        (close > sma20 and close > sma50, 8, "Price reclaimed SMA20 and SMA50"),
        (recently_below, 5, "A recent correction created a genuine reclaim"),
        (50 < rsi <= 70, 4, "RSI supports recovery"),
        (rs63 is not None and rs63 > 0, 4, "Relative strength is positive"),
        (volume_ratio >= 1.0, 4, "Volume confirms the reclaim"),
    ]

    short_trend = [
        (close < sma20 < sma50, 8, "Price < SMA20 < SMA50"),
        (sma50 < sma200, 5, "SMA50 is below SMA200"),
        (slope < 0, 4, "SMA50 is falling"),
        (rsi < 50, 4, "RSI is below 50"),
        (close < float(previous["Close"]), 3, "Latest close weakened"),
        (rs63 is not None and rs63 < 0, 3, "63-day relative strength trails SPY"),
    ]
    breakdown_checks = [
        (close <= prior_low, 9, "Close broke the prior 20-day low"),
        (volume_ratio >= 1.3, 6, "Breakdown volume is at least 1.3x average"),
        (close < sma50 < sma200, 6, "Long and intermediate trends are bearish"),
        (25 <= rsi <= 50, 4, "RSI confirms bearish momentum without being exhausted"),
        (prior_low - close <= atr, 3, "Breakdown is no more than one ATR extended"),
        (rs63 is not None and rs63 < 0, 2, "Relative strength trails SPY"),
    ]
    recently_above = bool(((recent["Close"] > recent["SMA20"]) | (recent["Close"] > recent["SMA50"])).any())
    failed_reclaim_checks = [
        (close < sma200, 5, "Price is below SMA200"),
        (close < sma20 and close < sma50, 8, "Price lost SMA20 and SMA50"),
        (recently_above, 5, "A recent rally created a failed reclaim"),
        (rsi < 50, 4, "RSI confirms weakness"),
        (rs63 is not None and rs63 < 0, 4, "Relative strength is negative"),
        (volume_ratio >= 1.0, 4, "Volume confirms the failed reclaim"),
    ]

    def best_setup(candidates: list[tuple[str, list[tuple[bool, int, str]], int, int]]) -> tuple[float, bool, str, list[tuple[bool, int, str]]]:
        scored = []
        for name, checks, pass_threshold, maximum in candidates:
            raw = sum(weight for passed, weight, _ in checks if passed)
            normalized = raw / maximum * 30
            scored.append((normalized, raw >= pass_threshold, name, checks))
        return max(scored, key=lambda item: item[0])

    long_score, long_pass, long_strategy, long_checks = best_setup([
        ("Trend pullback", long_checks, 21, 27),
        ("Confirmed breakout", breakout_checks, 24, 30),
        ("Recovery/reclaim", recovery_checks, 23, 30),
    ])
    short_score, short_pass, short_strategy, short_checks = best_setup([
        ("Bearish trend", short_trend, 21, 27),
        ("Confirmed breakdown", breakdown_checks, 24, 30),
        ("Failed reclaim", failed_reclaim_checks, 23, 30),
    ])

    common_evidence = [
        f"Close {close:.2f}; SMA20 {sma20:.2f}; SMA50 {sma50:.2f}; SMA200 {sma200:.2f}.",
        f"RSI14 {rsi:.1f}; ATR14 {atr:.2f} ({atr / close * 100:.1f}%); volume ratio {volume_ratio:.2f}x.",
        f"63-day relative strength vs SPY: {'unavailable' if rs63 is None else f'{rs63 * 100:+.1f}%' }.",
    ]
    long_evidence = common_evidence + [
        f"Selected strategy: {long_strategy}.",
        *[f"{'Passed' if ok else 'Failed'}: {label}." for ok, _, label in long_checks],
    ]
    short_evidence = common_evidence + [
        f"Selected strategy: {short_strategy}.",
        *[f"{'Passed' if ok else 'Failed'}: {label}." for ok, _, label in short_checks],
    ]
    long_gate = GateResult("Technical (long)", "PASS" if long_pass else "FAIL", round(long_score, 1), 30, long_evidence)
    short_gate = GateResult("Technical (short)", "PASS" if short_pass else "FAIL", round(short_score, 1), 30, short_evidence)
    facts = {
        "close": close,
        "atr14": atr,
        "rsi14": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "prior_high20": prior_high,
        "prior_low20": prior_low,
        "low10": float(row["LOW10"]),
        "volume_ratio": volume_ratio,
        "rs63": rs63 if rs63 is not None else np.nan,
        "latest_high": float(row["High"]),
        "latest_low": float(row["Low"]),
        "sma50_slope20": slope,
    }
    return long_gate, short_gate, long_strategy, short_strategy, facts


def _fundamental_gates(fundamentals: dict[str, Any]) -> tuple[GateResult, GateResult]:
    revenue_growth = _safe_number(fundamentals.get("revenue_growth"))
    earnings_growth = _safe_number(fundamentals.get("earnings_growth"))
    profit_margin = _safe_number(fundamentals.get("profit_margin"))
    free_cash_flow = _safe_number(fundamentals.get("free_cash_flow"))
    total_debt = _safe_number(fundamentals.get("total_debt"))
    total_cash = _safe_number(fundamentals.get("total_cash"))

    long_score = 0.0
    short_score = 0.0
    available = 0
    long_evidence: list[str] = []
    short_evidence: list[str] = []

    for label, value, long_points, short_points in [
        ("revenue growth", revenue_growth, 6, 6),
        ("earnings growth", earnings_growth, 6, 6),
        ("profit margin", profit_margin, 5, 5),
        ("free cash flow", free_cash_flow, 4, 4),
    ]:
        if value is not None:
            available += 1
            long_score += long_points if value > 0 else 0
            short_score += short_points if value < 0 else 0
            formatted = _pct(value) if "growth" in label or "margin" in label else f"${value:,.0f}"
            long_evidence.append(f"{label.title()}: {formatted}.")
            short_evidence.append(f"{label.title()}: {formatted}.")

    if total_debt is not None and total_cash is not None:
        available += 1
        ratio = total_debt / total_cash if total_cash > 0 else float("inf")
        long_score += 4 if ratio < 2 else (2 if ratio < 4 else 0)
        short_score += 4 if ratio >= 4 else (2 if ratio >= 2 else 0)
        long_evidence.append(f"Debt-to-cash ratio: {ratio:.2f}x.")
        short_evidence.append(f"Debt-to-cash ratio: {ratio:.2f}x.")

    if available < 3:
        long_status = short_status = "CAUTION"
        long_evidence.append("Too few fundamental fields were available for a directional pass.")
        short_evidence.append("Too few fundamental fields were available for a directional pass.")
    else:
        long_status = "PASS" if long_score >= 17 else ("CAUTION" if long_score >= 11 else "FAIL")
        short_status = "PASS" if short_score >= 17 else ("CAUTION" if short_score >= 11 else "FAIL")

    return (
        GateResult("Fundamentals (long)", long_status, round(long_score, 1), 25, long_evidence),
        GateResult("Fundamentals (short)", short_status, round(short_score, 1), 25, short_evidence),
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _news_gates(news: list[dict[str, Any]], earnings_date: Any) -> tuple[GateResult, GateResult]:
    now = datetime.now(timezone.utc)
    severe_words = {"fraud", "restatement", "bankruptcy", "accounting probe", "sec probe", "criminal investigation", "going concern", "data breach"}
    negative_words = {"lawsuit", "investigation", "downgrade", "misses", "warning", "cuts guidance", "recall", "antitrust", "regulatory probe", "layoffs", "weak demand", "shares fall", "shares drop", "profit warning"}
    positive_words = {"beat", "raises guidance", "upgrade", "record revenue", "new contract", "partnership", "approval", "buyback", "dividend increase"}
    negative_hits: list[str] = []
    positive_hits: list[str] = []
    severe_hits: list[str] = []
    recent_count = 0

    for item in news:
        title = str(item.get("title", ""))
        published = _parse_datetime(item.get("published"))
        if published and (now - published).days <= 7:
            recent_count += 1
            lowered = title.lower()
            if any(word in lowered for word in severe_words):
                severe_hits.append(title)
            if any(word in lowered for word in negative_words):
                negative_hits.append(title)
            if any(word in lowered for word in positive_words):
                positive_hits.append(title)

    base = [f"Reviewed {recent_count} headline(s) published in the last seven days."]
    event_block = False
    parsed_earnings = _parse_datetime(earnings_date)
    if parsed_earnings and parsed_earnings.date() >= now.date():
        sessions = int(np.busday_count(now.date(), parsed_earnings.date()))
        base.append(f"Next reported earnings date: {parsed_earnings.date()} ({sessions} business day(s)).")
        if sessions <= 2:
            event_block = True
            base.append("New entries are blocked within two trading sessions of earnings in both directions.")

    long_score = 15.0 if recent_count > 0 else 7.0
    short_score = 15.0 if recent_count > 0 else 7.0
    long_status = short_status = "PASS" if recent_count > 0 else "CAUTION"
    long_evidence = list(base)
    short_evidence = list(base)

    if positive_hits:
        long_score = min(15.0, long_score + 2)
        short_score = max(0.0, short_score - 3)
        long_evidence.extend(f"Positive headline: {title}" for title in positive_hits[:3])
        short_evidence.extend(f"Positive headline against short thesis: {title}" for title in positive_hits[:3])
    if negative_hits:
        short_score = min(15.0, short_score + 2)
        long_score = max(0.0, long_score - 3)
        short_evidence.extend(f"Negative headline: {title}" for title in negative_hits[:3])
        long_evidence.extend(f"Negative headline against long thesis: {title}" for title in negative_hits[:3])
    if severe_hits:
        long_status = "FAIL"
        short_status = "CAUTION"
        long_score = 0.0
        short_score = min(short_score, 8.0)
        long_evidence.extend(f"Severe headline flag: {title}" for title in severe_hits[:3])
        short_evidence.extend(f"Severe headline requires event-risk review: {title}" for title in severe_hits[:3])
    elif caution := (negative_hits or positive_hits):
        long_status = "CAUTION" if negative_hits else "PASS"
        short_status = "CAUTION" if positive_hits else "PASS"

    if event_block:
        long_status = "FAIL"
        short_status = "FAIL"
        long_score = 0.0
        short_score = 0.0

    return (
        GateResult("News/catalyst (long)", long_status, round(long_score, 1), 15, long_evidence),
        GateResult("News/catalyst (short)", short_status, round(short_score, 1), 15, short_evidence),
    )


def _valuation_gates(fundamentals: dict[str, Any]) -> tuple[GateResult, GateResult]:
    trailing_pe = _safe_number(fundamentals.get("trailing_pe"))
    forward_pe = _safe_number(fundamentals.get("forward_pe"))
    peg = _safe_number(fundamentals.get("peg_ratio"))
    reference = forward_pe if forward_pe is not None and forward_pe > 0 else trailing_pe
    valid_peg = peg if peg is not None and peg > 0 else None
    evidence = []
    if trailing_pe is not None:
        evidence.append(f"Trailing P/E: {trailing_pe:.1f}x.")
    if forward_pe is not None:
        evidence.append(f"Forward P/E: {forward_pe:.1f}x.")
    if peg is not None:
        evidence.append(f"PEG ratio: {peg:.2f}.")

    if reference is None and valid_peg is None:
        missing = GateResult("Valuation (long)", "CAUTION", 4, 10, evidence or ["Valuation data unavailable."])
        short = GateResult("Valuation (short)", "CAUTION", 4, 10, evidence or ["Valuation data unavailable."])
        return missing, short

    long_score = 10 if ((valid_peg is not None and valid_peg <= 1.5) or (reference is not None and 0 < reference <= 20)) else 7 if ((valid_peg is not None and valid_peg <= 2.5) or (reference is not None and reference <= 35)) else 2
    short_score = 2 if ((valid_peg is not None and valid_peg <= 1.5) or (reference is not None and 0 < reference <= 20)) else 7 if ((valid_peg is not None and valid_peg <= 2.5) or (reference is not None and reference <= 35)) else 10
    long_status = "PASS" if long_score >= 10 else ("CAUTION" if long_score >= 7 else "FAIL")
    short_status = "PASS" if short_score >= 10 else ("CAUTION" if short_score >= 7 else "FAIL")
    long_evidence = evidence + (["Higher valuation raises downside sensitivity for longs."] if long_status == "FAIL" else [])
    short_evidence = evidence + (["Elevated valuation can reinforce a bearish re-rating thesis."] if short_status == "PASS" else [])
    return GateResult("Valuation (long)", long_status, long_score, 10, long_evidence), GateResult("Valuation (short)", short_status, short_score, 10, short_evidence)


def _liquidity_gate(price_data: pd.DataFrame, minimum: float) -> GateResult:
    recent = price_data.sort_index().tail(20)
    avg_dollar_volume = float((recent["Close"] * recent["Volume"]).mean())
    evidence = [f"20-day average dollar volume: ${avg_dollar_volume:,.0f}."]
    if avg_dollar_volume >= minimum:
        return GateResult("Liquidity/execution", "PASS", 10, 10, evidence)
    if avg_dollar_volume >= minimum * 0.4:
        evidence.append("Liquidity is below the preferred threshold.")
        return GateResult("Liquidity/execution", "CAUTION", 6, 10, evidence)
    evidence.append("Liquidity fails the minimum execution threshold.")
    return GateResult("Liquidity/execution", "FAIL", 1, 10, evidence)


def _short_risk_gate(fundamentals: dict[str, Any], config: ModelConfig) -> GateResult:
    short_pct = _safe_number(fundamentals.get("short_percent_of_float"))
    short_ratio = _safe_number(fundamentals.get("short_ratio"))
    squeeze_flags: list[str] = []
    score = 5.0
    if short_pct is not None:
        if short_pct >= config.max_short_percent_float / 100:
            squeeze_flags.append(f"Short interest is {short_pct * 100:.1f}% of float.")
            score -= 2.5
        elif short_pct >= 0.15:
            squeeze_flags.append(f"Short interest is elevated at {short_pct * 100:.1f}% of float.")
            score -= 1
    else:
        squeeze_flags.append("Short percent of float is unavailable.")
        score -= 1
    if short_ratio is not None:
        if short_ratio >= config.max_short_ratio:
            squeeze_flags.append(f"Short ratio is {short_ratio:.1f} days-to-cover, above the model limit.")
            score -= 1
        elif short_ratio >= 5:
            squeeze_flags.append(f"Short ratio is elevated at {short_ratio:.1f} days-to-cover.")
            score -= 2
    else:
        squeeze_flags.append("Short ratio is unavailable.")
        score -= 1
    score = max(0.0, min(5.0, score))
    status = "PASS" if score >= 4 else ("CAUTION" if score >= 2.5 else "FAIL")
    evidence = squeeze_flags or ["No major short-interest warning was detected from available fields."]
    return GateResult("Short squeeze risk", status, round(score, 1), 5, evidence)


def _trade_plan(direction: str, strategy: str, facts: dict[str, float], config: ModelConfig) -> TradePlan | None:
    close = facts["close"]
    atr = facts["atr14"]
    if not isfinite(atr) or atr <= 0:
        return None

    if direction == "LONG":
        if strategy == "Confirmed breakout":
            entry = max(close, facts["prior_high20"] + 0.05 * atr)
            stop = facts["prior_high20"] - 0.75 * atr
        elif strategy == "Recovery/reclaim":
            entry = max(close, facts["latest_high"] + 0.05 * atr)
            supports = [value for value in [facts["sma20"], facts["sma50"]] if value < entry]
            if not supports:
                return None
            stop = max(supports) - 0.75 * atr
        else:
            entry = max(close, facts["latest_high"] + 0.05 * atr)
            supports = [value for value in [facts["sma20"], facts["sma50"], facts["low10"]] if value < entry]
            if not supports:
                return None
            stop = max(supports) - 0.50 * atr
        risk_per_share = entry - stop
        target_1 = entry + config.min_reward_risk * risk_per_share
        target_2 = entry + 3 * risk_per_share
        target_3 = entry + 4 * risk_per_share
    else:
        if strategy == "Confirmed breakdown":
            entry = min(close, facts["prior_low20"] - 0.05 * atr)
            stop = facts["prior_low20"] + 0.75 * atr
        elif strategy == "Failed reclaim":
            entry = min(close, facts["latest_low"] - 0.05 * atr)
            resistances = [value for value in [facts["sma20"], facts["sma50"], facts["latest_high"]] if value > entry]
            if not resistances:
                return None
            stop = min(resistances) + 0.75 * atr
        else:
            entry = min(close, facts["latest_low"] - 0.05 * atr)
            resistances = [value for value in [facts["sma20"], facts["sma50"], facts["sma200"]] if value > entry]
            if not resistances:
                return None
            stop = min(resistances) + 0.50 * atr
        risk_per_share = stop - entry
        target_1 = max(0.01, entry - config.min_reward_risk * risk_per_share)
        target_2 = max(0.01, entry - 3 * risk_per_share)
        target_3 = max(0.01, entry - 4 * risk_per_share)

    if risk_per_share <= 0:
        return None
    if risk_per_share < 0.75 * atr:
        risk_per_share = 0.75 * atr
        stop = entry - risk_per_share if direction == "LONG" else entry + risk_per_share
        target_1 = entry + config.min_reward_risk * risk_per_share if direction == "LONG" else max(0.01, entry - config.min_reward_risk * risk_per_share)
        target_2 = entry + 3 * risk_per_share if direction == "LONG" else max(0.01, entry - 3 * risk_per_share)
        target_3 = entry + 4 * risk_per_share if direction == "LONG" else max(0.01, entry - 4 * risk_per_share)

    shares_by_risk = config.risk_budget / risk_per_share
    shares_by_cap = config.max_position / entry if entry > 0 else 0
    shares = min(shares_by_risk, shares_by_cap)
    shares = floor(shares * 1000) / 1000 if config.fractional_shares else floor(shares)
    if shares <= 0:
        return None
    notional = shares * entry
    planned_risk = shares * risk_per_share
    return TradePlan(
        direction=direction,
        entry_trigger=round(entry, 2),
        stop=round(stop, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        target_3=round(target_3, 2),
        risk_per_share=round(risk_per_share, 2),
        reward_risk_1=round(config.min_reward_risk, 2),
        shares=shares,
        notional=round(notional, 2),
        planned_risk=round(planned_risk, 2),
    )


def _risk_gate(plan: TradePlan | None, config: ModelConfig, label: str) -> GateResult:
    if plan is None:
        return GateResult(f"Risk/setup ({label.lower()})", "FAIL", 0, 10, ["No valid entry and stop plan could be calculated."])
    evidence = [
        f"{plan.direction} entry {plan.entry_trigger:.2f}; stop {plan.stop:.2f}; risk/share {plan.risk_per_share:.2f}.",
        f"Size {plan.shares:.3f} share(s); notional ${plan.notional:.2f}; planned risk ${plan.planned_risk:.2f}.",
        f"Target 1 {plan.target_1:.2f}; Target 2 {plan.target_2:.2f}; Target 3 {plan.target_3:.2f}; reward/risk {plan.reward_risk_1:.1f}:1.",
    ]
    passed = plan.notional <= config.max_position + 0.02 and plan.planned_risk <= config.risk_budget + 0.02 and plan.reward_risk_1 >= config.min_reward_risk
    return GateResult(f"Risk/setup ({label.lower()})", "PASS" if passed else "FAIL", 10 if passed else 0, 10, evidence)


def _confidence(long_score: float, short_score: float, recommendation: str) -> str:
    top = max(long_score, short_score)
    gap = abs(long_score - short_score)
    if recommendation == "NO TRADE":
        return "LOW" if top < 60 else "MEDIUM"
    if top >= 85 and gap >= 10:
        return "HIGH"
    if top >= 75 and gap >= 5:
        return "MEDIUM"
    return "LOW"


def _directional_view(facts: dict[str, float], long_score: float, short_score: float, short_strategy: str, short_plan: TradePlan | None, short_risk_flags: list[str], recommendation: str, short_allowed: bool) -> DirectionalView:
    bias = {
        "LONG": "STRONG LONG" if long_score >= 85 else "LONG",
        "SHORT": "STRONG SHORT" if short_score >= 85 else "SHORT",
    }.get(recommendation, "NO TRADE")
    evidence = [
        f"Long score: {long_score:.1f}/100; short score: {short_score:.1f}/100.",
        f"Price {facts['close']:.2f} versus SMA20 {facts['sma20']:.2f}, SMA50 {facts['sma50']:.2f}, SMA200 {facts['sma200']:.2f}.",
        f"RSI14 {facts['rsi14']:.1f}; SMA50 slope {facts['sma50_slope20']:+.2f}; volume ratio {facts['volume_ratio']:.2f}x.",
    ]
    if short_plan:
        evidence.append(f"Short execution plan: entry {short_plan.entry_trigger:.2f}, stop {short_plan.stop:.2f}, targets {short_plan.target_1:.2f}/{short_plan.target_2:.2f}/{short_plan.target_3:.2f}.")
    if short_risk_flags:
        evidence.append("Short-specific risks: " + " ".join(short_risk_flags))
    return DirectionalView(
        bias=bias,
        evidence=evidence,
        bullish_score=round(long_score, 1),
        bearish_score=round(short_score, 1),
        bearish_strategy=short_strategy,
        bearish_trigger=short_plan.entry_trigger if short_plan else None,
        bearish_invalidation=short_plan.stop if short_plan else None,
        bearish_target_1=short_plan.target_1 if short_plan else None,
        bearish_target_2=short_plan.target_2 if short_plan else None,
        short_execution_allowed=short_allowed,
        short_risk_flags=short_risk_flags,
    )


def _make_rationale(direction: str, long_strategy: str, short_strategy: str, long_gates: list[GateResult], short_gates: list[GateResult], recommendation: str) -> tuple[list[str], list[str], list[str]]:
    thesis: list[str] = []
    bear_case: list[str] = []
    if direction == "LONG":
        thesis.append(f"The long thesis is led by the {long_strategy.lower()} setup and a long score that cleared the model threshold.")
        bear_case.append(f"The competing short score is {sum(g.score for g in short_gates):.1f}/100; a failed continuation or new negative catalyst can flip the setup.")
    elif direction == "SHORT":
        thesis.append(f"The short thesis is led by the {short_strategy.lower()} setup and a short score that cleared the model threshold.")
        bear_case.append(f"A short thesis can fail quickly through a squeeze, reclaim, gap, or positive catalyst even when the chart is bearish.")
    else:
        thesis.append("Neither direction cleared enough independent confirmation to justify a new position.")
        bear_case.append("The model prefers patience when the directional scores are too close, a hard gate fails, or risk/reward cannot be established cleanly.")
    long_failures = [g.name for g in long_gates if g.status == "FAIL"]
    short_failures = [g.name for g in short_gates if g.status == "FAIL"]
    if long_failures:
        bear_case.append("Long blockers: " + ", ".join(long_failures) + ".")
    if short_failures:
        bear_case.append("Short blockers: " + ", ".join(short_failures) + ".")
    invalidation = [
        "Exit when price reaches the model stop or the directional thesis is structurally invalidated.",
        "Re-score after earnings, material guidance, a major filing, or a high-impact headline.",
        "Do not average down after the invalidation level is breached.",
    ]
    if direction == "SHORT":
        invalidation.append("For shorts, reassess immediately after a high-volume reclaim of the broken support or a squeeze in short-interest names.")
    return thesis, bear_case, invalidation


def analyze_us_swing(ticker: str, price_data: pd.DataFrame, fundamentals: dict[str, Any], news: list[dict[str, Any]], benchmark: pd.DataFrame, config: ModelConfig | None = None) -> AnalysisResult:
    config = config or ModelConfig()
    portfolio_risk = _portfolio_risk_gate(config)
    market = assess_market_regime(benchmark)
    long_technical, short_technical, long_strategy, short_strategy, facts = _technical_scores(price_data, benchmark)
    long_fundamental, short_fundamental = _fundamental_gates(fundamentals)
    long_news, short_news = _news_gates(news, fundamentals.get("earnings_date"))
    long_valuation, short_valuation = _valuation_gates(fundamentals)
    liquidity = _liquidity_gate(price_data, config.min_avg_dollar_volume)
    short_liquidity = GateResult(
        "Liquidity/execution (short)",
        liquidity.status,
        round(liquidity.score / 2, 1),
        5,
        liquidity.evidence,
    )
    short_risk_flags: list[str] = []
    if _safe_number(fundamentals.get("short_percent_of_float")) is not None:
        pct = float(fundamentals["short_percent_of_float"]) * 100
        if pct >= 15:
            short_risk_flags.append(f"short float {pct:.1f}%")
    if _safe_number(fundamentals.get("short_ratio")) is not None:
        days = float(fundamentals["short_ratio"])
        if days >= 5:
            short_risk_flags.append(f"days-to-cover {days:.1f}")
    short_squeeze = _short_risk_gate(fundamentals, config)

    long_plan = _trade_plan("LONG", long_strategy, facts, config) if portfolio_risk.status != "FAIL" else None
    short_plan = _trade_plan("SHORT", short_strategy, facts, config) if portfolio_risk.status != "FAIL" and config.allow_shorts else None
    long_risk = _risk_gate(long_plan, config, "Long")
    short_execution_risk = _risk_gate(short_plan, config, "Short")

    long_gates = [long_technical, long_fundamental, long_news, long_valuation, liquidity, long_risk]
    short_gates = [short_technical, short_fundamental, short_news, short_valuation, short_liquidity, short_squeeze, short_execution_risk]
    long_score = round(sum(g.score for g in long_gates), 1)
    short_score = round(sum(g.score for g in short_gates), 1)

    long_hard_pass = all(g.status == "PASS" for g in [long_technical, long_fundamental, long_news, liquidity, long_risk]) and market.status != "FAIL" and portfolio_risk.status != "FAIL"
    short_hard_pass = config.allow_shorts and all(g.status == "PASS" for g in [short_technical, short_news, short_liquidity, short_squeeze, short_execution_risk]) and portfolio_risk.status != "FAIL"
    long_qualified = long_hard_pass and long_score >= 75
    short_qualified = short_hard_pass and short_score >= 75

    if long_qualified and short_qualified:
        direction = "LONG" if long_score >= short_score + 5 else "SHORT" if short_score >= long_score + 5 else "NONE"
    elif long_qualified:
        direction = "LONG"
    elif short_qualified:
        direction = "SHORT"
    else:
        direction = "NONE"

    if direction == "LONG":
        recommendation = "STRONG LONG" if long_score >= 85 else "LONG"
        status = "QUALIFIED LONG"
        selected_plan = long_plan
    elif direction == "SHORT":
        recommendation = "STRONG SHORT" if short_score >= 85 else "SHORT"
        status = "QUALIFIED SHORT"
        selected_plan = short_plan
    else:
        recommendation = "NO TRADE"
        selected_plan = None
        status = "WATCH" if max(long_score, short_score) >= 55 else "NO TRADE"
        if portfolio_risk.status == "FAIL" or liquidity.status == "FAIL":
            status = "REJECT FOR NOW"

    confidence = _confidence(long_score, short_score, recommendation)
    thesis, bear_case, invalidation = _make_rationale(direction, long_strategy, short_strategy, long_gates, short_gates, recommendation)
    directional_view = _directional_view(facts, long_score, short_score, short_strategy, short_plan if short_plan else (short_plan if direction == "SHORT" else None), short_risk_flags, direction, config.allow_shorts)
    warnings = [
        "Research model only; broker quotes, borrow availability, option liquidity, and order rules control actual execution.",
        "The short engine scores stock-specific weakness but cannot confirm real-time locate/borrow availability through the current data adapter.",
        "Headline classification is a triage tool; read the primary filing or article before committing capital.",
    ]

    all_gates = [portfolio_risk, market, *long_gates, *short_gates]
    facts = {
        **facts,
        **fundamentals,
        "short_risk_flags": short_risk_flags,
        "long_hard_pass": long_hard_pass,
        "short_hard_pass": short_hard_pass,
        "long_qualified": long_qualified,
        "short_qualified": short_qualified,
        "max_score": 100,
    }
    return AnalysisResult(
        ticker=ticker.upper(),
        as_of=pd.Timestamp(price_data.index[-1]).isoformat(),
        status=status,
        score=long_score if direction == "LONG" else short_score if direction == "SHORT" else max(long_score, short_score),
        strategy=long_strategy if direction == "LONG" else short_strategy if direction == "SHORT" else (long_strategy if long_score >= short_score else short_strategy),
        price=round(facts["close"], 2),
        gates=all_gates,
        trade_plan=selected_plan,
        thesis=thesis,
        bear_case=bear_case,
        invalidation=invalidation,
        facts=facts,
        news=news,
        warnings=warnings,
        directional_view=directional_view,
        long_score=long_score,
        short_score=short_score,
        recommendation=recommendation,
        confidence=confidence,
        selected_direction=direction,
        long_strategy=long_strategy,
        short_strategy=short_strategy,
    )
