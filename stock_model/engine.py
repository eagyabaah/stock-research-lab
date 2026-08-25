from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import floor, isfinite
from typing import Any

import numpy as np
import pandas as pd

from .indicators import add_indicators, relative_strength_63
from .types import AnalysisResult, GateResult, TradePlan


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
        evidence.append("Long-term trend is intact, but intermediate confirmation is mixed.")
        return GateResult("Market regime", "CAUTION", 0, 0, evidence)
    evidence.append("Benchmark is below its 200-day average; new long exposure is blocked.")
    return GateResult("Market regime", "FAIL", 0, 0, evidence)


def _portfolio_risk_gate(config: ModelConfig) -> GateResult:
    remaining = max(0.0, config.max_total_open_risk - config.current_open_risk)
    evidence = [
        f"Current drawdown: {config.current_drawdown_pct:.1f}% of the portfolio.",
        f"Current open risk: ${config.current_open_risk:.2f}; remaining open-risk capacity: ${remaining:.2f}.",
        f"Effective risk budget for a new trade: ${config.risk_budget:.2f}.",
    ]
    if config.current_drawdown_pct >= config.halt_drawdown_pct:
        evidence.append("The 10% drawdown halt has been reached; new trades are blocked.")
        return GateResult("Portfolio risk", "FAIL", 0, 0, evidence)
    if config.risk_budget <= 0:
        evidence.append("The maximum total open-risk allowance is already used.")
        return GateResult("Portfolio risk", "FAIL", 0, 0, evidence)
    if config.current_drawdown_pct >= 6:
        evidence.append("Drawdown breaker: reduce exposure and require only the strongest setups.")
        return GateResult("Portfolio risk", "CAUTION", 0, 0, evidence)
    return GateResult("Portfolio risk", "PASS", 0, 0, evidence)


def _technical_gate(
    price_data: pd.DataFrame, benchmark: pd.DataFrame | None
) -> tuple[GateResult, str, dict[str, float]]:
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
    volume_ratio = float(row["VOLUME_RATIO"])
    rs63 = relative_strength_63(data, benchmark)

    trend_checks = [
        (close > sma50 > sma200, 8, "Price > SMA50 > SMA200"),
        (float(row["SMA50_SLOPE20"]) > 0, 4, "SMA50 is rising"),
        (abs(close - sma20) <= 1.5 * atr and close >= sma20 * 0.98, 5, "Price is near SMA20"),
        (45 <= rsi <= 68, 4, "RSI is in the pullback zone"),
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
    recently_below = bool(
        ((recent["Close"] < recent["SMA20"]) | (recent["Close"] < recent["SMA50"])).any()
    )
    recovery_checks = [
        (close > sma200, 5, "Price remains above SMA200"),
        (close > sma20 and close > sma50, 8, "Price reclaimed SMA20 and SMA50"),
        (recently_below, 5, "A recent correction created a genuine reclaim"),
        (50 < rsi <= 70, 4, "RSI supports recovery"),
        (rs63 is not None and rs63 > 0, 4, "Relative strength is positive"),
        (volume_ratio >= 1.0, 4, "Volume confirms the reclaim"),
    ]

    candidates = []
    for name, checks, pass_threshold, maximum in [
        ("Trend pullback", trend_checks, 21, 27),
        ("Confirmed breakout", breakout_checks, 24, 30),
        ("Recovery/reclaim", recovery_checks, 23, 30),
    ]:
        raw = sum(weight for passed, weight, _ in checks if passed)
        normalized = raw / maximum * 30
        candidates.append((normalized, raw >= pass_threshold, name, checks))

    score, passed, strategy, checks = max(candidates, key=lambda item: item[0])
    evidence = [
        f"Close {close:.2f}; SMA20 {sma20:.2f}; SMA50 {sma50:.2f}; SMA200 {sma200:.2f}.",
        f"RSI14 {rsi:.1f}; ATR14 {atr:.2f} ({atr / close * 100:.1f}%); volume ratio {volume_ratio:.2f}x.",
        f"63-day relative strength vs SPY: {'unavailable' if rs63 is None else f'{rs63 * 100:+.1f}%'}.",
    ]
    evidence.extend(
        f"{'Passed' if ok else 'Failed'}: {label}." for ok, _, label in checks
    )
    status = "PASS" if passed else "FAIL"
    facts = {
        "close": close,
        "atr14": atr,
        "rsi14": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "prior_high20": prior_high,
        "low10": float(row["LOW10"]),
        "volume_ratio": volume_ratio,
        "rs63": rs63 if rs63 is not None else np.nan,
        "latest_high": float(row["High"]),
    }
    return GateResult("Technical", status, round(score, 1), 30, evidence), strategy, facts


def _fundamental_gate(fundamentals: dict[str, Any]) -> GateResult:
    revenue_growth = _safe_number(fundamentals.get("revenue_growth"))
    earnings_growth = _safe_number(fundamentals.get("earnings_growth"))
    profit_margin = _safe_number(fundamentals.get("profit_margin"))
    free_cash_flow = _safe_number(fundamentals.get("free_cash_flow"))
    total_debt = _safe_number(fundamentals.get("total_debt"))
    total_cash = _safe_number(fundamentals.get("total_cash"))

    score = 0.0
    available = 0
    evidence = []
    for value in [revenue_growth, earnings_growth, profit_margin, free_cash_flow]:
        available += value is not None
    available += total_debt is not None and total_cash is not None

    if revenue_growth is not None:
        score += 6 if revenue_growth > 0 else 0
        evidence.append(f"Latest reported revenue growth: {_pct(revenue_growth)}.")
    if earnings_growth is not None:
        score += 6 if earnings_growth > 0 else 0
        evidence.append(f"Latest reported earnings growth: {_pct(earnings_growth)}.")
    if profit_margin is not None:
        score += 5 if profit_margin > 0 else 0
        evidence.append(f"Profit margin: {_pct(profit_margin)}.")
    if free_cash_flow is not None:
        score += 4 if free_cash_flow > 0 else 0
        evidence.append(f"Free cash flow: ${free_cash_flow:,.0f}.")
    if total_debt is not None and total_cash is not None:
        ratio = total_debt / total_cash if total_cash > 0 else float("inf")
        score += 4 if ratio < 2 else (2 if ratio < 4 else 0)
        evidence.append(f"Debt-to-cash ratio: {ratio:.2f}x.")

    if available < 3:
        evidence.append("Too few fundamental fields were available for a pass.")
        return GateResult("Fundamentals", "CAUTION", round(score, 1), 25, evidence)
    status = "PASS" if score >= 17 else ("CAUTION" if score >= 11 else "FAIL")
    return GateResult("Fundamentals", status, round(score, 1), 25, evidence)


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


def _news_gate(news: list[dict[str, Any]], earnings_date: Any) -> GateResult:
    now = datetime.now(timezone.utc)
    severe_words = {
        "fraud", "restatement", "bankruptcy", "accounting probe", "sec probe",
        "criminal investigation", "going concern", "data breach",
    }
    caution_words = {
        "lawsuit", "investigation", "downgrade", "misses", "warning", "cuts guidance",
        "recall", "antitrust", "regulatory probe",
    }
    severe_hits: list[str] = []
    caution_hits: list[str] = []
    recent_count = 0
    for item in news:
        title = str(item.get("title", ""))
        published = _parse_datetime(item.get("published"))
        if published and (now - published).days <= 7:
            recent_count += 1
            lowered = title.lower()
            if any(word in lowered for word in severe_words):
                severe_hits.append(title)
            elif any(word in lowered for word in caution_words):
                caution_hits.append(title)

    evidence = [f"Reviewed {recent_count} headline(s) published in the last seven days."]
    event_block = False
    parsed_earnings = _parse_datetime(earnings_date)
    if parsed_earnings and parsed_earnings.date() >= now.date():
        sessions = int(np.busday_count(now.date(), parsed_earnings.date()))
        evidence.append(f"Next reported earnings date: {parsed_earnings.date()} ({sessions} business day(s)).")
        if sessions <= 2:
            event_block = True
            evidence.append("New entries are blocked within two trading sessions of earnings.")

    if severe_hits:
        evidence.extend(f"Severe headline flag: {title}" for title in severe_hits[:3])
    if caution_hits:
        evidence.extend(f"Headline requiring review: {title}" for title in caution_hits[:3])

    if event_block or severe_hits:
        return GateResult("News/catalyst", "FAIL", 0, 15, evidence)
    if recent_count == 0:
        evidence.append("A missing or stale headline feed cannot establish a news pass.")
        return GateResult("News/catalyst", "CAUTION", 7, 15, evidence)
    if caution_hits:
        return GateResult("News/catalyst", "CAUTION", 8, 15, evidence)
    evidence.append("No rule-based adverse headline or near-term earnings block was detected.")
    return GateResult("News/catalyst", "PASS", 15, 15, evidence)


def _valuation_gate(fundamentals: dict[str, Any]) -> GateResult:
    trailing_pe = _safe_number(fundamentals.get("trailing_pe"))
    forward_pe = _safe_number(fundamentals.get("forward_pe"))
    peg = _safe_number(fundamentals.get("peg_ratio"))
    evidence = []
    if trailing_pe is not None:
        evidence.append(f"Trailing P/E: {trailing_pe:.1f}x.")
    if forward_pe is not None:
        evidence.append(f"Forward P/E: {forward_pe:.1f}x.")
    if peg is not None:
        evidence.append(f"PEG ratio: {peg:.2f}.")
    reference = forward_pe if forward_pe is not None and forward_pe > 0 else trailing_pe
    if reference is not None and reference <= 0:
        reference = None
    valid_peg = peg if peg is not None and peg > 0 else None
    if reference is None and valid_peg is None:
        if trailing_pe is not None and trailing_pe <= 0:
            evidence.append("A non-positive P/E indicates that ordinary earnings valuation is not meaningful.")
            return GateResult("Valuation", "FAIL", 1, 10, evidence)
        return GateResult("Valuation", "CAUTION", 4, 10, ["Valuation data unavailable."])
    if (valid_peg is not None and valid_peg <= 1.5) or (reference is not None and reference <= 20):
        return GateResult("Valuation", "PASS", 10, 10, evidence)
    if (valid_peg is not None and valid_peg <= 2.5) or (reference is not None and reference <= 35):
        return GateResult("Valuation", "CAUTION", 7, 10, evidence)
    evidence.append("The multiple requires unusually strong future execution.")
    return GateResult("Valuation", "FAIL", 2, 10, evidence)


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


def _trade_plan(strategy: str, facts: dict[str, float], config: ModelConfig) -> TradePlan | None:
    close = facts["close"]
    atr = facts["atr14"]
    if not isfinite(atr) or atr <= 0:
        return None
    if strategy == "Confirmed breakout":
        entry = max(close, facts["prior_high20"] + 0.05 * atr)
        stop = facts["prior_high20"] - 0.75 * atr
    elif strategy == "Recovery/reclaim":
        entry = max(close, facts["latest_high"] + 0.05 * atr)
        supports = [value for value in [facts["sma20"], facts["sma50"]] if value < entry]
        if not supports:
            return None
        support = max(supports)
        stop = support - 0.75 * atr
    else:
        entry = max(close, facts["latest_high"] + 0.05 * atr)
        supports = [value for value in [facts["sma20"], facts["sma50"], facts["low10"]] if value < entry]
        if not supports:
            return None
        stop = max(supports) - 0.50 * atr

    if entry - stop < 0.75 * atr:
        stop = entry - 0.75 * atr
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    shares_by_risk = config.risk_budget / risk_per_share
    shares_by_cap = config.max_position / entry
    shares = min(shares_by_risk, shares_by_cap)
    shares = floor(shares * 1000) / 1000 if config.fractional_shares else floor(shares)
    if shares <= 0:
        return None
    notional = shares * entry
    planned_risk = shares * risk_per_share
    target_1 = entry + config.min_reward_risk * risk_per_share
    target_2 = entry + 3 * risk_per_share
    return TradePlan(
        entry_trigger=round(entry, 2),
        stop=round(stop, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        risk_per_share=round(risk_per_share, 2),
        reward_risk_1=config.min_reward_risk,
        shares=shares,
        notional=round(notional, 2),
        planned_risk=round(planned_risk, 2),
    )


def _risk_gate(plan: TradePlan | None, config: ModelConfig) -> GateResult:
    if plan is None:
        return GateResult("Risk/setup", "FAIL", 0, 10, ["No valid entry and stop plan could be calculated."])
    evidence = [
        f"Entry {plan.entry_trigger:.2f}; stop {plan.stop:.2f}; risk/share {plan.risk_per_share:.2f}.",
        f"Size {plan.shares:.3f} share(s); notional ${plan.notional:.2f}; planned risk ${plan.planned_risk:.2f}.",
        f"Target 1 {plan.target_1:.2f}; reward/risk {plan.reward_risk_1:.1f}:1.",
    ]
    passed = (
        plan.notional <= config.max_position + 0.02
        and plan.planned_risk <= config.risk_budget + 0.02
        and plan.reward_risk_1 >= config.min_reward_risk
    )
    return GateResult("Risk/setup", "PASS" if passed else "FAIL", 10 if passed else 0, 10, evidence)


def _make_rationale(
    strategy: str,
    technical: GateResult,
    fundamentals_gate: GateResult,
    valuation: GateResult,
    news_gate: GateResult,
) -> tuple[list[str], list[str], list[str]]:
    thesis = [
        f"The best-matching setup is {strategy.lower()}; its technical gate is {technical.status.lower()}.",
        fundamentals_gate.evidence[0] if fundamentals_gate.evidence else "Fundamental evidence is incomplete.",
        f"Valuation is classified {valuation.status.lower()} under the model's simple P/E/PEG guardrail.",
    ]
    bear_case = []
    if technical.status != "PASS":
        bear_case.append("The chart has not completed the required confirmation; an attractive company is not automatically a valid swing entry.")
    if fundamentals_gate.status != "PASS":
        bear_case.append("Fundamental strength is either insufficient or incompletely verified by the current data feed.")
    if valuation.status == "FAIL":
        bear_case.append("A high valuation raises the downside if growth expectations disappoint.")
    if news_gate.status != "PASS":
        bear_case.append("Current news or event risk has not cleared the hard gate.")
    if not bear_case:
        bear_case.append("Even a fully qualified setup can fail through a gap, market reversal, or unexpected company news.")
    invalidation = [
        "The setup is invalid if price reaches the model stop or the relevant moving-average structure breaks.",
        "Re-score immediately after earnings, guidance, a material filing, or a major adverse headline.",
    ]
    return thesis, bear_case, invalidation


def analyze_us_swing(
    ticker: str,
    price_data: pd.DataFrame,
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    benchmark: pd.DataFrame,
    config: ModelConfig | None = None,
) -> AnalysisResult:
    config = config or ModelConfig()
    portfolio_risk = _portfolio_risk_gate(config)
    market = assess_market_regime(benchmark)
    technical, strategy, facts = _technical_gate(price_data, benchmark)
    fundamental = _fundamental_gate(fundamentals)
    news_result = _news_gate(news, fundamentals.get("earnings_date"))
    valuation = _valuation_gate(fundamentals)
    liquidity = _liquidity_gate(price_data, config.min_avg_dollar_volume)
    plan = _trade_plan(strategy, facts, config)
    risk = _risk_gate(plan, config)
    scored_gates = [technical, fundamental, news_result, valuation, liquidity, risk]
    score = round(sum(gate.score for gate in scored_gates), 1)

    hard_pass = all(
        gate.status == "PASS"
        for gate in [technical, fundamental, news_result, liquidity, risk]
    ) and market.status != "FAIL" and portfolio_risk.status != "FAIL"
    if hard_pass and score >= 75:
        status = "QUALIFIED"
    elif fundamental.status == "FAIL" or liquidity.status == "FAIL" or news_result.status == "FAIL" or market.status == "FAIL" or portfolio_risk.status == "FAIL":
        status = "REJECT FOR NOW"
    else:
        status = "WATCH"

    thesis, bear_case, invalidation = _make_rationale(
        strategy, technical, fundamental, valuation, news_result
    )
    warnings = [
        "Research model only; broker quotes and order rules control actual execution.",
        "The rule-based headline screen cannot determine materiality as reliably as reading primary filings.",
    ]
    as_of = pd.Timestamp(price_data.index[-1]).isoformat()
    return AnalysisResult(
        ticker=ticker.upper(),
        as_of=as_of,
        status=status,
        score=score,
        strategy=strategy,
        price=round(facts["close"], 2),
        gates=[portfolio_risk, market, *scored_gates],
        trade_plan=plan,
        thesis=thesis,
        bear_case=bear_case,
        invalidation=invalidation,
        facts={**facts, **fundamentals},
        news=news,
        warnings=warnings,
    )
