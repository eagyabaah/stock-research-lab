from __future__ import annotations

from math import isfinite
from typing import Any


FIELDS = [
    "revenue_growth_pct",
    "earnings_growth_pct",
    "roe_pct",
    "pe_ratio",
    "dividend_yield_pct",
    "operating_cashflow_positive",
    "debt_to_equity",
    "governance_flag",
    "regulatory_flag",
    "avg_daily_value_ghs",
    "price_vs_200d_pct",
]


def _num(value: Any) -> float | None:
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def analyze_ghana_long_term(inputs: dict[str, Any]) -> dict[str, Any]:
    """Score a Ghana long-term candidate. Missing data lowers completeness, never fabricates a pass."""
    revenue = _num(inputs.get("revenue_growth_pct"))
    earnings = _num(inputs.get("earnings_growth_pct"))
    roe = _num(inputs.get("roe_pct"))
    pe = _num(inputs.get("pe_ratio"))
    dividend = _num(inputs.get("dividend_yield_pct"))
    cashflow = _bool(inputs.get("operating_cashflow_positive"))
    debt_equity = _num(inputs.get("debt_to_equity"))
    governance = _bool(inputs.get("governance_flag"))
    regulatory = _bool(inputs.get("regulatory_flag"))
    liquidity = _num(inputs.get("avg_daily_value_ghs"))
    timing = _num(inputs.get("price_vs_200d_pct"))

    available = [revenue, earnings, roe, pe, dividend, cashflow, debt_equity, governance, regulatory, liquidity, timing]
    completeness = sum(value is not None for value in available) / len(available)

    fundamentals = 0.0
    evidence: dict[str, list[str]] = {key: [] for key in ["Fundamentals", "Valuation", "Balance/cash flow", "News/governance", "Liquidity", "Technical timing"]}
    if revenue is not None:
        fundamentals += 10 if revenue >= 10 else (7 if revenue > 0 else 0)
        evidence["Fundamentals"].append(f"Revenue growth: {revenue:.1f}%.")
    if earnings is not None:
        fundamentals += 15 if earnings >= 15 else (10 if earnings > 0 else 0)
        evidence["Fundamentals"].append(f"Earnings growth: {earnings:.1f}%.")
    if roe is not None:
        fundamentals += 10 if roe >= 20 else (7 if roe >= 12 else (3 if roe > 0 else 0))
        evidence["Fundamentals"].append(f"ROE: {roe:.1f}%.")

    valuation = 0.0
    if pe is not None:
        valuation += 12 if 0 < pe <= 12 else (9 if pe <= 20 else (5 if pe <= 30 else 1))
        evidence["Valuation"].append(f"P/E: {pe:.1f}x.")
    if dividend is not None:
        valuation += 8 if dividend >= 5 else (5 if dividend >= 2 else (2 if dividend > 0 else 0))
        evidence["Valuation"].append(f"Dividend yield: {dividend:.1f}%.")

    balance = 0.0
    if cashflow is not None:
        balance += 8 if cashflow else 0
        evidence["Balance/cash flow"].append(f"Operating cash flow positive: {'yes' if cashflow else 'no'}.")
    if debt_equity is not None:
        balance += 7 if debt_equity <= 1 else (4 if debt_equity <= 2 else 1)
        evidence["Balance/cash flow"].append(f"Debt/equity: {debt_equity:.2f}x.")

    governance_score = 0.0
    if governance is not None:
        governance_score += 8 if not governance else 0
        evidence["News/governance"].append(f"Unresolved governance flag: {'yes' if governance else 'no'}.")
    if regulatory is not None:
        governance_score += 7 if not regulatory else 0
        evidence["News/governance"].append(f"Unresolved regulatory flag: {'yes' if regulatory else 'no'}.")

    liquidity_score = 0.0
    if liquidity is not None:
        liquidity_score = 10 if liquidity >= 1_000_000 else (7 if liquidity >= 250_000 else (4 if liquidity >= 100_000 else 0))
        evidence["Liquidity"].append(f"Average daily traded value: GHS {liquidity:,.0f}.")

    timing_score = 0.0
    if timing is not None:
        timing_score = 5 if -5 <= timing <= 10 else (3 if -15 <= timing <= 20 else 0)
        evidence["Technical timing"].append(f"Price versus 200-day average: {timing:+.1f}%.")

    components = [
        ("Fundamentals", fundamentals, 35),
        ("Valuation", valuation, 20),
        ("Balance/cash flow", balance, 15),
        ("News/governance", governance_score, 15),
        ("Liquidity", liquidity_score, 10),
        ("Technical timing", timing_score, 5),
    ]
    score = round(sum(component[1] for component in components), 1)
    hard_pass = (
        completeness >= 0.8
        and fundamentals >= 24
        and governance_score == 15
        and liquidity_score >= 4
    )
    if completeness < 0.7:
        status = "INSUFFICIENT DATA"
    elif hard_pass and score >= 75:
        status = "ACCUMULATE GRADUALLY"
    elif governance is True or regulatory is True or score < 45:
        status = "AVOID FOR NOW"
    else:
        status = "WATCH"

    gates = []
    for name, component_score, maximum in components:
        ratio = component_score / maximum if maximum else 0
        gate_status = "PASS" if ratio >= 0.7 else ("CAUTION" if ratio >= 0.4 else "FAIL")
        gates.append({
            "gate": name,
            "status": gate_status,
            "score": component_score,
            "max_score": maximum,
            "evidence": evidence[name],
        })
    automated = bool(inputs.get("_automated"))
    warnings = [
        "This is a long-term accumulation model, not a Ghana swing-trading signal.",
    ]
    if automated:
        warnings.append(
            "Use the current IC Wealth quote as the final executable price and open every linked source before buying."
        )
    else:
        warnings.append(
            "Use official GSE filings and the actual IC Wealth quote to update every field."
        )
    return {
        "ticker": str(inputs.get("ticker", "")).upper(),
        "status": status,
        "score": score,
        "completeness_pct": round(completeness * 100),
        "gates": gates,
        "warnings": warnings,
    }
