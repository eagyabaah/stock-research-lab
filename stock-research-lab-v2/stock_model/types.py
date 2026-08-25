from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    score: float
    max_score: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradePlan:
    entry_trigger: float
    stop: float
    target_1: float
    target_2: float
    risk_per_share: float
    reward_risk_1: float
    shares: float
    notional: float
    planned_risk: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalView:
    bias: str
    evidence: list[str] = field(default_factory=list)
    bearish_strategy: str | None = None
    bearish_trigger: float | None = None
    bearish_invalidation: float | None = None
    bearish_target_1: float | None = None
    bearish_target_2: float | None = None
    short_execution_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    ticker: str
    as_of: str
    status: str
    score: float
    strategy: str
    price: float
    gates: list[GateResult]
    trade_plan: TradePlan | None
    thesis: list[str]
    bear_case: list[str]
    invalidation: list[str]
    facts: dict[str, Any]
    news: list[dict[str, Any]]
    warnings: list[str]
    directional_view: DirectionalView

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gates": [gate.to_dict() for gate in self.gates],
            "trade_plan": self.trade_plan.to_dict() if self.trade_plan else None,
            "directional_view": self.directional_view.to_dict(),
        }
