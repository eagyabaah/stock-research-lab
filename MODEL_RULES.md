# Model rules

This document is the inspectable specification behind the website. The code is
deterministic: the same inputs produce the same gate decisions and position size.

## Portfolio mandate

- Initial portfolio: $1,000.
- Monthly contribution: $500 (tracked outside this single-run engine).
- Portfolio halt: 10% drawdown.
- US risk per trade: the smallest of 0.5% of portfolio, the $5 dollar cap, and
  remaining capacity under the $20 total-open-risk limit.
- Total concurrent open risk: $20 maximum. The website sizes one proposed trade;
  the user must compare the planned risk with existing positions.
- Maximum US position: $200.
- Minimum reward/risk: 2:1.
- No leverage or averaging down. US swing analysis now evaluates LONG, SHORT, and NO TRADE. Options are research-only until an option-chain/contract-quality module is added; Ghana remains long-term accumulation only.
- Ghana analysis is long-term accumulation only.

## US scoring and direction

The engine calculates a separate 100-point score for **Long** and **Short**. The
final decision is direction-neutral: `STRONG LONG`, `LONG`, `STRONG SHORT`,
`SHORT`, or `NO TRADE`. A trade is not recommended merely because a ticker was
searched.

| Component | Points | Direction-specific? |
|---|---:|---|
| Technical | 30 | Yes |
| Fundamentals | 25 | Yes |
| News/catalyst | 15 | Yes |
| Valuation | 10 | Yes |
| Liquidity/execution | 10 | Shared |
| Risk/setup | 10 | Yes |
| Short squeeze risk | 10 | Short only |

The long side uses trend pullback, confirmed breakout, and recovery/reclaim. The
short side uses bearish trend, confirmed breakdown, and failed reclaim. The model
compares both sides and only selects a direction when its directional hard gates
pass and its score is at least 75. If both sides qualify, the higher score must
lead by at least five points; otherwise the result is `NO TRADE`.

### Short-specific controls

- A short setup is not automatically blocked because SPY is strong; market regime
  is context rather than a universal short hard gate.
- Short interest and days-to-cover are used as squeeze-risk modifiers. Elevated
  readings reduce the short score; the model can block a short when the squeeze-risk gate fails.
- The current data adapter does not confirm real-time stock borrow/locate
  availability. Broker availability controls actual execution.
- Short position sizing uses the same dollar-risk cap and maximum notional as the
  long engine: planned risk is `(stop - entry) * shares` for a short.
- Short targets are below entry; invalidation is above entry.

### Trade-plan details

Every selected US setup includes an entry trigger, stop/invalidation, three profit
targets, risk per share, reward/risk to Target 1, position size, notional, planned
maximum loss, thesis, and invalidation rules. The engine does not model option
premiums, theta, or contract liquidity yet; those are separate contract-level
concerns and must be verified before an options trade.

### Market regime

- `PASS`: SPY is above SMA200, SMA50 is above SMA200, and SMA50 is rising over
  the last 20 sessions.
- `CAUTION`: SPY is above SMA200 but intermediate confirmation is mixed.
- `FAIL`: SPY is below SMA200. This blocks new longs, but does **not** automatically
  block shorts.

### Portfolio risk

- New trades in either direction are blocked at a 10% portfolio drawdown.
- New trades are also blocked when existing positions already use the $20
  total-open-risk allowance.
- At a drawdown of 6% or more, the gate changes to `CAUTION`.

## Ghana long-term scoring

| Component | Points |
|---|---:|
| Fundamentals and earnings quality | 35 |
| Valuation | 20 |
| Balance sheet and cash flow | 15 |
| News, governance, and regulation | 15 |
| Liquidity | 10 |
| Technical entry timing | 5 |

`ACCUMULATE GRADUALLY` requires at least 75 points, at least 80% data
completeness, a fundamentals score of at least 24/35, no unresolved governance
or regulatory flags, and minimum liquidity. With less than 70% completeness the
only possible result is `INSUFFICIENT DATA`.

## Known limitations

- yfinance is an unofficial, personal/research-use adapter and is not an
  exchange-grade feed.
- Headline keyword classification is a triage tool, not semantic due diligence.
- Adjusted daily data can differ from the prices and order types shown by a broker.
- Ghana market and financial fields are retrieved from delayed public research
  pages and attributed in the result. GSE/SEC search checks are limited headline
  screens, not a legal or governance opinion. IC Wealth remains the executable quote.
- Backtesting, transaction costs, taxes, FX changes, portfolio correlation, and
  slippage are not yet modeled.
