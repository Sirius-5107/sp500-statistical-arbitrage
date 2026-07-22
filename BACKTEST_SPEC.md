# Backtest Specification

## Status

**Frozen baseline specification â€” version 1.0**

This document defines the baseline portfolio backtest before its results are examined. Any later change must be recorded as a separate experiment and must not replace the baseline result.

## Objective

Test whether S&P 500 pairs selected using rolling cointegration produce economically meaningful out-of-sample mean reversion after realistic execution costs and portfolio constraints.

This is a research backtest, not a claim of deployable profitability.

## Known Limitation

The security universe is based on current S&P 500 constituents rather than point-in-time historical membership. Results therefore contain survivorship bias. This limitation must be disclosed in every report and prevents the study from being interpreted as a fully investable historical simulation.

## Data

- Source: Yahoo Finance via `yfinance`
- Frequency: daily
- Price fields: adjusted open and adjusted close
- Research period: 2000 onward, subject to data availability
- Signal time: market close
- Execution time: next available market open
- No execution is allowed on the same close that generated a signal
- Securities must pass the existing history, price, volume, liquidity, and data-quality eligibility rules

## Pair Selection

Pair selection occurs at each month-end using information available through that date only.

1. Restrict securities to the same GICS sector.
2. Use 252 daily log-return observations as the formation window.
3. Require at least 200 overlapping return observations.
4. Require return correlation of at least 0.50.
5. Retain at most 10 correlation partners per security.
6. Rank the remaining candidates within each sector and retain at most 20 pairs per sector.
7. Exclude securities belonging to the same issuer, identified using CIK.
8. Run Engle-Granger tests in both regression directions on log prices.
9. Use the larger of the two cointegration p-values as the conservative pair p-value.
10. Apply Benjamini-Hochberg false-discovery-rate control at 10% across the month's tested pairs.
11. Require a positive hedge ratio between 0.10 and 10.0.
12. Require an estimated spread half-life between 2 and 60 trading sessions.

A pair can be traded during the following month only if it qualifies at the preceding month-end.

## Spread and Z-score

For the selected regression direction:

```text
spread_t = log(price_A,t) - alpha - beta * log(price_B,t)
z_t = (spread_t - formation_mean) / formation_std
```

The intercept, hedge ratio, spread mean, and spread standard deviation are estimated using the formation window and remain fixed during the following trading month.

## Entry Rules

- Enter when the close-based z-score first reaches `|z| >= 2.0`.
- Execute both legs at the next available open.
- If `z >= 2.0`, short the spread: short A and long beta-adjusted B.
- If `z <= -2.0`, long the spread: long A and short beta-adjusted B.
- Do not average down or add to an open pair.
- Skip an entry if either opening price is unavailable or non-positive.
- Skip an entry if it violates a portfolio constraint.

## Exit Rules

An open position exits at the next available open after the first close satisfying any condition below:

1. **Mean reversion:** `|z| <= 0.5`
2. **Stop loss:** `|z| >= 4.0`
3. **Maximum holding period:** 20 trading sessions
4. **Qualification expiry:** the pair is absent from the newly selected monthly universe
5. **Data failure:** a required price becomes unavailable

If several conditions occur together, use the precedence order shown above for the recorded exit reason.

## Position Sizing

For a unit pair position:

```text
weight_A = 1 / (1 + |beta|)
weight_B = -beta / (1 + |beta|)
```

Reverse both signs for a short-spread position.

Portfolio constraints:

- Equal gross capital allocation across active pairs
- Maximum 20% portfolio gross exposure per pair
- Maximum 5 simultaneous pairs
- Maximum one active pair containing any given security
- Maximum 100% total gross exposure
- Unallocated capital remains cash
- No leverage beyond 100% gross exposure
- Pair weights are fixed from entry until exit; no daily re-hedging in the baseline

When more valid signals exist than capacity allows, rank them by descending absolute entry z-score, then ascending FDR-adjusted p-value, then pair ID for deterministic tie-breaking.

## Transaction Costs

Baseline costs apply to each leg whenever it is traded:

- Commission and market-impact proxy: 5 basis points
- Slippage: 5 basis points
- Total one-way execution cost: 10 basis points per leg
- Short borrow charge: 3% annualized, accrued daily on short market value using 252 trading days

The strategy must report results before and after costs.

## Accounting Assumptions

- Initial portfolio value: 1.0
- Cash return: 0%
- Daily mark-to-market uses adjusted close prices
- Entry and exit trades use adjusted open prices
- Corporate-action handling inherits Yahoo's adjusted price series
- Positions cannot trade through missing or non-positive execution prices
- No dividends are added separately because adjusted prices are used

## Benchmarks

- Primary benchmark: cash at 0%, because the strategy is intended to be market-neutral
- Context benchmark: SPY buy-and-hold over the same evaluable period
- Report the strategy's beta and correlation to SPY; do not present SPY as a like-for-like objective

## Required Outputs

Write machine-readable outputs under `data/results/`, which remains excluded from Git:

- `orders.csv`
- `trades.csv`
- `daily_positions.csv`
- `daily_exposure.csv`
- `equity_curve.csv`
- `performance_metrics.json`

Commit only compact, reproducible report tables and figures intended for the repository.

Each completed trade must record pair identifiers, dates, prices, formation statistics, z-scores, exit reason, exposure, gross and net returns, costs, holding period, and favourable/adverse excursion.

## Required Performance Metrics

- Total and annualized return
- Annualized volatility
- Sharpe and Sortino ratios
- Maximum drawdown and Calmar ratio
- SPY beta and correlation
- Average and maximum exposure
- Turnover and fraction of days invested
- Trade count, win rate, profit factor, and median holding period
- Return by sector and calendar year
- Exit-reason distribution
- Long-leg, short-leg, execution-cost, and borrow-cost contribution
- Best, worst, and tail-loss trades

## Predeclared Robustness Checks

These are sensitivity analyses, not parameters to optimize:

| Component | Values |
|---|---|
| Entry z-score | 1.5, 2.0, 2.5 |
| Exit z-score | 0.0, 0.5, 1.0 |
| Stop z-score | 3.5, 4.0, 5.0 |
| Formation window | 126, 252, 504 sessions |
| Costs | low, baseline, stressed |
| Subperiods | pre-2008, 2008â€“2012, 2013â€“2019, 2020 onward |
| Stability | remove best 5 and worst 5 trades |
| Segments | sector-level results |

The baseline remains entry 2.0, exit 0.5, stop 4.0, formation 252, and baseline costs regardless of which sensitivity cell performs best.

## Research Integrity Rules

- All formation estimates must use only observations available on or before the selection date.
- All close-generated signals execute no earlier than the next open.
- Failed or rejected trades remain in diagnostic logs.
- Baseline results must never be silently overwritten by a revised method.
- Any post-result methodological change must receive a new specification version and be labeled exploratory.
- Negative or insignificant performance is a valid research result.
- Machine learning is outside baseline version 1.0 and may be added only after this backtest and its diagnostics are complete.

## Acceptance Tests Before Results Are Trusted

- A formation window cannot contain a future observation.
- A signal cannot execute on its signal close.
- Same-issuer pairs cannot enter the candidate universe.
- Pair gross weights sum to one within numerical tolerance.
- A security cannot appear in two simultaneous positions.
- Portfolio gross exposure cannot exceed 100%.
- Transaction and borrow costs reduce, never increase, portfolio value.
- Every position has a deterministic exit or remains explicitly open at the dataset boundary.
- The equity curve reconciles exactly with positions, trades, and costs.
- Re-running with identical inputs produces identical outputs.