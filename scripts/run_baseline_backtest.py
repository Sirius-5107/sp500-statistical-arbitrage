"""Run the corrected frozen v1.0 walk-forward portfolio backtest.

Signals are calculated at the close and executed at the next open.  Read
BACKTEST_SPEC.md before changing any rule in this file.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BASELINE_PAIRS_FILE,
    CLOSE_PANEL_FILE,
    ELIGIBILITY_PANEL_FILE,
    OPEN_PANEL_FILE,
)


RESULTS_DIR = PROJECT_ROOT / "data" / "results"
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAX_HOLDING_SESSIONS = 20
MAX_ACTIVE_PAIRS = 5
PAIR_GROSS_FRACTION = 0.20
MAX_PORTFOLIO_GROSS = 1.00
ONE_WAY_COST_RATE = 0.0010
ANNUAL_BORROW_RATE = 0.03
TRADING_DAYS = 252
INITIAL_EQUITY = 1.0


@dataclass
class Position:
    pair_id: str
    ticker_a: str
    ticker_b: str
    sector: str
    selection_date: pd.Timestamp
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    entry_z: float
    fdr_pvalue: float
    direction: int
    shares_a: float
    shares_b: float
    entry_price_a: float
    entry_price_b: float
    entry_gross: float
    entry_cost: float
    sessions_held: int = 0
    borrow_cost: float = 0.0
    mae_z: float = 0.0
    mfe_z: float = 0.0
    last_z: float = np.nan
    pending_exit_reason: str | None = None
    exit_signal_date: pd.Timestamp | None = None

    @property
    def tickers(self) -> set[str]:
        return {self.ticker_a, self.ticker_b}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(BASELINE_PAIRS_FILE)
    open_prices = pd.read_parquet(OPEN_PANEL_FILE)
    close_prices = pd.read_parquet(CLOSE_PANEL_FILE)
    eligibility = pd.read_parquet(ELIGIBILITY_PANEL_FILE).astype(bool)

    pairs["selection_date"] = pd.to_datetime(pairs["selection_date"])
    for panel in (open_prices, close_prices, eligibility):
        panel.index = pd.to_datetime(panel.index)
        panel.sort_index(inplace=True)

    if not open_prices.index.equals(close_prices.index):
        raise ValueError("Open and close panel dates are not aligned")
    if not open_prices.columns.equals(close_prices.columns):
        raise ValueError("Open and close panel tickers are not aligned")
    if not eligibility.index.equals(close_prices.index):
        raise ValueError("Eligibility and close panel dates are not aligned")
    if not eligibility.columns.equals(close_prices.columns):
        raise ValueError("Eligibility and close panel tickers are not aligned")
    if pairs.duplicated(["selection_date", "pair_id"]).any():
        raise ValueError("Duplicate pair snapshots found")
    if "same_issuer" in pairs and pairs["same_issuer"].fillna(False).any():
        raise ValueError("Same-issuer pair found in baseline universe")

    required = {
        "pair_id", "ticker_1", "ticker_2", "dependent_ticker",
        "independent_ticker", "sector", "alpha", "hedge_ratio",
        "spread_mean", "spread_std", "baseline_fdr_pvalue",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Baseline pair columns missing: {sorted(missing)}")

    return pairs, open_prices, close_prices, eligibility


def z_score(pair: pd.Series | Position, date: pd.Timestamp, close: pd.DataFrame) -> float:
    if isinstance(pair, Position):
        dependent = pair.ticker_a
        independent = pair.ticker_b
        alpha = pair.alpha
        beta = pair.beta
        mean = pair.spread_mean
        std = pair.spread_std
    else:
        dependent = str(pair["dependent_ticker"])
        independent = str(pair["independent_ticker"])
        alpha = float(pair["alpha"])
        beta = float(pair["hedge_ratio"])
        mean = float(pair["spread_mean"])
        std = float(pair["spread_std"])

    p_dep = close.at[date, dependent]
    p_ind = close.at[date, independent]
    if not (np.isfinite(p_dep) and np.isfinite(p_ind) and p_dep > 0 and p_ind > 0 and std > 0):
        return np.nan
    spread = np.log(p_dep) - alpha - beta * np.log(p_ind)
    return float((spread - mean) / std)


def valid_price(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def pair_map_for_month(pairs: pd.DataFrame) -> dict[pd.Period, pd.DataFrame]:
    result: dict[pd.Period, pd.DataFrame] = {}
    for selection_date, group in pairs.groupby("selection_date", sort=True):
        result[pd.Timestamp(selection_date).to_period("M") + 1] = group.copy()
    return result


def portfolio_value(cash: float, positions: dict[str, Position], prices: pd.Series) -> float:
    value = cash
    for pos in positions.values():
        pa = prices.get(pos.ticker_a, np.nan)
        pb = prices.get(pos.ticker_b, np.nan)
        if valid_price(pa) and valid_price(pb):
            value += pos.shares_a * pa + pos.shares_b * pb
    return float(value)


def close_position(
    pos: Position,
    date: pd.Timestamp,
    prices: pd.Series,
    cash: float,
    reason: str,
) -> tuple[float, dict, list[dict]]:
    pa = float(prices[pos.ticker_a])
    pb = float(prices[pos.ticker_b])
    sale_value = pos.shares_a * pa + pos.shares_b * pb
    exit_gross = abs(pos.shares_a * pa) + abs(pos.shares_b * pb)
    exit_cost = exit_gross * ONE_WAY_COST_RATE
    cash += sale_value - exit_cost

    entry_net = pos.shares_a * pos.entry_price_a + pos.shares_b * pos.entry_price_b
    gross_pnl = (
        pos.shares_a * (pa - pos.entry_price_a)
        + pos.shares_b * (pb - pos.entry_price_b)
    )
    net_pnl = gross_pnl - pos.entry_cost - exit_cost - pos.borrow_cost
    trade = {
        "pair_id": pos.pair_id,
        "sector": pos.sector,
        "ticker_a": pos.ticker_a,
        "ticker_b": pos.ticker_b,
        "selection_date": pos.selection_date,
        "signal_date": pos.signal_date,
        "entry_date": pos.entry_date,
        "exit_signal_date": pos.exit_signal_date,
        "exit_date": date,
        "entry_price_a": pos.entry_price_a,
        "entry_price_b": pos.entry_price_b,
        "exit_price_a": pa,
        "exit_price_b": pb,
        "alpha": pos.alpha,
        "hedge_ratio": pos.beta,
        "spread_mean": pos.spread_mean,
        "spread_std": pos.spread_std,
        "entry_z_score": pos.entry_z,
        "exit_z_score": pos.last_z,
        "exit_reason": reason,
        "holding_sessions": pos.sessions_held,
        "entry_gross": pos.entry_gross,
        "gross_pnl": gross_pnl,
        "execution_cost": pos.entry_cost + exit_cost,
        "borrow_cost": pos.borrow_cost,
        "net_pnl": net_pnl,
        "gross_return": gross_pnl / pos.entry_gross,
        "net_return": net_pnl / pos.entry_gross,
        "maximum_favourable_excursion_z": pos.mfe_z,
        "maximum_adverse_excursion_z": pos.mae_z,
        "entry_net_cash_flow": entry_net,
    }
    orders = [
        {"date": date, "pair_id": pos.pair_id, "ticker": pos.ticker_a,
         "side": "SELL" if pos.shares_a > 0 else "BUY", "shares": abs(pos.shares_a),
         "price": pa, "reason": reason, "cost": abs(pos.shares_a * pa) * ONE_WAY_COST_RATE},
        {"date": date, "pair_id": pos.pair_id, "ticker": pos.ticker_b,
         "side": "SELL" if pos.shares_b > 0 else "BUY", "shares": abs(pos.shares_b),
         "price": pb, "reason": reason, "cost": abs(pos.shares_b * pb) * ONE_WAY_COST_RATE},
    ]
    return cash, trade, orders


def performance_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | None]:
    returns = equity["net_equity"].pct_change().dropna()
    years = len(returns) / TRADING_DAYS
    total_return = equity["net_equity"].iloc[-1] / equity["net_equity"].iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS) if returns.std(ddof=1) > 0 else np.nan
    downside = returns[returns < 0].std(ddof=1)
    sortino = returns.mean() / downside * np.sqrt(TRADING_DAYS) if pd.notna(downside) and downside > 0 else np.nan
    drawdown = equity["net_equity"] / equity["net_equity"].cummax() - 1
    max_drawdown = float(drawdown.min())
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum() if not trades.empty else 0.0
    losses = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum() if not trades.empty else 0.0
    metrics = {
        "start_date": str(equity["date"].iloc[0].date()),
        "end_date": str(equity["date"].iloc[-1].date()),
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": max_drawdown,
        "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "trade_count": int(len(trades)),
        "win_rate": float((trades["net_pnl"] > 0).mean()) if not trades.empty else np.nan,
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "median_holding_sessions": float(trades["holding_sessions"].median()) if not trades.empty else np.nan,
        "average_gross_exposure": float(equity["gross_exposure"].mean()),
        "maximum_gross_exposure": float(equity["gross_exposure"].max()),
        "average_net_exposure": float(equity["net_exposure"].mean()),
        "fraction_days_invested": float((equity["active_pairs"] > 0).mean()),
        "total_execution_cost": float(trades["execution_cost"].sum()) if not trades.empty else 0.0,
        "total_borrow_cost": float(trades["borrow_cost"].sum()) if not trades.empty else 0.0,
    }
    return {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in metrics.items()}


def run_backtest() -> None:
    pairs, open_prices, close_prices, eligibility = load_inputs()
    month_pairs = pair_map_for_month(pairs)
    dates = close_prices.index
    positions: dict[str, Position] = {}
    cash = INITIAL_EQUITY
    cumulative_borrow = 0.0
    cumulative_execution = 0.0
    orders: list[dict] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    position_rows: list[dict] = []
    exposure_rows: list[dict] = []
    pending_entries: list[dict] = []
    signal_armed: dict[tuple[pd.Period, str], bool] = {}

    first_evaluation = min(month_pairs)
    start_candidates = dates[dates.to_period("M") >= first_evaluation]
    if len(start_candidates) == 0:
        raise RuntimeError("No evaluation dates overlap the panels")
    start_idx = dates.get_loc(start_candidates[0])

    print("=" * 70)
    print("FROZEN BASELINE PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"Dates       : {dates[start_idx].date()} to {dates[-1].date()}")
    print(f"Pair rows   : {len(pairs):,}")
    print(f"Entry/exit : {ENTRY_Z:.1f} / {EXIT_Z:.1f}")
    print(f"Stop/limit : {STOP_Z:.1f} / {MAX_HOLDING_SESSIONS} sessions")
    print()

    for i in range(start_idx, len(dates)):
        date = dates[i]
        open_row = open_prices.loc[date]
        close_row = close_prices.loc[date]
        period = date.to_period("M")
        active_universe = month_pairs.get(period, pd.DataFrame())
        active_ids = set(active_universe["pair_id"]) if not active_universe.empty else set()

        # Exit orders generated at the previous close execute first.
        for pair_id in sorted(list(positions)):
            pos = positions[pair_id]
            reason = pos.pending_exit_reason
            if reason is None and pair_id not in active_ids:
                reason = "qualification_expiry"
                pos.exit_signal_date = dates[i - 1] if i > 0 else date
            if reason is None:
                continue
            pa, pb = open_row[pos.ticker_a], open_row[pos.ticker_b]
            if not (valid_price(pa) and valid_price(pb)):
                continue
            cash, trade, exit_orders = close_position(pos, date, open_row, cash, reason)
            cumulative_execution += trade["execution_cost"] - pos.entry_cost
            trades.append(trade)
            orders.extend(exit_orders)
            del positions[pair_id]

        # Entries generated at the previous close execute after exits.
        used_tickers = set().union(*(p.tickers for p in positions.values())) if positions else set()
        for signal in sorted(pending_entries, key=lambda x: (-abs(x["z"]), x["fdr"], x["pair_id"])):
            if len(positions) >= MAX_ACTIVE_PAIRS:
                break
            if signal["pair_id"] in positions or signal["pair_id"] not in active_ids:
                continue
            if {signal["dep"], signal["ind"]} & used_tickers:
                continue
            pa, pb = open_row[signal["dep"]], open_row[signal["ind"]]
            if not (valid_price(pa) and valid_price(pb)):
                continue
            if not (eligibility.at[date, signal["dep"]] and eligibility.at[date, signal["ind"]]):
                continue
            nav = portfolio_value(cash, positions, open_row)
            existing_gross = sum(
                abs(p.shares_a * open_row[p.ticker_a]) + abs(p.shares_b * open_row[p.ticker_b])
                for p in positions.values()
            )
            target_gross = min(PAIR_GROSS_FRACTION * nav, MAX_PORTFOLIO_GROSS * nav - existing_gross)
            if target_gross <= 0:
                continue
            beta = signal["beta"]
            direction = -1 if signal["z"] > 0 else 1
            wa = direction / (1 + abs(beta))
            wb = -direction * beta / (1 + abs(beta))
            notional_a, notional_b = target_gross * wa, target_gross * wb
            shares_a, shares_b = notional_a / pa, notional_b / pb
            entry_cost = target_gross * ONE_WAY_COST_RATE
            cash -= shares_a * pa + shares_b * pb + entry_cost
            cumulative_execution += entry_cost
            pos = Position(
                pair_id=signal["pair_id"], ticker_a=signal["dep"], ticker_b=signal["ind"],
                sector=signal["sector"], selection_date=signal["selection_date"],
                signal_date=signal["signal_date"], entry_date=date, alpha=signal["alpha"],
                beta=beta, spread_mean=signal["mean"], spread_std=signal["std"],
                entry_z=signal["z"], fdr_pvalue=signal["fdr"], direction=direction,
                shares_a=shares_a, shares_b=shares_b, entry_price_a=float(pa),
                entry_price_b=float(pb), entry_gross=target_gross, entry_cost=entry_cost,
                last_z=signal["z"],
            )
            positions[pos.pair_id] = pos
            used_tickers |= pos.tickers
            for ticker, shares, price in [(pos.ticker_a, shares_a, pa), (pos.ticker_b, shares_b, pb)]:
                orders.append({"date": date, "pair_id": pos.pair_id, "ticker": ticker,
                               "side": "BUY" if shares > 0 else "SELL", "shares": abs(shares),
                               "price": float(price), "reason": "entry",
                               "cost": abs(shares * price) * ONE_WAY_COST_RATE})
        pending_entries = []

        # Accrue one trading day's borrow and evaluate exits at the close.
        for pos in positions.values():
            short_value = max(-pos.shares_a * close_row[pos.ticker_a], 0) + max(-pos.shares_b * close_row[pos.ticker_b], 0)
            borrow = short_value * ANNUAL_BORROW_RATE / TRADING_DAYS
            cash -= borrow
            pos.borrow_cost += borrow
            cumulative_borrow += borrow
            pos.sessions_held += 1
            current_z = z_score(pos, date, close_prices)
            pos.last_z = current_z
            if np.isfinite(current_z):
                convergence = -np.sign(pos.entry_z) * (current_z - pos.entry_z)
                pos.mfe_z = max(pos.mfe_z, float(convergence))
                pos.mae_z = min(pos.mae_z, float(convergence))
                if abs(current_z) <= EXIT_Z:
                    pos.pending_exit_reason = "mean_reversion"
                elif abs(current_z) >= STOP_Z:
                    pos.pending_exit_reason = "stop_loss"
                elif pos.sessions_held >= MAX_HOLDING_SESSIONS:
                    pos.pending_exit_reason = "maximum_holding_period"
            else:
                pos.pending_exit_reason = "data_failure"
            if pos.pending_exit_reason:
                pos.exit_signal_date = date

        # Generate tomorrow's entries using today's close.  On a month-end,
        # the newly selected universe is already known at that close and may
        # therefore execute at the first open of the next month.
        occupied = set().union(*(p.tickers for p in positions.values())) if positions else set()
        if i < len(dates) - 1:
            next_period = dates[i + 1].to_period("M")
            signal_universe = month_pairs.get(next_period, pd.DataFrame())
        else:
            next_period = period
            signal_universe = pd.DataFrame()

        if not signal_universe.empty:
            for _, pair in signal_universe.iterrows():
                pair_id = str(pair["pair_id"])
                dep, ind = str(pair["dependent_ticker"]), str(pair["independent_ticker"])
                state_key = (next_period, pair_id)
                current_z = z_score(pair, date, close_prices)

                if not np.isfinite(current_z):
                    continue

                # A pair becomes eligible for another first-touch signal only
                # after returning inside the entry boundary.  This prevents
                # repeated entries while a spread remains structurally broken.
                if abs(current_z) < ENTRY_Z:
                    signal_armed[state_key] = True

                if pair_id in positions or {dep, ind} & occupied:
                    continue
                if not (eligibility.at[date, dep] and eligibility.at[date, ind]):
                    continue

                is_armed = signal_armed.setdefault(state_key, True)
                if is_armed and ENTRY_Z <= abs(current_z) < STOP_Z:
                    pending_entries.append({
                        "pair_id": pair_id, "dep": dep, "ind": ind, "sector": pair["sector"],
                        "selection_date": pair["selection_date"], "signal_date": date,
                        "z": current_z, "fdr": float(pair["baseline_fdr_pvalue"]),
                        "alpha": float(pair["alpha"]), "beta": float(pair["hedge_ratio"]),
                        "mean": float(pair["spread_mean"]), "std": float(pair["spread_std"]),
                    })
                    signal_armed[state_key] = False

        nav = portfolio_value(cash, positions, close_row)
        gross = sum(abs(p.shares_a * close_row[p.ticker_a]) + abs(p.shares_b * close_row[p.ticker_b]) for p in positions.values())
        net = sum(p.shares_a * close_row[p.ticker_a] + p.shares_b * close_row[p.ticker_b] for p in positions.values())
        gross_fraction = gross / nav if nav > 0 else np.nan
        net_fraction = net / nav if nav > 0 else np.nan
        if np.isfinite(gross_fraction) and gross_fraction > MAX_PORTFOLIO_GROSS + 0.10:
            # Price moves can take ex-post exposure above 100%; only entry-time exposure is constrained.
            pass
        equity_rows.append({"date": date, "net_equity": nav, "cash": cash,
                            "gross_exposure": gross_fraction, "net_exposure": net_fraction,
                            "active_pairs": len(positions), "cumulative_execution_cost": cumulative_execution,
                            "cumulative_borrow_cost": cumulative_borrow})
        exposure_rows.append({"date": date, "gross_market_value": gross, "net_market_value": net,
                              "gross_exposure": gross_fraction, "net_exposure": net_fraction,
                              "active_pairs": len(positions)})
        for pos in positions.values():
            position_rows.append({"date": date, "pair_id": pos.pair_id, "ticker_a": pos.ticker_a,
                                  "ticker_b": pos.ticker_b, "shares_a": pos.shares_a,
                                  "shares_b": pos.shares_b, "z_score": pos.last_z,
                                  "sessions_held": pos.sessions_held})

        if (i - start_idx + 1) % 500 == 0:
            print(f"Processed {i - start_idx + 1:,}/{len(dates) - start_idx:,} sessions")

    # Mark remaining positions explicitly at the dataset boundary; do not invent an unavailable next open.
    for pos in positions.values():
        pos.pending_exit_reason = "open_at_dataset_end"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    orders_df = pd.DataFrame(orders)
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    positions_df = pd.DataFrame(position_rows)
    exposure_df = pd.DataFrame(exposure_rows)
    orders_df.to_csv(RESULTS_DIR / "orders.csv", index=False)
    trades_df.to_csv(RESULTS_DIR / "trades.csv", index=False)
    positions_df.to_csv(RESULTS_DIR / "daily_positions.csv", index=False)
    exposure_df.to_csv(RESULTS_DIR / "daily_exposure.csv", index=False)
    equity_df.to_csv(RESULTS_DIR / "equity_curve.csv", index=False)
    metrics = performance_metrics(equity_df, trades_df)
    metrics["positions_open_at_dataset_end"] = len(positions)
    with (RESULTS_DIR / "performance_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print()
    print("=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    for key in ["trade_count", "total_return", "annualized_return", "sharpe_ratio",
                "maximum_drawdown", "win_rate", "profit_factor", "fraction_days_invested"]:
        print(f"{key:28s}: {metrics[key]}")
    print(f"{'open_at_dataset_end':28s}: {len(positions)}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    run_backtest()