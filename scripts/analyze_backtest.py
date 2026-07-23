"""Generate reproducible diagnostics for the frozen baseline backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

INITIAL_EQUITY = 1.0
TRADING_DAYS = 252

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#2563EB"
BLUE_LIGHT = "#BFDBFE"
ORANGE = "#EA580C"
ORANGE_LIGHT = "#FED7AA"
GOLD = "#CA8A04"
BACKGROUND = "#FFFFFF"


def configure_style() -> None:
    """Apply one consistent, publication-friendly chart style."""

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run scripts/run_baseline_backtest.py first."
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Load and normalize all baseline backtest artifacts."""

    files = {
        "trades": RESULTS_DIR / "trades.csv",
        "equity": RESULTS_DIR / "equity_curve.csv",
        "exposure": RESULTS_DIR / "daily_exposure.csv",
        "positions": RESULTS_DIR / "daily_positions.csv",
        "metrics": RESULTS_DIR / "performance_metrics.json",
    }

    for path in files.values():
        require_file(path)

    trades = pd.read_csv(
        files["trades"],
        parse_dates=[
            "selection_date",
            "signal_date",
            "entry_date",
            "exit_signal_date",
            "exit_date",
        ],
    )
    equity = pd.read_csv(
        files["equity"],
        parse_dates=["date"],
    )
    exposure = pd.read_csv(
        files["exposure"],
        parse_dates=["date"],
    )
    positions = pd.read_csv(
        files["positions"],
        parse_dates=["date"],
    )

    with files["metrics"].open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    trades.sort_values(["exit_date", "pair_id"], inplace=True)
    equity.sort_values("date", inplace=True)
    exposure.sort_values("date", inplace=True)
    positions.sort_values(["date", "pair_id"], inplace=True)

    return trades, equity, exposure, positions, metrics


def validate_results(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Run compact integrity tests and return an auditable QA table."""

    final_equity_change = (
        float(equity["net_equity"].iloc[-1])
        - INITIAL_EQUITY
    )
    trade_net_pnl = float(trades["net_pnl"].sum())
    reconciliation_error = (
        trade_net_pnl - final_equity_change
    )

    stacked_tickers = positions.melt(
        id_vars=["date", "pair_id"],
        value_vars=["ticker_a", "ticker_b"],
        value_name="ticker",
    )

    checks = [
        {
            "check": "Trade P&L reconciles to final equity",
            "value": reconciliation_error,
            "status": (
                "PASS"
                if abs(reconciliation_error) <= 1e-10
                else "FAIL"
            ),
        },
        {
            "check": "No entry occurs on/before signal date",
            "value": int(
                (trades["entry_date"]
                 <= trades["signal_date"]).sum()
            ),
            "status": (
                "PASS"
                if not (
                    trades["entry_date"]
                    <= trades["signal_date"]
                ).any()
                else "FAIL"
            ),
        },
        {
            "check": "No exit occurs on/before exit signal",
            "value": int(
                (trades["exit_date"]
                 <= trades["exit_signal_date"]).sum()
            ),
            "status": (
                "PASS"
                if not (
                    trades["exit_date"]
                    <= trades["exit_signal_date"]
                ).any()
                else "FAIL"
            ),
        },
        {
            "check": "No entry starts beyond stop boundary",
            "value": int(
                (trades["entry_z_score"].abs() >= 4).sum()
            ),
            "status": (
                "PASS"
                if not (
                    trades["entry_z_score"].abs() >= 4
                ).any()
                else "FAIL"
            ),
        },
        {
            "check": "No security appears in concurrent pairs",
            "value": int(
                stacked_tickers.duplicated(
                    ["date", "ticker"]
                ).sum()
            ),
            "status": (
                "PASS"
                if not stacked_tickers.duplicated(
                    ["date", "ticker"]
                ).any()
                else "FAIL"
            ),
        },
        {
            "check": "Maximum five active pairs",
            "value": int(
                equity["active_pairs"].max()
            ),
            "status": (
                "PASS"
                if equity["active_pairs"].max() <= 5
                else "FAIL"
            ),
        },
    ]

    validation = pd.DataFrame(checks)
    failures = validation.loc[
        validation["status"].eq("FAIL")
    ]

    if not failures.empty:
        raise RuntimeError(
            "Backtest validation failed:\n"
            + failures.to_string(index=False)
        )

    return validation


def calculate_tables(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: dict,
) -> dict[str, pd.DataFrame]:
    """Create report-ready summary tables."""

    gross_pnl = float(trades["gross_pnl"].sum())
    execution_cost = float(
        trades["execution_cost"].sum()
    )
    borrow_cost = float(
        trades["borrow_cost"].sum()
    )
    net_pnl = float(trades["net_pnl"].sum())

    headline = pd.DataFrame(
        [
            {
                "metric": "Gross trading P&L",
                "value": gross_pnl,
            },
            {
                "metric": "Execution costs",
                "value": -execution_cost,
            },
            {
                "metric": "Short-borrow costs",
                "value": -borrow_cost,
            },
            {
                "metric": "Net P&L",
                "value": net_pnl,
            },
            {
                "metric": "Net total return",
                "value": float(metrics["total_return"]),
            },
            {
                "metric": "Annualized return",
                "value": float(
                    metrics["annualized_return"]
                ),
            },
            {
                "metric": "Sharpe ratio",
                "value": float(metrics["sharpe_ratio"]),
            },
            {
                "metric": "Maximum drawdown",
                "value": float(
                    metrics["maximum_drawdown"]
                ),
            },
            {
                "metric": "Trades",
                "value": int(metrics["trade_count"]),
            },
            {
                "metric": "Win rate",
                "value": float(metrics["win_rate"]),
            },
            {
                "metric": "Profit factor",
                "value": float(
                    metrics["profit_factor"]
                ),
            },
            {
                "metric": "Days invested",
                "value": float(
                    metrics["fraction_days_invested"]
                ),
            },
        ]
    )

    exit_summary = (
        trades.groupby("exit_reason")
        .agg(
            trades=("pair_id", "size"),
            gross_pnl=("gross_pnl", "sum"),
            execution_cost=("execution_cost", "sum"),
            borrow_cost=("borrow_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_rate=(
                "net_pnl",
                lambda values: (values > 0).mean(),
            ),
            median_net_return=("net_return", "median"),
            median_holding_sessions=(
                "holding_sessions",
                "median",
            ),
        )
        .sort_values("net_pnl", ascending=False)
        .reset_index()
    )

    sector_summary = (
        trades.groupby("sector")
        .agg(
            trades=("pair_id", "size"),
            gross_pnl=("gross_pnl", "sum"),
            execution_cost=("execution_cost", "sum"),
            borrow_cost=("borrow_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_rate=(
                "net_pnl",
                lambda values: (values > 0).mean(),
            ),
            median_net_return=("net_return", "median"),
        )
        .sort_values("net_pnl", ascending=False)
        .reset_index()
    )

    daily_return = (
        equity["net_equity"]
        .pct_change()
        .fillna(
            equity["net_equity"].iloc[0]
            / INITIAL_EQUITY
            - 1
        )
    )
    yearly = pd.DataFrame(
        {
            "year": equity["date"].dt.year,
            "daily_return": daily_return,
        }
    )
    yearly_returns = (
        yearly.groupby("year")["daily_return"]
        .apply(lambda values: (1 + values).prod() - 1)
        .rename("net_return")
        .reset_index()
    )

    trade_columns = [
        "pair_id",
        "sector",
        "entry_date",
        "exit_date",
        "exit_reason",
        "entry_z_score",
        "exit_z_score",
        "holding_sessions",
        "gross_pnl",
        "execution_cost",
        "borrow_cost",
        "net_pnl",
        "net_return",
    ]
    best_trades = (
        trades.nlargest(10, "net_pnl")[trade_columns]
        .reset_index(drop=True)
    )
    worst_trades = (
        trades.nsmallest(10, "net_pnl")[trade_columns]
        .reset_index(drop=True)
    )

    return {
        "headline_metrics": headline,
        "exit_reason_summary": exit_summary,
        "sector_summary": sector_summary,
        "yearly_returns": yearly_returns,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
    }


def add_subtitle(
    figure: plt.Figure,
    text: str,
) -> None:
    figure.text(
        0.08,
        0.925,
        text,
        color=MUTED,
        fontsize=9,
        ha="left",
    )


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    path = FIGURES_DIR / filename
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_equity_curve(equity: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        equity["date"],
        equity["net_equity"],
        color=BLUE,
        linewidth=2,
        label="Net equity",
    )
    axis.axhline(
        INITIAL_EQUITY,
        color=INK,
        linewidth=1,
        linestyle="--",
        label="Initial capital",
    )
    axis.fill_between(
        equity["date"],
        equity["net_equity"],
        INITIAL_EQUITY,
        color=BLUE_LIGHT,
        alpha=0.35,
    )
    axis.set_title(
        "Baseline strategy net equity"
    )
    axis.set_ylabel("Portfolio value (initial = 1.0)")
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="best")
    add_subtitle(
        figure,
        "Daily mark-to-market equity after execution and "
        "short-borrow costs, 2001–2026",
    )
    save_figure(figure, "01_net_equity_curve.png")


def plot_drawdown(equity: pd.DataFrame) -> None:
    drawdown = (
        equity["net_equity"]
        / equity["net_equity"].cummax()
        - 1
    )
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.fill_between(
        equity["date"],
        drawdown,
        0,
        color=ORANGE_LIGHT,
        alpha=0.9,
    )
    axis.plot(
        equity["date"],
        drawdown,
        color=ORANGE,
        linewidth=1.4,
    )
    axis.set_title("Baseline strategy drawdown")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.grid(axis="y")
    add_subtitle(
        figure,
        "Decline from the running peak of net equity",
    )
    save_figure(figure, "02_drawdown.png")


def plot_pnl_attribution(
    trades: pd.DataFrame,
) -> None:
    values = pd.Series(
        {
            "Gross trading\nP&L": trades["gross_pnl"].sum(),
            "Execution\ncosts": -trades[
                "execution_cost"
            ].sum(),
            "Borrow\ncosts": -trades[
                "borrow_cost"
            ].sum(),
            "Net\nP&L": trades["net_pnl"].sum(),
        }
    )
    colors = [
        BLUE if value >= 0 else ORANGE
        for value in values
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(
        values.index,
        values.values,
        color=colors,
        edgecolor=INK,
        linewidth=0.6,
    )
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_title("Baseline P&L attribution")
    axis.set_ylabel("Portfolio-value contribution")
    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.grid(axis="y")
    for bar, value in zip(bars, values):
        offset = 0.002 if value >= 0 else -0.002
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontweight="bold",
        )
    add_subtitle(
        figure,
        "Gross realized trade P&L less recorded execution "
        "and short-borrow costs",
    )
    save_figure(figure, "03_pnl_attribution.png")


def plot_exit_diagnostics(
    exit_summary: pd.DataFrame,
) -> None:
    data = exit_summary.sort_values(
        "net_pnl",
        ascending=True,
    )
    labels = (
        data["exit_reason"]
        .str.replace("_", " ")
        .str.title()
    )
    colors = [
        BLUE if value >= 0 else ORANGE
        for value in data["net_pnl"]
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    axes[0].barh(
        labels,
        data["trades"],
        color=BLUE_LIGHT,
        edgecolor=BLUE,
    )
    axes[0].set_title("Trades by exit reason")
    axes[0].set_xlabel("Completed trades")
    axes[0].grid(axis="x")

    axes[1].barh(
        labels,
        data["net_pnl"],
        color=colors,
        edgecolor=INK,
        linewidth=0.5,
    )
    axes[1].axvline(0, color=INK, linewidth=1)
    axes[1].set_title("Net P&L by exit reason")
    axes[1].set_xlabel("Portfolio-value contribution")
    axes[1].xaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axes[1].grid(axis="x")

    figure.suptitle(
        "Exit-reason diagnostics",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.92,
        "Counts and aggregate net contribution across "
        "165 completed trades",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.88])
    save_figure(figure, "04_exit_diagnostics.png")


def plot_yearly_returns(
    yearly_returns: pd.DataFrame,
) -> None:
    colors = [
        BLUE if value >= 0 else ORANGE
        for value in yearly_returns["net_return"]
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(
        yearly_returns["year"].astype(str),
        yearly_returns["net_return"],
        color=colors,
        width=0.8,
    )
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_title("Baseline strategy calendar-year returns")
    axis.set_ylabel("Net return")
    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.tick_params(axis="x", rotation=60)
    axis.grid(axis="y")
    add_subtitle(
        figure,
        "Returns include inactive periods and all modeled costs",
    )
    save_figure(figure, "05_yearly_returns.png")


def plot_trade_distribution(
    trades: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.hist(
        trades["net_return"],
        bins=24,
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=0.8,
    )
    axis.axvline(
        0,
        color=INK,
        linewidth=1,
    )
    axis.axvline(
        trades["net_return"].median(),
        color=ORANGE,
        linewidth=1.8,
        linestyle="--",
        label=(
            "Median "
            f"{trades['net_return'].median():+.2%}"
        ),
    )
    axis.set_title("Distribution of completed-trade returns")
    axis.set_xlabel("Net return on allocated pair gross")
    axis.set_ylabel("Trades")
    axis.xaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.grid(axis="y")
    axis.legend(frameon=False)
    add_subtitle(
        figure,
        f"{len(trades)} completed trades; returns include "
        "execution and borrow costs",
    )
    save_figure(figure, "06_trade_return_distribution.png")


def plot_sector_contribution(
    sector_summary: pd.DataFrame,
) -> None:
    data = sector_summary.sort_values(
        "net_pnl",
        ascending=True,
    )
    colors = [
        BLUE if value >= 0 else ORANGE
        for value in data["net_pnl"]
    ]
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.barh(
        data["sector"],
        data["net_pnl"],
        color=colors,
        edgecolor=INK,
        linewidth=0.5,
    )
    axis.axvline(0, color=INK, linewidth=1)
    axis.set_title("Net P&L contribution by sector")
    axis.set_xlabel("Portfolio-value contribution")
    axis.xaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.grid(axis="x")
    add_subtitle(
        figure,
        "Aggregate realized net P&L; sector sample sizes "
        "are reported separately",
    )
    save_figure(figure, "07_sector_contribution.png")


def plot_entry_severity(
    trades: pd.DataFrame,
) -> None:
    marker_map = {
        "mean_reversion": "o",
        "stop_loss": "X",
        "maximum_holding_period": "^",
        "qualification_expiry": "s",
    }
    color_map = {
        "mean_reversion": BLUE,
        "stop_loss": ORANGE,
        "maximum_holding_period": GOLD,
        "qualification_expiry": MUTED,
    }
    figure, axis = plt.subplots(figsize=(9.5, 6))
    for reason, group in trades.groupby("exit_reason"):
        axis.scatter(
            group["entry_z_score"].abs(),
            group["net_return"],
            s=42,
            alpha=0.75,
            marker=marker_map.get(reason, "o"),
            color=color_map.get(reason, MUTED),
            label=reason.replace("_", " ").title(),
            edgecolor=BACKGROUND,
            linewidth=0.5,
        )
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_title(
        "Entry z-score magnitude and trade outcome"
    )
    axis.set_xlabel("Absolute entry z-score")
    axis.set_ylabel("Net trade return")
    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.grid()
    axis.legend(
        frameon=False,
        ncol=2,
        loc="best",
    )
    add_subtitle(
        figure,
        "One observation per completed trade, grouped by "
        "realized exit reason",
    )
    save_figure(figure, "08_entry_z_vs_return.png")


def plot_holding_period(
    trades: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 6))
    axis.scatter(
        trades["holding_sessions"],
        trades["net_return"],
        s=40,
        color=BLUE,
        alpha=0.65,
        edgecolor=BACKGROUND,
        linewidth=0.5,
    )
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_title(
        "Holding period and completed-trade return"
    )
    axis.set_xlabel("Holding period (trading sessions)")
    axis.set_ylabel("Net trade return")
    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1)
    )
    axis.set_xticks(
        sorted(
            set(
                [1, 5, 10, 15, 20]
                + trades["holding_sessions"]
                .dropna()
                .astype(int)
                .tolist()
            )
        )
    )
    axis.grid()
    add_subtitle(
        figure,
        "One observation per completed trade; maximum "
        "holding period is 20 sessions",
    )
    save_figure(figure, "09_holding_period_vs_return.png")


def save_tables(
    tables: dict[str, pd.DataFrame],
    validation: pd.DataFrame,
) -> None:
    for name, table in tables.items():
        table.to_csv(
            TABLES_DIR / f"{name}.csv",
            index=False,
        )
    validation.to_csv(
        TABLES_DIR / "validation_checks.csv",
        index=False,
    )


def main() -> None:
    configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    (
        trades,
        equity,
        _exposure,
        positions,
        metrics,
    ) = load_inputs()

    print("=" * 70)
    print("BASELINE BACKTEST DIAGNOSTICS")
    print("=" * 70)
    print(f"Trades      : {len(trades):,}")
    print(
        f"Date range  : "
        f"{equity['date'].min().date()} to "
        f"{equity['date'].max().date()}"
    )
    print()

    validation = validate_results(
        trades,
        equity,
        positions,
    )
    tables = calculate_tables(
        trades,
        equity,
        metrics,
    )
    save_tables(tables, validation)

    plot_equity_curve(equity)
    plot_drawdown(equity)
    plot_pnl_attribution(trades)
    plot_exit_diagnostics(
        tables["exit_reason_summary"]
    )
    plot_yearly_returns(
        tables["yearly_returns"]
    )
    plot_trade_distribution(trades)
    plot_sector_contribution(
        tables["sector_summary"]
    )
    plot_entry_severity(trades)
    plot_holding_period(trades)

    print("Validation checks:")
    print(validation.to_string(index=False))
    print()
    print("Generated:")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Tables  : {TABLES_DIR}")
    print("Figure count: 9")
    print(
        "Next step: inspect the figures before committing."
    )


if __name__ == "__main__":
    main()