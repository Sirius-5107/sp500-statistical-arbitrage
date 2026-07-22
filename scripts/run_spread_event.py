import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASELINE_PAIRS_FILE,
    CLOSE_PANEL_FILE,
    ELIGIBILITY_PANEL_FILE,
    ENTRY_Z_SCORE,
    PARTIAL_EXIT_Z_SCORE,
    SPREAD_EVENT_FILE,
    SPREAD_EVENT_SECTOR_FILE,
    SPREAD_EVENT_SUMMARY_FILE,
    STANDARD_EXIT_Z_SCORE,
)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load baseline pairs and aligned market panels."""

    pairs = pd.read_parquet(
        BASELINE_PAIRS_FILE
    )

    close = pd.read_parquet(
        CLOSE_PANEL_FILE
    )

    eligibility = pd.read_parquet(
        ELIGIBILITY_PANEL_FILE
    )

    pairs["selection_date"] = pd.to_datetime(
        pairs["selection_date"]
    )

    close.index = pd.to_datetime(close.index)
    eligibility.index = pd.to_datetime(
        eligibility.index
    )

    if not close.index.equals(eligibility.index):
        raise ValueError(
            "Close and eligibility dates are not aligned"
        )

    if not close.columns.equals(eligibility.columns):
        raise ValueError(
            "Close and eligibility tickers are not aligned"
        )

    return pairs, close, eligibility


def get_evaluation_dates(
    selection_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Return the next calendar month's available trading dates.

    The partial final month is excluded.
    """

    selection_period = selection_date.to_period("M")
    evaluation_period = selection_period + 1

    latest_period = trading_dates.max().to_period("M")

    # If the evaluation month is the final month in the
    # dataset, it may still be incomplete.
    if evaluation_period >= latest_period:
        return pd.DatetimeIndex([])

    mask = (
        trading_dates.to_period("M")
        == evaluation_period
    )

    return trading_dates[mask]


def calculate_z_scores(
    pair: pd.Series,
    dates: pd.DatetimeIndex,
    close: pd.DataFrame,
) -> pd.Series:
    """Calculate out-of-sample z-scores using frozen parameters."""

    dependent = pair["dependent_ticker"]
    independent = pair["independent_ticker"]

    alpha = float(pair["alpha"])
    beta = float(pair["hedge_ratio"])
    spread_mean = float(pair["spread_mean"])
    spread_std = float(pair["spread_std"])

    dependent_prices = close.loc[
        dates,
        dependent,
    ]

    independent_prices = close.loc[
        dates,
        independent,
    ]

    valid_prices = (
        dependent_prices.gt(0)
        & independent_prices.gt(0)
    )

    spread = pd.Series(
        np.nan,
        index=dates,
        dtype=float,
    )

    spread.loc[valid_prices] = (
        np.log(
            dependent_prices.loc[valid_prices]
        )
        - alpha
        - beta
        * np.log(
            independent_prices.loc[valid_prices]
        )
    )

    z_score = (
        spread - spread_mean
    ) / spread_std

    z_score.name = "z_score"

    return z_score


def first_true_position(
    mask: np.ndarray,
) -> int | None:
    """Return the first true array position."""

    positions = np.flatnonzero(mask)

    if len(positions) == 0:
        return None

    return int(positions[0])


def sessions_to_threshold(
    z_path: np.ndarray,
    threshold: float,
) -> float:
    """Find sessions required to enter an absolute-z region."""

    position = first_true_position(
        np.abs(z_path) <= threshold
    )

    if position is None:
        return np.nan

    # Position zero is the first session after the signal.
    return float(position + 1)


def sessions_to_mean_crossing(
    z_path: np.ndarray,
    entry_sign: float,
) -> float:
    """Find sessions until the spread crosses its fitted mean."""

    position = first_true_position(
        z_path * entry_sign <= 0
    )

    if position is None:
        return np.nan

    return float(position + 1)


def analyze_pair_month(
    pair: pd.Series,
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> dict | None:
    """Analyze the first divergence event for one pair-month."""

    selection_date = pd.Timestamp(
        pair["selection_date"]
    )

    evaluation_dates = get_evaluation_dates(
        selection_date,
        close.index,
    )

    if len(evaluation_dates) == 0:
        return None

    # The selection-date close can generate an order for
    # the first session of the following month.
    signal_dates = pd.DatetimeIndex(
        [selection_date]
    ).append(evaluation_dates)

    z_scores = calculate_z_scores(
        pair,
        signal_dates,
        close,
    )

    dependent = pair["dependent_ticker"]
    independent = pair["independent_ticker"]

    pair_eligible = (
        eligibility.loc[
            signal_dates,
            dependent,
        ]
        & eligibility.loc[
            signal_dates,
            independent,
        ]
    )

    # A signal on the final evaluation date cannot be acted
    # on within this event-study horizon.
    actionable_dates = signal_dates[:-1]

    actionable_signal = (
        z_scores.loc[actionable_dates].abs()
        >= ENTRY_Z_SCORE
    )

    actionable_signal &= pair_eligible.loc[
        actionable_dates
    ]

    signal_positions = np.flatnonzero(
        actionable_signal.to_numpy()
    )

    base_record = {
        "selection_date": selection_date,
        "evaluation_month": (
            evaluation_dates[0].to_period("M").to_timestamp()
        ),
        "pair_id": pair["pair_id"],
        "ticker_1": pair["ticker_1"],
        "ticker_2": pair["ticker_2"],
        "dependent_ticker": dependent,
        "independent_ticker": independent,
        "sector": pair["sector"],
        "same_sub_industry": (
            pair["same_sub_industry"]
        ),
        "return_correlation": (
            pair["return_correlation"]
        ),
        "fdr_pvalue": (
            pair["baseline_fdr_pvalue"]
        ),
        "hedge_ratio": pair["hedge_ratio"],
        "formation_half_life": (
            pair["half_life"]
        ),
        "evaluation_sessions": len(
            evaluation_dates
        ),
    }

    if len(signal_positions) == 0:
        return {
            **base_record,
            "entry_triggered": False,
            "signal_date": pd.NaT,
            "entry_evaluation_date": pd.NaT,
            "entry_z_score": np.nan,
            "entry_direction": "",
            "forward_sessions": 0,
            "touched_abs_z_1": False,
            "touched_abs_z_0_5": False,
            "crossed_mean": False,
            "sessions_to_abs_z_1": np.nan,
            "sessions_to_abs_z_0_5": np.nan,
            "sessions_to_mean_crossing": np.nan,
            "maximum_favourable_excursion_z": np.nan,
            "maximum_adverse_excursion_z": np.nan,
            "end_z_score": (
                z_scores.loc[
                    evaluation_dates[-1]
                ]
            ),
        }

    signal_position = int(
        signal_positions[0]
    )

    signal_date = actionable_dates[
        signal_position
    ]

    entry_z = float(
        z_scores.loc[signal_date]
    )

    entry_sign = float(
        np.sign(entry_z)
    )

    # The position would be entered during the session
    # following the close-based signal.
    next_position = (
        signal_dates.get_loc(signal_date) + 1
    )

    entry_evaluation_date = signal_dates[
        next_position
    ]

    forward_dates = signal_dates[
        next_position:
    ]

    forward_z = (
        z_scores.loc[forward_dates]
        .dropna()
    )

    if forward_z.empty:
        return {
            **base_record,
            "entry_triggered": False,
            "signal_date": signal_date,
            "entry_evaluation_date": pd.NaT,
            "entry_z_score": entry_z,
            "entry_direction": "",
            "forward_sessions": 0,
            "touched_abs_z_1": False,
            "touched_abs_z_0_5": False,
            "crossed_mean": False,
            "sessions_to_abs_z_1": np.nan,
            "sessions_to_abs_z_0_5": np.nan,
            "sessions_to_mean_crossing": np.nan,
            "maximum_favourable_excursion_z": np.nan,
            "maximum_adverse_excursion_z": np.nan,
            "end_z_score": np.nan,
        }

    z_path = forward_z.to_numpy(
        dtype=float
    )

    if entry_sign > 0:
        entry_direction = (
            f"short_{dependent}_long_{independent}"
        )
    else:
        entry_direction = (
            f"long_{dependent}_short_{independent}"
        )

    # A positive signed move represents convergence and
    # therefore favourable spread movement.
    signed_excursion = (
        -entry_sign
        * (z_path - entry_z)
    )

    sessions_z_1 = sessions_to_threshold(
        z_path,
        PARTIAL_EXIT_Z_SCORE,
    )

    sessions_z_0_5 = sessions_to_threshold(
        z_path,
        STANDARD_EXIT_Z_SCORE,
    )

    sessions_mean = sessions_to_mean_crossing(
        z_path,
        entry_sign,
    )

    return {
        **base_record,
        "entry_triggered": True,
        "signal_date": signal_date,
        "entry_evaluation_date": (
            entry_evaluation_date
        ),
        "entry_z_score": entry_z,
        "entry_direction": entry_direction,
        "forward_sessions": len(forward_z),
        "touched_abs_z_1": pd.notna(
            sessions_z_1
        ),
        "touched_abs_z_0_5": pd.notna(
            sessions_z_0_5
        ),
        "crossed_mean": pd.notna(
            sessions_mean
        ),
        "sessions_to_abs_z_1": sessions_z_1,
        "sessions_to_abs_z_0_5": (
            sessions_z_0_5
        ),
        "sessions_to_mean_crossing": (
            sessions_mean
        ),
        "maximum_favourable_excursion_z": float(
            np.max(signed_excursion)
        ),
        "maximum_adverse_excursion_z": float(
            np.min(signed_excursion)
        ),
        "end_z_score": float(
            forward_z.iloc[-1]
        ),
    }


def summarize_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build the overall event-study summary."""

    entered = events.loc[
        events["entry_triggered"]
    ]

    def safe_rate(column: str) -> float:
        if entered.empty:
            return np.nan

        return float(
            entered[column].mean()
        )

    metrics = {
        "pair_months_evaluated": len(events),
        "entry_events": len(entered),
        "entry_frequency": (
            len(entered) / len(events)
            if len(events)
            else np.nan
        ),
        "touched_abs_z_1_rate": safe_rate(
            "touched_abs_z_1"
        ),
        "touched_abs_z_0_5_rate": safe_rate(
            "touched_abs_z_0_5"
        ),
        "mean_crossing_rate": safe_rate(
            "crossed_mean"
        ),
        "median_entry_abs_z": (
            entered["entry_z_score"]
            .abs()
            .median()
        ),
        "median_sessions_to_abs_z_1": (
            entered["sessions_to_abs_z_1"]
            .median()
        ),
        "median_sessions_to_abs_z_0_5": (
            entered["sessions_to_abs_z_0_5"]
            .median()
        ),
        "median_sessions_to_mean_crossing": (
            entered[
                "sessions_to_mean_crossing"
            ].median()
        ),
        "median_favourable_excursion_z": (
            entered[
                "maximum_favourable_excursion_z"
            ].median()
        ),
        "median_adverse_excursion_z": (
            entered[
                "maximum_adverse_excursion_z"
            ].median()
        ),
    }

    return pd.DataFrame(
        {
            "metric": metrics.keys(),
            "value": metrics.values(),
        }
    )


def summarize_by_sector(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build sector-level event outcomes."""

    records = []

    for sector, group in events.groupby("sector"):
        entered = group.loc[
            group["entry_triggered"]
        ]

        records.append(
            {
                "sector": sector,
                "pair_months": len(group),
                "entry_events": len(entered),
                "entry_frequency": (
                    len(entered) / len(group)
                    if len(group)
                    else np.nan
                ),
                "touched_abs_z_1_rate": (
                    entered["touched_abs_z_1"].mean()
                    if not entered.empty
                    else np.nan
                ),
                "touched_abs_z_0_5_rate": (
                    entered[
                        "touched_abs_z_0_5"
                    ].mean()
                    if not entered.empty
                    else np.nan
                ),
                "mean_crossing_rate": (
                    entered["crossed_mean"].mean()
                    if not entered.empty
                    else np.nan
                ),
                "median_sessions_to_abs_z_0_5": (
                    entered[
                        "sessions_to_abs_z_0_5"
                    ].median()
                    if not entered.empty
                    else np.nan
                ),
                "median_favourable_excursion_z": (
                    entered[
                        "maximum_favourable_excursion_z"
                    ].median()
                    if not entered.empty
                    else np.nan
                ),
                "median_adverse_excursion_z": (
                    entered[
                        "maximum_adverse_excursion_z"
                    ].median()
                    if not entered.empty
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "entry_events",
            ascending=False,
        )
    )


def main() -> None:
    pairs, close, eligibility = load_inputs()

    records = []

    print("=" * 70)
    print("OUT-OF-SAMPLE SPREAD EVENT STUDY")
    print("=" * 70)
    print(f"Pair snapshots: {len(pairs)}")
    print(f"Entry threshold: {ENTRY_Z_SCORE:.2f}")
    print()

    for number, (_, pair) in enumerate(
        pairs.iterrows(),
        start=1,
    ):
        result = analyze_pair_month(
            pair,
            close,
            eligibility,
        )

        if result is not None:
            records.append(result)

        if number % 50 == 0 or number == len(pairs):
            print(
                f"Processed {number}/{len(pairs)}"
            )

    events = pd.DataFrame(records)

    if events.empty:
        raise RuntimeError(
            "No complete evaluation periods were found"
        )

    events = events.sort_values(
        [
            "selection_date",
            "sector",
            "pair_id",
        ]
    ).reset_index(drop=True)

    summary = summarize_events(events)
    sector_summary = summarize_by_sector(events)

    events.to_parquet(
        SPREAD_EVENT_FILE,
        engine="pyarrow",
        index=False,
    )

    summary.to_csv(
        SPREAD_EVENT_SUMMARY_FILE,
        index=False,
    )

    sector_summary.to_csv(
        SPREAD_EVENT_SECTOR_FILE,
        index=False,
    )

    entered = events.loc[
        events["entry_triggered"]
    ]

    print("\n" + "=" * 70)
    print("EVENT-STUDY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\nBy sector:")
    print(
        sector_summary.to_string(index=False)
    )

    if not entered.empty:
        print("\nWorst adverse excursions:")

        print(
            entered.sort_values(
                "maximum_adverse_excursion_z"
            )[
                [
                    "selection_date",
                    "pair_id",
                    "sector",
                    "entry_z_score",
                    "maximum_adverse_excursion_z",
                    "maximum_favourable_excursion_z",
                    "end_z_score",
                    "crossed_mean",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    print("\nSaved:")
    print(SPREAD_EVENT_FILE)
    print(SPREAD_EVENT_SUMMARY_FILE)
    print(SPREAD_EVENT_SECTOR_FILE)


if __name__ == "__main__":
    main()