import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CONSTITUENTS_FILE,
    QUALITY_SUMMARY_FILE,
    RAW_DATA_DIR,
    SECURITY_QUALITY_FILE,
)


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

MINIMUM_FORMATION_OBSERVATIONS = 252

# These observations are flagged for inspection, not deleted.
EXTREME_DAILY_RETURN = 0.50
MAX_MISSING_FRACTION = 0.05

# Adjusted Yahoo OHLC values may differ by tiny floating-point
# amounts after corporate-action adjustment.
OHLC_RELATIVE_TOLERANCE = 1e-6
OHLC_ABSOLUTE_TOLERANCE = 1e-8

def maximum_consecutive_missing(
    available: pd.Series,
) -> int:
    """
    Calculate the maximum consecutive run of missing observations.

    `available` must be a Boolean Series on the reference calendar.
    """

    missing = ~available

    if not missing.any():
        return 0

    groups = missing.ne(missing.shift()).cumsum()

    runs = missing.groupby(groups).sum()

    return int(runs.max())


def load_price_file(path: Path) -> pd.DataFrame:
    """Load and standardize one Parquet price file."""

    frame = pd.read_parquet(path)

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame.columns.name = None

    frame = frame.sort_index()

    return frame


def build_reference_calendar(
    paths: list[Path],
) -> pd.DatetimeIndex:
    """
    Construct a reference US trading calendar from all downloaded files.

    A date is included when at least 25% of securities that have already
    started trading contain an observation on that date.

    This removes isolated erroneous dates without requiring an external
    exchange-calendar package.
    """

    date_counts: dict[pd.Timestamp, int] = {}
    active_counts: dict[pd.Timestamp, int] = {}

    file_ranges = []
    file_dates = {}

    for path in paths:
        frame = load_price_file(path)

        if frame.empty:
            continue

        dates = pd.DatetimeIndex(frame.index.unique()).sort_values()

        file_dates[path.stem] = dates
        file_ranges.append(
            {
                "ticker": path.stem,
                "first_date": dates.min(),
                "last_date": dates.max(),
            }
        )

        for date in dates:
            date_counts[date] = date_counts.get(date, 0) + 1

    all_dates = pd.DatetimeIndex(
        sorted(date_counts.keys())
    )

    ranges = pd.DataFrame(file_ranges)

    for date in all_dates:
        active_counts[date] = int(
            (
                (ranges["first_date"] <= date)
                & (ranges["last_date"] >= date)
            ).sum()
        )

    valid_dates = []

    for date in all_dates:
        active = active_counts[date]
        observed = date_counts[date]

        if active == 0:
            continue

        coverage = observed / active

        if coverage >= 0.25:
            valid_dates.append(date)

    return pd.DatetimeIndex(valid_dates, name="date")


def validate_security(
    path: Path,
    reference_calendar: pd.DatetimeIndex,
) -> dict:
    """Run structural and statistical checks for one security."""

    ticker = path.stem

    try:
        frame = load_price_file(path)
    except Exception as error:
        return {
            "ticker": ticker,
            "status": "unreadable",
            "error": str(error),
        }

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        return {
            "ticker": ticker,
            "status": "missing_columns",
            "error": ", ".join(missing_columns),
        }

    duplicate_dates = int(frame.index.duplicated().sum())
    non_monotonic_index = not frame.index.is_monotonic_increasing

    first_date = frame.index.min()
    last_date = frame.index.max()
    observations = len(frame)

    relevant_calendar = reference_calendar[
        (reference_calendar >= first_date)
        & (reference_calendar <= last_date)
    ]

    available = pd.Series(
        relevant_calendar.isin(frame.index),
        index=relevant_calendar,
    )

    expected_observations = len(relevant_calendar)
    missing_observations = int((~available).sum())

    if expected_observations > 0:
        missing_fraction = (
            missing_observations / expected_observations
        )
    else:
        missing_fraction = np.nan

    maximum_missing_run = maximum_consecutive_missing(
        available
    )

    nonpositive_open = int((frame["Open"] <= 0).sum())
    nonpositive_high = int((frame["High"] <= 0).sum())
    nonpositive_low = int((frame["Low"] <= 0).sum())
    nonpositive_close = int((frame["Close"] <= 0).sum())

    negative_volume = int((frame["Volume"] < 0).sum())
    zero_volume = int((frame["Volume"] == 0).sum())

    daily_max_open_close = frame[
        ["Open", "Close"]
    ].max(axis=1)

    daily_min_open_close = frame[
        ["Open", "Close"]
    ].min(axis=1)


    high_below_low_mask = (
        (frame["High"] < frame["Low"])
        & ~np.isclose(
            frame["High"],
            frame["Low"],
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )

    high_below_open_close_mask = (
        (frame["High"] < daily_max_open_close)
        & ~np.isclose(
            frame["High"],
            daily_max_open_close,
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )

    low_above_open_close_mask = (
        (frame["Low"] > daily_min_open_close)
        & ~np.isclose(
            frame["Low"],
            daily_min_open_close,
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )


    high_below_low = int(
        high_below_low_mask.sum()
    )

    high_below_open_close = int(
        high_below_open_close_mask.sum()
    )

    low_above_open_close = int(
        low_above_open_close_mask.sum()
    )

    close_returns = frame["Close"].pct_change(
        fill_method=None
    )

    extreme_return_count = int(
        (close_returns.abs() > EXTREME_DAILY_RETURN).sum()
    )

    maximum_absolute_return = close_returns.abs().max()

    missing_ohlcv_values = int(
        frame[REQUIRED_COLUMNS].isna().sum().sum()
    )

    # Useful descriptive information for later universe filters.
    recent = frame.tail(252).copy()

    median_price_recent = recent["Close"].median()

    median_dollar_volume_recent = (
        recent["Close"] * recent["Volume"]
    ).median()

    sufficient_history = (
        observations >= MINIMUM_FORMATION_OBSERVATIONS
    )

    structural_error = any(
        [
            duplicate_dates > 0,
            non_monotonic_index,
            missing_ohlcv_values > 0,
            nonpositive_open > 0,
            nonpositive_high > 0,
            nonpositive_low > 0,
            nonpositive_close > 0,
            negative_volume > 0,
            high_below_low > 0,
            high_below_open_close > 0,
            low_above_open_close > 0,
        ]
    )

    coverage_warning = (
        pd.notna(missing_fraction)
        and missing_fraction > MAX_MISSING_FRACTION
    )

    statistical_warning = (
        extreme_return_count > 0
        or zero_volume > 0
        or coverage_warning
    )

    if structural_error:
        status = "error"
    elif statistical_warning:
        status = "warning"
    else:
        status = "pass"

    return {
        "ticker": ticker,
        "status": status,
        "first_date": first_date,
        "last_date": last_date,
        "observations": observations,
        "expected_observations": expected_observations,
        "missing_observations": missing_observations,
        "missing_fraction": missing_fraction,
        "maximum_missing_run": maximum_missing_run,
        "duplicate_dates": duplicate_dates,
        "non_monotonic_index": non_monotonic_index,
        "missing_ohlcv_values": missing_ohlcv_values,
        "nonpositive_open": nonpositive_open,
        "nonpositive_high": nonpositive_high,
        "nonpositive_low": nonpositive_low,
        "nonpositive_close": nonpositive_close,
        "negative_volume": negative_volume,
        "zero_volume": zero_volume,
        "high_below_low": high_below_low,
        "high_below_open_close": high_below_open_close,
        "low_above_open_close": low_above_open_close,
        "extreme_return_count": extreme_return_count,
        "maximum_absolute_return": maximum_absolute_return,
        "median_price_recent": median_price_recent,
        "median_dollar_volume_recent": (
            median_dollar_volume_recent
        ),
        "sufficient_history": sufficient_history,
        "error": "",
    }


def build_summary(
    quality: pd.DataFrame,
    reference_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a compact dataset-level quality summary."""

    status_counts = quality["status"].value_counts()

    summary = {
        "files_checked": len(quality),
        "pass": int(status_counts.get("pass", 0)),
        "warning": int(status_counts.get("warning", 0)),
        "error": int(status_counts.get("error", 0)),
        "unreadable": int(
            status_counts.get("unreadable", 0)
        ),
        "missing_columns": int(
            status_counts.get("missing_columns", 0)
        ),
        "reference_start": reference_calendar.min(),
        "reference_end": reference_calendar.max(),
        "reference_trading_days": len(reference_calendar),
        "earliest_security_date": quality[
            "first_date"
        ].min(),
        "latest_security_date": quality[
            "last_date"
        ].max(),
        "securities_with_252_observations": int(
            quality["sufficient_history"].fillna(False).sum()
        ),
        "securities_with_structural_errors": int(
            (quality["status"] == "error").sum()
        ),
        "securities_with_extreme_returns": int(
            (quality["extreme_return_count"].fillna(0) > 0).sum()
        ),
        "securities_with_zero_volume": int(
            (quality["zero_volume"].fillna(0) > 0).sum()
        ),
    }

    return pd.DataFrame(
        {
            "metric": summary.keys(),
            "value": summary.values(),
        }
    )


def main() -> None:
    if not CONSTITUENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing constituent file: {CONSTITUENTS_FILE}"
        )

    paths = sorted(RAW_DATA_DIR.glob("*.parquet"))

    if not paths:
        raise FileNotFoundError(
            f"No Parquet files found in {RAW_DATA_DIR}"
        )

    constituents = pd.read_csv(CONSTITUENTS_FILE)

    expected_tickers = set(
        constituents["yahoo_ticker"].dropna()
    )
    downloaded_tickers = {
        path.stem for path in paths
    }

    missing_files = sorted(
        expected_tickers - downloaded_tickers
    )
    unexpected_files = sorted(
        downloaded_tickers - expected_tickers
    )

    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)
    print(f"Expected securities   : {len(expected_tickers)}")
    print(f"Downloaded files      : {len(paths)}")
    print(f"Missing files         : {len(missing_files)}")
    print(f"Unexpected files      : {len(unexpected_files)}")

    if missing_files:
        print(f"Missing: {missing_files}")

    if unexpected_files:
        print(f"Unexpected: {unexpected_files}")

    print("\nBuilding reference trading calendar...")

    reference_calendar = build_reference_calendar(paths)

    print(
        f"Reference calendar: "
        f"{reference_calendar.min().date()} to "
        f"{reference_calendar.max().date()} "
        f"({len(reference_calendar)} sessions)"
    )

    records = []

    for number, path in enumerate(paths, start=1):
        records.append(
            validate_security(
                path,
                reference_calendar,
            )
        )

        if number % 50 == 0 or number == len(paths):
            print(
                f"Validated {number}/{len(paths)} securities"
            )

    quality = pd.DataFrame(records)
    quality = quality.sort_values("ticker").reset_index(
        drop=True
    )

    quality.to_csv(
        SECURITY_QUALITY_FILE,
        index=False,
    )

    summary = build_summary(
        quality,
        reference_calendar,
    )

    summary.to_csv(
        QUALITY_SUMMARY_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    warnings_table = quality.loc[
        quality["status"].isin(["warning", "error"]),
        [
            "ticker",
            "status",
            "first_date",
            "observations",
            "missing_fraction",
            "maximum_missing_run",
            "zero_volume",
            "extreme_return_count",
            "maximum_absolute_return",
        ],
    ]

    if warnings_table.empty:
        print("\nNo warnings or structural errors detected.")
    else:
        print("\nSecurities requiring inspection:")
        print(
            warnings_table.to_string(
                index=False,
            )
        )

    print(f"\nSecurity report: {SECURITY_QUALITY_FILE}")
    print(f"Summary report : {QUALITY_SUMMARY_FILE}")


if __name__ == "__main__":
    main()