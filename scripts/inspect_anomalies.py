import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    METADATA_DIR,
    RAW_DATA_DIR,
    SECURITY_QUALITY_FILE,
)


EXTREME_RETURN_THRESHOLD = 0.50
OHLC_RELATIVE_TOLERANCE = 1e-6
OHLC_ABSOLUTE_TOLERANCE = 1e-8

ANOMALY_EVENTS_FILE = (
    METADATA_DIR / "anomaly_events.csv"
)

ZERO_VOLUME_RUNS_FILE = (
    METADATA_DIR / "zero_volume_runs.csv"
)


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker's raw price file."""

    path = RAW_DATA_DIR / f"{ticker}.parquet"

    frame = pd.read_parquet(path)

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame.columns.name = None

    return frame.sort_index()


def find_extreme_returns(
    ticker: str,
    frame: pd.DataFrame,
) -> list[dict]:
    """Return every daily move above the inspection threshold."""

    returns = frame["Close"].pct_change(
        fill_method=None
    )

    mask = returns.abs() > EXTREME_RETURN_THRESHOLD

    events = []

    for date in frame.index[mask]:
        location = frame.index.get_loc(date)

        if location == 0:
            continue

        previous_date = frame.index[location - 1]
        previous_close = frame.iloc[location - 1]["Close"]
        current_close = frame.loc[date, "Close"]

        events.append(
            {
                "ticker": ticker,
                "date": date,
                "anomaly_type": "extreme_return",
                "previous_date": previous_date,
                "previous_close": previous_close,
                "open": frame.loc[date, "Open"],
                "high": frame.loc[date, "High"],
                "low": frame.loc[date, "Low"],
                "close": current_close,
                "volume": frame.loc[date, "Volume"],
                "value": returns.loc[date],
                "details": (
                    f"Absolute return exceeded "
                    f"{EXTREME_RETURN_THRESHOLD:.0%}"
                ),
            }
        )

    return events


def find_ohlc_violations(
    ticker: str,
    frame: pd.DataFrame,
) -> list[dict]:
    """Find genuine OHLC relationship violations."""

    daily_max = frame[
        ["Open", "Close"]
    ].max(axis=1)

    daily_min = frame[
        ["Open", "Close"]
    ].min(axis=1)

    high_below_low = (
        (frame["High"] < frame["Low"])
        & ~np.isclose(
            frame["High"],
            frame["Low"],
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )

    high_below_open_close = (
        (frame["High"] < daily_max)
        & ~np.isclose(
            frame["High"],
            daily_max,
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )

    low_above_open_close = (
        (frame["Low"] > daily_min)
        & ~np.isclose(
            frame["Low"],
            daily_min,
            rtol=OHLC_RELATIVE_TOLERANCE,
            atol=OHLC_ABSOLUTE_TOLERANCE,
        )
    )

    violation_definitions = {
        "high_below_low": high_below_low,
        "high_below_open_close": (
            high_below_open_close
        ),
        "low_above_open_close": (
            low_above_open_close
        ),
    }

    events = []

    for anomaly_type, mask in violation_definitions.items():
        for date in frame.index[mask]:
            if anomaly_type == "high_below_low":
                difference = (
                    frame.loc[date, "Low"]
                    - frame.loc[date, "High"]
                )

            elif anomaly_type == "high_below_open_close":
                difference = (
                    daily_max.loc[date]
                    - frame.loc[date, "High"]
                )

            else:
                difference = (
                    frame.loc[date, "Low"]
                    - daily_min.loc[date]
                )

            events.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "anomaly_type": anomaly_type,
                    "previous_date": pd.NaT,
                    "previous_close": np.nan,
                    "open": frame.loc[date, "Open"],
                    "high": frame.loc[date, "High"],
                    "low": frame.loc[date, "Low"],
                    "close": frame.loc[date, "Close"],
                    "volume": frame.loc[date, "Volume"],
                    "value": difference,
                    "details": (
                        "OHLC relationship violation; "
                        f"difference={difference:.10f}"
                    ),
                }
            )

    return events


def find_zero_volume_runs(
    ticker: str,
    frame: pd.DataFrame,
) -> list[dict]:
    """Summarize consecutive zero-volume observations."""

    zero_volume = frame["Volume"].fillna(0).eq(0)

    if not zero_volume.any():
        return []

    # Each nonzero row starts a new potential group.
    groups = (~zero_volume).cumsum()

    runs = []

    for _, group in frame.loc[zero_volume].groupby(
        groups[zero_volume]
    ):
        start_date = group.index.min()
        end_date = group.index.max()
        length = len(group)

        runs.append(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "length": length,
                "start_close": group.iloc[0]["Close"],
                "end_close": group.iloc[-1]["Close"],
                "unique_closes": group["Close"].nunique(),
            }
        )

    return runs


def main() -> None:
    quality = pd.read_csv(SECURITY_QUALITY_FILE)

    flagged = quality.loc[
        quality["status"].isin(["warning", "error"])
    ].copy()

    tickers = flagged["ticker"].tolist()

    anomaly_events = []
    zero_volume_runs = []

    print("=" * 70)
    print("ANOMALY INSPECTION")
    print("=" * 70)
    print(f"Flagged securities: {len(tickers)}")

    for number, ticker in enumerate(tickers, start=1):
        frame = load_prices(ticker)

        anomaly_events.extend(
            find_extreme_returns(
                ticker,
                frame,
            )
        )

        anomaly_events.extend(
            find_ohlc_violations(
                ticker,
                frame,
            )
        )

        zero_volume_runs.extend(
            find_zero_volume_runs(
                ticker,
                frame,
            )
        )

        if number % 10 == 0 or number == len(tickers):
            print(
                f"Inspected {number}/{len(tickers)} securities"
            )

    events = pd.DataFrame(anomaly_events)
    volume_runs = pd.DataFrame(zero_volume_runs)

    if not events.empty:
        events = events.sort_values(
            ["ticker", "date", "anomaly_type"]
        )

    if not volume_runs.empty:
        volume_runs = volume_runs.sort_values(
            ["length", "ticker"],
            ascending=[False, True],
        )

    events.to_csv(
        ANOMALY_EVENTS_FILE,
        index=False,
    )

    volume_runs.to_csv(
        ZERO_VOLUME_RUNS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ANOMALY SUMMARY")
    print("=" * 70)

    if events.empty:
        print("No extreme returns or OHLC violations.")
    else:
        print("\nEvents by type:")
        print(
            events["anomaly_type"]
            .value_counts()
            .to_string()
        )

        print("\nOHLC violations:")
        ohlc_events = events.loc[
            events["anomaly_type"] != "extreme_return"
        ]

        if ohlc_events.empty:
            print("None")
        else:
            print(
                ohlc_events.to_string(index=False)
            )

        print("\nExtreme-return events:")
        return_events = events.loc[
            events["anomaly_type"] == "extreme_return",
            [
                "ticker",
                "date",
                "previous_close",
                "close",
                "value",
                "volume",
            ],
        ]

        print(
            return_events.to_string(index=False)
        )

    print("\nLargest zero-volume runs:")

    if volume_runs.empty:
        print("None")
    else:
        print(
            volume_runs.head(30).to_string(index=False)
        )

    print(f"\nAnomaly events : {ANOMALY_EVENTS_FILE}")
    print(f"Volume runs    : {ZERO_VOLUME_RUNS_FILE}")


if __name__ == "__main__":
    main()