import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLOSE_PANEL_FILE,
    CONSTITUENTS_FILE,
    DOLLAR_VOLUME_PANEL_FILE,
    ELIGIBILITY_PANEL_FILE,
    LIQUIDITY_WINDOW,
    MINIMUM_HISTORY,
    MINIMUM_LIQUID_DAYS,
    MINIMUM_MEDIAN_DOLLAR_VOLUME,
    MINIMUM_PRICE,
    OPEN_PANEL_FILE,
    PROCESSED_DATA_DIR,
    UNIVERSE_COVERAGE_FILE,
    VOLUME_PANEL_FILE,
)


REQUIRED_COLUMNS = [
    "Open",
    "Close",
    "Volume",
]


def load_security(
    path: Path,
) -> pd.DataFrame:
    """Load one processed security file."""

    frame = pd.read_parquet(path)

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame.columns.name = None

    frame = frame.sort_index()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{path.stem}: missing columns {missing}"
        )

    return frame


def build_panel(
    series_by_ticker: dict[str, pd.Series],
) -> pd.DataFrame:
    """Combine ticker Series into an aligned wide matrix."""

    panel = pd.concat(
        series_by_ticker,
        axis=1,
    )

    panel.columns.name = "ticker"
    panel.index.name = "date"

    panel = panel.sort_index()
    panel = panel.reindex(
        sorted(panel.columns),
        axis=1,
    )

    return panel


def build_coverage_report(
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Build date-level universe coverage statistics."""

    available_count = close.notna().sum(axis=1)
    eligible_count = eligibility.sum(axis=1)

    coverage = pd.DataFrame(
        {
            "available_securities": available_count,
            "eligible_securities": eligible_count,
        }
    )

    coverage["ineligible_securities"] = (
        coverage["available_securities"]
        - coverage["eligible_securities"]
    )

    coverage["eligible_fraction"] = np.where(
        coverage["available_securities"] > 0,
        (
            coverage["eligible_securities"]
            / coverage["available_securities"]
        ),
        np.nan,
    )

    coverage.index.name = "date"

    return coverage


def main() -> None:
    paths = sorted(
        PROCESSED_DATA_DIR.glob("*.parquet")
    )

    if not paths:
        raise FileNotFoundError(
            f"No processed files found in "
            f"{PROCESSED_DATA_DIR}"
        )

    if not CONSTITUENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing constituent file: "
            f"{CONSTITUENTS_FILE}"
        )

    constituents = pd.read_csv(
        CONSTITUENTS_FILE
    )

    expected_tickers = set(
        constituents["yahoo_ticker"].dropna()
    )

    downloaded_tickers = {
        path.stem for path in paths
    }

    missing = sorted(
        expected_tickers - downloaded_tickers
    )

    unexpected = sorted(
        downloaded_tickers - expected_tickers
    )

    if missing:
        raise RuntimeError(
            f"Missing processed tickers: {missing}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected processed tickers: {unexpected}"
        )

    open_series = {}
    close_series = {}
    volume_series = {}

    print("=" * 70)
    print("BUILD ALIGNED DATA PANELS")
    print("=" * 70)
    print(f"Securities: {len(paths)}")

    for number, path in enumerate(paths, start=1):
        ticker = path.stem
        frame = load_security(path)

        open_series[ticker] = frame["Open"].rename(
            ticker
        )

        close_series[ticker] = frame["Close"].rename(
            ticker
        )

        volume_series[ticker] = frame["Volume"].rename(
            ticker
        )

        if number % 50 == 0 or number == len(paths):
            print(
                f"Loaded {number}/{len(paths)} securities"
            )

    print("\nAligning matrices...")

    open_prices = build_panel(open_series)
    close_prices = build_panel(close_series)
    volume = build_panel(volume_series)

    common_index = (
        open_prices.index
        .union(close_prices.index)
        .union(volume.index)
        .sort_values()
    )

    common_columns = sorted(
        set(open_prices.columns)
        | set(close_prices.columns)
        | set(volume.columns)
    )

    open_prices = open_prices.reindex(
        index=common_index,
        columns=common_columns,
    )

    close_prices = close_prices.reindex(
        index=common_index,
        columns=common_columns,
    )

    volume = volume.reindex(
        index=common_index,
        columns=common_columns,
    )

    print("Calculating eligibility filters...")

    valid_close = (
        close_prices.notna()
        & np.isfinite(close_prices)
        & (close_prices > 0)
    )

    valid_open = (
        open_prices.notna()
        & np.isfinite(open_prices)
        & (open_prices > 0)
    )

    valid_volume = (
        volume.notna()
        & np.isfinite(volume)
        & (volume >= 0)
    )

    history_count = valid_close.cumsum()

    sufficient_history = (
        history_count >= MINIMUM_HISTORY
    )

    sufficient_price = (
        close_prices >= MINIMUM_PRICE
    )

    positive_volume = (
        volume > 0
    )

    positive_volume_days = (
        positive_volume
        .astype(float)
        .rolling(
            window=LIQUIDITY_WINDOW,
            min_periods=MINIMUM_LIQUID_DAYS,
        )
        .sum()
    )

    sufficient_volume_history = (
        positive_volume_days
        >= MINIMUM_LIQUID_DAYS
    )

    dollar_volume = (
        close_prices * volume
    )

    median_dollar_volume = (
        dollar_volume
        .where(positive_volume)
        .rolling(
            window=LIQUIDITY_WINDOW,
            min_periods=MINIMUM_LIQUID_DAYS,
        )
        .median()
    )

    sufficient_dollar_volume = (
        median_dollar_volume
        >= MINIMUM_MEDIAN_DOLLAR_VOLUME
    )

    eligibility = (
        valid_close
        & valid_open
        & valid_volume
        & sufficient_history
        & sufficient_price
        & positive_volume
        & sufficient_volume_history
        & sufficient_dollar_volume
    )

    eligibility = eligibility.fillna(False)
    eligibility = eligibility.astype(bool)

    coverage = build_coverage_report(
        close_prices,
        eligibility,
    )

    print("Saving matrices...")

    open_prices.to_parquet(
        OPEN_PANEL_FILE,
        engine="pyarrow",
    )

    close_prices.to_parquet(
        CLOSE_PANEL_FILE,
        engine="pyarrow",
    )

    volume.to_parquet(
        VOLUME_PANEL_FILE,
        engine="pyarrow",
    )

    median_dollar_volume.to_parquet(
        DOLLAR_VOLUME_PANEL_FILE,
        engine="pyarrow",
    )

    eligibility.to_parquet(
        ELIGIBILITY_PANEL_FILE,
        engine="pyarrow",
    )

    coverage.to_csv(
        UNIVERSE_COVERAGE_FILE,
        index=True,
    )

    latest_date = common_index.max()
    latest_available = int(
        close_prices.loc[latest_date].notna().sum()
    )
    latest_eligible = int(
        eligibility.loc[latest_date].sum()
    )

    eligible_counts = eligibility.sum(axis=1)

    first_eligible_date = eligible_counts.loc[
        eligible_counts > 0
    ].index.min()

    print("\n" + "=" * 70)
    print("PANEL SUMMARY")
    print("=" * 70)
    print(
        f"Date range          : "
        f"{common_index.min().date()} to "
        f"{common_index.max().date()}"
    )
    print(
        f"Trading sessions    : {len(common_index)}"
    )
    print(
        f"Securities          : {len(common_columns)}"
    )
    print(
        f"First eligible date : "
        f"{first_eligible_date.date()}"
    )
    print(
        f"Latest available    : {latest_available}"
    )
    print(
        f"Latest eligible     : {latest_eligible}"
    )
    print(
        f"Minimum eligible    : "
        f"{int(eligible_counts.min())}"
    )
    print(
        f"Median eligible     : "
        f"{eligible_counts.median():.0f}"
    )
    print(
        f"Maximum eligible    : "
        f"{int(eligible_counts.max())}"
    )

    print("\nSaved files:")
    print(f"Open             : {OPEN_PANEL_FILE}")
    print(f"Close            : {CLOSE_PANEL_FILE}")
    print(f"Volume           : {VOLUME_PANEL_FILE}")
    print(
        f"Median dollar vol: "
        f"{DOLLAR_VOLUME_PANEL_FILE}"
    )
    print(
        f"Eligibility      : "
        f"{ELIGIBILITY_PANEL_FILE}"
    )
    print(
        f"Coverage report  : "
        f"{UNIVERSE_COVERAGE_FILE}"
    )


if __name__ == "__main__":
    main()