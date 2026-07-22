import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLEANING_LOG_FILE,
    DATA_OVERRIDES_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def load_overrides() -> pd.DataFrame:
    """Load and standardize documented data overrides."""

    if not DATA_OVERRIDES_FILE.exists():
        return pd.DataFrame()

    overrides = pd.read_csv(
        DATA_OVERRIDES_FILE,
        dtype=str,
    )

    for column in ["start_date", "end_date"]:
        overrides[column] = pd.to_datetime(
            overrides[column],
            errors="coerce",
        )

    return overrides


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load one raw price file without modifying it."""

    frame = pd.read_parquet(path)

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    frame.columns.name = None

    return frame.sort_index()


def apply_truncation(
    frame: pd.DataFrame,
    ticker: str,
    override: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Remove history before a verified security start date."""

    cutoff = override["start_date"]

    rows_before = len(frame)

    cleaned = frame.loc[
        frame.index >= cutoff
    ].copy()

    rows_removed = rows_before - len(cleaned)

    log = {
        "ticker": ticker,
        "action": "truncate_before",
        "date": cutoff,
        "field": "",
        "old_value": "",
        "new_value": "",
        "rows_affected": rows_removed,
        "reason": override["reason"],
    }

    return cleaned, log


def apply_ohlc_repair(
    frame: pd.DataFrame,
    ticker: str,
    override: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Repair one impossible OHLC field while preserving raw data.
    """

    date = override["start_date"]
    field = override["field"]

    if date not in frame.index:
        raise ValueError(
            f"{ticker}: override date {date.date()} "
            "does not exist"
        )

    if field not in frame.columns:
        raise ValueError(
            f"{ticker}: field {field} does not exist"
        )

    old_value = frame.loc[date, field]

    if field == "Low":
        new_value = frame.loc[
            date,
            ["Open", "High", "Low", "Close"],
        ].min()

    elif field == "High":
        new_value = frame.loc[
            date,
            ["Open", "High", "Low", "Close"],
        ].max()

    else:
        raise ValueError(
            f"Unsupported OHLC repair field: {field}"
        )

    cleaned = frame.copy()
    cleaned.loc[date, field] = new_value

    log = {
        "ticker": ticker,
        "action": "repair_ohlc",
        "date": date,
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "rows_affected": 1,
        "reason": override["reason"],
    }

    return cleaned, log


def apply_overrides(
    frame: pd.DataFrame,
    ticker: str,
    overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply every documented override for one security."""

    security_overrides = overrides.loc[
        overrides["ticker"] == ticker
    ]

    cleaned = frame.copy()
    logs = []

    for _, override in security_overrides.iterrows():
        action = override["action"]

        if action == "truncate_before":
            cleaned, log = apply_truncation(
                cleaned,
                ticker,
                override,
            )

        elif action == "repair_ohlc":
            cleaned, log = apply_ohlc_repair(
                cleaned,
                ticker,
                override,
            )

        else:
            raise ValueError(
                f"{ticker}: unknown override action {action}"
            )

        logs.append(log)

    return cleaned, logs


def validate_cleaned_frame(
    frame: pd.DataFrame,
    ticker: str,
) -> None:
    """Run essential checks before saving processed data."""

    if frame.empty:
        raise ValueError(
            f"{ticker}: processed frame is empty"
        )

    if frame.index.duplicated().any():
        raise ValueError(
            f"{ticker}: duplicate dates after cleaning"
        )

    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            f"{ticker}: dates are not sorted"
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ticker",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{ticker}: missing columns {missing}"
        )

    invalid_prices = (
        frame[["Open", "High", "Low", "Close"]] <= 0
    ).any().any()

    if invalid_prices:
        raise ValueError(
            f"{ticker}: nonpositive adjusted price"
        )


def main() -> None:
    raw_paths = sorted(
        RAW_DATA_DIR.glob("*.parquet")
    )

    if not raw_paths:
        raise FileNotFoundError(
            f"No raw files found in {RAW_DATA_DIR}"
        )

    overrides = load_overrides()
    cleaning_logs = []

    print("=" * 70)
    print("BUILD PROCESSED DATA")
    print("=" * 70)
    print(f"Raw securities : {len(raw_paths)}")
    print(f"Overrides      : {len(overrides)}")

    for number, path in enumerate(raw_paths, start=1):
        ticker = path.stem

        raw = load_raw_file(path)

        cleaned, logs = apply_overrides(
            raw,
            ticker,
            overrides,
        )

        validate_cleaned_frame(
            cleaned,
            ticker,
        )

        output_path = (
            PROCESSED_DATA_DIR / path.name
        )

        cleaned.to_parquet(
            output_path,
            engine="pyarrow",
            index=True,
        )

        cleaning_logs.extend(logs)

        if number % 50 == 0 or number == len(raw_paths):
            print(
                f"Processed {number}/{len(raw_paths)} securities"
            )

    cleaning_log = pd.DataFrame(cleaning_logs)

    cleaning_log.to_csv(
        CLEANING_LOG_FILE,
        index=False,
    )

    processed_count = len(
        list(PROCESSED_DATA_DIR.glob("*.parquet"))
    )

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Processed files : {processed_count}")
    print(f"Overrides used  : {len(cleaning_logs)}")
    print(f"Cleaning log    : {CLEANING_LOG_FILE}")

    if cleaning_logs:
        print("\nApplied changes:")
        print(
            cleaning_log.to_string(index=False)
        )


if __name__ == "__main__":
    main()