import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    CONSTITUENTS_FILE,
    DOWNLOAD_LOG_FILE,
    END_DATE,
    MAX_RETRIES,
    RAW_DATA_DIR,
    RETRY_WAIT_SECONDS,
    START_DATE,
)


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def split_into_batches(
    values: list[str],
    batch_size: int,
) -> list[list[str]]:
    """Split a list into fixed-size batches."""

    return [
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def clean_price_frame(
    frame: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Clean and validate one ticker's adjusted OHLCV data.
    """

    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()

    # Recent yfinance versions may return a MultiIndex even
    # when downloading a single ticker.
    if isinstance(frame.columns, pd.MultiIndex):
        ticker_levels = [
            str(value)
            for value in frame.columns.get_level_values(0).unique()
        ]

        if ticker in ticker_levels:
            frame = frame[ticker].copy()
        else:
            frame.columns = frame.columns.get_level_values(0)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        warnings.warn(
            f"{ticker}: missing columns {missing_columns}"
        )
        return pd.DataFrame()

    frame = frame[REQUIRED_COLUMNS].copy()

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"

    # Remove timezone information for consistent storage.
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ].sort_index()

    # A row without a close cannot be used.
    frame = frame.dropna(subset=["Close"])

    price_columns = ["Open", "High", "Low", "Close"]

    # Prices must be positive.
    for column in price_columns:
        frame.loc[frame[column] <= 0, column] = pd.NA

    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])

    frame["Volume"] = pd.to_numeric(
        frame["Volume"],
        errors="coerce",
    )

    frame["ticker"] = ticker

    return frame


def extract_ticker_from_batch(
    batch_data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Extract one ticker from a multi-ticker yfinance response.
    """

    if batch_data is None or batch_data.empty:
        return pd.DataFrame()

    if not isinstance(batch_data.columns, pd.MultiIndex):
        return clean_price_frame(batch_data, ticker)

    level_zero = batch_data.columns.get_level_values(0)

    if ticker not in level_zero:
        return pd.DataFrame()

    ticker_data = batch_data[ticker].copy()

    return clean_price_frame(ticker_data, ticker)


def save_ticker_data(
    ticker: str,
    frame: pd.DataFrame,
) -> Path:
    """Save one security to a Parquet file."""

    output_path = RAW_DATA_DIR / f"{ticker}.parquet"

    frame.to_parquet(
        output_path,
        engine="pyarrow",
        index=True,
    )

    return output_path


def download_batch(
    tickers: list[str],
) -> pd.DataFrame:
    """Download one batch of securities from Yahoo Finance."""

    return yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )


def download_single_ticker(
    ticker: str,
) -> pd.DataFrame:
    """Retry a failed security individually."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers=ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                actions=False,
                group_by="column",
                threads=False,
                progress=False,
                timeout=30,
            )

            data = clean_price_frame(data, ticker)

            if not data.empty:
                return data

        except Exception as error:
            warnings.warn(
                f"{ticker}: attempt {attempt} failed: {error}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_SECONDS * attempt)

    return pd.DataFrame()


def build_log_record(
    ticker: str,
    frame: pd.DataFrame,
    status: str,
    method: str,
    error: str = "",
) -> dict:
    """Create one download-log record."""

    if frame.empty:
        first_date = pd.NaT
        last_date = pd.NaT
        observations = 0
        missing_fraction = None
    else:
        first_date = frame.index.min()
        last_date = frame.index.max()
        observations = len(frame)

        missing_fraction = float(
            frame[REQUIRED_COLUMNS]
            .isna()
            .mean()
            .mean()
        )

    return {
        "ticker": ticker,
        "status": status,
        "method": method,
        "first_date": first_date,
        "last_date": last_date,
        "observations": observations,
        "missing_fraction": missing_fraction,
        "error": error,
    }


def main() -> None:
    if not CONSTITUENTS_FILE.exists():
        raise FileNotFoundError(
            f"{CONSTITUENTS_FILE} does not exist.\n"
            "Run download_constituents.py first."
        )

    constituents = pd.read_csv(CONSTITUENTS_FILE)
    tickers = constituents["yahoo_ticker"].dropna().tolist()

    batches = split_into_batches(
        tickers,
        BATCH_SIZE,
    )

    log_records = []
    failed_tickers = []

    print("=" * 60)
    print("YAHOO FINANCE PRICE DOWNLOAD")
    print("=" * 60)
    print(f"Requested tickers : {len(tickers)}")
    print(f"Start date        : {START_DATE}")
    print(f"End date          : {END_DATE or 'latest'}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Total batches     : {len(batches)}")
    print()

    for batch_number, batch in enumerate(batches, start=1):
        print(
            f"Downloading batch "
            f"{batch_number}/{len(batches)} "
            f"({len(batch)} tickers)..."
        )

        try:
            batch_data = download_batch(batch)

            for ticker in batch:
                ticker_data = extract_ticker_from_batch(
                    batch_data,
                    ticker,
                )

                if ticker_data.empty:
                    failed_tickers.append(ticker)
                    continue

                save_ticker_data(
                    ticker,
                    ticker_data,
                )

                log_records.append(
                    build_log_record(
                        ticker=ticker,
                        frame=ticker_data,
                        status="success",
                        method="batch",
                    )
                )

        except Exception as error:
            warnings.warn(
                f"Entire batch {batch_number} failed: {error}"
            )
            failed_tickers.extend(batch)

    failed_tickers = sorted(set(failed_tickers))

    if failed_tickers:
        print()
        print(
            f"Retrying {len(failed_tickers)} "
            "failed tickers individually..."
        )

    recovered_tickers = set()

    for retry_number, ticker in enumerate(
        failed_tickers,
        start=1,
    ):
        print(
            f"Retry {retry_number}/{len(failed_tickers)}: "
            f"{ticker}"
        )

        try:
            ticker_data = download_single_ticker(ticker)

            if ticker_data.empty:
                log_records.append(
                    build_log_record(
                        ticker=ticker,
                        frame=ticker_data,
                        status="failed",
                        method="individual_retry",
                        error="No valid OHLCV data returned",
                    )
                )
                continue

            save_ticker_data(
                ticker,
                ticker_data,
            )

            recovered_tickers.add(ticker)

            log_records.append(
                build_log_record(
                    ticker=ticker,
                    frame=ticker_data,
                    status="success",
                    method="individual_retry",
                )
            )

        except Exception as error:
            log_records.append(
                build_log_record(
                    ticker=ticker,
                    frame=pd.DataFrame(),
                    status="failed",
                    method="individual_retry",
                    error=str(error),
                )
            )

    log = pd.DataFrame(log_records)

    if not log.empty:
        # A successfully downloaded ticker should appear only once.
        log["status_order"] = log["status"].map(
            {"success": 0, "failed": 1}
        )

        log = (
            log.sort_values(
                ["ticker", "status_order"]
            )
            .drop_duplicates(
                subset=["ticker"],
                keep="first",
            )
            .drop(columns="status_order")
            .sort_values("ticker")
            .reset_index(drop=True)
        )

    log.to_csv(
        DOWNLOAD_LOG_FILE,
        index=False,
    )

    successful = int((log["status"] == "success").sum())
    failed = int((log["status"] == "failed").sum())

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful tickers : {successful}")
    print(f"Failed tickers     : {failed}")
    print(f"Recovered on retry: {len(recovered_tickers)}")
    print(f"Price files       : {RAW_DATA_DIR}")
    print(f"Download log      : {DOWNLOAD_LOG_FILE}")

    if failed:
        print("\nFailed symbols:")
        print(
            log.loc[
                log["status"] == "failed",
                ["ticker", "error"],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()