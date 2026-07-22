from io import StringIO
import sys
from pathlib import Path

import pandas as pd
import requests


# Allow scripts to import config.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CONSTITUENTS_FILE


WIKIPEDIA_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)


def download_constituents() -> pd.DataFrame:
    """
    Download the current S&P 500 constituent table from Wikipedia.

    Important:
    This is the current S&P 500 universe, not a historical
    point-in-time constituent dataset.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; AcademicResearchProject/1.0)"
        )
    }

    response = requests.get(
        WIKIPEDIA_URL,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise RuntimeError(
            "No HTML tables were found on the Wikipedia page."
        )

    constituents = tables[0].copy()

    required_columns = [
        "Symbol",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "Headquarters Location",
        "Date added",
        "CIK",
        "Founded",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in constituents.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Expected columns are missing: {missing_columns}"
        )

    constituents = constituents[required_columns].copy()

    constituents = constituents.rename(
        columns={
            "Symbol": "official_ticker",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters",
            "Date added": "date_added",
            "CIK": "cik",
            "Founded": "founded",
        }
    )

    # Yahoo Finance uses "-" where official tickers may use ".".
    # Example: BRK.B becomes BRK-B.
    constituents["yahoo_ticker"] = (
        constituents["official_ticker"]
        .str.strip()
        .str.replace(".", "-", regex=False)
    )

    constituents["date_added"] = pd.to_datetime(
        constituents["date_added"],
        errors="coerce",
    )

    constituents["cik"] = (
        constituents["cik"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )

    constituents["is_current_member"] = True

    constituents = constituents[
        [
            "official_ticker",
            "yahoo_ticker",
            "company",
            "sector",
            "sub_industry",
            "date_added",
            "cik",
            "headquarters",
            "founded",
            "is_current_member",
        ]
    ]

    constituents = constituents.sort_values(
        "yahoo_ticker"
    ).reset_index(drop=True)

    if constituents["yahoo_ticker"].duplicated().any():
        duplicates = constituents.loc[
            constituents["yahoo_ticker"].duplicated(
                keep=False
            ),
            "yahoo_ticker",
        ].tolist()

        raise RuntimeError(
            f"Duplicate Yahoo tickers found: {duplicates}"
        )

    return constituents


def main() -> None:
    constituents = download_constituents()

    constituents.to_csv(
        CONSTITUENTS_FILE,
        index=False,
    )

    print("=" * 60)
    print("S&P 500 CONSTITUENT DOWNLOAD")
    print("=" * 60)
    print(f"Companies retrieved : {len(constituents)}")
    print(
        f"Sectors retrieved   : "
        f"{constituents['sector'].nunique()}"
    )
    print(f"Saved to            : {CONSTITUENTS_FILE}")

    print("\nSector counts:")
    print(
        constituents["sector"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst five rows:")
    print(constituents.head().to_string(index=False))


if __name__ == "__main__":
    main()