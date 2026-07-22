import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CANDIDATE_PAIRS_FILE,
    CANDIDATE_SUMMARY_FILE,
    CLOSE_PANEL_FILE,
    CONSTITUENTS_FILE,
    ELIGIBILITY_PANEL_FILE,
    FORMATION_WINDOW,
    INCLUDE_PARTIAL_FINAL_MONTH,
    MAX_PARTNERS_PER_STOCK,
    MINIMUM_PAIR_OBSERVATIONS,
    MINIMUM_RETURN_CORRELATION,
)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load price, eligibility, and sector information."""

    close = pd.read_parquet(
        CLOSE_PANEL_FILE
    )

    eligibility = pd.read_parquet(
        ELIGIBILITY_PANEL_FILE
    )

    constituents = pd.read_csv(
        CONSTITUENTS_FILE
    )

    close.index = pd.to_datetime(close.index)
    eligibility.index = pd.to_datetime(
        eligibility.index
    )

    close.columns.name = "ticker"
    eligibility.columns.name = "ticker"

    if not close.index.equals(eligibility.index):
        raise ValueError(
            "Close and eligibility dates are not aligned"
        )

    if not close.columns.equals(eligibility.columns):
        raise ValueError(
            "Close and eligibility tickers are not aligned"
        )

    required_metadata = [
        "yahoo_ticker",
        "sector",
        "sub_industry",
    ]

    missing = [
        column
        for column in required_metadata
        if column not in constituents.columns
    ]

    if missing:
        raise ValueError(
            f"Constituent metadata missing: {missing}"
        )

    return close, eligibility, constituents


def get_month_end_dates(
    dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Return the last available trading session of each month.
    """

    date_series = pd.Series(
        dates,
        index=dates,
    )

    month_ends = (
        date_series
        .groupby(dates.to_period("M"))
        .max()
    )

    month_ends = pd.DatetimeIndex(
        month_ends.values
    )

    if (
        not INCLUDE_PARTIAL_FINAL_MONTH
        and len(month_ends) > 0
    ):
        month_ends = month_ends[:-1]

    return month_ends


def build_security_mappings(
    constituents: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """
    Build sector, sub-industry, and issuer mappings.

    CIK identifies the underlying SEC registrant and allows
    us to detect multiple share classes of the same company.
    """

    metadata = (
        constituents
        .drop_duplicates("yahoo_ticker")
        .set_index("yahoo_ticker")
    )

    sector_map = metadata["sector"].to_dict()

    sub_industry_map = (
        metadata["sub_industry"].to_dict()
    )

    issuer_map = (
        metadata["cik"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
        .to_dict()
    )

    return (
        sector_map,
        sub_industry_map,
        issuer_map,
    )


def generate_sector_candidates(
    selection_date: pd.Timestamp,
    sector: str,
    tickers: list[str],
    formation_returns: pd.DataFrame,
    sub_industry_map: dict[str, str],
    issuer_map: dict[str, str],
) -> list[dict]:
    """
    Generate unique correlation-filtered candidates
    for one sector and one selection date.
    """

    if len(tickers) < 2:
        return []

    sector_returns = formation_returns[
        tickers
    ].copy()

    correlation = sector_returns.corr(
        min_periods=MINIMUM_PAIR_OBSERVATIONS
    )

    valid = sector_returns.notna().astype(int)

    pair_observations = (
        valid.T @ valid
    )

    pairs: dict[tuple[str, str], dict] = {}

    for ticker_1 in tickers:
        correlations = (
            correlation[ticker_1]
            .drop(
                labels=[ticker_1],
                errors="ignore",
            )
            .dropna()
        )

        correlations = correlations.loc[
            correlations
            >= MINIMUM_RETURN_CORRELATION
        ]

        correlations = correlations.sort_values(
            ascending=False
        ).head(MAX_PARTNERS_PER_STOCK)

        for ticker_2, corr_value in correlations.items():
            pair = tuple(
                sorted([ticker_1, ticker_2])
            )
            issuer_1 = issuer_map.get(pair[0])
            issuer_2 = issuer_map.get(pair[1])

            same_issuer = (
                issuer_1 is not None
                and issuer_2 is not None
                and issuer_1 == issuer_2
            )

            # Multiple share classes of the same company
            # are excluded from the main pairs strategy.
            if same_issuer:
                continue

            observations = int(
                pair_observations.loc[
                    ticker_1,
                    ticker_2,
                ]
            )

            if observations < MINIMUM_PAIR_OBSERVATIONS:
                continue

            same_sub_industry = (
                sub_industry_map.get(pair[0])
                == sub_industry_map.get(pair[1])
            )

            record = {
                "selection_date": selection_date,
                "ticker_1": pair[0],
                "ticker_2": pair[1],
                "sector": sector,
                "sub_industry_1": (
                    sub_industry_map.get(pair[0])
                ),
                "sub_industry_2": (
                    sub_industry_map.get(pair[1])
                ),
                "same_sub_industry": same_sub_industry,
                "return_correlation": float(
                    corr_value
                ),
                "pair_observations": observations,
            }

            # The pair may be selected from both directions.
            # Retain the identical unique pair once.
            if pair not in pairs:
                pairs[pair] = record

    return list(pairs.values())


def generate_candidates(
    close: pd.DataFrame,
    eligibility: pd.DataFrame,
    constituents: pd.DataFrame,
) -> pd.DataFrame:
    """Generate candidates across all month-end dates."""

    (
        sector_map,
        sub_industry_map,
        issuer_map,
    ) = build_security_mappings(constituents)

    month_end_dates = get_month_end_dates(
        close.index
    )

    records = []

    print("=" * 70)
    print("GENERATE CANDIDATE PAIRS")
    print("=" * 70)
    print(
        f"Selection dates      : {len(month_end_dates)}"
    )
    print(
        f"Formation window     : {FORMATION_WINDOW} returns"
    )
    print(
        f"Minimum observations : "
        f"{MINIMUM_PAIR_OBSERVATIONS}"
    )
    print(
        f"Minimum correlation  : "
        f"{MINIMUM_RETURN_CORRELATION:.2f}"
    )
    print(
        f"Maximum partners     : "
        f"{MAX_PARTNERS_PER_STOCK}"
    )
    print()

    processed_dates = 0

    for selection_date in month_end_dates:
        location = close.index.get_loc(
            selection_date
        )

        # We need FORMATION_WINDOW returns, which requires
        # FORMATION_WINDOW + 1 price observations.
        if location < FORMATION_WINDOW:
            continue

        eligible_tickers = eligibility.columns[
            eligibility.loc[selection_date]
        ].tolist()

        if len(eligible_tickers) < 2:
            continue

        formation_prices = close.iloc[
            location - FORMATION_WINDOW:
            location + 1
        ][eligible_tickers]

        formation_prices = formation_prices.where(
            formation_prices > 0
        )

        formation_returns = np.log(
            formation_prices
        ).diff()

        tickers_by_sector: dict[str, list[str]] = {}

        for ticker in eligible_tickers:
            sector = sector_map.get(ticker)

            if pd.isna(sector) or sector is None:
                continue

            tickers_by_sector.setdefault(
                sector,
                [],
            ).append(ticker)

        date_records = []

        for sector, sector_tickers in (
            tickers_by_sector.items()
        ):
            sector_candidates = (
                generate_sector_candidates(
                    selection_date=selection_date,
                    sector=sector,
                    tickers=sector_tickers,
                    formation_returns=formation_returns,
                    sub_industry_map=sub_industry_map,
                    issuer_map=issuer_map,
                )
            )

            date_records.extend(
                sector_candidates
            )

        records.extend(date_records)
        processed_dates += 1

        if (
            processed_dates % 12 == 0
            or selection_date == month_end_dates[-1]
        ):
            print(
                f"{selection_date.date()} | "
                f"eligible={len(eligible_tickers):3d} | "
                f"candidates={len(date_records):4d}"
            )

    candidates = pd.DataFrame(records)

    if candidates.empty:
        raise RuntimeError(
            "No candidate pairs were generated"
        )

    candidates["selection_date"] = pd.to_datetime(
        candidates["selection_date"]
    )

    candidates = candidates.sort_values(
        [
            "selection_date",
            "sector",
            "return_correlation",
            "ticker_1",
            "ticker_2",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    duplicate_count = candidates.duplicated(
        subset=[
            "selection_date",
            "ticker_1",
            "ticker_2",
        ]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"Generated {duplicate_count} duplicate pairs"
        )

    return candidates


def build_summary(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate counts by selection date."""

    summary = (
        candidates
        .groupby("selection_date")
        .agg(
            candidate_pairs=(
                "return_correlation",
                "size",
            ),
            sectors=(
                "sector",
                "nunique",
            ),
            median_correlation=(
                "return_correlation",
                "median",
            ),
            minimum_correlation=(
                "return_correlation",
                "min",
            ),
            maximum_correlation=(
                "return_correlation",
                "max",
            ),
            same_sub_industry_pairs=(
                "same_sub_industry",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary


def main() -> None:
    close, eligibility, constituents = load_inputs()

    candidates = generate_candidates(
        close,
        eligibility,
        constituents,
    )

    summary = build_summary(candidates)

    candidates.to_parquet(
        CANDIDATE_PAIRS_FILE,
        engine="pyarrow",
        index=False,
    )

    summary.to_csv(
        CANDIDATE_SUMMARY_FILE,
        index=False,
    )

    pairs_per_date = candidates.groupby(
        "selection_date"
    ).size()

    print("\n" + "=" * 70)
    print("CANDIDATE SUMMARY")
    print("=" * 70)
    print(
        f"Selection dates       : "
        f"{candidates['selection_date'].nunique()}"
    )
    print(
        f"Total pair snapshots  : {len(candidates):,}"
    )
    print(
        f"Unique pairs overall  : "
        f"{candidates[['ticker_1', 'ticker_2']].drop_duplicates().shape[0]:,}"
    )
    print(
        f"Median pairs/date     : "
        f"{pairs_per_date.median():.0f}"
    )
    print(
        f"Minimum pairs/date    : "
        f"{pairs_per_date.min()}"
    )
    print(
        f"Maximum pairs/date    : "
        f"{pairs_per_date.max()}"
    )
    print(
        f"Median correlation    : "
        f"{candidates['return_correlation'].median():.3f}"
    )
    print(
        f"Same sub-industry     : "
        f"{candidates['same_sub_industry'].mean():.1%}"
    )
    print(
        f"Candidates saved      : "
        f"{CANDIDATE_PAIRS_FILE}"
    )
    print(
        f"Summary saved         : "
        f"{CANDIDATE_SUMMARY_FILE}"
    )

    print("\nCandidates by sector:")

    sector_summary = (
        candidates
        .groupby("sector")
        .agg(
            snapshots=(
                "return_correlation",
                "size",
            ),
            unique_pairs=(
                "ticker_1",
                lambda series: 0,
            ),
            median_correlation=(
                "return_correlation",
                "median",
            ),
        )
    )

    # Calculate unique pair count separately.
    unique_sector_pairs = (
        candidates
        .drop_duplicates(
            ["sector", "ticker_1", "ticker_2"]
        )
        .groupby("sector")
        .size()
    )

    sector_summary["unique_pairs"] = (
        unique_sector_pairs
    )

    print(
        sector_summary
        .sort_values(
            "snapshots",
            ascending=False,
        )
        .to_string()
    )


if __name__ == "__main__":
    main()