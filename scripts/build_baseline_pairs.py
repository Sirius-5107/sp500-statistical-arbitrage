import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASELINE_CANDIDATE_CAP,
    BASELINE_PAIRS_FILE,
    BASELINE_PAIR_SUMMARY_FILE,
    COINTEGRATION_RESULTS_FILE,
    CONSTITUENTS_FILE,
    FDR_THRESHOLD,
    MAXIMUM_HALF_LIFE,
    MAXIMUM_HEDGE_RATIO,
    MINIMUM_HALF_LIFE,
    MINIMUM_HEDGE_RATIO,
    MINIMUM_SPREAD_STANDARD_DEVIATION,
)


def benjamini_hochberg(
    pvalues: pd.Series,
) -> pd.Series:
    """Calculate Benjamini-Hochberg adjusted p-values."""

    adjusted = pd.Series(
        np.nan,
        index=pvalues.index,
        dtype=float,
    )

    valid = pvalues.dropna()

    if valid.empty:
        return adjusted

    ordered = valid.sort_values()
    number_of_tests = len(ordered)

    ranks = np.arange(
        1,
        number_of_tests + 1,
    )

    raw_adjusted = (
        ordered.to_numpy()
        * number_of_tests
        / ranks
    )

    monotonic = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    adjusted.loc[ordered.index] = np.clip(
        monotonic,
        0.0,
        1.0,
    )

    return adjusted


def add_issuer_information(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Attach SEC CIK identifiers to both pair legs."""

    constituents = pd.read_csv(
        CONSTITUENTS_FILE,
        dtype={
            "yahoo_ticker": str,
            "cik": str,
        },
    )

    constituents["cik"] = (
        constituents["cik"]
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )

    issuer_map = (
        constituents
        .drop_duplicates("yahoo_ticker")
        .set_index("yahoo_ticker")["cik"]
        .to_dict()
    )

    results = results.copy()

    results["issuer_1"] = (
        results["ticker_1"].map(issuer_map)
    )

    results["issuer_2"] = (
        results["ticker_2"].map(issuer_map)
    )

    results["same_issuer"] = (
        results["issuer_1"].notna()
        & results["issuer_2"].notna()
        & (
            results["issuer_1"]
            == results["issuer_2"]
        )
    )

    return results


def rank_candidates(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Rank candidate pairs within each date and sector."""

    ranked = results.sort_values(
        [
            "selection_date",
            "sector",
            "same_sub_industry",
            "return_correlation",
            "formation_observations",
            "ticker_1",
            "ticker_2",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            True,
            True,
        ],
    ).copy()

    ranked["sector_candidate_rank"] = (
        ranked
        .groupby(
            ["selection_date", "sector"]
        )
        .cumcount()
        + 1
    )

    return ranked


def qualify_pairs(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the locked baseline qualification rules."""

    results = results.copy()

    results["baseline_fdr_pvalue"] = (
        results
        .groupby("selection_date")[
            "conservative_pvalue"
        ]
        .transform(benjamini_hochberg)
    )

    results["passes_fdr"] = (
        results["baseline_fdr_pvalue"]
        <= FDR_THRESHOLD
    )

    results["passes_half_life"] = (
        results["half_life"].between(
            MINIMUM_HALF_LIFE,
            MAXIMUM_HALF_LIFE,
            inclusive="both",
        )
    )

    results["passes_hedge_ratio"] = (
        results["hedge_ratio"].between(
            MINIMUM_HEDGE_RATIO,
            MAXIMUM_HEDGE_RATIO,
            inclusive="both",
        )
    )

    results["passes_spread_variation"] = (
        results["spread_std"]
        >= MINIMUM_SPREAD_STANDARD_DEVIATION
    )

    results["baseline_qualified"] = (
        results["test_status"].eq("success")
        & results["passes_fdr"].fillna(False)
        & results["passes_half_life"].fillna(False)
        & results["passes_hedge_ratio"].fillna(False)
        & results[
            "passes_spread_variation"
        ].fillna(False)
        & ~results["same_issuer"]
    )

    return results


def build_summary(
    baseline: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build monthly baseline-pair summary."""

    monthly = (
        baseline
        .groupby("selection_date")
        .agg(
            qualified_pairs=(
                "pair_id",
                "size",
            ),
            unique_stocks=(
                "ticker_1",
                lambda values: 0,
            ),
            sectors=(
                "sector",
                "nunique",
            ),
            median_correlation=(
                "return_correlation",
                "median",
            ),
            median_half_life=(
                "half_life",
                "median",
            ),
            median_fdr_pvalue=(
                "baseline_fdr_pvalue",
                "median",
            ),
        )
        .reindex(
            all_dates,
            fill_value=0,
        )
    )

    unique_stocks_by_date = (
        baseline
        .groupby("selection_date")
        .apply(
            lambda group: len(
                set(group["ticker_1"])
                | set(group["ticker_2"])
            ),
            include_groups=False,
        )
    )

    monthly.loc[
        unique_stocks_by_date.index,
        "unique_stocks",
    ] = unique_stocks_by_date

    monthly.index.name = "selection_date"

    return monthly.reset_index()


def main() -> None:
    results = pd.read_parquet(
        COINTEGRATION_RESULTS_FILE
    )

    results["selection_date"] = pd.to_datetime(
        results["selection_date"]
    )

    all_dates = pd.DatetimeIndex(
        sorted(
            results["selection_date"].unique()
        ),
        name="selection_date",
    )

    results = add_issuer_information(results)

    same_issuer_count = int(
        results["same_issuer"].sum()
    )

    # Same-issuer pairs must be removed before ranking because
    # they should not consume one of the 20 candidate slots.
    results = results.loc[
        ~results["same_issuer"]
    ].copy()

    ranked = rank_candidates(results)

    selected = ranked.loc[
        ranked["sector_candidate_rank"]
        <= BASELINE_CANDIDATE_CAP
    ].copy()

    selected = qualify_pairs(selected)

    baseline = selected.loc[
        selected["baseline_qualified"]
    ].copy()

    baseline["pair_id"] = (
        baseline["ticker_1"]
        + "__"
        + baseline["ticker_2"]
    )

    baseline = baseline.sort_values(
        [
            "selection_date",
            "baseline_fdr_pvalue",
            "sector",
            "pair_id",
        ]
    ).reset_index(drop=True)

    duplicate_count = baseline.duplicated(
        [
            "selection_date",
            "ticker_1",
            "ticker_2",
        ]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate snapshots"
        )

    if baseline["same_issuer"].any():
        raise RuntimeError(
            "Same-issuer pair entered baseline"
        )

    summary = build_summary(
        baseline,
        all_dates,
    )

    baseline.to_parquet(
        BASELINE_PAIRS_FILE,
        engine="pyarrow",
        index=False,
    )

    summary.to_csv(
        BASELINE_PAIR_SUMMARY_FILE,
        index=False,
    )

    per_month = (
        baseline
        .groupby("selection_date")
        .size()
        .reindex(
            all_dates,
            fill_value=0,
        )
    )

    pair_recurrence = (
        baseline
        .groupby("pair_id")
        .size()
    )

    print("=" * 70)
    print("LOCKED BASELINE PAIR UNIVERSE")
    print("=" * 70)
    print(
        f"Candidate cap          : "
        f"{BASELINE_CANDIDATE_CAP}"
    )
    print(
        f"Same-issuer excluded   : "
        f"{same_issuer_count}"
    )
    print(
        f"Candidate snapshots    : "
        f"{len(selected):,}"
    )
    print(
        f"Qualified snapshots    : "
        f"{len(baseline):,}"
    )
    print(
        f"Unique pairs           : "
        f"{baseline['pair_id'].nunique()}"
    )
    print(
        f"Months with pairs      : "
        f"{int((per_month > 0).sum())}"
    )
    print(
        f"Months without pairs   : "
        f"{int((per_month == 0).sum())}"
    )
    print(
        f"Median pairs/month     : "
        f"{per_month.median():.0f}"
    )
    print(
        f"Maximum pairs/month    : "
        f"{per_month.max()}"
    )
    print(
        f"Pairs recurring 2+     : "
        f"{int((pair_recurrence >= 2).sum())}"
    )
    print(
        f"Pairs recurring 3+     : "
        f"{int((pair_recurrence >= 3).sum())}"
    )
    print(
        f"Baseline pairs saved   : "
        f"{BASELINE_PAIRS_FILE}"
    )
    print(
        f"Summary saved          : "
        f"{BASELINE_PAIR_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()