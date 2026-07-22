import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    COINTEGRATION_RESULTS_FILE,
    FDR_THRESHOLD,
    MAXIMUM_HALF_LIFE,
    MAXIMUM_HEDGE_RATIO,
    METADATA_DIR,
    MINIMUM_HALF_LIFE,
    MINIMUM_HEDGE_RATIO,
    MINIMUM_SPREAD_STANDARD_DEVIATION,
    CONSTITUENTS_FILE,
)


CANDIDATE_CAPS = [10, 20, 30, 50]

SENSITIVITY_SUMMARY_FILE = (
    METADATA_DIR
    / "cointegration_cap_sensitivity.csv"
)

SENSITIVITY_YEARLY_FILE = (
    METADATA_DIR
    / "cointegration_cap_yearly.csv"
)

PAIR_RECURRENCE_FILE = (
    METADATA_DIR
    / "qualified_pair_recurrence.csv"
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
    test_count = len(ordered)

    ranks = np.arange(
        1,
        test_count + 1,
    )

    raw_adjusted = (
        ordered.to_numpy()
        * test_count
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

def remove_same_issuer_pairs(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove pairs representing multiple share classes
    of the same SEC registrant.
    """

    constituents = pd.read_csv(
        CONSTITUENTS_FILE,
        dtype={
            "official_ticker": str,
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

    checked = results.copy()

    checked["issuer_1"] = (
        checked["ticker_1"].map(issuer_map)
    )

    checked["issuer_2"] = (
        checked["ticker_2"].map(issuer_map)
    )

    checked["same_issuer"] = (
        checked["issuer_1"].notna()
        & checked["issuer_2"].notna()
        & (
            checked["issuer_1"]
            == checked["issuer_2"]
        )
    )

    excluded = checked.loc[
        checked["same_issuer"]
    ].copy()

    filtered = checked.loc[
        ~checked["same_issuer"]
    ].copy()

    return filtered, excluded

def add_candidate_ranks(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the original within-sector candidate ranking.

    Same-sub-industry candidates are prioritized, followed
    by correlation and available formation observations.
    """

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


def qualify_for_cap(
    ranked: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:
    """Recompute FDR and qualifications for one cap."""

    subset = ranked.loc[
        ranked["sector_candidate_rank"] <= cap
    ].copy()

    subset["sensitivity_fdr_pvalue"] = (
        subset
        .groupby("selection_date")[
            "conservative_pvalue"
        ]
        .transform(benjamini_hochberg)
    )

    subset["sensitivity_qualified"] = (
        subset["test_status"].eq("success")
        & (
            subset["sensitivity_fdr_pvalue"]
            <= FDR_THRESHOLD
        )
        & subset["half_life"].between(
            MINIMUM_HALF_LIFE,
            MAXIMUM_HALF_LIFE,
            inclusive="both",
        )
        & subset["hedge_ratio"].between(
            MINIMUM_HEDGE_RATIO,
            MAXIMUM_HEDGE_RATIO,
            inclusive="both",
        )
        & (
            subset["spread_std"]
            >= MINIMUM_SPREAD_STANDARD_DEVIATION
        )
    )

    subset["candidate_cap"] = cap

    return subset


def summarize_cap(
    subset: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    cap: int,
) -> dict:
    """Summarize one candidate-cap experiment."""

    qualified = subset.loc[
        subset["sensitivity_qualified"]
    ]

    per_date = (
        qualified
        .groupby("selection_date")
        .size()
        .reindex(
            all_dates,
            fill_value=0,
        )
    )

    recurring_pairs = (
        qualified
        .groupby(["ticker_1", "ticker_2"])
        .size()
    )

    return {
        "candidate_cap": cap,
        "tests": len(subset),
        "qualified_snapshots": len(qualified),
        "qualification_rate": (
            len(qualified) / len(subset)
            if len(subset)
            else np.nan
        ),
        "months_total": len(all_dates),
        "months_with_pairs": int(
            (per_date > 0).sum()
        ),
        "months_without_pairs": int(
            (per_date == 0).sum()
        ),
        "coverage_fraction": float(
            (per_date > 0).mean()
        ),
        "median_pairs_all_months": float(
            per_date.median()
        ),
        "mean_pairs_all_months": float(
            per_date.mean()
        ),
        "maximum_pairs_month": int(
            per_date.max()
        ),
        "unique_qualified_pairs": int(
            qualified[
                ["ticker_1", "ticker_2"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "pairs_qualified_2plus_months": int(
            (recurring_pairs >= 2).sum()
        ),
        "pairs_qualified_3plus_months": int(
            (recurring_pairs >= 3).sum()
        ),
        "median_half_life": (
            qualified["half_life"].median()
            if not qualified.empty
            else np.nan
        ),
        "median_fdr_pvalue": (
            qualified[
                "sensitivity_fdr_pvalue"
            ].median()
            if not qualified.empty
            else np.nan
        ),
    }


def build_yearly_summary(
    subset: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    cap: int,
) -> pd.DataFrame:
    """Build annual qualification coverage."""

    qualified = subset.loc[
        subset["sensitivity_qualified"]
    ]

    monthly_series = (
        qualified
        .groupby("selection_date")
        .size()
        .reindex(
            all_dates,
            fill_value=0,
        )
        .rename("qualified_pairs")
    )

    monthly_series.index.name = "selection_date"

    monthly = monthly_series.reset_index()

    monthly["year"] = (
        monthly["selection_date"].dt.year
    )

    yearly = (
        monthly
        .groupby("year")
        .agg(
            months=("selection_date", "size"),
            months_with_pairs=(
                "qualified_pairs",
                lambda values: (values > 0).sum(),
            ),
            qualified_snapshots=(
                "qualified_pairs",
                "sum",
            ),
            median_pairs=(
                "qualified_pairs",
                "median",
            ),
            maximum_pairs=(
                "qualified_pairs",
                "max",
            ),
        )
        .reset_index()
    )

    yearly["candidate_cap"] = cap

    yearly["coverage_fraction"] = (
        yearly["months_with_pairs"]
        / yearly["months"]
    )

    return yearly


def build_recurrence_table(
    subset: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:
    """Summarize repeated qualification of the same pair."""

    qualified = subset.loc[
        subset["sensitivity_qualified"]
    ].copy()

    if qualified.empty:
        return pd.DataFrame()

    recurrence = (
        qualified
        .groupby(
            [
                "ticker_1",
                "ticker_2",
                "sector",
            ]
        )
        .agg(
            qualified_months=(
                "selection_date",
                "nunique",
            ),
            first_qualified=(
                "selection_date",
                "min",
            ),
            last_qualified=(
                "selection_date",
                "max",
            ),
            median_half_life=(
                "half_life",
                "median",
            ),
            median_correlation=(
                "return_correlation",
                "median",
            ),
            median_fdr_pvalue=(
                "sensitivity_fdr_pvalue",
                "median",
            ),
        )
        .reset_index()
    )

    recurrence["candidate_cap"] = cap

    return recurrence


def main() -> None:
    results = pd.read_parquet(
        COINTEGRATION_RESULTS_FILE
    )

    results["selection_date"] = pd.to_datetime(
        results["selection_date"]
    )

    results, excluded_same_issuer = (
        remove_same_issuer_pairs(results)
    )

    ranked = add_candidate_ranks(results)

    print(
        f"Same-issuer snapshots excluded: "
        f"{len(excluded_same_issuer):,}"
    )

    if not excluded_same_issuer.empty:
        print("\nExcluded same-issuer pairs:")

        excluded_pairs = (
            excluded_same_issuer[
                ["ticker_1", "ticker_2"]
            ]
            .drop_duplicates()
            .sort_values(
                ["ticker_1", "ticker_2"]
            )
        )

        print(
            excluded_pairs.to_string(index=False)
        )

        print()

    all_dates = pd.DatetimeIndex(
        sorted(
            results["selection_date"].unique()
        ),
        name="selection_date",
    )

    summary_records = []
    yearly_tables = []
    recurrence_tables = []

    print("=" * 70)
    print("COINTEGRATION CANDIDATE-CAP SENSITIVITY")
    print("=" * 70)

    for cap in CANDIDATE_CAPS:
        subset = qualify_for_cap(
            ranked,
            cap,
        )

        summary_records.append(
            summarize_cap(
                subset,
                all_dates,
                cap,
            )
        )

        yearly_tables.append(
            build_yearly_summary(
                subset,
                all_dates,
                cap,
            )
        )

        recurrence = build_recurrence_table(
            subset,
            cap,
        )

        if not recurrence.empty:
            recurrence_tables.append(
                recurrence
            )

    summary = pd.DataFrame(
        summary_records
    )

    yearly = pd.concat(
        yearly_tables,
        ignore_index=True,
    )

    recurrence = pd.concat(
        recurrence_tables,
        ignore_index=True,
    )

    summary.to_csv(
        SENSITIVITY_SUMMARY_FILE,
        index=False,
    )

    yearly.to_csv(
        SENSITIVITY_YEARLY_FILE,
        index=False,
    )

    recurrence = recurrence.sort_values(
        [
            "candidate_cap",
            "qualified_months",
            "median_fdr_pvalue",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    recurrence.to_csv(
        PAIR_RECURRENCE_FILE,
        index=False,
    )

    display_columns = [
        "candidate_cap",
        "tests",
        "qualified_snapshots",
        "qualification_rate",
        "months_with_pairs",
        "months_without_pairs",
        "coverage_fraction",
        "median_pairs_all_months",
        "mean_pairs_all_months",
        "unique_qualified_pairs",
        "pairs_qualified_2plus_months",
        "pairs_qualified_3plus_months",
    ]

    print(
        summary[display_columns]
        .to_string(index=False)
    )

    print("\nMost recurrent pairs under each cap:")

    for cap in CANDIDATE_CAPS:
        cap_recurrence = recurrence.loc[
            recurrence["candidate_cap"] == cap
        ].head(10)

        print(f"\nCandidate cap = {cap}")

        if cap_recurrence.empty:
            print("No qualified pairs")
        else:
            print(
                cap_recurrence[
                    [
                        "ticker_1",
                        "ticker_2",
                        "sector",
                        "qualified_months",
                        "first_qualified",
                        "last_qualified",
                        "median_half_life",
                        "median_fdr_pvalue",
                    ]
                ].to_string(index=False)
            )

    print("\nSaved:")
    print(SENSITIVITY_SUMMARY_FILE)
    print(SENSITIVITY_YEARLY_FILE)
    print(PAIR_RECURRENCE_FILE)


if __name__ == "__main__":
    main()