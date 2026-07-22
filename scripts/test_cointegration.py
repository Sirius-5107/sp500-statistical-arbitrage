import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CANDIDATE_PAIRS_FILE,
    CLOSE_PANEL_FILE,
    COINTEGRATION_MAX_LAG,
    COINTEGRATION_RESULTS_FILE,
    COINTEGRATION_SUMMARY_FILE,
    FDR_THRESHOLD,
    FORMATION_WINDOW,
    MAXIMUM_HALF_LIFE,
    MAXIMUM_HEDGE_RATIO,
    MAX_CANDIDATES_PER_SECTOR,
    MINIMUM_HALF_LIFE,
    MINIMUM_HEDGE_RATIO,
    MINIMUM_PAIR_OBSERVATIONS,
    MINIMUM_SPREAD_STANDARD_DEVIATION,
    QUALIFIED_PAIRS_FILE,
)


def benjamini_hochberg(
    pvalues: pd.Series,
) -> pd.Series:
    """
    Calculate Benjamini-Hochberg adjusted p-values.

    Missing p-values remain missing.
    """

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

    # Enforce monotonic adjusted p-values.
    monotonic = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    monotonic = np.clip(
        monotonic,
        0.0,
        1.0,
    )

    adjusted.loc[ordered.index] = monotonic

    return adjusted


def select_test_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retain the strongest candidates within every
    date-sector group.
    """

    ranked = candidates.sort_values(
        [
            "selection_date",
            "sector",
            "same_sub_industry",
            "return_correlation",
            "pair_observations",
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
    )

    selected = (
        ranked
        .groupby(
            ["selection_date", "sector"],
            group_keys=False,
        )
        .head(MAX_CANDIDATES_PER_SECTOR)
        .reset_index(drop=True)
    )

    return selected


def estimate_ols_spread(
    dependent: np.ndarray,
    independent: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate:

        dependent = alpha + beta * independent + residual
    """

    design = np.column_stack(
        [
            np.ones(len(independent)),
            independent,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        dependent,
        rcond=None,
    )

    alpha = float(coefficients[0])
    beta = float(coefficients[1])

    spread = (
        dependent
        - alpha
        - beta * independent
    )

    return alpha, beta, spread


def estimate_half_life(
    spread: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate spread mean-reversion half-life using:

        delta_spread = a + b * lagged_spread

        half_life = -log(2) / b
    """

    spread = np.asarray(
        spread,
        dtype=float,
    )

    lagged = spread[:-1]
    delta = np.diff(spread)

    valid = (
        np.isfinite(lagged)
        & np.isfinite(delta)
    )

    lagged = lagged[valid]
    delta = delta[valid]

    if len(lagged) < 20:
        return np.nan, np.nan

    design = np.column_stack(
        [
            np.ones(len(lagged)),
            lagged,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        delta,
        rcond=None,
    )

    mean_reversion_coefficient = float(
        coefficients[1]
    )

    if mean_reversion_coefficient >= 0:
        return np.inf, mean_reversion_coefficient

    half_life = (
        -np.log(2)
        / mean_reversion_coefficient
    )

    return float(half_life), mean_reversion_coefficient


def count_mean_crossings(
    spread: np.ndarray,
) -> int:
    """Count spread crossings around its formation mean."""

    centered = spread - np.nanmean(spread)

    signs = np.sign(centered)

    # Replace exact zeros with missing so they do not
    # create artificial double crossings.
    signs[signs == 0] = np.nan

    signs = pd.Series(signs).ffill().bfill().to_numpy()

    if len(signs) < 2:
        return 0

    return int(
        np.sum(signs[1:] != signs[:-1])
    )


def test_pair(
    candidate: pd.Series,
    close: pd.DataFrame,
) -> dict:
    """Run bidirectional cointegration tests on one pair."""

    selection_date = pd.Timestamp(
        candidate["selection_date"]
    )

    ticker_1 = candidate["ticker_1"]
    ticker_2 = candidate["ticker_2"]

    location = close.index.get_loc(
        selection_date
    )

    prices = close.iloc[
        location - FORMATION_WINDOW:
        location + 1
    ][[ticker_1, ticker_2]].dropna()

    prices = prices.loc[
        (prices[ticker_1] > 0)
        & (prices[ticker_2] > 0)
    ]

    observations = len(prices)

    base_result = {
        "selection_date": selection_date,
        "ticker_1": ticker_1,
        "ticker_2": ticker_2,
        "sector": candidate["sector"],
        "same_sub_industry": candidate[
            "same_sub_industry"
        ],
        "return_correlation": candidate[
            "return_correlation"
        ],
        "formation_observations": observations,
    }

    if observations < MINIMUM_PAIR_OBSERVATIONS:
        return {
            **base_result,
            "test_status": "insufficient_observations",
        }

    log_1 = np.log(
        prices[ticker_1].to_numpy(dtype=float)
    )

    log_2 = np.log(
        prices[ticker_2].to_numpy(dtype=float)
    )

    if (
        np.std(log_1) < 1e-10
        or np.std(log_2) < 1e-10
    ):
        return {
            **base_result,
            "test_status": "constant_price_series",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            statistic_1_on_2, pvalue_1_on_2, _ = coint(
                log_1,
                log_2,
                trend="c",
                maxlag=COINTEGRATION_MAX_LAG,
                autolag=None,
            )

            statistic_2_on_1, pvalue_2_on_1, _ = coint(
                log_2,
                log_1,
                trend="c",
                maxlag=COINTEGRATION_MAX_LAG,
                autolag=None,
            )

    except Exception as error:
        return {
            **base_result,
            "test_status": "test_failure",
            "error": str(error),
        }

    if not (
        np.isfinite(pvalue_1_on_2)
        and np.isfinite(pvalue_2_on_1)
    ):
        return {
            **base_result,
            "test_status": "invalid_pvalue",
        }

    # Conservative symmetric p-value:
    # the relationship must work in both directions.
    conservative_pvalue = max(
        pvalue_1_on_2,
        pvalue_2_on_1,
    )

    # Use the more stationary direction to construct
    # the eventual trading spread.
    if pvalue_1_on_2 <= pvalue_2_on_1:
        dependent_ticker = ticker_1
        independent_ticker = ticker_2

        dependent = log_1
        independent = log_2

        chosen_statistic = statistic_1_on_2
        chosen_pvalue = pvalue_1_on_2
        direction = "ticker_1_on_ticker_2"

    else:
        dependent_ticker = ticker_2
        independent_ticker = ticker_1

        dependent = log_2
        independent = log_1

        chosen_statistic = statistic_2_on_1
        chosen_pvalue = pvalue_2_on_1
        direction = "ticker_2_on_ticker_1"

    alpha, beta, spread = estimate_ols_spread(
        dependent,
        independent,
    )

    half_life, mean_reversion_coefficient = (
        estimate_half_life(spread)
    )

    spread_mean = float(
        np.mean(spread)
    )

    spread_std = float(
        np.std(spread, ddof=1)
    )

    crossings = count_mean_crossings(spread)

    return {
        **base_result,
        "test_status": "success",
        "pvalue_1_on_2": float(
            pvalue_1_on_2
        ),
        "pvalue_2_on_1": float(
            pvalue_2_on_1
        ),
        "statistic_1_on_2": float(
            statistic_1_on_2
        ),
        "statistic_2_on_1": float(
            statistic_2_on_1
        ),
        "conservative_pvalue": float(
            conservative_pvalue
        ),
        "chosen_pvalue": float(
            chosen_pvalue
        ),
        "chosen_statistic": float(
            chosen_statistic
        ),
        "spread_direction": direction,
        "dependent_ticker": dependent_ticker,
        "independent_ticker": independent_ticker,
        "alpha": alpha,
        "hedge_ratio": beta,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "half_life": half_life,
        "mean_reversion_coefficient": (
            mean_reversion_coefficient
        ),
        "mean_crossings": crossings,
        "test_error": "",
    }


def apply_qualification_rules(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate monthly FDR and qualification flags."""

    results = results.copy()

    results["fdr_pvalue"] = (
        results
        .groupby("selection_date")[
            "conservative_pvalue"
        ]
        .transform(benjamini_hochberg)
    )

    results["passes_fdr"] = (
        results["fdr_pvalue"]
        <= FDR_THRESHOLD
    )

    results["passes_half_life"] = (
        results["half_life"]
        .between(
            MINIMUM_HALF_LIFE,
            MAXIMUM_HALF_LIFE,
            inclusive="both",
        )
    )

    results["passes_hedge_ratio"] = (
        results["hedge_ratio"]
        .between(
            MINIMUM_HEDGE_RATIO,
            MAXIMUM_HEDGE_RATIO,
            inclusive="both",
        )
    )

    results["passes_spread_variation"] = (
        results["spread_std"]
        >= MINIMUM_SPREAD_STANDARD_DEVIATION
    )

    results["qualified"] = (
        results["test_status"].eq("success")
        & results["passes_fdr"].fillna(False)
        & results["passes_half_life"].fillna(False)
        & results["passes_hedge_ratio"].fillna(False)
        & results[
            "passes_spread_variation"
        ].fillna(False)
    )

    return results


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize monthly cointegration outcomes."""

    summary = (
        results
        .groupby("selection_date")
        .agg(
            pairs_tested=(
                "ticker_1",
                "size",
            ),
            successful_tests=(
                "test_status",
                lambda values: (
                    values == "success"
                ).sum(),
            ),
            qualified_pairs=(
                "qualified",
                "sum",
            ),
            median_conservative_pvalue=(
                "conservative_pvalue",
                "median",
            ),
            median_half_life=(
                "half_life",
                lambda values: (
                    values.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).median()
                ),
            ),
        )
        .reset_index()
    )

    summary["qualification_rate"] = (
        summary["qualified_pairs"]
        / summary["pairs_tested"]
    )

    return summary


def main() -> None:
    close = pd.read_parquet(
        CLOSE_PANEL_FILE
    )

    candidates = pd.read_parquet(
        CANDIDATE_PAIRS_FILE
    )

    close.index = pd.to_datetime(close.index)
    candidates["selection_date"] = pd.to_datetime(
        candidates["selection_date"]
    )

    selected = select_test_candidates(
        candidates
    )

    dates = sorted(
        selected["selection_date"].unique()
    )

    print("=" * 70)
    print("ROLLING COINTEGRATION TESTS")
    print("=" * 70)
    print(
        f"Original snapshots : {len(candidates):,}"
    )
    print(
        f"Selected for tests : {len(selected):,}"
    )
    print(
        f"Selection dates    : {len(dates)}"
    )
    print(
        f"Maximum per sector : "
        f"{MAX_CANDIDATES_PER_SECTOR}"
    )
    print(
        f"FDR threshold      : {FDR_THRESHOLD:.0%}"
    )
    print()

    records = []

    for date_number, selection_date in enumerate(
        dates,
        start=1,
    ):
        date_candidates = selected.loc[
            selected["selection_date"]
            == selection_date
        ]

        for _, candidate in date_candidates.iterrows():
            records.append(
                test_pair(
                    candidate,
                    close,
                )
            )

        if (
            date_number % 12 == 0
            or date_number == len(dates)
        ):
            print(
                f"{pd.Timestamp(selection_date).date()} | "
                f"dates={date_number}/{len(dates)} | "
                f"tests={len(records):,}"
            )

    results = pd.DataFrame(records)

    results = apply_qualification_rules(
        results
    )

    results = results.sort_values(
        [
            "selection_date",
            "qualified",
            "fdr_pvalue",
            "return_correlation",
        ],
        ascending=[
            True,
            False,
            True,
            False,
        ],
    ).reset_index(drop=True)

    qualified = results.loc[
        results["qualified"]
    ].copy()

    summary = build_summary(results)

    results.to_parquet(
        COINTEGRATION_RESULTS_FILE,
        engine="pyarrow",
        index=False,
    )

    qualified.to_parquet(
        QUALIFIED_PAIRS_FILE,
        engine="pyarrow",
        index=False,
    )

    summary.to_csv(
        COINTEGRATION_SUMMARY_FILE,
        index=False,
    )

    successful = int(
        results["test_status"].eq("success").sum()
    )

    print("\n" + "=" * 70)
    print("COINTEGRATION SUMMARY")
    print("=" * 70)
    print(f"Tests attempted   : {len(results):,}")
    print(f"Tests successful  : {successful:,}")
    print(f"Qualified snapshots: {len(qualified):,}")
    print(
        f"Qualification rate: "
        f"{len(qualified) / len(results):.2%}"
    )
    print(
        f"Dates with pairs  : "
        f"{qualified['selection_date'].nunique()}"
    )

    if not qualified.empty:
        qualified_per_date = qualified.groupby(
            "selection_date"
        ).size()

        print(
            f"Median qualified/date: "
            f"{qualified_per_date.median():.0f}"
        )
        print(
            f"Median half-life     : "
            f"{qualified['half_life'].median():.2f}"
        )
        print(
            f"Median hedge ratio   : "
            f"{qualified['hedge_ratio'].median():.3f}"
        )

        print("\nQualified pairs by sector:")
        print(
            qualified["sector"]
            .value_counts()
            .to_string()
        )

    print(
        f"\nAll results : "
        f"{COINTEGRATION_RESULTS_FILE}"
    )
    print(
        f"Qualified   : {QUALIFIED_PAIRS_FILE}"
    )
    print(
        f"Summary     : "
        f"{COINTEGRATION_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()