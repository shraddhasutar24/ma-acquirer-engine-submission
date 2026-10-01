"""Compare ranking methods using strictly prior-year history."""

import json
from pathlib import Path

import pandas as pd

from data_loader import load_transactions
from scoring import (
    calculate_margin_similarity,
    calculate_sector_similarities,
    calculate_size_similarity,
    rank_acquirers,
)


METHODS = [
    "random_expected",
    "prior_closed_count",
    "sector_weighted_count",
    "original",
    "log_size_only",
    "shrinkage_only",
    "log_size_shrinkage",
]


def score_historical_transactions(
    history_df,
    target_row,
    log_size=True,
):
    """Score prior closed deals against one held-out target."""
    sector_similarities = calculate_sector_similarities(
        history_df,
        target_row["sector"],
    )

    scored = history_df[
        history_df["outcome"] == "Closed"
    ].copy()

    scored["sector_score"] = (
        scored["sector"]
        .map(sector_similarities)
        .fillna(0.0)
    )

    if log_size:
        scored["size_score"] = scored["deal_size_mm"].apply(
            lambda value: calculate_size_similarity(
                value,
                target_row["deal_size_mm"],
            )
        )
    else:
        # Original linear-size formula for comparison.
        scored["size_score"] = (
            1
            - (
                scored["deal_size_mm"]
                - target_row["deal_size_mm"]
            ).abs()
            / target_row["deal_size_mm"]
        ).clip(lower=0.0)

    scored["margin_score"] = scored["ebitda_margin_pct"].apply(
        lambda value: calculate_margin_similarity(
            value,
            target_row["ebitda_margin_pct"],
        )
    )

    scored["relevance_score"] = (
        0.50 * scored["sector_score"]
        + 0.25 * scored["size_score"]
        + 0.25 * scored["margin_score"]
    )

    scored["evidence_score"] = scored["relevance_score"]
    scored["outcome_score"] = 1.0

    return scored


def calculate_buyer_scores(scored_df, method):
    """Produce one score per buyer for the chosen method."""
    if method == "prior_closed_count":
        return scored_df.groupby("acquirer").size()

    if method == "sector_weighted_count":
        # Exact-sector deals count 1; other sectors receive
        # their historical buyer-overlap weight.
        return (
            scored_df.groupby("acquirer")["sector_score"]
            .sum()
        )

    if method in [
        "shrinkage_only",
        "log_size_shrinkage",
    ]:
        rankings = rank_acquirers(
            scored_df,
            top_n=scored_df["acquirer"].nunique(),
        )

        return (
            rankings.set_index("acquirer")["acquirer_score"]
        )

    if method not in ["original", "log_size_only"]:
        raise ValueError(f"Unsupported method: {method}")

    top_three = (
        scored_df
        .sort_values(
            ["relevance_score", "transaction_id"],
            ascending=[False, True],
        )
        .groupby("acquirer")
        .head(3)
    )

    summary = (
        top_three
        .groupby("acquirer")["relevance_score"]
        .agg(["mean", "count"])
    )

    # Original top-three mean and repeat-evidence bonus.
    return (
        summary["mean"]
        * (
            1
            + 0.025 * (summary["count"] - 1)
        )
    ).clip(upper=1.0)


def evaluate_all_methods(df, start_year=2018):
    """
    Evaluate every method on the same eligible closed deals.

    History contains only deals from earlier years.

    Eligibility requires prior history for the observed buyer
    and target sector. Buyers with only non-closed history
    remain eligible but cannot be recommended by any method.
    """
    cutoffs = [1, 5, 10]

    totals = {
        method: {
            cutoff: 0.0
            for cutoff in cutoffs
        }
        for method in METHODS
    }

    eligible_tests = 0
    excluded_tests = 0

    test_deals = df[
        (df["deal_year"] >= start_year)
        & (df["outcome"] == "Closed")
    ]

    for _, target_row in test_deals.iterrows():
        history_df = df[
            df["deal_year"] < target_row["deal_year"]
        ].copy()

        buyer_has_history = (
            target_row["acquirer"]
            in set(history_df["acquirer"])
        )

        sector_has_history = (
            target_row["sector"]
            in set(history_df["sector"])
        )

        if not buyer_has_history or not sector_has_history:
            excluded_tests += 1
            continue

        eligible_tests += 1

        linear_scores = score_historical_transactions(
            history_df,
            target_row,
            log_size=False,
        )

        log_scores = score_historical_transactions(
            history_df,
            target_row,
            log_size=True,
        )

        # Identical candidate universe for every method:
        # buyers with at least one prior closed transaction.
        candidates = set(linear_scores["acquirer"])
        actual_buyer = target_row["acquirer"]

        for method in METHODS:
            if method == "random_expected":
                # Exact expected hit rate for a uniformly
                # random shortlist; no random simulation.
                for cutoff in cutoffs:
                    probability = (
                        min(cutoff / len(candidates), 1.0)
                        if actual_buyer in candidates
                        else 0.0
                    )

                    totals[method][cutoff] += probability

                continue

            scored_df = (
                log_scores
                if method in [
                    "log_size_only",
                    "log_size_shrinkage",
                ]
                else linear_scores
            )

            buyer_scores = calculate_buyer_scores(
                scored_df,
                method,
            )

            ranked_buyers = (
                buyer_scores
                .rename("score")
                .reset_index()
                .sort_values(
                    ["score", "acquirer"],
                    ascending=[False, True],
                )["acquirer"]
                .tolist()
            )

            for cutoff in cutoffs:
                totals[method][cutoff] += (
                    actual_buyer
                    in ranked_buyers[:cutoff]
                )

    results = []

    for method in METHODS:
        result = {
            "method": method,
            "eligible_tests": eligible_tests,
            "excluded_tests": excluded_tests,
        }

        for cutoff in cutoffs:
            result[f"hit_at_{cutoff}"] = (
                totals[method][cutoff] / eligible_tests
                if eligible_tests
                else 0.0
            )

        results.append(result)

    return results


if __name__ == "__main__":
    transactions = load_transactions()

    results = evaluate_all_methods(transactions)

    project_root = Path(__file__).resolve().parent.parent
    output_path = (
        project_root
        / "outputs"
        / "validation_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    results_table = pd.DataFrame(results)

    for cutoff in [1, 5, 10]:
        column = f"hit_at_{cutoff}"
        results_table[column] = (
            results_table[column] * 100
        ).round(1)

    print("\nHistorical validation results (%):")
    print(results_table.to_string(index=False))
    print(f"\nSaved results to: {output_path}")