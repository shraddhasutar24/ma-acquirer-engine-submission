import math

import pandas as pd

from config import TARGET_PROFILE
from data_loader import load_transactions


def calculate_size_similarity(deal_size, target_size):
    """Compare enterprise values proportionally on a 0–1 scale."""
    if pd.isna(target_size) or target_size <= 0:
        raise ValueError("Target enterprise value must be positive.")

    if pd.isna(deal_size) or deal_size <= 0:
        return 0.0

    return math.exp(
        -abs(math.log(deal_size / target_size))
    )


def calculate_sector_similarities(df, target_sector):
    """
    Calculate sector relevance using historical buyer overlap.

    Jaccard similarity:
        shared buyers / unique buyers across both sectors
    """
    target_acquirers = set(
        df.loc[
            df["sector"] == target_sector,
            "acquirer",
        ].dropna()
    )

    if not target_acquirers:
        raise ValueError(
            f"No transactions available for target sector: "
            f"{target_sector}"
        )

    sector_similarities = {}

    for sector in df["sector"].dropna().unique():
        if sector == target_sector:
            sector_similarities[sector] = 1.0
            continue

        sector_acquirers = set(
            df.loc[
                df["sector"] == sector,
                "acquirer",
            ].dropna()
        )

        shared_acquirers = (
            target_acquirers & sector_acquirers
        )
        all_acquirers = (
            target_acquirers | sector_acquirers
        )

        sector_similarities[sector] = (
            len(shared_acquirers) / len(all_acquirers)
            if all_acquirers
            else 0.0
        )

    return sector_similarities


def derive_target_margin(df, target_sector, ebitda_profile):
    """
    Represent 'Strong' using the target sector's 75th-percentile
    historical EBITDA margin.

    This is a modeling proxy, not the target's actual margin.
    """
    sector_margins = df.loc[
        df["sector"] == target_sector,
        "ebitda_margin_pct",
    ].dropna()

    if sector_margins.empty:
        raise ValueError(
            f"No EBITDA margin data available for sector: "
            f"{target_sector}"
        )

    if ebitda_profile.lower() == "strong":
        return sector_margins.quantile(0.75)

    raise ValueError(
        f"Unsupported EBITDA profile: {ebitda_profile}"
    )


def calculate_margin_similarity(margin, target_margin):
    """Score EBITDA-margin similarity on a 0–1 scale."""
    if pd.isna(margin):
        return 0.0

    difference = abs(margin - target_margin)
    similarity = 1 - (difference / target_margin)

    return max(0.0, similarity)


def calculate_outcome_score(outcome):
    """Only closed transactions count as completed precedent."""
    return 1.0 if outcome == "Closed" else 0.0


def score_transactions(df, target_profile):
    """
    Score transactions using:
        50% sector relevance
        25% enterprise-value similarity
        25% EBITDA-margin similarity
    """
    scored_df = df.copy()

    sector_similarities = calculate_sector_similarities(
        scored_df,
        target_profile["sector"],
    )

    target_margin = derive_target_margin(
        scored_df,
        target_profile["sector"],
        target_profile["ebitda_profile"],
    )

    scored_df["sector_score"] = (
        scored_df["sector"]
        .map(sector_similarities)
        .fillna(0.0)
    )

    scored_df["size_score"] = (
        scored_df["deal_size_mm"].apply(
            lambda value: calculate_size_similarity(
                value,
                target_profile["deal_size_mm"],
            )
        )
    )

    scored_df["margin_score"] = (
        scored_df["ebitda_margin_pct"].apply(
            lambda value: calculate_margin_similarity(
                value,
                target_margin,
            )
        )
    )

    scored_df["outcome_score"] = (
        scored_df["outcome"].apply(
            calculate_outcome_score
        )
    )

    scored_df["relevance_score"] = (
        0.50 * scored_df["sector_score"]
        + 0.25 * scored_df["size_score"]
        + 0.25 * scored_df["margin_score"]
    )

    scored_df["evidence_score"] = (
        scored_df["relevance_score"]
        * scored_df["outcome_score"]
    )

    return scored_df


def rank_acquirers(
    scored_df,
    top_n=10,
    shrinkage_strength=2.0,
):
    """
    Rank buyers by top-three fit, adjusted for evidence depth.

    Buyer fit:
        Mean score of up to three strongest closed precedents.

    Evidence depth:
        Sum of sector relevance across closed buyer deals,
        capped at six equivalent exact-sector deals.

    Prior:
        Average top-three fit across buyers.

    Thin histories receive more weight on the prior.
    """
    closed_deals = scored_df[
        scored_df["outcome_score"] == 1.0
    ].copy()

    if closed_deals.empty:
        return pd.DataFrame(
            columns=[
                "acquirer",
                "mean_precedent_score",
                "best_precedent_score",
                "precedent_count",
                "evidence_depth",
                "reliability_weight",
                "prior_score",
                "acquirer_score",
            ]
        )

    top_precedents = (
        closed_deals
        .sort_values(
            ["evidence_score", "transaction_id"],
            ascending=[False, True],
        )
        .groupby("acquirer")
        .head(3)
    )

    summary = (
        top_precedents
        .groupby("acquirer")
        .agg(
            mean_precedent_score=(
                "evidence_score",
                "mean",
            ),
            best_precedent_score=(
                "evidence_score",
                "max",
            ),
            precedent_count=(
                "transaction_id",
                "count",
            ),
        )
    )

    # Exact-sector deals contribute 1; other sectors contribute
    # their buyer-overlap score. Cap depth to limit volume dominance.
    summary["evidence_depth"] = (
        closed_deals
        .groupby("acquirer")["sector_score"]
        .sum()
        .clip(upper=6.0)
    )

    # Average across buyers, rather than transactions.
    prior_score = summary["mean_precedent_score"].mean()

    summary["prior_score"] = prior_score

    summary["reliability_weight"] = (
        summary["evidence_depth"]
        / (
            summary["evidence_depth"]
            + shrinkage_strength
        )
    )

    summary["acquirer_score"] = (
        summary["reliability_weight"]
        * summary["mean_precedent_score"]
        + (
            1 - summary["reliability_weight"]
        )
        * summary["prior_score"]
    )

    return (
        summary
        .reset_index()
        .sort_values(
            ["acquirer_score", "acquirer"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    transactions = load_transactions()

    scored_transactions = score_transactions(
        transactions,
        TARGET_PROFILE,
    )

    top_acquirers = rank_acquirers(
        scored_transactions,
        top_n=10,
    )

    print("\nTop 10 candidate acquirers:")
    print(top_acquirers.to_string(index=False))