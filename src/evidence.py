"""Build buyer evidence and target-relevant market comps."""

import json

import pandas as pd

from config import TARGET_PROFILE


PROVIDER_SECTORS = {
    "Healthcare Services",
    "Physician Groups",
    "Behavioral Health",
    "Home Health/Hospice",
    "Dental",
}

PRECEDENT_FIELDS = [
    "transaction_id",
    "target_company",
    "sector",
    "sub_sector",
    "deal_year",
    "deal_quarter",
    "deal_size_mm",
    "deal_type",
    "geography",
    "outcome",
    "num_bidders",
    "financing_type",
    "days_to_close",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "strategic_rationale_tags",
    "target_ownership_pre",
]


def clean_records(df, fields):
    """Convert missing values to JSON null, not NaN."""
    available = [
        field for field in fields
        if field in df.columns
    ]

    return json.loads(
        df[available].to_json(orient="records")
    )


def rounded_number(value):
    if pd.isna(value):
        return None

    return round(float(value), 1)


def relevant_sectors(target_sector):
    """Explicit comp-selection assumption, not ranking weights."""
    if target_sector in PROVIDER_SECTORS:
        return PROVIDER_SECTORS

    return {target_sector}


def summarize_transactions(df):
    """Compute statistics and retain contributing IDs."""
    summary = {
        "count": len(df),
        "transaction_ids": df["transaction_id"].tolist(),
        "sector_counts": (
            df["sector"].value_counts().to_dict()
        ),
        "financing_counts": (
            df["financing_type"].value_counts().to_dict()
        ),
        "deal_type_counts": (
            df["deal_type"].value_counts().to_dict()
        ),
    }

    for field in [
        "deal_size_mm",
        "ev_ebitda_multiple",
        "ev_revenue_multiple",
        "num_bidders",
        "days_to_close",
    ]:
        values = df[field].dropna()

        summary[field] = {
            "n": len(values),
            "median": rounded_number(values.median()),
            "q25": rounded_number(values.quantile(0.25)),
            "q75": rounded_number(values.quantile(0.75)),
            "minimum": rounded_number(values.min()),
            "maximum": rounded_number(values.max()),
        }

    return summary


def summarize_tags(df):
    """Count tags only within the explicitly supplied scope."""
    tags = (
        df["strategic_rationale_tags"]
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )

    tags = tags[tags != ""]

    counts = tags.value_counts().to_dict()
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:5])


def build_process_summary(closed_df, scope):
    """
    Summarize observed process facts from closed deals only.

    These are historical observations, not causal findings
    or evidence of current capacity.
    """
    known_bidders = closed_df[
        closed_df["num_bidders"].notna()
    ]

    competitive = known_bidders[
        known_bidders["num_bidders"] >= 3
    ]

    known_financing = closed_df[
        closed_df["financing_type"].notna()
    ]

    leveraged = known_financing[
        known_financing["financing_type"] == "Leveraged"
    ]

    known_closing_times = closed_df[
        closed_df["days_to_close"].notna()
    ]

    long_closing_times = known_closing_times[
        known_closing_times["days_to_close"] >= 365
    ]

    return {
        "scope": scope,
        "closed_deal_count": len(closed_df),

        "bidder_observations": {
            "known_count": len(known_bidders),
            "median": rounded_number(
                known_bidders["num_bidders"].median()
            ),
            "maximum": rounded_number(
                known_bidders["num_bidders"].max()
            ),
            "three_or_more_bidders_count": len(competitive),
            "three_or_more_bidders_ids": (
                competitive["transaction_id"].tolist()
            ),
            "interpretation_limit": (
                "Past competition does not establish current "
                "bidders or a competitive process for this target."
            ),
        },

        "financing_observations": {
            "known_count": len(known_financing),
            "type_counts": (
                known_financing["financing_type"]
                .value_counts()
                .to_dict()
            ),
            "leveraged_financing_count": len(leveraged),
            "leveraged_financing_share_pct": (
                round(
                    100 * len(leveraged) / len(known_financing),
                    1,
                )
                if len(known_financing)
                else None
            ),
            "leveraged_financing_ids": (
                leveraged["transaction_id"].tolist()
            ),
            "interpretation_limit": (
                "Financing type is not the buyer's current "
                "leverage, liquidity or ability to fund a deal. "
                "Deal type Leveraged Buyout and financing type "
                "Leveraged are separate CSV fields."
            ),
        },

        "closing_time_observations": {
            "known_count": len(known_closing_times),
            "median_days": rounded_number(
                known_closing_times["days_to_close"].median()
            ),
            "maximum_days": rounded_number(
                known_closing_times["days_to_close"].max()
            ),
            "at_least_365_days_count": len(long_closing_times),
            "at_least_365_days_ids": (
                long_closing_times["transaction_id"].tolist()
            ),
            "interpretation_limit": (
                "Closing duration does not identify regulatory, "
                "financing or operational causes."
            ),
        },
    }


def build_deal_structure_summary(closed_df):
    """
    Identify explicitly minority deals without assuming that
    every other transaction transferred control.
    """
    minority = closed_df[
        closed_df["deal_type"] == "Minority Investment"
    ]

    other = closed_df[
        closed_df["deal_type"] != "Minority Investment"
    ]

    return {
        "minority_investment_count": len(minority),
        "minority_investment_ids": (
            minority["transaction_id"].tolist()
        ),
        "other_closed_deal_count": len(other),
        "deal_type_counts": (
            closed_df["deal_type"].value_counts().to_dict()
        ),
        "interpretation_limit": (
            "Minority investments demonstrate investment "
            "activity, not control-acquisition experience. "
            "For other deal types, the CSV does not fully "
            "establish ownership percentages or control rights. "
            "Completed transactions do not establish successful "
            "post-deal integration."
        ),
    }


def build_market_comps(scored_df, target_profile):
    target_ev = target_profile["deal_size_mm"]

    size_matches = scored_df[
        (scored_df["outcome"] == "Closed")
        & scored_df["deal_size_mm"].between(
            target_ev / 2,
            target_ev * 2,
        )
    ].copy()

    primary = size_matches[
        size_matches["sector"] == target_profile["sector"]
    ]

    expanded = size_matches[
        size_matches["sector"].isin(
            relevant_sectors(target_profile["sector"])
        )
    ]

    if len(primary) >= 5:
        comps = primary
        selection = "exact sector"
    else:
        comps = expanded
        selection = "expanded provider-services sectors"

    return {
        "selection": selection,
        "size_band_mm": [target_ev / 2, target_ev * 2],
        "minimum_primary_count": 5,
        "primary_count": len(primary),
        "summary": summarize_transactions(comps),
        "records": clean_records(comps, PRECEDENT_FIELDS),
        "limitation": (
            "Historical context, not a target valuation. "
            "Sector and size matching do not establish "
            "identical growth, geography or control rights."
        ),
    }


def build_acquirer_evidence(
    scored_df,
    ranked_acquirers,
    target_profile=None,
):
    if target_profile is None:
        target_profile = TARGET_PROFILE

    target_sector = target_profile["sector"]
    target_ev = target_profile["deal_size_mm"]
    comp_sectors = relevant_sectors(target_sector)

    market_comps = build_market_comps(
        scored_df,
        target_profile,
    )

    packages = []

    for rank, ranking_row in ranked_acquirers.iterrows():
        acquirer = ranking_row["acquirer"]

        buyer = scored_df[
            scored_df["acquirer"] == acquirer
        ].copy()

        closed = buyer[
            buyer["outcome"] == "Closed"
        ].copy()

        exact_sector_closed = closed[
            closed["sector"] == target_sector
        ].copy()

        relevant = closed[
            closed["sector"].isin(comp_sectors)
            & closed["deal_size_mm"].between(
                target_ev / 2,
                target_ev * 2,
            )
        ].copy()

        exact_size_matched = relevant[
            relevant["sector"] == target_sector
        ]

        top = (
            closed.sort_values(
                ["evidence_score", "transaction_id"],
                ascending=[False, True],
            )
            .head(3)
            .copy()
        )

        top["relevance_score"] = (
            top["evidence_score"].round(3)
        )

        recent = (
            buyer.sort_values(
                ["deal_year", "deal_quarter"],
                ascending=False,
            )
            .head(3)
        )

        relevant_summary = summarize_transactions(relevant)
        acquirer_types = buyer["acquirer_type"].mode()

        packages.append(
            {
                "rank": rank + 1,
                "acquirer": acquirer,
                "acquirer_score": round(
                    float(ranking_row["acquirer_score"]),
                    3,
                ),
                "acquirer_type": (
                    acquirer_types.iloc[0]
                    if not acquirer_types.empty
                    else "Unknown"
                ),

                "total_transactions": len(buyer),
                "closed_transactions": len(closed),

                # Retained for compatibility with the current UI.
                "healthcare_services_transactions": int(
                    (buyer["sector"] == target_sector).sum()
                ),

                # Explicitly distinguish these three scopes.
                "exact_sector_closed_count": (
                    len(exact_sector_closed)
                ),
                "exact_sector_size_matched_closed_count": (
                    len(exact_size_matched)
                ),
                "relevant_closed_count": len(relevant),

                "relevant_scope": {
                    "sectors": sorted(comp_sectors),
                    "size_band_mm": [
                        target_ev / 2,
                        target_ev * 2,
                    ],
                    "meaning": (
                        "Closed, size-matched provider-sector "
                        "comps. This count is not all sector "
                        "experience and not a ranking eligibility "
                        "requirement."
                    ),
                },

                "evidence_depth": round(
                    float(ranking_row["evidence_depth"]),
                    3,
                ),
                "reliability_weight": round(
                    float(ranking_row["reliability_weight"]),
                    3,
                ),

                "relevant_summary": relevant_summary,
                "all_closed_summary": (
                    summarize_transactions(closed)
                ),

                "median_deal_size_mm": (
                    relevant_summary["deal_size_mm"]["median"]
                ),
                "median_ev_ebitda": (
                    relevant_summary[
                        "ev_ebitda_multiple"
                    ]["median"]
                ),
                "median_ev_revenue": (
                    relevant_summary[
                        "ev_revenue_multiple"
                    ]["median"]
                ),

                "outcome_counts": (
                    buyer["outcome"].value_counts().to_dict()
                ),

                "top_rationale_tags": summarize_tags(relevant),
                "all_closed_rationale_tags": summarize_tags(
                    closed
                ),

                "top_precedents": clean_records(
                    top,
                    PRECEDENT_FIELDS + ["relevance_score"],
                ),
                "relevant_precedents": clean_records(
                    relevant,
                    PRECEDENT_FIELDS,
                ),
                "recent_activity": clean_records(
                    recent,
                    PRECEDENT_FIELDS,
                ),

                "relevant_process_summary": (
                    build_process_summary(
                        relevant,
                        "size-matched provider-sector closed deals",
                    )
                ),
                "all_closed_process_summary": (
                    build_process_summary(
                        closed,
                        "all-sector closed buyer deals",
                    )
                ),

                "relevant_deal_structure_summary": (
                    build_deal_structure_summary(relevant)
                ),
                "all_closed_deal_structure_summary": (
                    build_deal_structure_summary(closed)
                ),

                "market_comps": market_comps,

                "diligence_limits": [
                    (
                        "Historical rationale tags describe "
                        "past transactions, not current priorities."
                    ),
                    (
                        "Target region, local shares, portfolio "
                        "overlaps and balance sheets are absent. "
                        "These are information gaps, not findings "
                        "of antitrust or financing exposure."
                    ),
                    (
                        "Non-closed transactions must retain "
                        "their actual outcome when discussed."
                    ),
                    (
                        "Withdrawn or terminated outcomes do "
                        "not establish why the deal failed."
                    ),
                    (
                        "Variety in deal types or financing "
                        "structures alone is not evidence of "
                        "integration problems or funding weakness."
                    ),
                    (
                        "All-sector buyer medians must not "
                        "replace missing size-matched comps."
                    ),
                ],
            }
        )

    peers = [
        {
            "acquirer": item["acquirer"],
            "acquirer_type": item["acquirer_type"],
            "exact_sector_closed_count": (
                item["exact_sector_closed_count"]
            ),
            "relevant_closed_count": (
                item["relevant_closed_count"]
            ),
            "minority_size_matched_count": (
                item["relevant_deal_structure_summary"]
                ["minority_investment_count"]
            ),
            "precedent_ids": [
                deal["transaction_id"]
                for deal in item["top_precedents"]
            ],
        }
        for item in packages
    ]

    for item in packages:
        item["shortlist_peers"] = [
            peer for peer in peers
            if peer["acquirer"] != item["acquirer"]
        ]

    return packages


if __name__ == "__main__":
    from data_loader import load_transactions
    from scoring import rank_acquirers, score_transactions

    transactions = load_transactions()

    scored = score_transactions(
        transactions,
        TARGET_PROFILE,
    )

    packages = build_acquirer_evidence(
        scored,
        rank_acquirers(scored),
    )

    for item in packages:
        print(
            f"\n#{item['rank']} {item['acquirer']}"
        )
        print(
            "Exact-sector closed deals:",
            item["exact_sector_closed_count"],
        )
        print(
            "Size-matched provider-sector comps:",
            item["relevant_closed_count"],
        )
        print(
            "Size-matched minority investments:",
            item["relevant_deal_structure_summary"]
            ["minority_investment_count"],
        )
        print(
            "Relevant median EV ($M):",
            item["median_deal_size_mm"],
        )
        print(
            "Relevant bidder observations:",
            item["relevant_process_summary"]
            ["bidder_observations"],
        )

    # Confirm every package serializes without NaN.
    json.dumps(packages, allow_nan=False)

    print("\nEvidence packages are JSON-safe.")