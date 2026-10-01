"""Streamlit interface for buyer rankings and grounded memos."""

import json
import os

import pandas as pd
import streamlit as st

from config import TARGET_PROFILE
from data_loader import load_transactions
from evidence import build_acquirer_evidence
from rationale import (
    MODEL_NAME,
    OUTPUT_PATH,
    build_target_context,
    export_markdown,
    generate_rationale,
    make_result,
    save_results,
    valid_cached_results,
)
from scoring import rank_acquirers, score_transactions


st.set_page_config(
    page_title="M&A Acquirer Identification Engine",
    page_icon="📊",
    layout="wide",
)


TABLE_LABELS = {
    "transaction_id": "Deal ID",
    "target_company": "Target",
    "sector": "Sector",
    "deal_year": "Year",
    "deal_size_mm": "EV ($M)",
    "deal_type": "Deal Type",
    "outcome": "Outcome",
    "geography": "Geography",
    "financing_type": "Financing",
    "num_bidders": "Bidders",
    "days_to_close": "Days to Close",
    "ev_ebitda_multiple": "EV/EBITDA",
    "ev_revenue_multiple": "EV/Revenue",
    "ebitda_margin_pct": "EBITDA Margin (%)",
}


def markdown_safe(text):
    """Escape dollar signs in Markdown and captions."""
    return str(text).replace("$", r"\$")


def metric_value(value, prefix="", suffix=""):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{prefix}{value:,.1f}{suffix}"


def show_deal_table(records, columns):
    """Show a compact selection of clearly named fields."""
    if not records:
        st.info("No transactions meet this scope.")
        return

    df = pd.DataFrame(records)

    available = [
        column for column in columns
        if column in df.columns
    ]

    displayed = df[available].rename(columns=TABLE_LABELS)

    number_formats = {
        "EV ($M)": "%.1f",
        "EV/EBITDA": "%.1fx",
        "EV/Revenue": "%.1fx",
        "EBITDA Margin (%)": "%.1f",
        "Year": "%d",
        "Bidders": "%d",
        "Days to Close": "%.0f",
    }

    configuration = {
        label: st.column_config.NumberColumn(
            label,
            format=number_format,
        )
        for label, number_format in number_formats.items()
        if label in displayed.columns
    }

    st.dataframe(
        displayed,
        column_config=configuration,
        hide_index=True,
        width="stretch",
    )


def show_valuation_metrics(summary):
    cols = st.columns(3)

    cols[0].metric(
        "Closed Comps",
        str(summary["count"]),
    )

    for col, field, label in [
        (cols[1], "ev_ebitda_multiple", "Median EV/EBITDA"),
        (cols[2], "ev_revenue_multiple", "Median EV/Revenue"),
    ]:
        values = summary[field]

        col.metric(
            label,
            metric_value(values["median"], suffix="x"),
        )

        col.caption(
            f"IQR: {metric_value(values['q25'], suffix='x')}–"
            f"{metric_value(values['q75'], suffix='x')} "
            f"• {values['n']} observations"
        )


@st.cache_data
def load_engine_results():
    transactions = load_transactions()

    scored = score_transactions(
        transactions,
        TARGET_PROFILE,
    )

    rankings = rank_acquirers(scored)

    packages = build_acquirer_evidence(
        scored,
        rankings,
    )

    target = build_target_context(
        transactions,
        TARGET_PROFILE,
    )

    return rankings, packages, target


def load_saved_results():
    if not OUTPUT_PATH.exists():
        return []

    results = json.loads(
        OUTPUT_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(results, list):
        raise ValueError("Saved rationales must be a JSON list.")

    return results


def refresh_rationales(packages, target, existing):
    """Save each successful generation before moving on."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY in the project's .env file."
        )

    client = genai.Client(api_key=api_key)

    results = {
        item["acquirer"]: item
        for item in existing
    }

    progress = st.progress(0)
    status = st.empty()

    try:
        for index, evidence in enumerate(packages, start=1):
            status.write(
                f"Generating {index}/{len(packages)}: "
                f"{evidence['acquirer']}"
            )

            structured = generate_rationale(
                client,
                target,
                evidence,
            )

            results[evidence["acquirer"]] = make_result(
                target,
                evidence,
                structured,
            )

            save_results(list(results.values()))

            progress.progress(index / len(packages))

    finally:
        progress.empty()
        status.empty()


try:
    rankings, packages, target_context = load_engine_results()
    saved_results = load_saved_results()

except Exception as error:
    st.error(f"Could not load the project: {error}")
    st.stop()


current_results = valid_cached_results(
    saved_results,
    target_context,
    packages,
)

evidence_by_name = {
    item["acquirer"]: item
    for item in packages
}

results_by_name = {
    item["acquirer"]: item
    for item in current_results
}


st.title("M&A Acquirer Identification Engine")

st.caption(
    "Historical buyer screening with grounded acquisition memos"
)

if st.session_state.pop("generation_success", False):
    st.success("All ten rationales were generated and saved.")

if len(current_results) < len(saved_results):
    st.warning(
        "Some saved memos are stale or fail validation. "
        "They are hidden until regenerated."
    )

st.caption(
    f"Current validated rationales: {len(current_results)}/10 "
    f"• Model: {MODEL_NAME}"
)

if current_results:
    st.download_button(
        "Download all current memos",
        data=export_markdown(current_results),
        file_name="acquirer_rationales.md",
        mime="text/markdown",
    )


st.subheader("Target Profile")

cols = st.columns(4)

cols[0].metric("Sector", TARGET_PROFILE["sector"])

cols[1].metric(
    "Enterprise Value",
    f"${TARGET_PROFILE['deal_size_mm']} million",
)

cols[2].metric(
    "Market Position",
    TARGET_PROFILE["market_position"],
)

cols[3].metric(
    "EBITDA Profile",
    TARGET_PROFILE["ebitda_profile"],
)

st.caption(
    f"{TARGET_PROFILE['company_type']} • "
    f"{TARGET_PROFILE['geography_profile']} • "
    f"Modeled margin benchmark: "
    f"{target_context['derived_target_ebitda_margin_pct']}%. "
    "This is a sector-percentile proxy, not an actual target margin."
)


with st.expander("Methodology and interpretation"):
    st.markdown(
    r"""
**Rank — relative historical fit**

Rank orders buyers using precedent sector similarity, deal size
and EBITDA-margin similarity, with an adjustment for evidence depth.
Rank 1 has the highest score under this formula.
The score is not an acquisition probability.

**Conviction — strength of comparable historical evidence**

Conviction is calculated in Python, rather than assigned by Gemini:

| Conviction | Closed provider-sector deals within the size band |
|---|---|
| High | 4 or more |
| Medium | 1–3 |
| Low | 0 |

For this approximately \$200M target, the size band is
\$100–\$400 million. Provider sectors include Healthcare Services,
Physician Groups, Behavioral Health, Home Health/Hospice and Dental.
Adjacent-sector and minority-investment precedents have comparability
limitations.

**Why rank and conviction can differ**

A lower-ranked buyer with High conviction has substantial comparable
history but a weaker relative fit score. A higher-ranked buyer with
Low conviction scores well on historical fit but lacks size-matched
provider-sector precedents.

Zero size-matched comps does not mean zero sector experience.
Historical activity does not establish current interest, funding
capacity or willingness to bid.

**Role of Gemini**

Gemini synthesizes buyer narratives from supplied evidence.
Python calculates rankings, conviction tiers, valuation statistics
and eligible risk wording. Generated narratives require review.
"""
)


st.subheader("Top 10 Candidate Acquirers")

if st.session_state.get("selected_acquirer") not in evidence_by_name:
    st.session_state.selected_acquirer = rankings.iloc[0]["acquirer"]

columns = st.columns([0.5, 4, 1, 1.5, 1])

for column, label in zip(
    columns,
    ["Rank", "Acquirer", "Fit", "Size-matched comps", "Conviction"],
):
    column.markdown(f"**{label}**")

for index, row in rankings.iterrows():
    name = row["acquirer"]
    evidence = evidence_by_name[name]
    result = results_by_name.get(name)

    columns = st.columns([0.5, 4, 1, 1.5, 1])

    # Markdown strings avoid Streamlit's numeric code styling.
    columns[0].markdown(str(index + 1))

    label = (
        f"✓ {name}"
        if st.session_state.selected_acquirer == name
        else name
    )

    if columns[1].button(
        label,
        key=f"buyer_{index}",
        type="secondary",
        width="stretch",
    ):
        st.session_state.selected_acquirer = name
        st.rerun()

    columns[2].markdown(f"{row['acquirer_score']:.3f}")

    columns[3].markdown(
        str(evidence["relevant_closed_count"])
    )

    columns[4].markdown(
        result["structured_rationale"]["conviction"]
        if result
        else "Not generated"
    )


name = st.session_state.selected_acquirer
evidence = evidence_by_name[name]
result = results_by_name.get(name)

st.divider()
st.subheader(f"Candidate Deep Dive — {name}")

cols = st.columns(4)

cols[0].metric("Buyer Type", evidence["acquirer_type"])

cols[1].metric(
    "Closed Deals — All Sectors",
    str(evidence["closed_transactions"]),
)

cols[2].metric(
    "Exact-Sector Closed Deals",
    str(evidence["exact_sector_closed_count"]),
)

cols[3].metric(
    "Size-Matched Provider Comps",
    str(evidence["relevant_closed_count"]),
)


st.markdown("### Acquirer Rationale")

if result:
    st.markdown(
        markdown_safe(result["rationale"])
    )

    st.download_button(
        "Download this buyer memo",
        data=export_markdown([result]),
        file_name=f"buyer_{evidence['rank']}_memo.md",
        mime="text/markdown",
    )

else:
    st.info(
        "No current validated memo is available. "
        "Run `python src/rationale.py` to generate missing memos."
    )


with st.expander("Strongest completed ranking precedents"):
    st.caption(
        "Highest-scoring historical deals; some may fall "
        "outside the size-matched comp band."
    )

    show_deal_table(
        evidence["top_precedents"],
        [
            "transaction_id",
            "target_company",
            "sector",
            "deal_size_mm",
            "deal_type",
            "ev_ebitda_multiple",
        ],
    )


with st.expander("Size-matched buyer comps"):
    scope = evidence["relevant_scope"]

    st.caption(
        markdown_safe(
            f"Closed provider-sector deals at "
            f"${scope['size_band_mm'][0]:,.0f}–"
            f"${scope['size_band_mm'][1]:,.0f} million EV."
        )
    )

    st.caption("Sectors: " + ", ".join(scope["sectors"]))

    show_valuation_metrics(evidence["relevant_summary"])

    show_deal_table(
        evidence["relevant_precedents"],
        [
            "transaction_id",
            "target_company",
            "sector",
            "deal_size_mm",
            "deal_type",
            "ev_ebitda_multiple",
            "ev_revenue_multiple",
        ],
    )


with st.expander("Target-relevant market comps"):
    market = evidence["market_comps"]

    st.caption(
        f"Selection: {market['selection']}. "
        "Historical context, not a target valuation."
    )

    show_valuation_metrics(market["summary"])

    show_deal_table(
        market["records"],
        [
            "transaction_id",
            "target_company",
            "deal_size_mm",
            "deal_type",
            "ev_ebitda_multiple",
            "ev_revenue_multiple",
        ],
    )


with st.expander("Recent activity and historical themes"):
    st.markdown("**Recent observed activity**")

    show_deal_table(
        evidence["recent_activity"],
        [
            "target_company",
            "sector",
            "deal_year",
            "deal_size_mm",
            "deal_type",
            "outcome",
        ],
    )

    st.markdown("**Observed outcomes**")

    st.caption(
        " • ".join(
            f"{outcome}: {count}"
            for outcome, count in evidence["outcome_counts"].items()
        )
    )

    st.markdown("**Themes in size-matched closed comps**")

    themes = evidence["top_rationale_tags"]

    if themes:
        st.caption(
            " • ".join(
                f"{theme} ({count})"
                for theme, count in themes.items()
            )
        )
    else:
        st.caption(
            "No size-matched comps; no themes inferred for this scope."
        )


with st.expander("Historical process observations"):
    process = evidence["relevant_process_summary"]

    st.caption(
        "Scope: size-matched provider-sector closed deals. "
        "These observations do not establish current capacity "
        "or causes of historical outcomes."
    )

    bidders = process["bidder_observations"]
    financing = process["financing_observations"]
    timing = process["closing_time_observations"]

    cols = st.columns(3)

    cols[0].metric(
        "Median Bidders",
        metric_value(bidders["median"]),
    )
    cols[0].caption(
        f"{bidders['known_count']} observed deals; "
        f"{bidders['three_or_more_bidders_count']} "
        "had at least three bidders."
    )

    cols[1].metric(
        "Leveraged Financing",
        metric_value(
            financing["leveraged_financing_share_pct"],
            suffix="%",
        ),
    )
    cols[1].caption(
        f"{financing['leveraged_financing_count']} of "
        f"{financing['known_count']} known financing records. "
        "Not current buyer leverage."
    )

    cols[2].metric(
        "Median Days to Close",
        metric_value(timing["median_days"]),
    )
    cols[2].caption(
        f"{timing['known_count']} observations. "
        "Closing-time causes are unknown."
    )

    show_deal_table(
        evidence["relevant_precedents"],
        [
            "transaction_id",
            "target_company",
            "financing_type",
            "num_bidders",
            "days_to_close",
        ],
    )


st.divider()
st.subheader("LLM Generation")

st.caption(
    "Current memos load without API calls. Refreshing all ten "
    "makes Gemini API calls and saves each completed memo. "
    "The CLI resumes missing or stale memos."
)

if st.button("Refresh All Ten Memos", type="secondary"):
    try:
        refresh_rationales(
            packages,
            target_context,
            current_results,
        )

        st.session_state.generation_success = True
        st.rerun()

    except Exception as error:
        st.error(
            f"Generation stopped: {error}. "
            "Completed memos remain saved. "
            "Run `python src/rationale.py` to resume."
        )


st.divider()

st.caption(
    "Historical evidence supports screening, not confirmed "
    "buyer interest. Current strategy, financing and market "
    "overlap require further diligence."
)