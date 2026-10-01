from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ma_transactions_500.csv"


# Columns required by the ranking engine
REQUIRED_COLUMNS = [
    "target_company",
    "acquirer",
    "sector",
    "sub_sector",
    "deal_size_mm",
    "deal_type",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "geography",
    "acquirer_type",
    "strategic_rationale_tags",
    "outcome",
    "deal_year",
]


def load_transactions(data_path=DATA_PATH):
    """Load and validate the historical M&A transaction dataset."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return df


if __name__ == "__main__":
    transactions = load_transactions()

    print(f"Loaded {len(transactions)} transactions.")
    print(f"Unique acquirers: {transactions['acquirer'].nunique()}")
    print("\nSector distribution:")
    print(transactions["sector"].value_counts())