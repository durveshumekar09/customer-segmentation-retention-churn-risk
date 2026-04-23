import pandas as pd
from pathlib import Path

# Set file paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "segment_summary.csv"


def assign_rfm_scores(df):
    # Recency: lower is better, so reverse the labels
    df["r_score"] = pd.qcut(df["recency"], 4, labels=[4, 3, 2, 1]).astype(int)

    # Frequency: higher is better
    df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    # Monetary: higher is better
    df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    return df


def assign_segment(row):
    r = row["r_score"]
    f = row["f_score"]
    m = row["m_score"]

    if r >= 3 and f >= 3 and m >= 3:
        return "High-Value"
    elif r >= 3 and f >= 2:
        return "Repeat"
    elif r <= 2 and f <= 2 and m <= 2:
        return "At-Risk"
    else:
        return "Low-Engagement"


def create_segments():
    # Read customer-level RFM table
    df = pd.read_csv(INPUT_PATH)

    # Add RFM scores
    df = assign_rfm_scores(df)

    # Create a combined RFM score if needed later
    df["rfm_score"] = (
        df["r_score"].astype(str)
        + df["f_score"].astype(str)
        + df["m_score"].astype(str)
    )

    # Assign business segment
    df["segment"] = df.apply(assign_segment, axis=1)

    # Save detailed customer segment file
    df.to_csv(OUTPUT_PATH, index=False)

    # Build a summary table
    summary = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values(by="customers", ascending=False)
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nSegment preview:")
    print(df.head())

    print("\nSegment counts:")
    print(df["segment"].value_counts())

    print(f"\nCustomer segments saved to: {OUTPUT_PATH}")
    print(f"Segment summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    create_segments()