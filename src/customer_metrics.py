import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "customer_project.db"

CUSTOMER_FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
CUSTOMER_SEGMENTS_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"

COHORT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cohort_retention.csv"
FINAL_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_model_base.csv"


def load_transactions():
    # Load the transactions table from SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()

    # Convert the transaction date column to datetime
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def build_cohort_retention(transactions_df):
    # Create a month column from each transaction date
    transactions_df["order_month"] = transactions_df["transaction_date"].dt.to_period("M").astype(str)

    # Find the first purchase date for each customer
    first_purchase = (
        transactions_df.groupby("customer_id")["transaction_date"]
        .min()
        .reset_index()
        .rename(columns={"transaction_date": "first_purchase_date"})
    )

    # Assign each customer to a cohort month based on first purchase
    first_purchase["cohort_month"] = first_purchase["first_purchase_date"].dt.to_period("M").astype(str)

    # Merge cohort month back into the transaction-level data
    cohort_df = transactions_df.merge(
        first_purchase[["customer_id", "cohort_month"]],
        on="customer_id",
        how="left"
    )

    # Convert the month fields into datetime for month-difference calculation
    cohort_df["order_period"] = pd.to_datetime(cohort_df["order_month"])
    cohort_df["cohort_period"] = pd.to_datetime(cohort_df["cohort_month"])

    # Calculate month index within the cohort
    cohort_df["cohort_index"] = (
        (cohort_df["order_period"].dt.year - cohort_df["cohort_period"].dt.year) * 12
        + (cohort_df["order_period"].dt.month - cohort_df["cohort_period"].dt.month)
        + 1
    )

    # Count active customers in each cohort by month number
    cohort_counts = (
        cohort_df.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "customers"})
    )

    # Get starting cohort size
    cohort_sizes = (
        cohort_counts[cohort_counts["cohort_index"] == 1][["cohort_month", "customers"]]
        .rename(columns={"customers": "cohort_size"})
    )

    # Calculate retention rate
    cohort_counts = cohort_counts.merge(cohort_sizes, on="cohort_month", how="left")
    cohort_counts["retention_rate"] = (
        cohort_counts["customers"] / cohort_counts["cohort_size"]
    ).round(4)

    # Save cohort output
    cohort_counts.to_csv(COHORT_OUTPUT_PATH, index=False)
    print(f"\nCohort retention saved to: {COHORT_OUTPUT_PATH}")

    return cohort_counts


def build_customer_model_base(transactions_df):
    # Use the latest transaction date as the dataset end date
    dataset_end_date = transactions_df["transaction_date"].max()

    # Create customer-level aggregated features
    customer_metrics = (
        transactions_df.groupby("customer_id")
        .agg(
            last_purchase_date=("transaction_date", "max"),
            first_purchase_date=("transaction_date", "min"),
            total_transactions=("transaction_id", "count"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            active_months=("transaction_date", lambda x: x.dt.to_period("M").nunique()),
            quantity_total=("quantity", "sum"),
            discount_usage_rate=("discount_used_pct", "mean"),
        )
        .reset_index()
    )

    # Calculate days since last purchase
    customer_metrics["days_since_last_purchase"] = (
        dataset_end_date - customer_metrics["last_purchase_date"]
    ).dt.days

    # Calculate customer lifespan in days
    customer_metrics["customer_lifespan_days"] = (
        customer_metrics["last_purchase_date"] - customer_metrics["first_purchase_date"]
    ).dt.days

    # Define churn: no purchase in the last 90 days
    customer_metrics["churn_label"] = (
        customer_metrics["days_since_last_purchase"] > 90
    ).astype(int)

    # Create a simple CLV proxy
    customer_metrics["clv_proxy"] = (
        customer_metrics["total_transactions"]
        * customer_metrics["avg_order_value"]
        * (customer_metrics["active_months"] / 12)
    ).round(2)

    # Create recent activity features based on the last 90 days
    recent_cutoff = dataset_end_date - pd.Timedelta(days=90)
    recent_df = transactions_df[transactions_df["transaction_date"] >= recent_cutoff]

    recent_metrics = (
        recent_df.groupby("customer_id")
        .agg(
            transactions_last_90_days=("transaction_id", "count"),
            spend_last_90_days=("order_value", "sum"),
        )
        .reset_index()
    )

    # Merge recent activity into the customer-level table
    customer_metrics = customer_metrics.merge(recent_metrics, on="customer_id", how="left")
    customer_metrics["transactions_last_90_days"] = customer_metrics["transactions_last_90_days"].fillna(0)
    customer_metrics["spend_last_90_days"] = customer_metrics["spend_last_90_days"].fillna(0)

    return customer_metrics


def merge_all_outputs(customer_metrics):
    # Load RFM and segment outputs
    rfm_df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    segments_df = pd.read_csv(CUSTOMER_SEGMENTS_PATH)

    # Keep only the segment-related columns to avoid duplicates
    segment_cols = ["customer_id", "r_score", "f_score", "m_score", "rfm_score", "segment"]
    segments_df = segments_df[segment_cols]

    # Merge everything into one final model base
    final_df = (
        customer_metrics
        .merge(rfm_df, on="customer_id", how="left")
        .merge(segments_df, on="customer_id", how="left")
    )

    # Save final customer-level table
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)

    print(f"\nCustomer model base saved to: {FINAL_OUTPUT_PATH}")
    print("\nPreview:")
    print(final_df.head())

    print("\nChurn label distribution:")
    print(final_df["churn_label"].value_counts())

    return final_df


if __name__ == "__main__":
    transactions_df = load_transactions()
    build_cohort_retention(transactions_df)
    customer_metrics = build_customer_model_base(transactions_df)
    merge_all_outputs(customer_metrics)