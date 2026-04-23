import sqlite3
import pandas as pd
from pathlib import Path

# Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "customer_project.db"
SQL_PATH = BASE_DIR / "sql" / "rfm_queries.sql"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

def run_rfm_query():
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)

    # Read SQL from file
    with open(SQL_PATH, "r") as file:
        query = file.read()

    # Run query and load result into pandas
    rfm_df = pd.read_sql_query(query, conn)

    # Show preview
    print("\nRFM preview:")
    print(rfm_df.head())

    print("\nShape:")
    print(rfm_df.shape)

    # Save output
    rfm_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nRFM table saved to: {OUTPUT_PATH}")

    # Close connection
    conn.close()

if __name__ == "__main__":
    run_rfm_query()