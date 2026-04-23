import sqlite3
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "customer_project.db"
SQL_DIR = BASE_DIR / "sql"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# SQL files and matching output files
SQL_JOBS = {
    "rfm_queries.sql": "rfm_sql_output.csv",
    "cohort_queries.sql": "cohort_sql_output.csv",
    "churn_label_queries.sql": "churn_sql_output.csv",
}


def run_sql_file(conn, sql_file_path):
    with open(sql_file_path, "r", encoding="utf-8") as file:
        query = file.read()
    return pd.read_sql_query(query, conn)


def main():
    conn = sqlite3.connect(DB_PATH)

    for sql_file, output_file in SQL_JOBS.items():
        sql_path = SQL_DIR / sql_file
        output_path = OUTPUT_DIR / output_file

        df = run_sql_file(conn, sql_path)
        df.to_csv(output_path, index=False)

        print(f"\nRan: {sql_file}")
        print(f"Saved: {output_path}")
        print("Preview:")
        print(df.head())
        print(f"Shape: {df.shape}")

    conn.close()
    print("\nSQL execution complete.")


if __name__ == "__main__":
    main()
    