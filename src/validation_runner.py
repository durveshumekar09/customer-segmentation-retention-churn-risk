import sqlite3
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "customer_project.db"
VALIDATION_SQL_PATH = BASE_DIR / "sql" / "validation_queries.sql"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read the full validation SQL script
    with open(VALIDATION_SQL_PATH, "r", encoding="utf-8") as file:
        sql_script = file.read()

    # Split the script into separate queries
    queries = [q.strip() for q in sql_script.split(";") if q.strip()]

    for i, query in enumerate(queries, start=1):
        print(f"\nValidation Query {i}")
        print("-" * 50)

        try:
            cursor.execute(query)
            rows = cursor.fetchall()

            # Print column names if the query returns a result set
            if cursor.description:
                col_names = [desc[0] for desc in cursor.description]
                print(col_names)

            for row in rows[:10]:
                print(row)

            if len(rows) > 10:
                print(f"... showing first 10 of {len(rows)} rows")

        except Exception as e:
            print(f"Error: {e}")

    conn.close()
    print("\nValidation checks complete.")


if __name__ == "__main__":
    main()
    