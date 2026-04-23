import sqlite3
import pandas as pd
from pathlib import Path

# Set the main project folder path
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the raw transaction file
DATA_PATH = BASE_DIR / "data" / "raw" / "customer_transactions_synthetic_2025.csv"

# Path where the SQLite database will be stored
DB_PATH = BASE_DIR / "database" / "customer_project.db"


def load_csv():
    """
    Load the transaction dataset from the CSV file.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(DATA_PATH)

    # Show a quick preview of the data
    print("\nDataset preview:")
    print(df.head())

    # Print the column names
    print("\nColumns:")
    print(df.columns.tolist())

    # Print the number of rows and columns
    print("\nShape:")
    print(df.shape)

    return df


def create_sqlite_db(df):
    """
    Create the SQLite database and load the data into a table.
    """
    # Open a connection to the SQLite database
    conn = sqlite3.connect(DB_PATH)

    # Write the DataFrame into a table called 'transactions'
    # Replace the table if it already exists
    df.to_sql("transactions", conn, if_exists="replace", index=False)

    print("\nLoaded data into SQLite table: transactions")

    # Run a quick check to confirm the row count
    row_count = pd.read_sql_query(
        "SELECT COUNT(*) AS total_rows FROM transactions",
        conn
    )

    print("\nRow count in SQLite:")
    print(row_count)

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    # Load the CSV file
    df = load_csv()

    # Save the data into SQLite
    create_sqlite_db(df)

    print("\nData loading complete.")
    