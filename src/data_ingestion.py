import pandas as pd
from src.db import get_mssql_engine

engine = get_mssql_engine()

df = pd.read_csv("data/raw/creditcard.csv")

print("Rows:", len(df))
print("Columns:", len(df.columns))

df.to_sql(
    "fraud_transactions",
    engine,
    if_exists="replace",
    index=False
)

print("Fraud data loaded successfully!")