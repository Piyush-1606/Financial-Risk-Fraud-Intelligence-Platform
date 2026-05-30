import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from src.db import get_mssql_engine

load_dotenv()

def load_fred_series(series_code, label):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    data = fred.get_series(series_code)

    df = pd.DataFrame({
        "indicator_date": data.index,
        "value": data.values
    })
    df["indicator_name"] = label
    return df[["indicator_name", "indicator_date", "value"]]

if __name__ == "__main__":
    engine = get_mssql_engine()

    series_map = {
        "CPIAUCSL": "Inflation",
        "FEDFUNDS": "Interest_Rate",
        "UNRATE": "Unemployment",
        "GDP": "GDP",
        "USREC": "Recession"
    }

    all_data = []

    for code, label in series_map.items():
        df = load_fred_series(code, label)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_sql(
        "macro_indicators",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("Macro data loaded successfully!")
    print("Rows:", len(final_df))