import pandas as pd
import yfinance as yf
from src.db import get_mssql_engine


def fetch_stock_data(ticker: str, start="2020-01-01"):
    df = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    df = df.reset_index()

    # Make date timezone-free so SQL Server can store it properly
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date

    df = df.rename(columns={
        "Date": "trade_date",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close_price"]

    df["ticker"] = ticker

    return df[[
        "trade_date",
        "ticker",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adj_close",
        "volume",
    ]]


if __name__ == "__main__":
    engine = get_mssql_engine()

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    all_data = []

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df.to_sql(
        "stock_prices",
        con=engine,
        if_exists="append",
        index=False
    )

    print("Stock data loaded successfully!")
    print("Rows:", len(final_df))