import pandas as pd
from src.db import get_mssql_engine

def compute_fraud_metrics(engine):
    fraud_df = pd.read_sql("SELECT * FROM fraud_transactions", con=engine)

    total_transactions = len(fraud_df)
    total_frauds = fraud_df["Class"].sum()
    fraud_rate = (total_frauds / total_transactions) * 100 if total_transactions else 0
    avg_fraud_amount = fraud_df.loc[fraud_df["Class"] == 1, "Amount"].mean()

    return [
        ("total_transactions", float(total_transactions), "All fraud dataset records"),
        ("total_frauds", float(total_frauds), "Number of fraud cases"),
        ("fraud_rate", float(fraud_rate), "Fraud percentage"),
        ("avg_fraud_amount", float(avg_fraud_amount if pd.notna(avg_fraud_amount) else 0), "Average fraud amount"),
    ]


def compute_stock_metrics(engine):
    stock_df = pd.read_sql("SELECT * FROM stock_prices", con=engine)

    # Handle likely column names from yfinance
    close_col = "Close" if "Close" in stock_df.columns else "close_price"
    ticker_col = "ticker" if "ticker" in stock_df.columns else "Ticker"

    stock_df[close_col] = pd.to_numeric(stock_df[close_col], errors="coerce")
    stock_df = stock_df.dropna(subset=[close_col])

    stock_df["return"] = stock_df.groupby(ticker_col)[close_col].pct_change()

    avg_return = stock_df["return"].mean() * 100
    volatility = stock_df["return"].std() * 100

    return [
        ("avg_stock_return", float(avg_return if pd.notna(avg_return) else 0), "Average daily return %"),
        ("stock_volatility", float(volatility if pd.notna(volatility) else 0), "Return volatility %"),
    ]


def compute_macro_metrics(engine):
    macro_df = pd.read_sql("SELECT * FROM macro_indicators", con=engine)

    metrics = []
    for indicator in macro_df["indicator_name"].dropna().unique():
        sub = macro_df[macro_df["indicator_name"] == indicator].sort_values("indicator_date")
        latest_value = sub["value"].iloc[-1]
        metrics.append((f"macro_{indicator.lower()}", float(latest_value), f"Latest value for {indicator}"))

    return metrics


def save_metrics(engine, metrics):
    out_df = pd.DataFrame(metrics, columns=["metric_name", "metric_value", "metric_text"])
    out_df.to_sql("analytics_summary", con=engine, if_exists="append", index=False)
    return out_df


if __name__ == "__main__":
    engine = get_mssql_engine()


    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM analytics_summary;")

    all_metrics = []
    all_metrics.extend(compute_fraud_metrics(engine))
    all_metrics.extend(compute_stock_metrics(engine))
    all_metrics.extend(compute_macro_metrics(engine))

    result_df = save_metrics(engine, all_metrics)
    print("Analytics summary saved successfully!")
    print(result_df)