from __future__ import annotations

import pandas as pd
import numpy as np

from src.db import get_mssql_engine


def safe_read_sql(query: str) -> pd.DataFrame:
    engine = get_mssql_engine()
    try:
        return pd.read_sql(query, con=engine)
    except Exception:
        return pd.DataFrame()


def latest_metric(analytics_df: pd.DataFrame, metric_name: str, default=None):
    if analytics_df.empty or "metric_name" not in analytics_df.columns:
        return default

    sub = analytics_df[analytics_df["metric_name"] == metric_name].copy()
    if sub.empty:
        return default

    if "created_at" in sub.columns:
        sub["created_at"] = pd.to_datetime(sub["created_at"], errors="coerce")
        sub = sub.sort_values("created_at", ascending=False)

    value = sub["metric_value"].iloc[0]
    return value if pd.notna(value) else default


def build_fraud_summary(pred_df: pd.DataFrame, analytics_df: pd.DataFrame) -> str:
    if pred_df.empty:
        return "No fraud prediction data found yet."

    df = pred_df.copy()
    if "prediction_score" in df.columns:
        df["prediction_score"] = pd.to_numeric(df["prediction_score"], errors="coerce")
        df = df.dropna(subset=["prediction_score"])
    else:
        return "Fraud predictions are available, but prediction_score is missing."

    total_predictions = len(df)
    high_risk = int((df["prediction_score"] >= 0.5).sum())
    avg_score = float(df["prediction_score"].mean())
    max_score = float(df["prediction_score"].max())

    fraud_rate = latest_metric(analytics_df, "fraud_rate", default=None)
    total_frauds = latest_metric(analytics_df, "total_frauds", default=None)

    lines = [
        "FRAUD SUMMARY",
        f"- Total predictions stored: {total_predictions:,}",
        f"- High-risk predictions (score >= 0.5): {high_risk:,}",
        f"- Average fraud risk score: {avg_score:.4f}",
        f"- Highest fraud risk score: {max_score:.4f}",
    ]

    if fraud_rate is not None:
        lines.append(f"- Historical fraud rate: {float(fraud_rate):.2f}%")
    if total_frauds is not None:
        lines.append(f"- Total fraud cases in dataset: {float(total_frauds):,.0f}")

    if high_risk > 0:
        lines.append(
            "- Interpretation: the model is flagging a small but meaningful set of transactions for manual review."
        )
    else:
        lines.append(
            "- Interpretation: the model is not flagging many transactions, which may indicate a conservative threshold."
        )

    return "\n".join(lines)


def build_market_summary(stock_df: pd.DataFrame) -> str:
    if stock_df.empty:
        return "No stock market data found yet."

    df = stock_df.copy()

    date_col = "trade_date" if "trade_date" in df.columns else "Date"
    ticker_col = "ticker" if "ticker" in df.columns else "Ticker"
    close_col = "close_price" if "close_price" in df.columns else "Close"

    if date_col not in df.columns or ticker_col not in df.columns or close_col not in df.columns:
        return "Stock data exists, but the expected columns were not found."

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, ticker_col, close_col])
    df = df.sort_values([ticker_col, date_col])

    df["return"] = df.groupby(ticker_col)[close_col].pct_change()

    tickers = df[ticker_col].dropna().unique().tolist()
    latest_prices = (
        df.groupby(ticker_col)[close_col]
        .last()
        .sort_values(ascending=False)
    )

    avg_return = df["return"].mean()
    volatility = df["return"].std()
    top_ticker = latest_prices.index[0] if len(latest_prices) > 0 else None
    top_price = latest_prices.iloc[0] if len(latest_prices) > 0 else None

    lines = [
        "MARKET SUMMARY",
        f"- Number of tracked stocks: {len(tickers)}",
        f"- Average daily return: {avg_return:.4%}" if pd.notna(avg_return) else "- Average daily return: not available",
        f"- Return volatility: {volatility:.4%}" if pd.notna(volatility) else "- Return volatility: not available",
    ]

    if top_ticker is not None:
        lines.append(f"- Highest latest closing price: {top_ticker} at {top_price:,.2f}")

    if pd.notna(volatility):
        if volatility > 0.03:
            lines.append("- Interpretation: the market basket is showing relatively high short-term volatility.")
        else:
            lines.append("- Interpretation: the market basket is showing moderate short-term volatility.")

    return "\n".join(lines)


def build_macro_summary(macro_df: pd.DataFrame) -> str:
    if macro_df.empty:
        return "No macroeconomic data found yet."

    df = macro_df.copy()

    indicator_col = "indicator_name" if "indicator_name" in df.columns else "indicator"
    date_col = "indicator_date" if "indicator_date" in df.columns else "date"

    if indicator_col not in df.columns or date_col not in df.columns or "value" not in df.columns:
        return "Macro data exists, but the expected columns were not found."

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=[indicator_col, date_col, "value"])
    df = df.sort_values([indicator_col, date_col])

    lines = ["MACRO SUMMARY"]

    latest_indicators = {}
    for indicator in df[indicator_col].dropna().unique():
        sub = df[df[indicator_col] == indicator]
        latest_val = sub["value"].iloc[-1]
        latest_indicators[str(indicator)] = float(latest_val)
        lines.append(f"- Latest {indicator}: {latest_val:,.2f}")

    recession_val = latest_indicators.get("Recession", None)
    if recession_val is not None:
        if recession_val >= 1:
            lines.append("- Interpretation: recession conditions are currently present in the macro dataset.")
        else:
            lines.append("- Interpretation: recession conditions are not currently active in the macro dataset.")

    inflation = latest_indicators.get("Inflation", None)
    interest = latest_indicators.get("Interest_Rate", None)

    if inflation is not None and interest is not None:
        if inflation > 0 and interest > 0:
            lines.append("- Interpretation: inflation and interest rates both need close monitoring for risk planning.")

    return "\n".join(lines)


def build_recommendations(fraud_summary: str, market_summary: str, macro_summary: str) -> str:
    recommendations = [
        "RECOMMENDATIONS",
        "- Review the highest-risk fraud predictions first and manually inspect the top flagged records.",
        "- Monitor correlated stocks more carefully before building a concentrated portfolio.",
        "- Keep an eye on inflation, rates, and recession indicators before making aggressive investment decisions.",
    ]

    if "high-risk" in fraud_summary.lower():
        recommendations.append("- Consider using a stricter fraud alert threshold for the dashboard.")

    if "high short-term volatility" in market_summary.lower():
        recommendations.append("- Reduce exposure to highly volatile positions or rebalance the portfolio.")

    if "recession conditions are currently present" in macro_summary.lower():
        recommendations.append("- Use a defensive allocation mindset until macro conditions improve.")

    return "\n".join(recommendations)


def generate_financial_copilot_report() -> str:
    analytics_df = safe_read_sql("SELECT * FROM analytics_summary")
    pred_df = safe_read_sql("SELECT * FROM model_predictions")
    stock_df = safe_read_sql("SELECT * FROM stock_prices")
    macro_df = safe_read_sql("SELECT * FROM macro_indicators")

    fraud_summary = build_fraud_summary(pred_df, analytics_df)
    market_summary = build_market_summary(stock_df)
    macro_summary = build_macro_summary(macro_df)
    recommendations = build_recommendations(fraud_summary, market_summary, macro_summary)

    report = (
        "# Financial Copilot Report\n\n"
        "## Executive Summary\n"
        "This platform combines fraud detection, market intelligence, and macro risk signals to help identify "
        "suspicious activity, monitor market conditions, and support better financial decisions.\n\n"
        f"## {fraud_summary}\n\n"
        f"## {market_summary}\n\n"
        f"## {macro_summary}\n\n"
        f"## {recommendations}\n"
    )
    return report


if __name__ == "__main__":
    print(generate_financial_copilot_report())