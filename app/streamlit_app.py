from pathlib import Path
import sys
import os
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import os

from src.db import get_mssql_engine

logo_path = os.path.join(ROOT_DIR, "assets", "logo.png")
logo = Image.open(logo_path)

st.set_page_config(
    page_title="Financial Risk & Fraud Intelligence Platform",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

engine = get_mssql_engine()

logo_path = os.path.join(ROOT_DIR, "assets", "logo.png")

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)

st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    **Financial Risk & Fraud Intelligence Platform**

    AI-powered fintech intelligence suite for:

    - Fraud detection
    - Portfolio analytics
    - Quant finance
    - Macro risk monitoring
    - Financial intelligence dashboards
    """
)

st.title("Financial Risk & Fraud Intelligence Platform")
st.markdown(
    """
    ### AI-Powered Fintech Intelligence Suite

    Fraud Detection • Portfolio Analytics • Quantitative Finance •
    Macroeconomic Intelligence • Risk Management • AI Financial Copilot
    """
)

st.caption(
    "Fraud analytics, market intelligence, macro risk, and quant insights in one place."
)

@st.cache_data(ttl=300)
def load_table(query: str) -> pd.DataFrame:
    try:
        return pd.read_sql(query, con=engine)
    except Exception:
        return pd.DataFrame()

def latest_metric(df: pd.DataFrame, metric_name: str, default=0.0):
    if df.empty or "metric_name" not in df.columns:
        return default
    sub = df[df["metric_name"] == metric_name]
    if sub.empty:
        return default
    if "created_at" in sub.columns:
        sub = sub.sort_values("created_at", ascending=False)
    return float(sub["metric_value"].iloc[0])

def safe_to_datetime(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

analytics_df = load_table("SELECT * FROM analytics_summary")
pred_df = load_table("SELECT * FROM model_predictions")
stock_df = load_table("SELECT * FROM stock_prices")
macro_df = load_table("SELECT * FROM macro_indicators")

fraud_rate = latest_metric(analytics_df, "fraud_rate")
total_transactions = latest_metric(analytics_df, "total_transactions")
total_frauds = latest_metric(analytics_df, "total_frauds")
avg_fraud_amount = latest_metric(analytics_df, "avg_fraud_amount")
avg_stock_return = latest_metric(analytics_df, "avg_stock_return")
stock_volatility = latest_metric(analytics_df, "stock_volatility")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Transactions", f"{total_transactions:,.0f}")
k2.metric("Total Frauds", f"{total_frauds:,.0f}")
k3.metric("Fraud Rate %", f"{fraud_rate:.2f}")
k4.metric("Avg Fraud Amount", f"{avg_fraud_amount:,.2f}")
k5.metric("Avg Stock Return %", f"{avg_stock_return:.2f}")
k6.metric("Stock Volatility %", f"{stock_volatility:.2f}")

st.markdown("---")

st.markdown(
    """
    ## Platform Overview

    This platform combines machine learning, quantitative finance,
    fraud detection, portfolio analytics, macroeconomic intelligence,
    and AI-powered financial explanations into a single decision-support system.

    ### Key Capabilities
    - Fraud Detection & Risk Scoring
    - Portfolio Performance Analytics
    - Market Intelligence Dashboard
    - Macroeconomic Monitoring
    - Quantitative Finance Models
    - AI Financial Copilot
    """
)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Fraud Intelligence",
    "Market Intelligence",
    "Macro Risk",
    "Model Predictions"
])

with tab1:
    st.subheader("Project Summary")
    st.write(
        "This platform combines fraud detection, market analytics, macroeconomic indicators, "
        "and quant-style financial metrics into one dashboard."
    )

    st.subheader("Latest Analytics Metrics")
    if analytics_df.empty:
        st.info("No analytics data found yet.")
    else:
        if "created_at" in analytics_df.columns:
            analytics_display = analytics_df.sort_values("created_at", ascending=False)
        else:
            analytics_display = analytics_df
        st.dataframe(analytics_display, use_container_width=True)

with tab2:
    st.subheader("Fraud Intelligence")

    if pred_df.empty:
        st.info("No prediction data found yet.")
    else:
        fraud_only = pred_df.copy()

        if "prediction_score" in fraud_only.columns:
            fraud_only["prediction_score"] = pd.to_numeric(fraud_only["prediction_score"], errors="coerce")
        else:
            st.warning("prediction_score column not found in model_predictions.")
            st.stop()

        if "prediction_label" in fraud_only.columns:
            fraud_only["prediction_label"] = pd.to_numeric(fraud_only["prediction_label"], errors="coerce")

        fraud_only = fraud_only.dropna(subset=["prediction_score"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predictions Stored", f"{len(fraud_only):,}")
        with c2:
            high_risk_count = int((fraud_only["prediction_score"] >= 0.5).sum())
            st.metric("High Risk Predictions", f"{high_risk_count:,}")
        with c3:
            avg_score = fraud_only["prediction_score"].mean() * 100
            st.metric("Avg Risk Score %", f"{avg_score:.4f}")

        st.markdown("### Score Distribution")
        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = px.histogram(
                fraud_only,
                x="prediction_score",
                nbins=100,
                title="Prediction Score Distribution (Full Range)"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            zoom_df = fraud_only[fraud_only["prediction_score"] <= 0.2].copy()
            fig2 = px.histogram(
                zoom_df,
                x="prediction_score",
                nbins=80,
                range_x=[0, 0.2],
                title="Prediction Score Distribution (Zoomed: 0 to 0.2)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### High-Risk Predictions")
        high_risk_df = fraud_only[fraud_only["prediction_score"] >= 0.5].copy()

        if high_risk_df.empty:
            st.info("No rows crossed the 0.5 risk threshold.")
        else:
            high_risk_df = high_risk_df.sort_values("prediction_score", ascending=False)

            top_col1, top_col2 = st.columns(2)
            with top_col1:
                fig3 = px.histogram(
                    high_risk_df,
                    x="prediction_score",
                    nbins=30,
                    title="High-Risk Score Distribution"
                )
                st.plotly_chart(fig3, use_container_width=True)

            with top_col2:
                st.dataframe(
                    high_risk_df.head(20),
                    use_container_width=True
                )

        st.markdown("### Top Suspicious Records")
        top_suspicious = fraud_only.sort_values("prediction_score", ascending=False).head(20)
        st.dataframe(top_suspicious, use_container_width=True)

with tab3:
    st.subheader("Market Intelligence")

    if stock_df.empty:
        st.info("No stock data found yet.")
    else:
        stock_df = stock_df.copy()

        date_col = "trade_date" if "trade_date" in stock_df.columns else "Date"
        close_col = "close_price" if "close_price" in stock_df.columns else "Close"
        ticker_col = "ticker" if "ticker" in stock_df.columns else "Ticker"

        stock_df = safe_to_datetime(stock_df, date_col)
        stock_df[close_col] = pd.to_numeric(stock_df[close_col], errors="coerce")
        stock_df = stock_df.dropna(subset=[date_col, ticker_col, close_col])
        stock_df = stock_df.sort_values([ticker_col, date_col])

        tickers = stock_df[ticker_col].dropna().unique().tolist()
        selected_ticker = st.selectbox("Select Ticker", tickers)

        ticker_df = stock_df[stock_df[ticker_col] == selected_ticker].copy()
        ticker_df["daily_return"] = ticker_df[close_col].pct_change()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Latest Close Price", f"{ticker_df[close_col].iloc[-1]:,.2f}")
        with c2:
            latest_ret = ticker_df["daily_return"].iloc[-1] * 100 if len(ticker_df) > 1 else 0
            st.metric("Latest Daily Return %", f"{latest_ret:.2f}")

        fig1 = px.line(
            ticker_df,
            x=date_col,
            y=close_col,
            title=f"{selected_ticker} Closing Price"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            ticker_df,
            x=date_col,
            y="daily_return",
            title=f"{selected_ticker} Daily Returns"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Macro Risk")

    if macro_df.empty:
        st.info("No macro data found yet.")
    else:
        macro_df = macro_df.copy()

        indicator_col = "indicator_name" if "indicator_name" in macro_df.columns else "indicator"
        date_col = "indicator_date" if "indicator_date" in macro_df.columns else "date"

        macro_df = safe_to_datetime(macro_df, date_col)
        macro_df["value"] = pd.to_numeric(macro_df["value"], errors="coerce")
        macro_df = macro_df.dropna(subset=[indicator_col, date_col, "value"])
        macro_df = macro_df.sort_values([indicator_col, date_col])

        indicators = macro_df[indicator_col].dropna().unique().tolist()
        selected_indicator = st.selectbox("Select Macro Indicator", indicators)

        ind_df = macro_df[macro_df[indicator_col] == selected_indicator].copy()

        st.metric("Latest Value", f"{ind_df['value'].iloc[-1]:,.2f}")

        fig = px.line(
            ind_df,
            x=date_col,
            y="value",
            title=f"{selected_indicator} Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            ind_df.sort_values(date_col, ascending=False).head(20),
            use_container_width=True
        )

with tab5:
    st.subheader("Stored Model Predictions")

    if pred_df.empty:
        st.info("No predictions saved yet.")
    else:
        show_df = pred_df.copy()
        if "created_at" in show_df.columns:
            show_df["created_at"] = pd.to_datetime(show_df["created_at"], errors="coerce")

        if "created_at" in show_df.columns:
            st.dataframe(
                show_df.sort_values("created_at", ascending=False).head(50),
                use_container_width=True
            )
        else:
            st.dataframe(show_df.head(50), use_container_width=True)

        if "prediction_score" in show_df.columns:
            show_df["prediction_score"] = pd.to_numeric(show_df["prediction_score"], errors="coerce")
            fig = px.box(
                show_df,
                y="prediction_score",
                title="Prediction Score Spread"
            )
            st.plotly_chart(fig, use_container_width=True)