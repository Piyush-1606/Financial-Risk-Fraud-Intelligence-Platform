import pandas as pd
import streamlit as st
import plotly.express as px

from src.db import get_mssql_engine

st.set_page_config(
    page_title="Financial Risk & Fraud Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

engine = get_mssql_engine()

st.title("Financial Risk & Fraud Intelligence Platform")
st.caption("Fraud analytics, market intelligence, macro risk, and quant insights in one place.")

@st.cache_data(ttl=300)
def load_table(query: str) -> pd.DataFrame:
    return pd.read_sql(query, con=engine)

def latest_metric(df: pd.DataFrame, metric_name: str, default=0.0):
    if df.empty:
        return default
    sub = df[df["metric_name"] == metric_name]
    if sub.empty:
        return default
    value = sub.sort_values("created_at", ascending=False)["metric_value"].iloc[0]
    return float(value)

def safe_to_datetime(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# Load data
analytics_df = load_table("SELECT * FROM analytics_summary")
pred_df = load_table("SELECT * FROM model_predictions")
stock_df = load_table("SELECT * FROM stock_prices")
macro_df = load_table("SELECT * FROM macro_indicators")

# Top KPIs
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
        st.dataframe(
            analytics_df.sort_values("created_at", ascending=False),
            use_container_width=True
        )

with tab2:
    st.subheader("Fraud Intelligence")

    if pred_df.empty:
        st.info("No prediction data found yet.")
    else:
        fraud_only = pred_df.copy()
        fraud_only["prediction_score"] = pd.to_numeric(fraud_only["prediction_score"], errors="coerce")
        fraud_only["prediction_label"] = pd.to_numeric(fraud_only["prediction_label"], errors="coerce")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predictions Stored", f"{len(fraud_only):,}")
        with c2:
            high_risk_count = int((fraud_only["prediction_score"] >= 0.5).sum())
            st.metric("High Risk Predictions", f"{high_risk_count:,}")

        fig = px.histogram(
            fraud_only,
            x="prediction_score",
            nbins=30,
            title="Prediction Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest Prediction Records")
        st.dataframe(
            fraud_only.sort_values("created_at", ascending=False).head(20),
            use_container_width=True
        )

with tab3:
    st.subheader("Market Intelligence")

    if stock_df.empty:
        st.info("No stock data found yet.")
    else:
        stock_df = stock_df.copy()
        stock_df = safe_to_datetime(stock_df, "trade_date")
        stock_df["close_price"] = pd.to_numeric(stock_df["close_price"], errors="coerce")
        stock_df = stock_df.dropna(subset=["trade_date", "ticker", "close_price"])
        stock_df = stock_df.sort_values(["ticker", "trade_date"])

        tickers = stock_df["ticker"].dropna().unique().tolist()
        selected_ticker = st.selectbox("Select Ticker", tickers)

        ticker_df = stock_df[stock_df["ticker"] == selected_ticker].copy()
        ticker_df["daily_return"] = ticker_df["close_price"].pct_change()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Latest Close Price", f"{ticker_df['close_price'].iloc[-1]:,.2f}")
        with c2:
            latest_ret = ticker_df["daily_return"].iloc[-1] * 100 if len(ticker_df) > 1 else 0
            st.metric("Latest Daily Return %", f"{latest_ret:.2f}")

        fig1 = px.line(
            ticker_df,
            x="trade_date",
            y="close_price",
            title=f"{selected_ticker} Closing Price"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(
            ticker_df,
            x="trade_date",
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
        macro_df = safe_to_datetime(macro_df, "indicator_date")
        macro_df["value"] = pd.to_numeric(macro_df["value"], errors="coerce")
        macro_df = macro_df.dropna(subset=["indicator_name", "indicator_date", "value"])
        macro_df = macro_df.sort_values(["indicator_name", "indicator_date"])

        indicators = macro_df["indicator_name"].dropna().unique().tolist()
        selected_indicator = st.selectbox("Select Macro Indicator", indicators)

        ind_df = macro_df[macro_df["indicator_name"] == selected_indicator].copy()

        st.metric(
            "Latest Value",
            f"{ind_df['value'].iloc[-1]:,.2f}"
        )

        fig = px.line(
            ind_df,
            x="indicator_date",
            y="value",
            title=f"{selected_indicator} Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            ind_df.sort_values("indicator_date", ascending=False).head(20),
            use_container_width=True
        )

with tab5:
    st.subheader("Stored Model Predictions")

    if pred_df.empty:
        st.info("No predictions saved yet.")
    else:
        show_df = pred_df.copy()
        st.dataframe(
            show_df.sort_values("created_at", ascending=False).head(50),
            use_container_width=True
        )

        if "prediction_score" in show_df.columns:
            show_df["prediction_score"] = pd.to_numeric(show_df["prediction_score"], errors="coerce")
            fig = px.box(
                show_df,
                y="prediction_score",
                title="Prediction Score Spread"
            )
            st.plotly_chart(fig, use_container_width=True)