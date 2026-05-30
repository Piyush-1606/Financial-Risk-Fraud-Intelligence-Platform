import pandas as pd
import numpy as np


def calculate_returns(df, price_col="close_price"):
    df = df.copy()
    df["return"] = df[price_col].pct_change()
    return df


def calculate_volatility(df, return_col="return"):
    return df[return_col].std()


def calculate_drawdown(df, price_col="close_price"):
    df = df.copy()
    df["rolling_max"] = df[price_col].cummax()
    df["drawdown"] = (df[price_col] - df["rolling_max"]) / df["rolling_max"]
    return df


def correlation_matrix(df, columns):
    return df[columns].corr()