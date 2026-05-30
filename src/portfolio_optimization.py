import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.db import get_mssql_engine


def load_stock_data():
    engine = get_mssql_engine()
    df = pd.read_sql("SELECT * FROM stock_prices", con=engine)

    # Standardize date column
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    elif "Date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Standardize price column
    if "close_price" not in df.columns and "Close" in df.columns:
        df["close_price"] = df["Close"]

    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df = df.dropna(subset=["trade_date", "ticker", "close_price"])

    return df


def prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="trade_date",
        columns="ticker",
        values="close_price"
    ).sort_index()

    returns = pivot.pct_change().dropna()
    return returns


def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
    return portfolio_return, portfolio_volatility, sharpe


def minimize_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_max_sharpe(mean_returns, cov_matrix, num_assets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(
        lambda w, mean_returns, cov_matrix: -portfolio_performance(w, mean_returns, cov_matrix)[2],
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x


def optimize_min_variance(mean_returns, cov_matrix, num_assets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(
        lambda w, mean_returns, cov_matrix: portfolio_performance(w, mean_returns, cov_matrix)[1],
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x


def efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    num_assets = len(mean_returns)
    results = []

    for target_return in np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, num_portfolios):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, target=target_return: portfolio_performance(x, mean_returns, cov_matrix)[0] - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        result = minimize(
            lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            ret, vol, sharpe = portfolio_performance(result.x, mean_returns, cov_matrix)
            results.append((ret, vol, sharpe, result.x))

    return results

def monte_carlo_portfolio(
    start_value,
    mean_return,
    volatility,
    years=1,
    simulations=500
):
    days = 252 * years

    results = np.zeros((days, simulations))

    for sim in range(simulations):
        prices = [start_value]

        for _ in range(days - 1):
            shock = np.random.normal(mean_return/252, volatility/np.sqrt(252))
            prices.append(prices[-1] * (1 + shock))

        results[:, sim] = prices

    return results


def save_portfolio_metrics(portfolio_name, weights, mean_returns, cov_matrix):
    engine = get_mssql_engine()

    portfolio_return, portfolio_volatility, sharpe = portfolio_performance(
        weights, mean_returns, cov_matrix
    )

    rows = [
        (portfolio_name, "expected_return", float(portfolio_return)),
        (portfolio_name, "volatility", float(portfolio_volatility)),
        (portfolio_name, "sharpe_ratio", float(sharpe)),
    ]

    out_df = pd.DataFrame(rows, columns=["portfolio_name", "metric_name", "metric_value"])
    out_df.to_sql("portfolio_results", con=engine, if_exists="append", index=False)

    return out_df


if __name__ == "__main__":
    df = load_stock_data()
    returns = prepare_returns(df)

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    max_sharpe_weights = optimize_max_sharpe(mean_returns, cov_matrix, num_assets)
    min_var_weights = optimize_min_variance(mean_returns, cov_matrix, num_assets)

    print("\nAssets:", list(returns.columns))
    print("\nMax Sharpe Weights:", max_sharpe_weights)
    print("Min Variance Weights:", min_var_weights)

    max_ret, max_vol, max_sharpe = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
    min_ret, min_vol, min_sharpe = portfolio_performance(min_var_weights, mean_returns, cov_matrix)

    print("\nMax Sharpe Portfolio:")
    print("Return:", max_ret)
    print("Volatility:", max_vol)
    print("Sharpe:", max_sharpe)

    print("\nMin Variance Portfolio:")
    print("Return:", min_ret)
    print("Volatility:", min_vol)
    print("Sharpe:", min_sharpe)

    save_portfolio_metrics("Max Sharpe Portfolio", max_sharpe_weights, mean_returns, cov_matrix)
    save_portfolio_metrics("Min Variance Portfolio", min_var_weights, mean_returns, cov_matrix)

    frontier = efficient_frontier(mean_returns, cov_matrix, num_portfolios=20)
    print(f"\nEfficient frontier generated with {len(frontier)} points.")