import numpy as np


def future_value(pv, rate, n):
    return pv * ((1 + rate) ** n)


def present_value(fv, rate, n):
    return fv / ((1 + rate) ** n)


def capm(risk_free_rate, beta, market_return):
    return risk_free_rate + beta * (market_return - risk_free_rate)


def sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns)


def monte_carlo_simulation(start_price, daily_return, daily_volatility, days=30, simulations=500):
    paths = []

    for _ in range(simulations):
        prices = [start_price]
        for _ in range(days):
            shock = np.random.normal(daily_return, daily_volatility)
            prices.append(prices[-1] * (1 + shock))
        paths.append(prices)

    return np.array(paths)


def value_at_risk(returns, confidence_level=0.05):
    returns = np.array(returns)
    return np.percentile(returns, confidence_level * 100)