import pandas as pd
from src.portfolio_analysis import calculate_returns, calculate_volatility, calculate_drawdown

df = pd.DataFrame({
    "close_price": [100, 102, 101, 105, 103, 108]
})

df = calculate_returns(df)
df = calculate_drawdown(df)

print(df)
print("Volatility:", calculate_volatility(df))