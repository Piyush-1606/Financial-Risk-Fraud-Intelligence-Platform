from src.quant_model import future_value, present_value, capm, sharpe_ratio, value_at_risk

print("Future Value:", future_value(10000, 0.10, 5))
print("Present Value:", present_value(16105.1, 0.10, 5))
print("CAPM:", capm(0.05, 1.2, 0.12))
print("Sharpe Ratio:", sharpe_ratio([0.02, 0.01, -0.03, 0.04, 0.015], 0.005))
print("VaR:", value_at_risk([0.02, -0.01, 0.03, -0.05, 0.01]))