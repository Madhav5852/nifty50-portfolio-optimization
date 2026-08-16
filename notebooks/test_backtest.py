import sys
sys.path.append('../src')
import numpy as np
import pandas as pd

# Reuse the same simulated price data generation as before, but longer
# so we have a meaningful train/test split (5 years)
np.random.seed(7)
n_days = 252 * 5
tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "MARUTI"]

annual_drift = np.array([0.14, 0.18, 0.12, 0.16, 0.10, 0.13])
annual_vol   = np.array([0.28, 0.24, 0.22, 0.26, 0.18, 0.30])
daily_drift = annual_drift / 252
daily_vol = annual_vol / np.sqrt(252)

corr = np.array([
    [1.00, 0.30, 0.20, 0.25, 0.15, 0.10],
    [0.30, 1.00, 0.15, 0.55, 0.10, 0.05],
    [0.20, 0.15, 1.00, 0.10, 0.20, 0.15],
    [0.25, 0.55, 0.10, 1.00, 0.05, 0.05],
    [0.15, 0.10, 0.20, 0.05, 1.00, 0.20],
    [0.10, 0.05, 0.15, 0.05, 0.20, 1.00],
])
cov_daily = np.outer(daily_vol, daily_vol) * corr
daily_returns = np.random.multivariate_normal(daily_drift, cov_daily, size=n_days)
prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
price_df = pd.DataFrame(prices, columns=tickers)

from backtest import run_backtest

results, w_minvar, w_sharpe, w_eq = run_backtest(price_df, train_frac=0.7, risk_free_rate=0.065)

pd.set_option('display.width', 120)
print("=== Backtest results: EXPECTED (from training data) vs REALIZED (on unseen test data) ===\n")
print(results.round(4))

print("\n=== Weights chosen (from training data only) ===")
print("\nMin-Variance weights:")
print(w_minvar.round(3))
print("\nMax-Sharpe weights:")
print(w_sharpe.round(3))
