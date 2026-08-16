"""
Runs the train/test backtest against REAL Nifty50 data
(data/nifty_prices.csv, produced by src/fetch_data.py) --
as opposed to test_backtest.py, which validates the same
logic against synthetic data.

Usage:
    cd notebooks
    python run_real_backtest.py
"""

import sys
sys.path.append('../src')
import pandas as pd
from backtest import run_backtest

RISK_FREE_RATE = 0.0676  # India 10Y G-Sec yield, Aug 2026

price_df = pd.read_csv("../data/nifty_prices.csv", index_col=0, parse_dates=True)

results, w_minvar, w_sharpe, w_eq = run_backtest(
    price_df, train_frac=0.7, risk_free_rate=RISK_FREE_RATE
)

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', None)

print("=== REAL Nifty50 Backtest: EXPECTED (train) vs REALIZED (test) ===\n")
print(results.round(4))

print("\n=== Min-Variance weights (chosen from training data only) ===")
print(w_minvar.round(3))

print("\n=== Max-Sharpe weights (chosen from training data only) ===")
print(w_sharpe.round(3))
