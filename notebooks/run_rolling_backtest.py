import sys
sys.path.append('../src')
import pandas as pd
from backtest import rolling_backtest, expanding_window_splits

RISK_FREE_RATE = 0.0676  # India 10Y G-Sec yield, Aug 2026

price_df = pd.read_csv("../data/nifty_prices.csv", index_col=0, parse_dates=True)

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)

# Show the windows being used, for transparency
windows = expanding_window_splits(price_df, min_train_years=2, test_years=1)
print("=== Windows ===")
for train_prices, test_prices, train_label, test_label in windows:
    print(f"  train {train_label} ({len(train_prices)} days) -> test {test_label} ({len(test_prices)} days)")
print()

per_window, summary = rolling_backtest(
    price_df, min_train_years=2, test_years=1, risk_free_rate=RISK_FREE_RATE
)

print("=== Per-window results ===")
print(per_window.round(4).to_string(index=False))
print()

print("=== Summary: mean +/- std realized Sharpe across all windows ===")
print(summary.round(4))