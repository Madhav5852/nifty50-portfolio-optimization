# Nifty50 Portfolio Optimization (Markowitz Mean-Variance)

A from-scratch implementation of mean-variance portfolio optimization on
Nifty50 stocks: minimum-variance portfolio, max-Sharpe (tangency) portfolio,
efficient frontier, and a train/test backtest against equal-weight and
index benchmarks.

## Status

Work in progress, built incrementally. See commit history for the build order:
1. Data collection (`src/fetch_data.py`)
2. Return & covariance estimation (`src/portfolio.py`)
3. Optimization engine — min-variance, max-Sharpe, efficient frontier
4. Train/test backtest (`src/backtest.py`)
5. Visualization
6. Full write-up
