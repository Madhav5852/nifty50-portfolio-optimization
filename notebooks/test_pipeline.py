import sys
sys.path.append('../src')
import numpy as np
import pandas as pd
from portfolio import (
    compute_returns, compute_mu_sigma, min_variance_portfolio,
    max_sharpe_portfolio, efficient_frontier, equal_weight_portfolio,
    portfolio_return, portfolio_volatility
)

np.random.seed(42)

# ---- Simulate realistic multi-year daily prices for 6 stocks ----
n_days = 252 * 3   # 3 years of trading days
tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "MARUTI"]
n = len(tickers)

# Different annualized drift & vol per stock, with some correlation structure
annual_drift = np.array([0.14, 0.18, 0.12, 0.16, 0.10, 0.13])
annual_vol   = np.array([0.28, 0.24, 0.22, 0.26, 0.18, 0.30])

daily_drift = annual_drift / 252
daily_vol = annual_vol / np.sqrt(252)

# correlation matrix -- plausible sector correlations
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

print("=== Simulated price data ===")
print(price_df.head(3))
print("...")
print(price_df.tail(3))
print()

# ---- Run the actual pipeline ----
returns = compute_returns(price_df, log_returns=True)
mu, sigma = compute_mu_sigma(returns, annualize=True)

print("=== Annualized mu (expected returns) ===")
print(mu.round(4))
print()
print("=== Annualized covariance matrix ===")
print(sigma.round(4))
print()

w_minvar = min_variance_portfolio(sigma)
print("=== Min-Variance Portfolio ===")
print(w_minvar.round(4))
print("Return:", round(portfolio_return(w_minvar.values, mu.values), 4))
print("Volatility:", round(portfolio_volatility(w_minvar.values, sigma.values), 4))
print()

w_sharpe = max_sharpe_portfolio(mu, sigma, risk_free_rate=0.065)
print("=== Max-Sharpe Portfolio ===")
print(w_sharpe.round(4))
ret_s = portfolio_return(w_sharpe.values, mu.values)
vol_s = portfolio_volatility(w_sharpe.values, sigma.values)
print("Return:", round(ret_s, 4))
print("Volatility:", round(vol_s, 4))
print("Sharpe:", round((ret_s - 0.065)/vol_s, 4))
print()

w_eq = equal_weight_portfolio(tickers)
print("=== Equal-Weight Portfolio (benchmark) ===")
print("Return:", round(portfolio_return(w_eq.values, mu.values), 4))
print("Volatility:", round(portfolio_volatility(w_eq.values, sigma.values), 4))
print()

frontier = efficient_frontier(mu, sigma, n_points=8)
print("=== Efficient Frontier (8 points) ===")
print(frontier[["target_return", "volatility"]].round(4))
