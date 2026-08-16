import sys
sys.path.append('../src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio import (
    compute_returns, compute_mu_sigma, min_variance_portfolio,
    max_sharpe_portfolio, efficient_frontier, equal_weight_portfolio,
    portfolio_return, portfolio_volatility
)

np.random.seed(42)
n_days = 252 * 3
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

returns = compute_returns(price_df, log_returns=True)
mu, sigma = compute_mu_sigma(returns, annualize=True)

frontier = efficient_frontier(mu, sigma, n_points=40)
w_minvar = min_variance_portfolio(sigma)
w_sharpe = max_sharpe_portfolio(mu, sigma, risk_free_rate=0.065)
w_eq = equal_weight_portfolio(tickers)

rf = 0.065
minvar_ret = portfolio_return(w_minvar.values, mu.values)
minvar_vol = portfolio_volatility(w_minvar.values, sigma.values)
sharpe_ret = portfolio_return(w_sharpe.values, mu.values)
sharpe_vol = portfolio_volatility(w_sharpe.values, sigma.values)
eq_ret = portfolio_return(w_eq.values, mu.values)
eq_vol = portfolio_volatility(w_eq.values, sigma.values)

fig, ax = plt.subplots(figsize=(9, 6))

# Efficient frontier curve
ax.plot(frontier["volatility"], frontier["target_return"], 'b-', linewidth=2, label="Efficient Frontier")

# Individual stocks
ax.scatter(np.sqrt(np.diag(sigma)), mu, c='gray', marker='x', s=60, label="Individual stocks")
for i, t in enumerate(tickers):
    ax.annotate(t, (np.sqrt(sigma.values[i,i]), mu.values[i]), fontsize=8, xytext=(5,5), textcoords='offset points')

# Key portfolios
ax.scatter([minvar_vol], [minvar_ret], c='green', s=120, marker='*', zorder=5, label="Min-Variance")
ax.scatter([sharpe_vol], [sharpe_ret], c='red', s=120, marker='*', zorder=5, label="Max-Sharpe (Tangency)")
ax.scatter([eq_vol], [eq_ret], c='orange', s=100, marker='D', zorder=5, label="Equal-Weight")

# Capital Market Line: from (0, rf) through tangency portfolio
cml_x = np.linspace(0, frontier["volatility"].max(), 10)
cml_slope = (sharpe_ret - rf) / sharpe_vol
cml_y = rf + cml_slope * cml_x
ax.plot(cml_x, cml_y, 'r--', linewidth=1.2, label="Capital Market Line")
ax.scatter([0], [rf], c='black', s=50, zorder=5, label="Risk-free rate")

ax.set_xlabel("Volatility (annualized)")
ax.set_ylabel("Expected Return (annualized)")
ax.set_title("Efficient Frontier — Simulated Nifty50-style Portfolio")
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/portfolio_optimization/outputs/efficient_frontier.png', dpi=150)
print("Saved plot to outputs/efficient_frontier.png")
