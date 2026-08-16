"""
Step 7: Visualization.

Draws the efficient frontier, marks the three key portfolios
(min-variance, max-Sharpe, equal-weight), plots individual stocks
for comparison, and draws the Capital Market Line through the
max-Sharpe (tangency) portfolio.

Usage:
    cd notebooks
    python plot_frontier.py

Reads prices from ../data/nifty_prices.csv (run fetch_data.py first).
"""

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

RISK_FREE_RATE = 0.0676  # placeholder -- replace with the actual rate for your period


def main():
    price_df = pd.read_csv("../data/nifty_prices.csv", index_col=0, parse_dates=True)
    tickers = price_df.columns.tolist()

    returns = compute_returns(price_df, log_returns=True)
    mu, sigma = compute_mu_sigma(returns, annualize=True)

    frontier = efficient_frontier(mu, sigma, n_points=40)
    w_minvar = min_variance_portfolio(sigma)
    w_sharpe = max_sharpe_portfolio(mu, sigma, risk_free_rate=RISK_FREE_RATE)
    w_eq = equal_weight_portfolio(tickers)

    rf = RISK_FREE_RATE
    minvar_ret = portfolio_return(w_minvar.values, mu.values)
    minvar_vol = portfolio_volatility(w_minvar.values, sigma.values)
    sharpe_ret = portfolio_return(w_sharpe.values, mu.values)
    sharpe_vol = portfolio_volatility(w_sharpe.values, sigma.values)
    eq_ret = portfolio_return(w_eq.values, mu.values)
    eq_vol = portfolio_volatility(w_eq.values, sigma.values)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Efficient frontier curve
    ax.plot(frontier["volatility"], frontier["target_return"], 'b-', linewidth=2, label="Efficient Frontier")

    # Individual stocks -- should all sit outside/right of the frontier
    ax.scatter(np.sqrt(np.diag(sigma)), mu, c='gray', marker='x', s=60, label="Individual stocks")
    for i, t in enumerate(tickers):
        ax.annotate(t, (np.sqrt(sigma.values[i, i]), mu.values[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')

    # Key portfolios
    ax.scatter([minvar_vol], [minvar_ret], c='green', s=120, marker='*', zorder=5, label="Min-Variance")
    ax.scatter([sharpe_vol], [sharpe_ret], c='red', s=120, marker='*', zorder=5, label="Max-Sharpe (Tangency)")
    ax.scatter([eq_vol], [eq_ret], c='orange', s=100, marker='D', zorder=5, label="Equal-Weight")

    # Capital Market Line: from (0, rf) through the tangency portfolio
    cml_x = np.linspace(0, frontier["volatility"].max(), 10)
    cml_slope = (sharpe_ret - rf) / sharpe_vol
    cml_y = rf + cml_slope * cml_x
    ax.plot(cml_x, cml_y, 'r--', linewidth=1.2, label="Capital Market Line")
    ax.scatter([0], [rf], c='black', s=50, zorder=5, label="Risk-free rate")

    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    ax.set_title("Efficient Frontier — Nifty50 Portfolio")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = '../outputs/efficient_frontier.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
