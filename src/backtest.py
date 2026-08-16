"""
Step 5 & 6: Train/test split and benchmarking.

This is the "honesty check" step: estimate mu and sigma ONLY on the
training window, lock in the resulting weights, then see how those
fixed weights actually perform on a test window the optimizer never saw.
"""

import numpy as np
import pandas as pd
from portfolio import (
    compute_returns, compute_mu_sigma, min_variance_portfolio,
    max_sharpe_portfolio, equal_weight_portfolio, portfolio_return,
    portfolio_volatility,
)


def train_test_split_prices(prices: pd.DataFrame, train_frac: float = 0.7):
    """Chronological split -- never shuffle time series data."""
    split_idx = int(len(prices) * train_frac)
    return prices.iloc[:split_idx], prices.iloc[split_idx:]


def realized_performance(weights: pd.Series, test_returns: pd.DataFrame, annualize: bool = True):
    """
    Given fixed weights (chosen using only training data) and the ACTUAL
    returns that happened in the test window, compute what really happened
    -- not what the optimizer predicted would happen.
    """
    # daily portfolio return series over the test window
    port_daily = test_returns @ weights
    realized_mean = port_daily.mean()
    realized_vol = port_daily.std()
    cumulative_return = (1 + port_daily).prod() - 1

    if annualize:
        realized_mean *= 252
        realized_vol *= np.sqrt(252)

    return {
        "annualized_return": realized_mean,
        "annualized_volatility": realized_vol,
        "cumulative_return": cumulative_return,
        "sharpe": realized_mean / realized_vol if realized_vol > 0 else np.nan,
    }


def run_backtest(prices: pd.DataFrame, train_frac: float = 0.7, risk_free_rate: float = 0.065):
    """
    Full walk-forward test:
      1. Split prices chronologically into train/test.
      2. Estimate mu, sigma on TRAIN only.
      3. Solve min-variance, max-Sharpe, and equal-weight portfolios using
         only that training information.
      4. Apply those FIXED weights to the TEST window's actual returns.
      5. Return a comparison table.
    """
    train_prices, test_prices = train_test_split_prices(prices, train_frac)

    train_returns = compute_returns(train_prices, log_returns=True)
    test_returns = compute_returns(test_prices, log_returns=True)

    mu_train, sigma_train = compute_mu_sigma(train_returns, annualize=True)

    w_minvar = min_variance_portfolio(sigma_train)
    w_sharpe = max_sharpe_portfolio(mu_train, sigma_train, risk_free_rate)
    w_eq = equal_weight_portfolio(prices.columns)

    results = {}
    for name, w in [
        ("Min-Variance", w_minvar),
        ("Max-Sharpe", w_sharpe),
        ("Equal-Weight", w_eq),
    ]:
        # What the optimizer EXPECTED, based on training data
        expected_return = portfolio_return(w.values, mu_train.values)
        expected_vol = portfolio_volatility(w.values, sigma_train.values)

        # What ACTUALLY happened on unseen test data
        realized = realized_performance(w, test_returns)

        results[name] = {
            "expected_return_train": expected_return,
            "expected_vol_train": expected_vol,
            **realized,
        }

    return pd.DataFrame(results).T, w_minvar, w_sharpe, w_eq
