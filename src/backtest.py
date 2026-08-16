import numpy as np
import pandas as pd
from portfolio import (
    compute_returns, compute_mu_sigma, min_variance_portfolio,
    max_sharpe_portfolio, equal_weight_portfolio, portfolio_return,
    portfolio_volatility,
)


def train_test_split_prices(prices: pd.DataFrame, train_frac: float = 0.7):
    
    split_idx = int(len(prices) * train_frac)
    return prices.iloc[:split_idx], prices.iloc[split_idx:]


def realized_performance(weights: pd.Series, test_returns: pd.DataFrame, annualize: bool = True):
    
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


def run_backtest(prices: pd.DataFrame, train_frac: float = 0.7, risk_free_rate: float = 0.0676):
    
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


def expanding_window_splits(prices: pd.DataFrame, min_train_years: int = 2, test_years: int = 1):
    
    years = sorted(prices.index.year.unique())
    windows = []

    for i in range(min_train_years, len(years) - test_years + 1):
        train_years = years[:i]
        test_year_list = years[i:i + test_years]

        train_prices = prices[prices.index.year.isin(train_years)]
        test_prices = prices[prices.index.year.isin(test_year_list)]

        train_label = f"{train_years[0]}-{train_years[-1]}"
        test_label = f"{test_year_list[0]}" if len(test_year_list) == 1 else f"{test_year_list[0]}-{test_year_list[-1]}"

        windows.append((train_prices, test_prices, train_label, test_label))

    return windows


def rolling_backtest(prices: pd.DataFrame, min_train_years: int = 2,
                      test_years: int = 1, risk_free_rate: float = 0.0676):
    
    windows = expanding_window_splits(prices, min_train_years, test_years)

    per_window_rows = []

    for train_prices, test_prices, train_label, test_label in windows:
        train_returns = compute_returns(train_prices, log_returns=True)
        test_returns = compute_returns(test_prices, log_returns=True)

        if len(train_returns) < 30 or len(test_returns) < 5:
            # Not enough data in this window to estimate/test meaningfully -- skip it
            continue

        mu_train, sigma_train = compute_mu_sigma(train_returns, annualize=True)

        w_minvar = min_variance_portfolio(sigma_train)
        w_sharpe = max_sharpe_portfolio(mu_train, sigma_train, risk_free_rate)
        w_eq = equal_weight_portfolio(prices.columns)

        for name, w in [
            ("Min-Variance", w_minvar),
            ("Max-Sharpe", w_sharpe),
            ("Equal-Weight", w_eq),
        ]:
            realized = realized_performance(w, test_returns)
            per_window_rows.append({
                "train_window": train_label,
                "test_window": test_label,
                "portfolio": name,
                **realized,
            })

    per_window = pd.DataFrame(per_window_rows)

    summary = (
        per_window.groupby("portfolio")[["annualized_return", "annualized_volatility", "sharpe"]]
        .agg(["mean", "std"])
    )

    return per_window, summary