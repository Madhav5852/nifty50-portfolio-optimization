"""
Step 2 & 3: Estimation + Optimization engine.

Every function here is a direct code version of something we derived by hand:
- compute_returns / compute_mu_sigma  -> Steps 1-2 of the worked example
- portfolio_return / portfolio_variance -> the w^T Sigma w quadratic form
- min_variance_portfolio -> the Lagrangian-derived w_MVP (numerically, with w>=0)
- max_sharpe_portfolio   -> the tangency portfolio (numerically, with w>=0)
- efficient_frontier     -> sweeping target returns, same as the frontier table
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    """
    Convert a price DataFrame (rows=days, cols=tickers) into daily returns.
    Log returns are used by default -- they're additive over time, which
    makes multi-day compounding and later statistics better behaved than
    simple returns.
    """
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def compute_mu_sigma(returns: pd.DataFrame, annualize: bool = True):
    """
    mu    -> mean return per stock (Step 2 in the worked example)
    sigma -> full covariance matrix (Step 3 in the worked example)

    annualize=True scales daily stats to yearly, using the standard
    sqrt(time) convention for volatility and linear scaling for mean
    return. 252 = approx. number of trading days in a year.
    """
    mu = returns.mean()
    sigma = returns.cov()
    if annualize:
        mu = mu * 252
        sigma = sigma * 252
    return mu, sigma


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """w . mu -- weighted average return."""
    return float(np.dot(weights, mu))


def portfolio_variance(weights: np.ndarray, sigma: np.ndarray) -> float:
    """w^T Sigma w -- the quadratic form we expanded by hand for 3 assets."""
    return float(weights @ sigma @ weights)


def portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, sigma)))


def _base_constraints(n):
    """Weights sum to 1, no short-selling (0 <= w_i <= 1)."""
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.array([1 / n] * n)
    return cons, bounds, w0


def min_variance_portfolio(sigma: pd.DataFrame):
    """
    The Global Minimum Variance portfolio. Ignores mu entirely --
    exactly as derived: this is the one portfolio that never needs
    expected returns as an input.
    """
    n = len(sigma)
    cons, bounds, w0 = _base_constraints(n)
    result = minimize(
        portfolio_variance, w0, args=(sigma.values,),
        method="SLSQP", bounds=bounds, constraints=cons,
    )
    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed: {result.message}")
    return pd.Series(result.x, index=sigma.columns)


def max_sharpe_portfolio(mu: pd.Series, sigma: pd.DataFrame, risk_free_rate: float = 0.0676):
    """
    The tangency / max-Sharpe portfolio. risk_free_rate defaults to a
    rough Indian 10Y G-Sec-ish figure (annualized) -- replace with the
    actual prevailing rate for your test period when you run this for real.
    """
    n = len(sigma)
    cons, bounds, w0 = _base_constraints(n)

    def neg_sharpe(w, mu, sigma, rf):
        ret = portfolio_return(w, mu.values)
        vol = portfolio_volatility(w, sigma.values)
        return -(ret - rf) / vol

    result = minimize(
        neg_sharpe, w0, args=(mu, sigma, risk_free_rate),
        method="SLSQP", bounds=bounds, constraints=cons,
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    return pd.Series(result.x, index=sigma.columns)


def efficient_frontier(mu: pd.Series, sigma: pd.DataFrame, n_points: int = 30):
    """
    Sweep target returns from the lowest to highest single-asset mean,
    minimizing variance at each -- same process as the frontier table
    we built by hand, just at finer resolution.

    Returns a DataFrame with columns: target_return, volatility, and
    one column per ticker holding that point's weights.
    """
    n = len(sigma)
    _, bounds, w0 = _base_constraints(n)
    targets = np.linspace(mu.min(), mu.max(), n_points)

    rows = []
    for target in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu.values) - t},
        )
        result = minimize(
            portfolio_variance, w0, args=(sigma.values,),
            method="SLSQP", bounds=bounds, constraints=cons,
        )
        if result.success:
            row = {"target_return": target, "volatility": np.sqrt(result.fun)}
            row.update(dict(zip(sigma.columns, result.x)))
            rows.append(row)

    return pd.DataFrame(rows)


def equal_weight_portfolio(tickers):
    n = len(tickers)
    return pd.Series([1 / n] * n, index=tickers)
