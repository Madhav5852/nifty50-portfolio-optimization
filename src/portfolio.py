import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def compute_mu_sigma(returns: pd.DataFrame, annualize: bool = True):
    
    mu = returns.mean()
    sigma = returns.cov()
    if annualize:
        mu = mu * 252
        sigma = sigma * 252
    return mu, sigma


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    
    return float(np.dot(weights, mu))


def portfolio_variance(weights: np.ndarray, sigma: np.ndarray) -> float:
    
    return float(weights @ sigma @ weights)


def portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, sigma)))


def _base_constraints(n):
    
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.array([1 / n] * n)
    return cons, bounds, w0


def min_variance_portfolio(sigma: pd.DataFrame):
    
    n = len(sigma)
    cons, bounds, w0 = _base_constraints(n)
    result = minimize(
        portfolio_variance, w0, args=(sigma.values,),
        method="SLSQP", bounds=bounds, constraints=cons,
    )
    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed: {result.message}")
    return pd.Series(result.x, index=sigma.columns)


def max_sharpe_portfolio(mu: pd.Series, sigma: pd.DataFrame, risk_free_rate: float = 0.065):
    
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