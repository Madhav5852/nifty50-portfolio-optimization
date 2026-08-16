# Findings — Nifty50 Portfolio Optimization

Notes on what this project actually taught, worked through by testing my
own understanding rather than just reading an explanation.

## 1. Why the minimum-variance portfolio never needs expected returns

The optimizer's objective, `portfolio_variance(w) = wᵀΣw`, is defined
purely in terms of weights and the covariance matrix — expected return
(μ) never appears in that formula at all. It's not that returns get
calculated and discarded; the question "how do I make this number as
small as possible" is structurally defined without ever referencing μ.
Given only a covariance matrix — no stock names, no historical returns —
there's enough information to solve for the minimum-variance portfolio
in full.

Contrast with max-Sharpe, where the objective is
`(return - risk_free) / risk` — μ is baked directly into what's being
optimized, so that portfolio *cannot* be solved without it.

## 2. Correlation determines portfolio risk — not portfolio return

Portfolio return is always the weighted average of individual returns,
regardless of correlation. Correlation only affects **risk**:

- **Correlation = +1**: no diversification benefit at all. Portfolio
  volatility becomes exactly the weighted average of the individual
  volatilities (the terms collapse into a perfect square). Two 25%-vol
  stocks, 50/50, correlation +1 → portfolio volatility is still 25%.
- **Correlation = 0**: real diversification benefit. Same two 25%-vol
  stocks, 50/50, correlation 0 → portfolio volatility drops to ~17.7%
  (computed via the full variance formula, then square-rooted to get
  back to volatility units — variance and volatility are not the same
  number, and it's easy to stop one step early).
- **Correlation → -1**: risk keeps shrinking toward near-total
  cancellation.

Two stocks with identical individual volatility can produce wildly
different portfolio outcomes purely based on the one number — correlation
— that isn't visible from the individual volatilities alone.

## 3. Why the solver is needed instead of the closed-form formulas

The clean formulas derived by hand (`w = Σ⁻¹𝟙 / 𝟙ᵀΣ⁻¹𝟙`, etc.) come from
Lagrange multipliers, which only handle **equality** constraints
("weights sum to exactly 1"). No-short-selling (`w_i ≥ 0`) is a **range**
constraint, not an equality — Lagrangians have no way to express it. It's
not that the manual formula "can error negative" — the formula has no
mechanism to even represent "don't go negative" in the first place. A
numerical solver (`scipy.optimize.minimize`, SLSQP) is the tool that can
search for an optimum while actively respecting a boundary the algebra
itself can't encode.

## 4. Why some stocks get exactly zero weight, not just "small"

Zero weight isn't a rounding artifact or the constraint kicking in by
default — it's the solver's genuine finding that a small positive
allocation to that stock would make the objective *worse*, given every
other stock already competing for the same weight budget (weights sum to
1). If any positive amount of a stock would improve the Sharpe ratio, the
solver finds it — however small. Zero means it checked, and the honest
answer was none.

This ties directly to estimation error (below): a stock getting excluded
isn't a verdict on the stock — it's a statement that its estimated μ and
its correlations with the *other* holdings, measured over *this specific
training window*, didn't clear the bar. A different window could flip
that easily.

## 5. The optimizer's confidence is not evidence the confidence is deserved

Two backtests in this project gave opposite headlines:

- **Synthetic data test**: Max-Sharpe expected 33.7% return, realized
  only 25.1% — equal-weight ended up with the better realized Sharpe.
- **Real Nifty50 data (2019–2024, 70/30 split)**: Max-Sharpe expected
  22.4%, realized 25.9% — beating both min-variance and equal-weight.

Both results are consistent with the same underlying cause: μ is
estimated from a historical average, which is a genuinely noisy predictor
of future returns. A concentrated bet (Bharti Airtel at 37% weight in the
real portfolio) can either be catching a real, persistent trend or riding
a lucky historical stretch — nothing in the optimization process itself
can tell those two apart in advance. How concentrated an optimizer's
weights are is not, by itself, a signal that the concentration is
justified.

## Real backtest result

| Portfolio | Expected return (train) | Realized return (test) | Realized Sharpe |
|---|---|---|---|
| Min-Variance | 13.9% | 12.1% | 1.06 |
| Max-Sharpe | 22.4% | 25.9% | **1.72** |
| Equal-Weight | 14.7% | 15.9% | 1.37 |

Max-Sharpe won this particular window. Given point 5 above, that result
should be read as one data point from one training/test split — not
proof the approach reliably beats simpler benchmarks. A rolling/walk-forward
backtest across multiple windows would be the honest next step to
actually test that.