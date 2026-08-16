# Findings: Nifty50 Portfolio Optimization

Notes on what I actually learned from this project, worked out by testing
my own understanding rather than just reading an explanation and moving on.

## 1. Why the minimum-variance portfolio never needs expected returns

The thing being minimized, `portfolio_variance(w) = wᵀΣw`, is defined
purely in terms of the weights and the covariance matrix. Expected return
(mu) never shows up in that formula at all. It's not that returns get
calculated and then thrown away, the question "how do I make this number
as small as possible" is just never framed in terms of mu to begin with.
If you handed me only a covariance matrix, no stock names, no historical
returns, I'd still have everything I need to solve for the minimum
variance portfolio.

Compare that to max-Sharpe, where the objective is
`(return - risk free) / risk`. Mu is baked directly into what's being
optimized there, so that portfolio genuinely can't be solved without it.

## 2. Correlation determines portfolio risk, not portfolio return

Portfolio return is always just the weighted average of the individual
returns, no matter what the correlation is. Correlation only changes
risk:

- Correlation = +1: no diversification benefit at all. The portfolio's
  volatility ends up being exactly the weighted average of the
  individual volatilities (the math collapses into a perfect square).
  Two 25% volatility stocks, split 50/50, correlation of +1, and you're
  still sitting at 25% volatility.
- Correlation = 0: real benefit shows up. Same two stocks, same 50/50
  split, but correlation of 0 drops the portfolio volatility to around
  17.7%. Getting there means running the full variance formula and then
  taking the square root to get back to volatility units. Variance and
  volatility aren't the same number, and it's an easy step to skip.
- Correlation heading toward -1: risk keeps shrinking, toward near total
  cancellation.

Two stocks with identical individual volatility can end up producing
wildly different portfolio outcomes, purely because of one number,
correlation, that you'd never see just by looking at the volatilities
alone.

## 3. Why you need a solver instead of the closed-form formulas

The clean formulas I derived by hand (things like
`w = Σ⁻¹𝟙 / 𝟙ᵀΣ⁻¹𝟙`) come from Lagrange multipliers, and Lagrange
multipliers only work for equality constraints, like "weights have to sum
to exactly 1." No-short-selling (`w_i ≥ 0`) isn't an equality, it's a
range, and Lagrangians have no way to express that. It's not that the
manual formula "can error into negative weights," it's that the formula
has no mechanism to even represent "don't go negative" in the first
place. A numerical solver like `scipy.optimize.minimize` is the tool that
can actually search for an answer while respecting a boundary the algebra
itself can't encode.

## 4. Why some stocks get exactly zero weight, not just a small amount

Zero weight isn't a rounding artifact and it isn't the constraint
kicking in by default. It's the solver genuinely finding that giving
that stock even a small positive weight would make the result worse,
given that every other stock is competing for the same budget (weights
have to sum to 1). If there's any amount of a stock that would improve
the Sharpe ratio, the solver finds it, no matter how small. Zero means it
checked, and the honest answer was none.

This connects straight back to estimation error below. A stock getting
excluded isn't a verdict on the stock itself, it's a statement that its
estimated mu and its correlations with everything else in the portfolio,
measured over that specific training window, just didn't clear the bar.
A different window could flip that easily, and that's basically what
ended up happening (see point 5).

## 5. The optimizer's confidence isn't evidence that the confidence is deserved

I tested this three separate ways in this project, and each one pointed
at the same underlying cause: mu is just a historical average, and a
historical average is a genuinely noisy way to predict the future.

- On synthetic data: Max-Sharpe expected a 33.7% return but only realized
  25.1%. Equal-Weight ended up with the better realized Sharpe.
- On real Nifty50 data with one 70/30 split: Max-Sharpe expected 22.4%
  and realized 25.9%, beating both other portfolios convincingly with a
  Sharpe of 1.72.
- On real Nifty50 data with a rolling backtest across four expanding
  windows (train 2019-2020, test 2021, all the way to train 2019-2023,
  test 2024): this is the one that actually settles it. Averaged across
  all four windows, Max-Sharpe's mean realized Sharpe (1.03) didn't beat
  Equal-Weight (1.12), and its window-to-window standard deviation (1.06)
  was roughly double the other two portfolios'. Its per-window Sharpe
  went from 1.98 down to -0.43, an actual risk-adjusted loss, back up to
  1.58.

The single-split result, the one that looked like a clean win for
Max-Sharpe, turned out to be its best window, not a typical one. If I'd
stopped after running just that one split, I would have walked away with
a confidently wrong conclusion. That's the concrete version of the
abstract lesson: how concentrated an optimizer's weights are, or how good
a single backtest looks, isn't evidence that the underlying bet was
sound. It takes more than one test to tell a real edge apart from a
lucky draw.

## The real final result

| Portfolio | Mean realized Sharpe | Std dev across windows |
|---|---|---|
| Equal-Weight | **1.12** | 0.54 |
| Min-Variance | 0.95 | 0.61 |
| Max-Sharpe | 1.03 | **1.06** |

Equal-weight, the portfolio with no optimization behind it at all, was
the steadiest performer across every window I tested. Max-Sharpe's higher
peaks (1.98 in 2021) got wiped out by an actual loss in 2022, so it
netted out to a similar average but with a lot more variation in outcomes
along the way. Min-Variance, even though it doesn't use any return
information at all, still slightly underperformed equal-weight on
average, which is a good reminder that "safe" (low realized volatility)
and "good risk-adjusted return" aren't automatically the same thing.