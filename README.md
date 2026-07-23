Quantitative Portfolio Optimization & Risk Analytics

A five-asset universe (XOM, CVX, VST, NRG, SPY), optimized for maximum Sharpe ratio, then stress-tested with Monte Carlo simulation and benchmarked against SPY on return, volatility, VaR, CVaR, and drawdown.

Companion project: End-to-End ERCOT Battery Trading & Optimization Platform. Two of these five tickers, VST and NRG, are ERCOT generators/retailers, and the optimizer ended up leaning on one of them heavily. More on that below.

Every number here comes from a single historical backtest, a mean-variance optimization over that same window, and a 10,000-path Monte Carlo simulation. None of it has been forward-tested or traded. Read the limitations section before taking any of it at face value.

What this actually is

Five tickers, five years of daily prices (Jan 2021 – Jan 2026, pulled via yfinance). Portfolio weights are solved by maximizing the Sharpe ratio with scipy.optimize (long-only, weights summing to 1) against the historical mean return and covariance matrix. The optimized portfolio is then projected forward 252 trading days with a Monte Carlo simulation that resamples from a multivariate normal fit to those same historical returns.

Optimized weights: SPY around 52%, NRG around 34%, CVX around 14%, XOM and VST both essentially 0%. (Read off portfolio_allocation.png — exact values print in the notebook output.)

The optimizer dropped two of the five assets entirely, and that's not a bug. XOM is 0.86-correlated with CVX, NRG is 0.65-correlated with VST, so once a Sharpe-maximizer finds the better-performing half of a redundant pair, it has little reason to hold the other. Unconstrained mean-variance optimization does this constantly. It's one of the standard critiques of Markowitz-style optimization in practice, showing up here in miniature.

Results
Metric	Portfolio	SPY
Annualized Return	39.88%	15.20%
Annualized Volatility	26.82%	17.11%
Sharpe Ratio (5% risk-free)	1.30	0.60
VaR (95%, 1-year)	-8.05%	-14.39%
CVaR (95%, 1-year)	-17.33%	-16.41%
Maximum Drawdown	-18.41%	-24.50%

Portfolio drawdown is the average peak-to-trough across 10,000 simulated one-year Monte Carlo paths (worst single path: -52.01%). SPY's is the actual historical compounded drawdown over the full five-year window. Both use the same calculation, peak-to-trough on a compounded path, but one is a 1-year simulated figure and the other a 5-year realized one, so treat the pairing as directionally comparable rather than exact.

From the simulation: mean final value $148,514 on a $100,000 start, median $143,182, 95% confidence interval $85,013–$242,023, worst simulated path $54,009, best $405,219, probability of a loss over the horizon 9.02%.

Portfolio and SPY VaR/CVaR are now on the same one-year horizon, which fixes the bug this notebook originally had. They're still not measuring quite the same thing, though. The portfolio's VaR comes out of a Monte Carlo simulation built on its own historical mean return of 39.88%/year, projected forward assuming normally distributed returns. SPY's VaR comes directly from actual historical rolling-year outcomes, real bad years included. That's part of why the higher-volatility optimized portfolio shows a better VaR than SPY here: the simulation is effectively betting the future looks like this portfolio's unusually strong recent past, while SPY's number reflects what actually happened. Don't read too much into the portfolio "beating" SPY on this specific metric.

Why NRG carries a third of this portfolio

NRG and VST are both ERCOT-facing power companies, NRG a major generator and retailer, Vistra one of the largest battery and generation fleets in Texas. NRG's daily returns show real tails out past plus or minus 10%, which plausibly still carries some signature of the February 2021 ERCOT winter storm sitting inside this five-year window. The optimizer wasn't told any of that. It just found NRG's return-to-volatility trade-off attractive enough to hand it a third of the portfolio on pure Sharpe grounds. So this "optimized" portfolio ends up more concentrated in ERCOT-specific risk than the original hand-picked version, not less. If you're evaluating this alongside the battery project rather than on its own, a real event in that market would now hit a third of this portfolio directly.

Methodology, briefly
Prices pulled once via yfinance, daily percent returns computed from adjusted close.
Mean returns and covariance matrix estimated from the full five-year sample. No train/test split.
Weights solved via scipy.optimize.minimize maximizing Sharpe ratio, long-only, fully invested, against that same in-sample mean/covariance.
The Monte Carlo engine draws a joint return vector from numpy.random.multivariate_normal for each of 252 simulated days, compounds it into a portfolio path, and repeats for 10,000 independent paths.
Portfolio VaR/CVaR come from the distribution of simulated one-year returns. SPY's come from actual historical rolling 252-day returns (prices["SPY"].pct_change(252), NaNs dropped). Same horizon now, different sampling method, see above.
Maximum drawdown for both portfolio and SPY uses the same peak-to-trough calculation on a compounded (cumprod) path, not the additive cumsum approximation the notebook used originally.
Sharpe ratio uses a flat 5% assumed risk-free rate for both portfolio and SPY.
Limitations & notable details

The optimizer solves in-sample, on a universe I picked by hand, and it produces a known unstable pattern. Mean-variance optimization is sensitive to estimation error in expected returns, and collapsing to 3 of 5 assets, with none of the diversification a "portfolio" is supposed to offer, is the textbook symptom of that sensitivity. A small perturbation to the historical mean-return estimates could send the weights somewhere completely different. Adding a minimum-weight or maximum-concentration constraint, or comparing against a minimum-variance solution instead of max-Sharpe, would make this more robust, and is the natural next step.

The tickers were picked with the outcome already known. All five were chosen for a backtest running through the same five years they're evaluated on, and the two the optimizer ended up favoring, SPY and NRG, both had strong runs in this specific window. Building the universe and then optimizing and testing on the exact period it was picked for isn't a genuine out-of-sample test. A real one would fix the universe before the window, or test on a period it wasn't chosen to fit.

The Monte Carlo simulation assumes jointly normal returns. No fat tails, no skew, no volatility clustering, even though NRG's and VST's own daily return histograms in this notebook show real fat tails the simulation has no way to reproduce. It's likely understating how bad the bad days can actually get.

Correlation is treated as static too, with no regime-switching. One covariance matrix covers the whole five years, but real correlations move, usually rising in a crisis, right when diversification is needed most and hardest to get. This simulation can't capture that.

No rebalancing, no transaction costs, no taxes. Weights are fixed at the top of every simulated path and never adjusted.

Everything here is in-sample now, at every stage. The optimizer, the Monte Carlo statistics, and the benchmark comparison all draw from the same five-year window with no holdout. A stronger version of this project would optimize on one period and evaluate on a later, unseen one.

Repo contents
montecarlo_var.ipynb          # everything, single notebook top to bottom
plots/                         # all 14 saved figures
results/
  summary_statistics.csv
  comparison_with_spy.csv
Setup
bash
pip install -r requirements.txt
jupyter notebook montecarlo_var.ipynb

Needs a working internet connection on first run. yfinance pulls prices live rather than reading from a local cache.

What I'd change next

Add a diversification constraint to the optimizer so it doesn't collapse to three assets, or add a minimum-variance solution alongside the max-Sharpe one and compare the two. Swap the normal-distribution Monte Carlo for something that keeps the fat tails the historical data actually has. Pick the universe and optimize on one window, then genuinely test on a later one the weights were never fit to.

License

MIT