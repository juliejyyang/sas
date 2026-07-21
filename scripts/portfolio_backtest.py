"""
07_portfolio_backtest.py

For each quarter, sort stocks by predicted SAS into quantiles.
Go long the top quantile. Compute portfolio returns using actual
next-quarter stock returns.

Inputs:
    - data/processed/sas_predictions.csv
    - data/processed/benchmark_weights.csv

Output:
    - data/processed/portfolio_returns.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ── 1. Load predictions ──
preds = pd.read_csv('data/processed/sas_predictions.csv')
preds['date'] = pd.to_datetime(preds['date'])
preds['quarter'] = preds['date'].dt.to_period('Q')

print(f"Predictions: {preds.shape}")
print(f"Models: {preds['model'].unique()}")
print(f"Quarters: {sorted(preds['quarter'].unique())}")

# ── 2. Get all unique tickers we need returns for ──
all_tickers = preds['ticker'].unique().tolist()
print(f"\nDownloading returns for {len(all_tickers)} tickers...")

# Download monthly prices for computing forward returns
# Extend end date by 3 months to capture next-quarter returns
min_date = preds['date'].min() - pd.Timedelta(days=30)
max_date = preds['date'].max() + pd.Timedelta(days=120)

prices = yf.download(all_tickers, start=min_date, end=max_date, interval='1mo')['Close']

# Compute monthly returns
returns = prices.pct_change()

# Reshape to long format: date, ticker, return
returns_long = returns.stack().reset_index()
returns_long.columns = ['date', 'ticker', 'monthly_return']
returns_long['date'] = pd.to_datetime(returns_long['date'])

# Compute forward 3-month return (next quarter's return)
# For each stock-month, sum the next 3 months of returns
returns_wide = returns.copy()
forward_returns = {}
for ticker in returns_wide.columns:
    series = returns_wide[ticker]
    # Rolling 3-month forward return: (1+r1)*(1+r2)*(1+r3) - 1
    fwd = (1 + series).rolling(3).apply(lambda x: x.prod(), raw=True).shift(-3) - 1
    forward_returns[ticker] = fwd

fwd_returns = pd.DataFrame(forward_returns)
fwd_returns_long = fwd_returns.stack().reset_index()
fwd_returns_long.columns = ['date', 'ticker', 'fwd_3m_return']
fwd_returns_long['date'] = pd.to_datetime(fwd_returns_long['date'])

# Align forward returns to the prediction dates
fwd_returns_long['yyyymm'] = fwd_returns_long['date'].dt.year * 100 + fwd_returns_long['date'].dt.month
preds['yyyymm'] = preds['date'].dt.year * 100 + preds['date'].dt.month

# ── 3. Merge predictions with forward returns ──
preds_with_returns = preds.merge(
    fwd_returns_long[['ticker', 'yyyymm', 'fwd_3m_return']],
    on=['ticker', 'yyyymm'],
    how='left'
)

missing_returns = preds_with_returns['fwd_3m_return'].isna().mean()
print(f"\nMissing forward returns: {missing_returns:.0%}")
preds_with_returns = preds_with_returns.dropna(subset=['fwd_3m_return'])
print(f"Predictions with returns: {preds_with_returns.shape}")

# ── 4. Form portfolios for each model and quarter ──
all_portfolio_returns = []

for model_name in preds_with_returns['model'].unique():
    model_preds = preds_with_returns[preds_with_returns['model'] == model_name]

    for quarter in sorted(model_preds['quarter'].unique()):
        q_data = model_preds[model_preds['quarter'] == quarter].copy()

        if len(q_data) < 10:
            continue

        # Sort stocks into quantiles based on predicted SAS
        q_data['quantile'] = pd.qcut(q_data['predicted_log_sas'], q=5, labels=False) + 1

        # Compute returns for each quantile portfolio (equal-weighted)
        for q in range(1, 6):
            portfolio = q_data[q_data['quantile'] == q]
            if len(portfolio) == 0:
                continue

            # Equal-weighted portfolio return
            port_return = portfolio['fwd_3m_return'].mean()

            all_portfolio_returns.append({
                'model': model_name,
                'quarter': str(quarter),
                'quantile': q,
                'portfolio_return': port_return,
                'n_stocks': len(portfolio),
                'tickers': ','.join(portfolio['ticker'].tolist())
            })

        # Also compute benchmark return (equal-weighted all stocks)
        bench_return = q_data['fwd_3m_return'].mean()
        all_portfolio_returns.append({
            'model': model_name,
            'quarter': str(quarter),
            'quantile': 0,  # 0 = benchmark
            'portfolio_return': bench_return,
            'n_stocks': len(q_data),
            'tickers': 'BENCHMARK'
        })

portfolio_returns = pd.DataFrame(all_portfolio_returns)

# ── 5. Summary ──
print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE SUMMARY")
print("=" * 70)

for model_name in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model_name]
    print(f"\n--- {model_name} ---")

    for q in range(0, 6):
        q_data = model_data[model_data['quantile'] == q]
        if len(q_data) == 0:
            continue

        label = 'Benchmark' if q == 0 else f'Q{q}'
        mean_ret = q_data['portfolio_return'].mean() * 100
        std_ret = q_data['portfolio_return'].std() * 100
        sharpe = q_data['portfolio_return'].mean() / q_data['portfolio_return'].std() if q_data['portfolio_return'].std() > 0 else 0

        print(f"  {label:10s} | Mean: {mean_ret:6.2f}% | Std: {std_ret:6.2f}% | Sharpe: {sharpe:.3f} | Avg stocks: {q_data['n_stocks'].mean():.0f}")

# ── 6. Check if top quantile beats benchmark ──
print("\n" + "=" * 70)
print("TOP QUINTILE vs BENCHMARK")
print("=" * 70)

for model_name in portfolio_returns['model'].unique():
    model_data = portfolio_returns[portfolio_returns['model'] == model_name]
    top = model_data[model_data['quantile'] == 5]['portfolio_return'].mean()
    bench = model_data[model_data['quantile'] == 0]['portfolio_return'].mean()
    diff = top - bench

    print(f"{model_name:20s} | Top Q5: {top*100:6.2f}% | Bench: {bench*100:6.2f}% | Diff: {diff*100:+6.2f}%")

# ── 7. Save ──
portfolio_returns.to_csv('data/processed/portfolio_returns.csv', index=False)
print(f"\nSaved to data/processed/portfolio_returns.csv")