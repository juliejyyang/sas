"""
09_live_portfolio.py

Generate current stock picks based on predicted SAS values.
Uses the most recent data to recommend stocks for next quarter.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load your full modeling dataset and retrain on ALL data ──
df = pd.read_csv('data/processed/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

feature_cols = ['mom1m', 'mom6m', 'mom12m', 'volatility_12m', 'max_ret',
                'dollar_vol', 'sharpe_12m', 'high_52w',
                'D12', 'E12', 'dfy', 'ntis', 'tbl', 'tms', 'svar']
feature_cols = [c for c in feature_cols if c in df.columns]
target = 'log_sas'

df[feature_cols] = df.groupby('yyyymm')[feature_cols].transform(
    lambda x: x.fillna(x.mean())
)
df = df.dropna(subset=feature_cols + [target])

# Train Random Forest on ALL historical data (it performed best)
print("Training model on all historical data...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[feature_cols], df[target])
print("Model trained.")

# ── 2. Get current S&P 500 constituents ──
sp500 = pd.read_csv('data/processed/cleaned_sp500.csv')
sp500['date'] = pd.to_datetime(sp500['date'])
latest = sp500.iloc[-1]
tickers = latest['tickers'].split(',')
print(f"\nCurrent S&P 500 stocks: {len(tickers)}")

# ── 3. Download current price features ──
print("Downloading current price data...")
prices = yf.download(tickers, period='2y', interval='1mo')['Close']

current_features = []
for ticker in tickers:
    try:
        close = prices[ticker].dropna()
        if len(close) < 13:
            continue

        latest_data = {
            'ticker': ticker,
            'mom1m': close.pct_change(1).iloc[-1],
            'mom6m': close.pct_change(6).iloc[-1],
            'mom12m': close.pct_change(12).iloc[-1],
            'volatility_12m': close.pct_change(1).rolling(12).std().iloc[-1],
            'max_ret': close.pct_change(1).iloc[-1],
            'dollar_vol': (close * prices['Volume'][ticker]).iloc[-1] if 'Volume' in prices.columns.get_level_values(0) else 0,
            'sharpe_12m': close.pct_change(1).rolling(12).mean().iloc[-1] / close.pct_change(1).rolling(12).std().iloc[-1],
            'high_52w': close.iloc[-1] / close.rolling(12).max().iloc[-1],
        }
        current_features.append(latest_data)
    except:
        continue

current = pd.DataFrame(current_features)
print(f"Got features for {len(current)} stocks")

# ── 4. Add current macro variables ──
macro = pd.read_csv('data/processed/cleaned_macro.csv')
latest_macro = macro.iloc[-1]
for col in macro.columns:
    if col != 'yyyymm' and col in feature_cols:
        current[col] = latest_macro[col]

# ── 5. Fill NaN with cross-sectional mean ──
for col in feature_cols:
    if col in current.columns:
        current[col] = current[col].fillna(current[col].mean())

# ── 6. Predict SAS and rank stocks ──
available_features = [c for c in feature_cols if c in current.columns]
current['predicted_sas'] = model.predict(current[available_features])

# Sort by predicted SAS — highest = most manager conviction
current = current.sort_values('predicted_sas', ascending=False)

# ── 7. Output portfolio picks ──
n_stocks = len(current) // 5  # top quintile

top_picks = current.head(n_stocks)
print(f"\n{'=' * 60}")
print(f"TOP QUINTILE PORTFOLIO ({n_stocks} stocks)")
print(f"{'=' * 60}")
print(f"\n{'Rank':<6}{'Ticker':<10}{'Predicted SAS':<15}{'Mom12m':<10}{'Sharpe12m':<10}")
print("-" * 51)

for i, (_, row) in enumerate(top_picks.iterrows()):
    print(f"{i+1:<6}{row['ticker']:<10}{row['predicted_sas']:<15.4f}{row.get('mom12m', 0):<10.2%}{row.get('sharpe_12m', 0):<10.3f}")

# Save full rankings
current.to_csv('data/processed/current_rankings.csv', index=False)
print(f"\nFull rankings saved to data/processed/current_rankings.csv")

# Top picks list
print(f"\nTickers to buy:")
print(', '.join(top_picks['ticker'].tolist()))