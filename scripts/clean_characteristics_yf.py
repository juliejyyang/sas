"""
03_clean_characteristics.py

Compute price-based stock characteristics from yfinance.
Uses only features derivable from price/volume history (no fundamentals).

Inputs:
    - data/processed/cleaned_sp500.csv

Output:
    - data/processed/cleaned_characteristics.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ── 1. Get all unique tickers from S&P 500 history ──
sp500 = pd.read_csv('data/processed/cleaned_sp500.csv')
sp500['date'] = pd.to_datetime(sp500['date'])

all_tickers = set()
for _, row in sp500.iterrows():
    all_tickers.update(row['tickers'].split(','))
tickers = sorted(list(all_tickers))

print(f"Total unique tickers: {len(tickers)}")

# ── 2. Download monthly price data for all stocks at once ──
print("Downloading monthly price data...")
prices = yf.download(tickers, start='2020-01-01', end='2024-07-01', interval='1mo')

# ── 3. Compute price-based features for each stock ──
print("Computing features...")
all_features = []

for ticker in tickers:
    try:
        close = prices['Close'][ticker].dropna()
        volume = prices['Volume'][ticker].dropna()

        if len(close) < 13:  # need at least 13 months for 12m momentum
            continue

        df = pd.DataFrame({'close': close, 'volume': volume})

        # Momentum features
        df['mom1m'] = df['close'].pct_change(1)
        df['mom6m'] = df['close'].pct_change(6)
        df['mom12m'] = df['close'].pct_change(12)

        # Volatility (12-month rolling std of monthly returns)
        df['volatility_12m'] = df['mom1m'].rolling(12).std()

        # Max monthly return
        df['max_ret'] = df['mom1m']

        # Dollar volume
        df['dollar_vol'] = df['close'] * df['volume']

        # Rolling Sharpe ratio (12-month)
        df['sharpe_12m'] = df['mom1m'].rolling(12).mean() / df['mom1m'].rolling(12).std()

        # Log size (log of price as proxy)
        df['log_size'] = np.log(df['close'])

        # 52-week (12-month) high ratio
        df['high_52w'] = df['close'] / df['close'].rolling(12).max()

        df['ticker'] = ticker
        df['date'] = df.index

        all_features.append(df)
    except Exception as e:
        continue

price_features = pd.concat(all_features, ignore_index=True)
price_features['month_end'] = pd.to_datetime(price_features['date']) + pd.offsets.MonthEnd(0)
price_features['yyyymm'] = price_features['month_end'].dt.year * 100 + price_features['month_end'].dt.month

print(f"Raw features shape: {price_features.shape}")

# ── 4. Define feature columns ──
feature_cols = ['mom1m', 'mom6m', 'mom12m', 'volatility_12m', 'max_ret',
                'dollar_vol', 'sharpe_12m', 'log_size', 'high_52w']

# ── 5. Drop columns with too many NaNs ──
for col in feature_cols.copy():
    pct_missing = price_features[col].isna().mean()
    if pct_missing > 0.5:
        print(f"Dropping {col}: {pct_missing:.0%} missing")
        feature_cols.remove(col)

# ── 6. Fill remaining NaNs with cross-sectional mean ──
price_features[feature_cols] = price_features.groupby('yyyymm')[feature_cols].transform(
    lambda x: x.fillna(x.mean())
)

# ── 7. Standardize: subtract cross-sectional mean, divide by std ──
price_features[feature_cols] = price_features.groupby('yyyymm')[feature_cols].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ── 8. Save ──
output = price_features[['ticker', 'yyyymm', 'month_end'] + feature_cols]
output.to_csv('data/processed/cleaned_characteristics.csv', index=False)

print(f"\nSaved: {output.shape}")
print(f"Date range: {output['month_end'].min().date()} to {output['month_end'].max().date()}")
print(f"Unique tickers: {output['ticker'].nunique()}")
print(f"Features: {feature_cols}")