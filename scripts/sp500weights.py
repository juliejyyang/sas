"""
Compute S&P 500 historical benchmark weights using yfinance.

Inputs:
    - data/processed/cleaned_sp500.csv

Output:
    - data/processed/benchmark_weights.csv

Run from the sas/ directory:
    pip install yfinance
    python scripts/sp500weights.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

START       = "2020-10-30"
END         = "2021-09-08"
MAX_WORKERS = 20  # parallel threads for shares fetching

# ── 1. Load and filter S&P 500 constituents ───────────────────────────────────
print("Loading S&P 500 constituents...")
sp500 = pd.read_csv('data/processed/cleaned_sp500.csv', parse_dates=['date'])
sp500 = sp500[(sp500['date'] >= START) & (sp500['date'] <= END)]

if sp500.empty:
    raise ValueError(f"No observations in cleaned_sp500.csv between {START} and {END}.")

sp500_long = (
    sp500
    .assign(ticker=sp500['tickers'].str.split(','))
    .explode('ticker')
    .assign(ticker=lambda df: df['ticker'].str.strip())
    [['date', 'ticker']]
    .drop_duplicates()
)

obs_dates   = sorted(sp500_long['date'].unique())
all_tickers = sorted(sp500_long['ticker'].unique())
date_tickers = sp500_long.groupby('date')['ticker'].apply(list).to_dict()

print(f"  {len(obs_dates)} observation dates, {len(all_tickers)} unique tickers")

# ── 2. Download closing prices (single bulk call) ─────────────────────────────
print("\nDownloading prices from yfinance...")
dl_start = (pd.Timestamp(START) - pd.offsets.BDay(5)).strftime('%Y-%m-%d')
dl_end   = (pd.Timestamp(END)   + pd.offsets.BDay(2)).strftime('%Y-%m-%d')

raw = yf.download(
    all_tickers, start=dl_start, end=dl_end,
    auto_adjust=True, progress=False, threads=True,
)
prices = raw['Close']
if isinstance(prices, pd.Series):          # single-ticker edge case
    prices = prices.to_frame(all_tickers[0])
prices.index = prices.index.tz_localize(None) if prices.index.tz else prices.index
prices = prices.ffill()

# Snapshot prices on each observation date (forward-fill to nearest trading day)
price_snap = prices.reindex(obs_dates, method='ffill')
print(f"  Price matrix: {price_snap.shape[0]} dates × {price_snap.shape[1]} tickers")

# ── 3. Fetch historical shares outstanding (parallel) ─────────────────────────
print(f"\nFetching shares outstanding ({MAX_WORKERS} threads)...")

def fetch_shares(ticker):
    t = yf.Ticker(ticker)
    try:
        s = t.get_shares_full(start=dl_start, end=dl_end)
        if s is not None and not s.empty:
            s.index = s.index.tz_convert(None) if s.index.tz else s.index
            return ticker, s
    except Exception:
        pass
    try:
        val = t.fast_info.shares
        if val and val > 0:
            return ticker, pd.Series([val], index=[pd.Timestamp(START)])
    except Exception:
        pass
    return ticker, None

shares_map = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(fetch_shares, t): t for t in all_tickers}
    for i, fut in enumerate(as_completed(futures), 1):
        ticker, series = fut.result()
        if series is not None:
            shares_map[ticker] = series
        if i % 100 == 0:
            print(f"  {i}/{len(all_tickers)} tickers fetched")

print(f"  Shares data available for {len(shares_map)}/{len(all_tickers)} tickers")

# ── 4. Build market cap table ─────────────────────────────────────────────────
print("\nComputing market caps...")

def shares_on_date(ticker, date):
    if ticker not in shares_map:
        return None
    prior = shares_map[ticker]
    prior = prior[prior.index <= date]
    return float(prior.iloc[-1]) if not prior.empty else None

rows = []
for date in obs_dates:
    for ticker in date_tickers.get(date, []):
        if ticker not in price_snap.columns:
            continue
        price  = price_snap.at[date, ticker]
        shares = shares_on_date(ticker, date)
        if pd.isna(price) or shares is None or shares <= 0:
            continue
        rows.append({'date': date, 'ticker': ticker, 'market_cap': price * shares})

df = pd.DataFrame(rows)
missing = len(sp500_long) - len(df)
print(f"  {missing} stock-date observations dropped (no price or shares data)")

# ── 5. Compute weights ────────────────────────────────────────────────────────
df['total_mc'] = df.groupby('date')['market_cap'].transform('sum')
df['weight']   = df['market_cap'] / df['total_mc'] * 100
df = df.drop(columns='total_mc').sort_values(['date', 'ticker']).reset_index(drop=True)

# ── 6. Save ───────────────────────────────────────────────────────────────────
out_path = 'data/processed/benchmark_weights.csv'
df.to_csv(out_path, index=False)

print(f"\nSaved to {out_path}")
print(f"  Rows: {len(df):,} | Dates: {df['date'].nunique()} | Tickers: {df['ticker'].nunique()}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

sample_date = df['date'].iloc[len(df) // 2]
top10 = df[df['date'] == sample_date].nlargest(10, 'weight')[['ticker', 'weight']]
print(f"\nTop 10 weights on {sample_date.date()}:")
print(top10.to_string(index=False))
