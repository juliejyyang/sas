import pandas as pd
import csv

holdings = pd.read_csv('data/processed/clean_holdings.csv')
benchmark_weights = pd.read_csv('data/processed/benchmark_weights.csv')

# Merge holdings with benchmark weights
merged = holdings.merge(
    benchmark_weights[['date', 'ticker', 'weight']],
    on=['date', 'ticker'],
    how='outer'
)

# Fill missing weights
# If stock in benchmark but not held by ETF, set the weight to 0
# Stock in ETF but not in benchmark, skip it 
merged['weight(%)'] = merged['weight(%)'].fillna(0)
merged = merged.dropna(subset=['weight'])

# Absolute deviation for each stock-fund-date
merged['adfb'] = abs(merged['weight(%)'] - merged['weight'])

# Calculate the SAS by summing all abs deviations of the same tickers in a given date
sas = merged.groupby(['date', 'ticker'])['adfb'].sum().reset_index()
sas.columns = ['date', 'ticker', 'sas']

sas.to_csv('data/processed/sas.csv', index=False)










