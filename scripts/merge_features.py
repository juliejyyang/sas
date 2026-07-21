import pandas as pd
import numpy as np

sas = pd.read_csv('data/processed/sas.csv')
chars = pd.read_csv('data/processed/cleaned_characteristics.csv')
macro = pd.read_csv('data/processed/cleaned_macro.csv')

# Align dates
sas['date'] = pd.to_datetime(sas['date'])
sas['yyyymm'] = sas['date'].dt.year * 100 + sas['date'].dt.month

# Merge SAS with characteristics
merged = sas.merge(chars, on=['ticker', 'yyyymm'], how='inner')
print(f"After SAS + chars: {merged.shape}")

# Merge macro (same values for all stocks in a given month)
macro['yyyymm'] = macro['yyyymm'].astype(int)
merged = merged.merge(macro, on='yyyymm', how='left')
print(f"After adding macro: {merged.shape}")

# Create target
merged['log_sas'] = np.log(merged['sas'] + 1)

print(f"\nFinal modeling dataset: {merged.shape}")
print(f"Stocks: {merged['ticker'].nunique()}")
print(f"Months: {merged['yyyymm'].nunique()}")
print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
print(f"Columns: {merged.columns.tolist()}")

merged.to_csv('data/processed/modeling_dataset.csv', index=False)
print("\nSaved to data/processed/modeling_dataset.csv")