"""
08_evaluate_performance.py

Compute risk-adjusted performance metrics and compare to Fama-French factors.

Inputs:
    - data/processed/portfolio_returns.csv
    - data/raw/fama_factor.csv

Output:
    - Prints performance summary
"""

import pandas as pd
import numpy as np

# Load portfolio returns
port = pd.read_csv('data/processed/portfolio_returns.csv')

# Load Fama-French factors
ff = pd.read_csv('data/raw/fama_factor.csv', skiprows=4)
ff = ff.rename(columns={ff.columns[0]: 'yyyymm'})
ff = ff[pd.to_numeric(ff['yyyymm'], errors='coerce').notna()]
ff['yyyymm'] = ff['yyyymm'].astype(int)
for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
    ff[col] = pd.to_numeric(ff[col], errors='coerce') / 100

# Add momentum factor
mom = pd.read_csv('data/raw/momentum_factor.csv', skiprows=13)
mom = mom.rename(columns={mom.columns[0]: 'yyyymm', mom.columns[1]: 'Mom'})
mom = mom[pd.to_numeric(mom['yyyymm'], errors='coerce').notna()]
mom['yyyymm'] = mom['yyyymm'].astype(int)
mom['Mom'] = pd.to_numeric(mom['Mom'], errors='coerce') / 100
ff = ff.merge(mom, on='yyyymm', how='left')

# Then update factor_cols later in the script:
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

# You'll need to align the FF dates with your quarter dates
# and regress portfolio excess returns on the factors to compute alpha
# For now, print the summary metrics we already have

print("\n" + "=" * 70)
print("FINAL PERFORMANCE REPORT")
print("=" * 70)

for model_name in port['model'].unique():
    model_data = port[port['model'] == model_name]
    
    top = model_data[model_data['quantile'] == 5]
    bottom = model_data[model_data['quantile'] == 1]
    bench = model_data[model_data['quantile'] == 0]
    
    print(f"\n--- {model_name} ---")
    print(f"  Top Q5 mean return:    {top['portfolio_return'].mean()*100:6.2f}%")
    print(f"  Bottom Q1 mean return: {bottom['portfolio_return'].mean()*100:6.2f}%")
    print(f"  Benchmark mean return: {bench['portfolio_return'].mean()*100:6.2f}%")
    print(f"  Q5 - Benchmark:        {(top['portfolio_return'].mean() - bench['portfolio_return'].mean())*100:+6.2f}%")
    print(f"  Q5 - Q1 spread:        {(top['portfolio_return'].mean() - bottom['portfolio_return'].mean())*100:+6.2f}%")