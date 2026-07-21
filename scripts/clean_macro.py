"""
04_clean_macro.py

Clean Goyal-Welch macroeconomic predictors.
Selects the 8 variables used in the paper, aligns date format.

Inputs:
    - data/raw/golay_monthly.csv

Output:
    - data/processed/cleaned_macro.csv
"""

import pandas as pd

# ── 1. Load raw data ──
df = pd.read_csv('data/raw/golay_monthly.csv')

print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# ── 2. Find the date column ──
# The Goyal-Welch file usually has a date column like 'yyyymm' or 'Date'
# Print first few values to identify format
for col in df.columns:
    print(f"\n{col}: {df[col].head(3).tolist()}")

# ── 3. Map to the 8 variables from the paper ──
# The paper uses: D12, E12, bm, dfy, ntis, tbl, tms, svar
# The Goyal-Welch file may use different names
# Common mappings:
#   D12  -> dp or D12 (log dividend-price ratio)
#   E12  -> ep or E12 (log earnings-price ratio)
#   bm   -> bm (book-to-market for DJIA)
#   dfy  -> dfy (default yield spread)
#   ntis -> ntis (net equity expansion)
#   tbl  -> tbl (Treasury bill rate)
#   tms  -> tms (term spread)
#   svar -> svar (stock variance)

# Try to find matching columns (case-insensitive)
target_vars = ['D12', 'E12', 'bm', 'dfy', 'ntis', 'tbl', 'tms', 'svar']

# Build a case-insensitive mapping
col_lower_map = {col.lower().strip(): col for col in df.columns}

rename_map = {}
found_vars = []
missing_vars = []

for var in target_vars:
    if var in df.columns:
        found_vars.append(var)
    elif var.lower() in col_lower_map:
        rename_map[col_lower_map[var.lower()]] = var
        found_vars.append(var)
    else:
        missing_vars.append(var)

if rename_map:
    df = df.rename(columns=rename_map)

print(f"\nFound variables: {found_vars}")
if missing_vars:
    print(f"Missing variables: {missing_vars}")
    print("You may need to manually map column names.")
    print(f"Available columns: {df.columns.tolist()}")

# ── 4. Identify and format the date column ──
# Look for a date-like column
date_col = None
for col in df.columns:
    if col.lower() in ['yyyymm', 'date', 'yearmon', 'year_month']:
        date_col = col
        break

if date_col is None:
    # Try the first column
    date_col = df.columns[0]
    print(f"\nAssuming '{date_col}' is the date column")

# Convert to yyyymm integer format
sample_val = str(df[date_col].iloc[0])
print(f"\nDate sample: {sample_val}")

if len(sample_val) == 6:
    # Already yyyymm format like 202001
    df['yyyymm'] = df[date_col].astype(int)
elif '-' in sample_val:
    # Format like 2020-01 or 2020-01-01
    df['yyyymm'] = pd.to_datetime(df[date_col]).dt.year * 100 + pd.to_datetime(df[date_col]).dt.month
elif '/' in sample_val:
    # Format like 01/2020
    df['yyyymm'] = pd.to_datetime(df[date_col]).dt.year * 100 + pd.to_datetime(df[date_col]).dt.month
else:
    # Try direct conversion
    df['yyyymm'] = df[date_col].astype(int)

# ── 5. Select only the columns we need ──
keep_cols = ['yyyymm'] + [v for v in target_vars if v in df.columns]
output = df[keep_cols].copy()

# ── 6. Convert to numeric (some columns may be strings) ──
for col in target_vars:
    if col in output.columns:
        output[col] = pd.to_numeric(output[col], errors='coerce')

# ── 7. Drop rows with missing dates ──
output = output.dropna(subset=['yyyymm'])
output['yyyymm'] = output['yyyymm'].astype(int)

# ── 8. Summary and save ──
print(f"\nCleaned macro data:")
print(f"  Shape: {output.shape}")
print(f"  Date range: {output['yyyymm'].min()} to {output['yyyymm'].max()}")
print(f"  Columns: {output.columns.tolist()}")
print(f"\nSample:")
print(output.tail(10))

output.to_csv('data/processed/cleaned_macro.csv', index=False)
print(f"\nSaved to data/processed/cleaned_macro.csv")