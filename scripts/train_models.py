# Use the expanding window approach 
# Round 1: Train on Oct-Dec 2020 → Predict Q1 2021 (Jan-Mar)
# Round 2: Train on Oct 2020 - Mar 2021 → Predict Q2 2021 (Apr-Jun)
# Round 3: Train on Oct 2020 - Jun 2021 → Predict Q3 2021 (Jul-Sep) 

"""
06_train_models.py

Train ML models to predict log(SAS) using expanding window.

Input:
    - data/processed/modeling_dataset.csv

Output:
    - data/processed/sas_predictions.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('data/processed/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Define features and target
feature_cols = ['mom1m', 'mom6m', 'mom12m', 'volatility_12m', 'max_ret',
                'dollar_vol', 'sharpe_12m', 'high_52w',
                'D12', 'E12', 'dfy', 'ntis', 'tbl', 'tms', 'svar']

# Drop any feature columns that don't exist
feature_cols = [c for c in feature_cols if c in df.columns]
target = 'log_sas'

print(f"Features: {feature_cols}")
print(f"Total rows: {len(df)}")

# Instead of dropping NaN rows, fill them
df[feature_cols] = df.groupby('yyyymm')[feature_cols].transform(
    lambda x: x.fillna(x.mean())
)
df = df.dropna(subset=feature_cols + [target])
print(f"After handling NaN: {len(df)}")

# Get quarterly periods
df['quarter'] = df['date'].dt.to_period('Q')
quarters = sorted(df['quarter'].unique())
print(f"Quarters: {quarters}")

# Store predictions
all_predictions = []

for i in range(1, len(quarters)):
    test_quarter = quarters[i]

    train = df[df['quarter'] < test_quarter]
    test = df[df['quarter'] == test_quarter]

    if len(train) < 50 or len(test) == 0:
        print(f"Skipping {test_quarter}: train={len(train)}, test={len(test)}")
        continue

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    models = {
        'OLS': LinearRegression(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)

        results = test[['date', 'ticker', 'sas', 'log_sas']].copy()
        results['predicted_log_sas'] = preds
        results['model'] = name
        all_predictions.append(results)

        print(f"  {test_quarter} | {name:20s} | R2: {r2:.4f} | Train: {len(train)} | Test: {len(test)}")

predictions = pd.concat(all_predictions, ignore_index=True)
predictions.to_csv('data/processed/sas_predictions.csv', index=False)
print(f"\nSaved predictions: {predictions.shape}")
print(f"Models: {predictions['model'].unique()}")

