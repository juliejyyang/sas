import pandas as pd
import csv

df = pd.read_csv('data/raw/sp500.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.groupby(df['date'].dt.to_period('M')).tail(1)
df.to_csv('data/processed/cleaned_sp500.csv', index=False)

print(df.head()) 