import pandas as pd
import csv

df = pd.read_csv('data/raw/etf_holdings.csv')

df['date'] = pd.to_datetime(df['date'])

#Keep only the last trading day per month, per fund
df = df.groupby(['fund', df['date'].dt.to_period('M')]).apply(
    lambda x: x[x['date'] == x['date'].max()]
).reset_index(drop=True)

df.to_csv('data/processed/clean_holdings.csv', index=False)

print(df.head()) 




