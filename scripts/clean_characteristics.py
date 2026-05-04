import pandas as pd

df = pd.read_csv('data/raw/characteristics.csv')

id_cols = ['permno', 'yyyymm']

#drop columns with more than 70% of values as NaN
df = df.dropna(thresh=len(df)*0.70, axis=1) 
feature_cols = [col for col in df.columns if col not in id_cols]

#replace the remaining NaNs with cross sectional averages across the monthly periods 
df[feature_cols] = df.groupby('yyyymm')[feature_cols].transform(lambda x: x.fillna(x.mean()))

#standardize based on the paper's specifications
df[feature_cols] = df.groupby('yyyymm')[feature_cols].transform(lambda x: (x - x.mean())/x.std())

#export as a csv
df.to_csv('data/processed/cleaned_characteristics.csv', index=False)
print(df.head())

