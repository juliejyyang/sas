import pandas as pd

with open('data/raw/fama_factor.csv', 'r') as f:
    for i, line in enumerate(f):
        if 'Annual' in line or 'annual' in line:
            print(f"Line {i}: {line.strip()}")