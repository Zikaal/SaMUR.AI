import pandas as pd
import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

file_path = os.path.join(DATA_PATH, 'accounting_data.csv')

df = pd.read_csv(file_path)

bank_symbols = ['WFC', 'JPM', 'BAC', 'C', 'USB', 'PNC', 'TFC', 'HBAN', 'FITB', 'RF', 'KEY', 'MTB', 'CFG', 'BBT', 'NUE']

df['Symbol'] = np.random.choice(bank_symbols, size=len(df), replace=True)

# Добавление нового столбца: 'Liquidity Ratio' = (Cash Flow + Net Income) / (Expenditure + 1)
df['Liquidity Ratio'] = (df['Cash Flow'] + df['Net Income']) / (df['Expenditure'] + 1)

print(df.head())

df.to_csv('improved_accounting_data.csv', index=False)
print("Датасет улучшен и сохранен как 'improved_accounting_data.csv'")