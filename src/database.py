import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Text

DB_URL = "postgresql://postgres.zolipjvrqejnhbendclq:fLXxkf42l6NtY@aws-1-eu-north-1.pooler.supabase.com:6543/postgres"

# Use raw string or double backslashes to avoid escape sequence issues
FOLDER = r"C:\Users\User\samurai0022\data\improved_accounting_data.csv"

engine = create_engine(DB_URL)

# Since FOLDER is a file, no need for os.listdir loop
table = os.path.splitext(os.path.basename(FOLDER))[0]  # Extract table name from file name
path = FOLDER
print("Loading", path, "->", table)
df = pd.read_csv(path)  # можно добавить encoding='cp1251' или sep=';' if needed
# Загружаем в базу (заменяем таблицу, все колонки как TEXT)
df.to_sql(table, engine, if_exists='replace', index=False,
          dtype={c: Text() for c in df.columns})