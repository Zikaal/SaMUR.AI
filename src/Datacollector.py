import requests
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import json
import os

# Указываем абсолютный путь к папке с данными
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Tool 1: Фетч курсов валют из ЦБ РФ (реальный API)
def get_currency_rates(date: str = None) -> Dict[str, float]:
    """
    Фетчит ежедневные курсы валют из XML ЦБ РФ.
    :param date: Опционально, дата в формате YYYY-MM-DD (по умолчанию — текущая).
    :return: Dict {code: rate} для major валют (USD, EUR, CNY, GBP и т.д.).
    """
    url = "https://www.cbr.ru/scripts/XML_daily.asp"
    if date:
        day, month, year = date.split('-')
        url += f"?date_req={day}/{month}/{year}"
    
    response = requests.get(url)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    rates = {}
    for valute in root.findall('Valute'):
        code = valute.find('CharCode').text
        nominal = int(valute.find('Nominal').text)
        value_str = valute.find('Value').text.replace(',', '.')
        value = float(value_str) / nominal  # Нормализуем на 1 единицу
        rates[code] = value
    
    major_codes = ['USD', 'EUR', 'CNY', 'GBP', 'JPY']
    return {code: rates.get(code, 0) for code in major_codes}

# Tool 2: Загрузка и обработка транзакций из датасета (mock для банков/платежей)
def get_transactions(start_date: str, end_date: str, account_type: str = None, handle_noisy: bool = True) -> pd.DataFrame:
    """
    Загружает реальные транзакции из CSV-датасета.
    Фильтрует по дате и типу аккаунта.
    Обработка noisy: заполняет missing (median), добавляет ~5% gaussian noise для теста.
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :param account_type: 'Revenue', 'Expense', etc.
    :param handle_noisy: Включить обработку dirty data.
    :return: DataFrame с данными.
    """
    # Загружаем датасет из указанной папки
    file_path = os.path.join(DATA_PATH, 'accounting_data.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}. Убедитесь, что accounting_data.csv находится в папке {DATA_PATH}.")
    df = pd.read_csv(file_path)
    
    # Конверт Date в datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Фильтр по дате
    mask_date = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask_date]
    
    # Фильтр по типу аккаунта
    if account_type:
        df = df[df['Account Type'] == account_type]
    
    if handle_noisy:
        # Обработка missing (используя Missing Data Indicator)
        df.loc[df['Missing Data Indicator'] == 1, 'Transaction Amount'] = df['Transaction Amount'].median()
        df.loc[df['Missing Data Indicator'] == 1, 'Cash Flow'] = df['Cash Flow'].median()
        df['Missing Data Indicator'] = 0  # Сброс флага после fill
        
        # Добавляем шум для теста robustness (5% от amount)
        noise = np.random.normal(0, 0.05 * df['Transaction Amount'])
        df['Transaction Amount'] += noise
        df['Normalized Transaction Amount'] = (df['Transaction Amount'] - df['Transaction Amount'].min()) / (df['Transaction Amount'].max() - df['Transaction Amount'].min())
    
    return df

# Интеграция: Пример end-to-end сбора (для теста/демо)
def collect_integrated_data(start_date: str, end_date: str, account_type: str = None) -> Dict[str, Any]:
    """
    Комбинирует курсы и транзакции в один JSON для агента/ERP.
    Готово для REST: return как response.json().
    """
    rates = get_currency_rates()
    transactions = get_transactions(start_date, end_date, account_type, handle_noisy=True)  # Передаем handle_noisy
    
    # Простая интеграция: добавляем колонку с курсом USD к транзакциям (предполагаем все в RUB, конверт если нужно)
    usd_rate = rates.get('USD', 1.0)
    transactions['USD_Equivalent'] = transactions['Transaction Amount'] / usd_rate
    
    data = {
        'date_range': {'start': start_date, 'end': end_date},
        'currency_rates': rates,
        'transactions': transactions.to_dict('records'),  # JSON-serializable
        'summary': {
            'total_transactions': len(transactions),
            'total_cash_flow': transactions['Cash Flow'].sum(),
            'avg_profit_margin': transactions['Profit Margin'].mean()
        }
    }
    return data

# Тест на noisy data (без вывода сэмплов — только метрики)
if __name__ == "__main__":
    # Пример вызова (замените на реальные даты из датасета)
    start = '2025-01-01'
    end = '2025-09-16'  # Текущая дата
    
    # Clean data
    clean_data = collect_integrated_data(start, end)  # handle_noisy=False по умолчанию в get_transactions
    print(f"Clean: {len(clean_data['transactions'])} records, Cash Flow sum: {clean_data['summary']['total_cash_flow']:.2f}")
    
    # Noisy data
    noisy_data = collect_integrated_data(start, end)  # handle_noisy=True по умолчанию в get_transactions
    print(f"Noisy: {len(noisy_data['transactions'])} records, Cash Flow sum: {noisy_data['summary']['total_cash_flow']:.2f}")
    
    # Вывод в JSON для API
    print(json.dumps(collect_integrated_data(start, end), indent=2, default=str))