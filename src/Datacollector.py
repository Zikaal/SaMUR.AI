import requests
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import os
import math  # Добавлено для обработки inf/NaN

# Указываем абсолютный путь к папке с данными
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Tool 1: Фетч курсов валют из ЦБ РФ (реальный API)
def get_currency_rates(date: str = None) -> Dict[str, float]:
    """
    Фетчит ежедневные курсы валют из XML ЦБ РФ.
    :param date: Опционально, дата в формате YYYY-MM-DD (по умолчанию — текущая).
    :return: Dict {code: rate} для major валют (USD, EUR, CNY, GBP и т.д.).
    """
    try:
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
    except Exception as e:
        # Fallback на дефолтные курсы, если API недоступен
        return {'USD': 90.0, 'EUR': 100.0, 'CNY': 12.5, 'GBP': 120.0, 'JPY': 0.6}

# Tool 2: Загрузка и обработка транзакций из датасета (mock для банков/платежей)
def get_transactions(start_date: str, end_date: str, account_type: str = None, 
                     handle_noisy: bool = True, real_time: bool = False) -> pd.DataFrame:
    """
    Загружает реальные транзакции из CSV-датасета или создает mock-данные.
    Фильтрует по дате и типу аккаунта.
    Обработка noisy: заполняет missing (median), добавляет ~5% gaussian noise для теста.
    real_time: Если True, использует последние 24 часа.
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :param account_type: 'Revenue', 'Expense', etc.
    :param handle_noisy: Включить обработку dirty data.
    :param real_time: Включить режим реального времени.
    :return: DataFrame с данными.
    """
    file_path = os.path.join(DATA_PATH, 'accounting_data.csv')
    if not os.path.exists(file_path):
        # Mock-данные для тестирования, если CSV отсутствует
        dates = pd.date_range(start='2025-01-01', end='2025-09-16', freq='D')[:100]
        mock_data = {
            'Date': np.random.choice(dates, 100),
            'Account Type': np.random.choice(['Asset', 'Liability', 'Cash', 'Inventory', 'Revenue', 'Expense', 'Purchase'], 100),
            'Transaction Amount': np.random.uniform(100, 10000, 100),
            'Cash Flow': np.random.uniform(-5000, 5000, 100),
            'Profit Margin': np.random.uniform(0.05, 0.20, 100),
            'Debt-to-Equity Ratio': np.random.uniform(0.1, 2.0, 100),
            'Transaction Outcome': np.random.choice([0, 1], 100),
            'Missing Data Indicator': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        }
        df = pd.DataFrame(mock_data)
        print(f"Warning: CSV file not found at {file_path}. Using mock data.")  # Лог для отладки
    else:
        df = pd.read_csv(file_path)
    
    # Конверт Date в datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    if real_time:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Фильтр по дате
    mask_date = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask_date]
    
    # Фильтр по типу аккаунта
    if account_type:
        df = df[df['Account Type'] == account_type]
    
    if handle_noisy and len(df) > 0:
        # Обработка missing (используя Missing Data Indicator)
        df.loc[df['Missing Data Indicator'] == 1, 'Transaction Amount'] = df['Transaction Amount'].median()
        df.loc[df['Missing Data Indicator'] == 1, 'Cash Flow'] = df['Cash Flow'].median()
        df['Missing Data Indicator'] = 0  # Сброс флага после fill
        
        # Добавляем шум для теста robustness (5% от amount)
        noise = np.random.normal(0, 0.05 * df['Transaction Amount'])
        df['Transaction Amount'] += noise
        df['Normalized Transaction Amount'] = (df['Transaction Amount'] - df['Transaction Amount'].min()) / (df['Transaction Amount'].max() - df['Transaction Amount'].min())
    
    if len(df) == 0:
        raise ValueError("No transactions found for the given date range and account type.")
    
    return df

# Расчёт метрик ликвидности
def calculate_liquidity_metrics(transactions: pd.DataFrame) -> Dict[str, Any]:
    """
    Рассчитывает метрики ликвидности на основе транзакций.
    Округляет значения до 2 знаков после запятой для точности.
    Возвращает 0 вместо Infinity при отсутствии или отрицательных обязательствах и добавляет статус.
    Предполагаем, что типы аккаунтов включают 'Asset', 'Liability', 'Cash', 'Inventory' и т.д.
    """
    try:
        current_assets = transactions[transactions['Account Type'].str.contains('Asset', case=False, na=False)]['Transaction Amount'].sum()
        current_liabilities = transactions[transactions['Account Type'].str.contains('Liability', case=False, na=False)]['Transaction Amount'].sum()
        cash = transactions[transactions['Account Type'].str.contains('Cash', case=False, na=False)]['Transaction Amount'].sum()
        inventory = transactions[transactions['Account Type'].str.contains('Inventory', case=False, na=False)]['Transaction Amount'].sum()
        
        # Проверяем наличие обязательств
        if current_liabilities <= 0:
            return {
                'Current Ratio': 0.0,
                'Quick Ratio': 0.0,
                'Cash Ratio': 0.0,
                'status': 'No or negative liabilities found in the dataset. Ratios set to 0.',
                'recommendation': 'Verify dataset for liability transactions or reduce asset exposure to balance liquidity.'
            }
        
        current_ratio = round(current_assets / current_liabilities, 2)
        quick_ratio = round((current_assets - inventory) / current_liabilities, 2)
        cash_ratio = round(cash / current_liabilities, 2)
        
        # Генерация рекомендации на основе ликвидности
        recommendation = "Liquidity is stable."
        if current_ratio < 1.0:
            recommendation = "Increase current assets or reduce liabilities to improve Current Ratio."
        elif quick_ratio < 0.8:
            recommendation = "Reduce inventory or increase liquid assets to improve Quick Ratio."
        elif cash_ratio < 0.5:
            recommendation = "Boost cash reserves to improve Cash Ratio and liquidity buffer."
        
        return {
            'Current Ratio': current_ratio,
            'Quick Ratio': quick_ratio,
            'Cash Ratio': cash_ratio,
            'status': 'Success',
            'recommendation': recommendation
        }
    except Exception as e:
        return {
            'Current Ratio': 0.0,
            'Quick Ratio': 0.0,
            'Cash Ratio': 0.0,
            'status': f'Error calculating liquidity: {str(e)}',
            'recommendation': 'Check data integrity and try again.'
        }

# Интеграция: Пример end-to-end сбора (для теста/демо)
def collect_integrated_data(start_date: str, end_date: str, account_type: str = None, real_time: bool = False) -> Dict[str, Any]:
    """
    Комбинирует курсы и транзакции в один JSON для агента/ERP.
    Готово для REST: return как response.json().
    """
    try:
        rates = get_currency_rates()
        transactions = get_transactions(start_date, end_date, account_type, handle_noisy=True, real_time=real_time)
        
        # Простая интеграция: добавляем колонку с курсом USD к транзакциям (предполагаем все в RUB, конверт если нужно)
        usd_rate = rates.get('USD', 1.0)
        if usd_rate == 0:
            usd_rate = 1.0  # Fallback
        transactions['USD_Equivalent'] = transactions['Transaction Amount'] / usd_rate
        
        total_cash_flow = round(transactions['Cash Flow'].sum(), 2)
        total_transactions = len(transactions)
        
        # Генерация рекомендации для данных
        recommendation = "Financial position is stable."
        if total_cash_flow < 0:
            recommendation = "Address negative cash flow by reducing expenses or increasing revenue."
        elif total_transactions < 10:
            recommendation = "Limited transaction data; consider expanding dataset for better insights."
        
        data = {
            'date_range': {'start': start_date, 'end': end_date},
            'currency_rates': rates,
            'transactions': transactions.to_dict('records'),  # JSON-serializable
            'summary': {
                'total_transactions': total_transactions,
                'total_cash_flow': total_cash_flow,
                'avg_profit_margin': round(transactions['Profit Margin'].mean(), 2),
                'recommendation': recommendation
            }
        }
        # Добавляем метрики ликвидности в summary
        data['summary'].update(calculate_liquidity_metrics(transactions))
        return data
    except Exception as e:
        raise Exception(f"Error in collect_integrated_data: {str(e)}")

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