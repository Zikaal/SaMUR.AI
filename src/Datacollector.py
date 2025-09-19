import requests
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import os
import math  # Для обработки inf/NaN

# Указываем абсолютный путь к папке с данными
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Список поддерживаемых банков (S&P 500 + Sberbank)
BANK_SYMBOLS = ['WFC', 'JPM', 'BAC', 'C', 'USB', 'PNC', 'TFC', 'HBAN', 'FITB', 'RF', 'KEY', 'MTB', 'CFG', 'BBT', 'NUE', 'SBER']

# Tool 1: Фетч курсов валют из ЦБ РФ (реальный API)
def get_currency_rates(date: str = None) -> Dict[str, float]:
    """
    Фетчит ежедневные курсы валют из XML ЦБ РФ.
    :param date: Опционально, дата в формате YYYY-MM-DD (по умолчанию — текущая).
    :return: Dict {code: rate} для major валют (USD, EUR, CNY, GBP, JPY).
    """
    try:
        url = "https://www.cbr.ru/scripts/XML_daily.asp"
        if date:
            day, month, year = date.split('-')
            url += f"?date_req={day}/{month}/{year}"
        else:
            # Текущая дата с учетом часового пояса +05
            current_time = datetime.now() + timedelta(hours=5)
            date = current_time.strftime('%Y-%m-%d')
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
def get_transactions(start_date: str, end_date: str, account_type: str = None, symbol: str = None,
                     handle_noisy: bool = True, real_time: bool = False) -> pd.DataFrame:
    """
    Загружает реальные транзакции из CSV-датасета или создает mock-данные.
    Фильтрует по дате, типу аккаунта и символу банка (Symbol).
    Обработка noisy: заполняет missing (median), добавляет ~5% gaussian noise для теста.
    real_time: Если True, использует последние 24 часа с учетом +05.
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :param account_type: 'Revenue', 'Expense', etc.
    :param symbol: 'WFC', 'JPM', etc. (фильтр по Symbol)
    :param handle_noisy: Включить обработку dirty data.
    :param real_time: Включить режим реального времени.
    :return: DataFrame с данными.
    """
    file_path = os.path.join(DATA_PATH, 'improved_accounting_data.csv')
    if not os.path.exists(file_path):
        # Mock-данные для тестирования, если CSV отсутствует (с Symbol и Liquidity Ratio)
        dates = pd.date_range(start='2025-01-01', end='2025-09-19', freq='D')[:100]  # До текущей даты
        mock_data = {
            'Date': np.random.choice(dates, 100),
            'Account Type': np.random.choice(['Asset', 'Liability', 'Cash', 'Inventory', 'Revenue', 'Expense', 'Purchase'], 100),
            'Transaction Amount': np.random.uniform(100, 10000, 100),
            'Cash Flow': np.random.uniform(-5000, 5000, 100),
            'Net Income': np.random.uniform(1000, 4000, 100),
            'Revenue': np.random.uniform(2000, 6000, 100),
            'Expenditure': np.random.uniform(500, 3000, 100),
            'Profit Margin': np.random.uniform(0.05, 0.20, 100),
            'Debt-to-Equity Ratio': np.random.uniform(0.1, 2.0, 100),
            'Operating Expenses': np.random.uniform(1000, 5000, 100),
            'Gross Profit': np.random.uniform(500, 3000, 100),
            'Transaction Volume': np.random.randint(1, 10, 100),
            'Processing Time (seconds)': np.random.uniform(0.5, 3.0, 100),
            'Accuracy Score': np.random.uniform(0.8, 1.0, 100),
            'Missing Data Indicator': np.random.choice([False, True], 100, p=[0.95, 0.05]),
            'Normalized Transaction Amount': np.random.uniform(0, 1, 100),
            'Transaction Outcome': np.random.choice([0, 1], 100),
            'Symbol': np.random.choice(BANK_SYMBOLS, 100),  # Случайные символы банков
            'Liquidity Ratio': np.random.uniform(0.5, 3.0, 100)  # Liquidity Ratio
        }
        df = pd.DataFrame(mock_data)
        print(f"Warning: CSV file not found at {file_path}. Using mock data with Symbol and Liquidity Ratio.")
    else:
        df = pd.read_csv(file_path)
    
    # Конверт Date в datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    if real_time:
        current_time = datetime.now() + timedelta(hours=5)  # +05
        end_date = current_time.strftime('%Y-%m-%d')
        start_date = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Фильтр по дате
    mask_date = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask_date]
    
    # Фильтр по типу аккаунта
    if account_type:
        df = df[df['Account Type'] == account_type]
    
    # Фильтр по символу банка
    if symbol and symbol in BANK_SYMBOLS:
        df = df[df['Symbol'] == symbol]
    
    if handle_noisy and len(df) > 0:
        # Обработка missing (используя Missing Data Indicator)
        for col in ['Transaction Amount', 'Cash Flow', 'Liquidity Ratio']:
            median_val = df[col].median()
            df.loc[df['Missing Data Indicator'] == True, col] = median_val
        df['Missing Data Indicator'] = False
        
        # Добавляем шум для теста robustness (5% от amount)
        noise = np.random.normal(0, 0.05 * df['Transaction Amount'])
        df['Transaction Amount'] += noise
        df['Normalized Transaction Amount'] = (df['Transaction Amount'] - df['Transaction Amount'].min()) / (df['Transaction Amount'].max() - df['Transaction Amount'].min())
        
        # Шум для Liquidity Ratio
        liquidity_noise = np.random.normal(0, 0.02 * df['Liquidity Ratio'])
        df['Liquidity Ratio'] = np.clip(df['Liquidity Ratio'] + liquidity_noise, 0.1, 5.0)
    
    if len(df) == 0:
        raise ValueError("No transactions found for the given date range, account type, or symbol.")
    
    return df

# Расчёт метрик ликвидности (улучшено с Liquidity Ratio)
def calculate_liquidity_metrics(transactions: pd.DataFrame) -> Dict[str, Any]:
    """
    Рассчитывает метрики ликвидности на основе транзакций.
    Приоритет: среднее 'Liquidity Ratio' из датасета; fallback на расчет.
    Округляет значения до 2 знаков после запятой.
    Возвращает 0 вместо Infinity при отсутствии обязательств.
    """
    try:
        if 'Liquidity Ratio' in transactions.columns and len(transactions) > 0:
            avg_liquidity = round(transactions['Liquidity Ratio'].mean(), 2)
            status = 'Success (using direct Liquidity Ratio)'
        else:
            # Fallback: традиционный расчет
            current_assets = transactions[transactions['Account Type'].str.contains('Asset', case=False, na=False)]['Transaction Amount'].sum()
            current_liabilities = transactions[transactions['Account Type'].str.contains('Liability', case=False, na=False)]['Transaction Amount'].sum()
            cash = transactions[transactions['Account Type'].str.contains('Cash', case=False, na=False)]['Transaction Amount'].sum()
            inventory = transactions[transactions['Account Type'].str.contains('Inventory', case=False, na=False)]['Transaction Amount'].sum()
            
            if current_liabilities <= 0:
                return {
                    'Liquidity Ratio': 0.0,
                    'Current Ratio': 0.0,
                    'Quick Ratio': 0.0,
                    'Cash Ratio': 0.0,
                    'status': 'No or negative liabilities found. Ratios set to 0.',
                    'recommendation': 'Verify dataset for liability transactions or reduce asset exposure.'
                }
            
            avg_liquidity = round(current_assets / current_liabilities, 2)
            current_ratio = round(current_assets / current_liabilities, 2)
            quick_ratio = round((current_assets - inventory) / current_liabilities, 2)
            cash_ratio = round(cash / current_liabilities, 2)
            status = 'Success (calculated from assets/liabilities)'
        
        # Генерация рекомендации на основе Liquidity Ratio
        recommendation = "Liquidity is stable."
        if avg_liquidity < 1.0:
            recommendation = "Low liquidity risk: Increase current assets or reduce liabilities immediately."
        elif avg_liquidity < 1.5:
            recommendation = "Monitor liquidity closely; consider short-term financing."
        elif avg_liquidity > 2.0:
            recommendation = "Strong liquidity position; explore investment opportunities."
        
        return {
            'Liquidity Ratio': avg_liquidity,
            'Current Ratio': avg_liquidity,  # Совместимо с fallback
            'Quick Ratio': round(avg_liquidity * 0.8, 2),
            'Cash Ratio': round(avg_liquidity * 0.5, 2),
            'status': status,
            'recommendation': recommendation
        }
    except Exception as e:
        return {
            'Liquidity Ratio': 0.0,
            'Current Ratio': 0.0,
            'Quick Ratio': 0.0,
            'Cash Ratio': 0.0,
            'status': f'Error calculating liquidity: {str(e)}',
            'recommendation': 'Check data integrity and try again.'
        }

# Интеграция: Пример end-to-end сбора (для теста/демо)
def collect_integrated_data(start_date: str, end_date: str, account_type: str = None, symbol: str = None, real_time: bool = False) -> Dict[str, Any]:
    """
    Комбинирует курсы и транзакции в один JSON для агента/ERP.
    Фильтр по symbol; summary per-symbol если multiple.
    Готово для REST: return как response.json().
    """
    try:
        rates = get_currency_rates()
        transactions = get_transactions(start_date, end_date, account_type, symbol, handle_noisy=True, real_time=real_time)
        
        # Простая интеграция: добавляем колонку с курсом USD к транзакциям
        usd_rate = rates.get('USD', 1.0)
        if usd_rate == 0:
            usd_rate = 1.0  # Fallback
        transactions['USD_Equivalent'] = transactions['Transaction Amount'] / usd_rate
        
        total_cash_flow = round(transactions['Cash Flow'].sum(), 2)
        total_transactions = len(transactions)
        avg_liquidity = round(transactions['Liquidity Ratio'].mean(), 2) if 'Liquidity Ratio' in transactions.columns else 0.0
        
        # Группировка по Symbol для multi-bank (если symbol=None)
        if symbol is None and len(transactions) > 0:
            summary_by_symbol = transactions.groupby('Symbol').agg({
                'Cash Flow': 'sum',
                'Liquidity Ratio': 'mean',
                'Transaction Outcome': 'mean'
            }).round(2).to_dict('index')
        else:
            summary_by_symbol = {}
        
        # Генерация рекомендации
        recommendation = "Financial position is stable."
        if total_cash_flow < 0:
            recommendation = f"Negative cash flow ({total_cash_flow:.2f}); reduce expenses."
        elif avg_liquidity < 1.0:
            recommendation += f" Low liquidity ({avg_liquidity}); alert for {symbol or 'portfolio'}."
        elif total_transactions < 10:
            recommendation = "Limited data; expand for better insights."
        
        data = {
            'date_range': {'start': start_date, 'end': end_date},
            'currency_rates': rates,
            'transactions': transactions.to_dict('records'),  # JSON-serializable
            'summary': {
                'total_transactions': total_transactions,
                'total_cash_flow': total_cash_flow,
                'avg_profit_margin': round(transactions['Profit Margin'].mean(), 2),
                'avg_liquidity_ratio': avg_liquidity,
                'recommendation': recommendation
            },
            'summary_by_symbol': summary_by_symbol
        }
        # Добавляем метрики ликвидности в summary
        data['summary'].update(calculate_liquidity_metrics(transactions))
        return data
    except Exception as e:
        raise Exception(f"Error in collect_integrated_data: {str(e)}")

# Тест на noisy data (с учетом текущей даты и времени)
if __name__ == "__main__":
    # Текущая дата и время: 2025-09-19 11:15 AM +05
    current_time = datetime.now() + timedelta(hours=5)
    start = '2025-01-01'
    end = current_time.strftime('%Y-%m-%d')
    
    # Clean data для JPM
    clean_data = collect_integrated_data(start, end, symbol='JPM')
    print(f"Clean JPM: {len(clean_data['transactions'])} records, Cash Flow sum: {clean_data['summary']['total_cash_flow']:.2f}, Liquidity: {clean_data['summary']['avg_liquidity_ratio']:.2f}")
    
    # Noisy data для SBER
    noisy_data = collect_integrated_data(start, end, symbol='SBER', handle_noisy=True)
    print(f"Noisy SBER: {len(noisy_data['transactions'])} records, Cash Flow sum: {noisy_data['summary']['total_cash_flow']:.2f}, Liquidity: {noisy_data['summary']['avg_liquidity_ratio']:.2f}")
    
    # Вывод в JSON для API (multi-symbol)
    print(json.dumps(collect_integrated_data(start, end), indent=2, default=str))