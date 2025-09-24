import requests
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import os
import math
from sqlalchemy import create_engine




# Список поддерживаемых банков
BANK_SYMBOLS = ['WFC', 'JPM', 'BAC', 'C', 'USB', 'PNC', 'TFC', 'HBAN', 'FITB', 'RF', 'KEY', 'MTB', 'CFG', 'BBT', 'NUE', 'SBER']

# Tool 1: Фетч курсов валют из ЦБ РФ
def get_currency_rates(date: str = None) -> Dict[str, float]:
    try:
        url = "https://www.cbr.ru/scripts/XML_daily.asp"
        if date:
            day, month, year = date.split('-')
            url += f"?date_req={day}/{month}/{year}"
        else:
            current_time = datetime.now() + timedelta(hours=5)  # 05:59 PM +05, 2025-09-19
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
            value = float(value_str) / nominal
            rates[code] = value
        
        major_codes = ['USD', 'EUR', 'CNY', 'GBP', 'JPY']
        return {code: rates.get(code, 0) for code in major_codes}
    except Exception as e:
        return {'USD': 90.0, 'EUR': 100.0, 'CNY': 12.5, 'GBP': 120.0, 'JPY': 0.6}

# Tool 2: Загрузка и обработка транзакций
def get_transactions(start_date: str, end_date: str, account_type: str = None, symbol: str = None,
                     handle_noisy: bool = True, real_time: bool = False) -> pd.DataFrame:
    # Build WHERE clause conditions first, then add LIMIT at the end
    query = 'SELECT * FROM improved_accounting_data WHERE "Date" BETWEEN %s AND %s'
    params = (start_date, end_date)
    
    if real_time:
        current_time = datetime.now() + timedelta(hours=5)  # 05:59 PM +05, 2025-09-19
        end_date = current_time.strftime('%Y-%m-%d')
        start_date = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')
        params = (start_date, end_date)
    
    if account_type:
        query += ' AND "Account Type" = %s'
        params += (account_type,)
    if symbol and symbol in BANK_SYMBOLS:
        query += ' AND "Symbol" = %s'
        params += (symbol,)
    
    # Add LIMIT at the end of the query
    query += ' LIMIT 700'
    
    df = pd.read_sql(query, engine, params=params)
    
    # Since dates are stored as text, convert to datetime for processing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Use errors='coerce' to handle invalid dates
    
    if handle_noisy and len(df) > 0:
        # Handle missing data columns with proper column name references
        numeric_columns = ['Transaction Amount', 'Cash Flow', 'Liquidity Ratio']
        for col in numeric_columns:
            if col in df.columns:
                # Convert string columns to numeric first
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                # Handle missing data indicator if it exists
                if 'Missing Data Indicator' in df.columns:
                    df.loc[df['Missing Data Indicator'] == True, col] = median_val
                    
        # Set missing data indicator to False if it exists
        if 'Missing Data Indicator' in df.columns:
            df['Missing Data Indicator'] = False
            
        # Add noise to transaction amounts if column exists
        if 'Transaction Amount' in df.columns and len(df) > 0:
            noise = np.random.normal(0, 0.05 * df['Transaction Amount'].std())
            df['Transaction Amount'] += noise
            df['Normalized Transaction Amount'] = (
                (df['Transaction Amount'] - df['Transaction Amount'].min()) / 
                (df['Transaction Amount'].max() - df['Transaction Amount'].min() + 1e-10)  # Add small epsilon to avoid division by zero
            )
            
        # Add noise to liquidity ratio if column exists
        if 'Liquidity Ratio' in df.columns and len(df) > 0:
            liquidity_noise = np.random.normal(0, 0.02 * df['Liquidity Ratio'].std())
            df['Liquidity Ratio'] = np.clip(df['Liquidity Ratio'] + liquidity_noise, 0.1, 5.0)
    
    if len(df) == 0:
        raise ValueError("No transactions found.")
    
    return df

# Расчёт метрик ликвидности
def calculate_liquidity_metrics(transactions: pd.DataFrame) -> Dict[str, Any]:
    try:
        # Convert numeric columns to proper numeric types first
        numeric_columns = ['Transaction Amount', 'Cash Flow', 'Liquidity Ratio']
        for col in numeric_columns:
            if col in transactions.columns:
                transactions[col] = pd.to_numeric(transactions[col], errors='coerce')
        
        if 'Liquidity Ratio' in transactions.columns and len(transactions) > 0:
            avg_liquidity = round(transactions['Liquidity Ratio'].mean(), 2)
            status = 'Success (using direct Liquidity Ratio)'
        else:
            # Calculate liquidity using account types if direct ratio not available
            current_assets = transactions[
                transactions['Account Type'].str.contains('Asset', case=False, na=False)
            ]['Transaction Amount'].sum()
            
            current_liabilities = transactions[
                transactions['Account Type'].str.contains('Liability', case=False, na=False)
            ]['Transaction Amount'].sum()
            
            cash = transactions[
                transactions['Account Type'].str.contains('Cash', case=False, na=False)
            ]['Transaction Amount'].sum()
            
            inventory = transactions[
                transactions['Account Type'].str.contains('Inventory', case=False, na=False)
            ]['Transaction Amount'].sum()
            
            if current_liabilities <= 0:
                return {
                    'Liquidity Ratio': 0.0, 
                    'status': 'No liabilities found.', 
                    'recommendation': 'Verify account data structure.'
                }
            
            avg_liquidity = round(current_assets / current_liabilities, 2)
            status = 'Success (calculated from account types)'
        
        # Generate recommendation based on liquidity level
        recommendation = "Liquidity is stable."
        if avg_liquidity < 1.0: 
            recommendation = "Low liquidity detected: Consider increasing liquid assets or reducing short-term liabilities."
        elif avg_liquidity > 2.0: 
            recommendation = "Strong liquidity position; consider strategic investments for better returns."
        
        return {
            'Liquidity Ratio': avg_liquidity,
            'Current Ratio': avg_liquidity,
            'Quick Ratio': round(avg_liquidity * 0.8, 2),  # Approximate quick ratio
            'Cash Ratio': round(avg_liquidity * 0.5, 2),    # Approximate cash ratio
            'status': status,
            'recommendation': recommendation
        }
    except Exception as e:
        return {
            'Liquidity Ratio': 0.0,
            'status': f'Error calculating liquidity: {str(e)}',
            'recommendation': 'Please check data quality and column names.'
        }

# Интеграция данных
def collect_integrated_data(start_date: str, end_date: str, account_type: str = None, symbol: str = None, real_time: bool = False) -> Dict[str, Any]:
    try:
        # Get currency rates
        rates = get_currency_rates()
        
        # Get transactions with error handling
        transactions = get_transactions(start_date, end_date, account_type, symbol, handle_noisy=True, real_time=real_time)
        
        # Convert numeric columns to proper numeric types
        numeric_columns = ['Transaction Amount', 'Cash Flow', 'Liquidity Ratio']
        for col in numeric_columns:
            if col in transactions.columns:
                transactions[col] = pd.to_numeric(transactions[col], errors='coerce')
        
        # Calculate USD equivalent if possible
        usd_rate = rates.get('USD', 1.0) or 1.0
        if 'Transaction Amount' in transactions.columns:
            transactions['USD_Equivalent'] = transactions['Transaction Amount'] / usd_rate
        
        # Calculate summary metrics with error handling
        total_cash_flow = 0
        if 'Cash Flow' in transactions.columns:
            total_cash_flow = round(transactions['Cash Flow'].sum(), 2)
            
        total_transactions = len(transactions)
        
        avg_liquidity = 0.0
        if 'Liquidity Ratio' in transactions.columns:
            avg_liquidity = round(transactions['Liquidity Ratio'].mean(), 2)
        
        # Generate summary by symbol if no specific symbol requested
        summary_by_symbol = {}
        if symbol is None and len(transactions) > 0 and 'Symbol' in transactions.columns:
            agg_dict = {}
            if 'Cash Flow' in transactions.columns:
                agg_dict['Cash Flow'] = 'sum'
            if 'Liquidity Ratio' in transactions.columns:
                agg_dict['Liquidity Ratio'] = 'mean'
            
            if agg_dict:
                summary_by_symbol = transactions.groupby('Symbol').agg(agg_dict).round(2).to_dict('index')
        
        # Generate recommendation
        recommendation = "Financial position appears stable."
        if total_cash_flow < 0:
            recommendation = f"Warning: Negative cash flow detected ({total_cash_flow:.2f}). Consider expense reduction."
        elif avg_liquidity < 1.0 and avg_liquidity > 0:
            recommendation += f" Liquidity concern: ratio is {avg_liquidity}, consider improving liquid asset position."
        
        # Build response data structure
        data = {
            'date_range': {
                'start': start_date, 
                'end': end_date
            },
            'currency_rates': rates,
            'transactions': transactions.to_dict('records'),
            'summary': {
                'total_transactions': total_transactions,
                'total_cash_flow': total_cash_flow,
                'avg_liquidity_ratio': avg_liquidity,
                'recommendation': recommendation
            },
            'summary_by_symbol': summary_by_symbol
        }
        
        # Add liquidity metrics to summary
        liquidity_metrics = calculate_liquidity_metrics(transactions)
        data['summary'].update(liquidity_metrics)
        
        return data
        
    except Exception as e:
        raise Exception(f"Error in collect_integrated_data: {str(e)}")

if __name__ == "__main__":
    # Test the function
    current_time = datetime.now() + timedelta(hours=5)  # 05:59 PM +05, 2025-09-19
    start = '2025-01-01'
    end = current_time.strftime('%Y-%m-%d')
    
    try:
        data = collect_integrated_data(start, end, symbol='JPM')
        print("Data collection successful!")
        print(json.dumps(data, indent=2, default=str))
    except Exception as e:

        print(f"Error: {e}")
