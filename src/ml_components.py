import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, f1_score
import json
import os
import sys
from flask import Flask, request, jsonify
from typing import Dict, Any
from datetime import datetime, timedelta
import math  # Для обработки inf/NaN
import traceback  # Для детальных ошибок

# Добавляем src в sys.path, если запускаем из корня
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импорт функций из Datacollector.py
try:
    from Datacollector import collect_integrated_data, calculate_liquidity_metrics, BANK_SYMBOLS
except ImportError as e:
    print(f"Import error: {e}. Ensure Datacollector.py is in the same directory.")
    # Fallback: Определим заглушки
    def collect_integrated_data(start_date, end_date, account_type=None, symbol=None, real_time=False):
        raise ImportError("Datacollector not available.")
    def calculate_liquidity_metrics(transactions):
        return {'Liquidity Ratio': 0.0, 'status': 'Import error', 'recommendation': 'Fix Datacollector import.'}
    BANK_SYMBOLS = ['WFC', 'JPM', 'SBER']  # Минимальный fallback

# Путь к данным
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

app = Flask(__name__)

# Функция для обработки выбросов (IQR метод)
def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Удаляет выбросы из столбца с помощью IQR."""
    if len(df) == 0:
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 1. Постройте модели: Прогноз потоков (Prophet с регрессорами, per-symbol)
def train_prophet_model(transactions: pd.DataFrame, symbol: str = None) -> Dict[str, Prophet]:
    """Обучает Prophet на агрегированных данных Cash Flow с регрессорами."""
    try:
        if len(transactions) == 0:
            raise ValueError("No data for Prophet model.")
        
        models = {}
        if symbol:
            df_filtered = transactions[transactions['Symbol'] == symbol]
            if len(df_filtered) < 2:
                raise ValueError(f"Insufficient data for {symbol}.")
            df_prophet = df_filtered.groupby('Date').agg({'Cash Flow': 'sum', 'USD_Equivalent': 'mean', 'Liquidity Ratio': 'mean'}).reset_index()
            df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Cash Flow': 'y', 'USD_Equivalent': 'usd_rate', 'Liquidity Ratio': 'liq_ratio'})
            df_prophet = remove_outliers(df_prophet, 'y').dropna()
            if len(df_prophet) < 2 or df_prophet['y'].isna().all():
                raise ValueError(f"Insufficient clean data for {symbol} after outlier removal.")
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.add_regressor('usd_rate')
            model.add_regressor('liq_ratio')
            model.fit(df_prophet)
            models[symbol] = model
        else:
            for sym in transactions['Symbol'].unique():
                if sym in BANK_SYMBOLS:
                    sym_data = train_prophet_model(transactions[transactions['Symbol'] == sym], sym)
                    models.update(sym_data)
        
        if not models:
            raise ValueError("No valid models trained.")
        return models
    except Exception as e:
        raise Exception(f"Error training Prophet model: {str(e)}")

# 2. Сценарии "what-if" (Monte Carlo, per-symbol)
def monte_carlo_simulation(transactions: pd.DataFrame, num_simulations: int = 2000, 
                           currency_growth: float = 0.1, currency_std: float = 0.01, 
                           delay_factor: float = 0.05, purchase_shift_days: int = 0,
                           scenario_type: str = 'all', symbol: str = None) -> Dict[str, pd.Series]:
    """Симулирует сценарии 'что если' per-symbol."""
    try:
        if len(transactions) == 0:
            raise ValueError("No data for Monte Carlo simulation.")
        
        results = {}
        if symbol:
            sim_data = transactions[transactions['Symbol'] == symbol].copy()
            if len(sim_data) == 0:
                raise ValueError(f"No data for {symbol}.")
            sim_results = _run_monte_carlo(sim_data, num_simulations, currency_growth, currency_std, delay_factor, purchase_shift_days, scenario_type)
            results[symbol] = sim_results
        else:
            for sym in transactions['Symbol'].unique():
                if sym in BANK_SYMBOLS:
                    sym_data = transactions[transactions['Symbol'] == sym]
                    if len(sym_data) > 0:
                        sim_results = _run_monte_carlo(sym_data, num_simulations, currency_growth, currency_std, delay_factor, purchase_shift_days, scenario_type)
                        results[sym] = sim_results
        
        if not results:
            raise ValueError("No simulations run.")
        return results
    except Exception as e:
        raise Exception(f"Error in Monte Carlo simulation: {str(e)}")

def _run_monte_carlo(transactions: pd.DataFrame, num_simulations: int, currency_growth: float, currency_std: float, 
                     delay_factor: float, purchase_shift_days: int, scenario_type: str) -> pd.Series:
    """Внутренняя функция для single run."""
    sim_results = []
    usd_rate = transactions['USD_Equivalent'].mean()
    if np.isnan(usd_rate):
        usd_rate = 1.0
    for _ in range(num_simulations):
        sim_transactions = transactions.copy()
        if scenario_type in ['all', 'currency_growth']:
            simulated_rate = max(usd_rate * (1 + np.random.normal(currency_growth, currency_std)), 0.01)  # Избегаем нуля
            sim_transactions['USD_Equivalent'] = sim_transactions['Transaction Amount'] / simulated_rate
            sim_transactions['Cash Flow'] /= simulated_rate
            sim_transactions['Liquidity Ratio'] *= simulated_rate
        if scenario_type in ['all', 'payment_delay']:
            sim_transactions['Cash Flow'] *= np.random.uniform(1 - delay_factor, 1 + delay_factor)
            delay_days = np.random.randint(0, int(delay_factor * 30))
            sim_transactions['Date'] = pd.to_datetime(sim_transactions['Date']) + pd.Timedelta(days=delay_days)
        if scenario_type in ['all', 'purchase_schedule']:
            purchase_mask = sim_transactions['Account Type'].str.contains('Purchase|Procurement', case=False, na=False)
            shift_days = np.random.randint(-purchase_shift_days, purchase_shift_days)
            sim_transactions.loc[purchase_mask, 'Date'] = pd.to_datetime(sim_transactions.loc[purchase_mask, 'Date']) + pd.Timedelta(days=shift_days)
        sim_cash_flow = sim_transactions.groupby('Date')['Cash Flow'].sum().sum()
        sim_results.append(sim_cash_flow)
    return pd.Series(sim_results)

# 3. Рекомендации (Decision Tree с Liquidity Ratio)
def train_recommendation_model(transactions: pd.DataFrame, symbol: str = None) -> Dict[str, DecisionTreeClassifier]:
    """Обучает Decision Tree для рекомендаций по кредитам/инвестициям, per-symbol."""
    try:
        if len(transactions) == 0:
            raise ValueError("No data for Decision Tree model.")
        
        models = {}
        if symbol:
            df_filtered = transactions[transactions['Symbol'] == symbol]
            if len(df_filtered) < 2:
                raise ValueError(f"Insufficient data for {symbol}.")
            X = df_filtered[['Debt-to-Equity Ratio', 'Profit Margin', 'Transaction Amount', 'Liquidity Ratio']].fillna(0)
            y = (df_filtered['Transaction Outcome'] == 1).astype(int)
            if len(np.unique(y)) < 2:
                raise ValueError(f"Unbalanced data for {symbol}.")
        else:
            for sym in transactions['Symbol'].unique():
                if sym in BANK_SYMBOLS:
                    sym_data = transactions[transactions['Symbol'] == sym]
                    if len(sym_data) >= 2:
                        X_sym = sym_data[['Debt-to-Equity Ratio', 'Profit Margin', 'Transaction Amount', 'Liquidity Ratio']].fillna(0)
                        y_sym = (sym_data['Transaction Outcome'] == 1).astype(int)
                        if len(np.unique(y_sym)) >= 2:
                            model = DecisionTreeClassifier(max_depth=3, random_state=42)
                            model.fit(X_sym, y_sym)
                            models[sym] = model
        
        if not models:
            raise ValueError("No valid models trained.")
        return models
    except Exception as e:
        raise Exception(f"Error training Decision Tree model: {str(e)}")

# 4. Измерьте метрики (per-symbol)
def evaluate_models(prophet_models: Dict[str, Prophet], transactions: pd.DataFrame, rec_models: Dict[str, DecisionTreeClassifier], symbol: str = None) -> Dict[str, Any]:
    """Оценивает точность моделей (MAE для Prophet, F1 для Decision Tree), per-symbol."""
    try:
        metrics = {}
        if symbol and symbol in prophet_models:
            prophet_model = prophet_models[symbol]
            rec_model = rec_models.get(symbol)
            sym_trans = transactions[transactions['Symbol'] == symbol]
            if len(sym_trans) == 0:
                raise ValueError(f"No data for {symbol}.")
            future = prophet_model.make_future_dataframe(periods=30)
            last_usd_rate = sym_trans.groupby('Date').agg({'USD_Equivalent': 'mean'}).iloc[-1]['USD_Equivalent'] if not sym_trans['USD_Equivalent'].isna().all() else 1.0
            last_liq = sym_trans['Liquidity Ratio'].mean() if 'Liquidity Ratio' in sym_trans.columns and not sym_trans['Liquidity Ratio'].isna().all() else 1.0
            future['usd_rate'] = last_usd_rate
            future['liq_ratio'] = last_liq
            forecast = prophet_model.predict(future)
            true_values = sym_trans.groupby('Date').agg({'Cash Flow': 'sum'}).reindex(future['ds']).fillna(0)['Cash Flow']
            if len(true_values) < 30 or len(forecast['yhat']) < 30:
                mae = 0.0
            else:
                mae = round(mean_absolute_error(true_values[-30:], forecast['yhat'][-30:]), 2)
            X_test = sym_trans[['Debt-to-Equity Ratio', 'Profit Margin', 'Transaction Amount', 'Liquidity Ratio']].fillna(0)
            y_true = (sym_trans['Transaction Outcome'] == 1).astype(int)
            if rec_model and len(X_test) > 0 and len(y_true) > 0:
                y_pred = rec_model.predict(X_test)
                f1 = round(f1_score(y_true, y_pred, zero_division=0), 2)
            else:
                f1 = 0.0
            metrics[symbol] = {'MAE (Prophet)': mae, 'F1-Score (Recommendations)': f1}
        else:
            for sym in set(list(prophet_models.keys()) + list(rec_models.keys())):
                if sym in BANK_SYMBOLS:
                    sym_metrics = evaluate_models({sym: prophet_models[sym]}, transactions, {sym: rec_models[sym]}, sym)
                    metrics.update(sym_metrics)
        
        avg_mae = np.mean([m['MAE (Prophet)'] for m in metrics.values() if m['MAE (Prophet)'] > 0])
        avg_f1 = np.mean([m['F1-Score (Recommendations)'] for m in metrics.values() if m['F1-Score (Recommendations)'] > 0])
        recommendation = "Model performance is adequate."
        if avg_mae > 1000 or np.isnan(avg_mae):
            recommendation = "Improve forecasting by adding more bank-specific features."
        elif avg_f1 < 0.7 or np.isnan(avg_f1):
            recommendation = "Enhance recommendations with liquidity thresholds."
        
        overall = {'overall': {'avg_MAE': round(avg_mae, 2) if not np.isnan(avg_mae) else 0.0, 
                              'avg_F1': round(avg_f1, 2) if not np.isnan(avg_f1) else 0.0, 
                              'recommendation': recommendation}}
        return {**metrics, **overall}
    except Exception as e:
        return {'overall': {'avg_MAE': 0.0, 'avg_F1': 0.0, 'recommendation': f"Error evaluating models: {str(e)}"}}

# Вспомогательная функция для сериализации
def convert_to_serializable(obj):
    """Преобразует NumPy-типы в стандартные Python-типы, фиксит inf/NaN."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict) and 'models' in obj:
        return {'note': 'Models not serialized; use for prediction only'}
    else:
        return obj

# Обработка ошибок в эндпоинтах
def handle_error(e, endpoint_name):
    error_details = {
        'error': str(e),
        'endpoint': endpoint_name,
        'traceback': traceback.format_exc()[:500]
    }
    return jsonify(error_details), 500

# Эндпоинт: Список банков
@app.route('/api/banks', methods=['GET'])
def get_banks():
    return jsonify({'banks': BANK_SYMBOLS})

# Эндпоинт: Получение данных
@app.route('/api/data', methods=['GET'])
def get_data():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        serialized_data = json.loads(json.dumps(data, default=convert_to_serializable))
        return jsonify(serialized_data)
    except Exception as e:
        return handle_error(e, '/api/data')

# Эндпоинт: Метрики моделей (исправлен)
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        if transactions.empty:
            raise ValueError("No transaction data available.")
        prophet_models = train_prophet_model(transactions, symbol)
        rec_models = train_recommendation_model(transactions, symbol)
        if not prophet_models or not rec_models:
            raise ValueError("Model training failed for all symbols.")
        metrics = evaluate_models(prophet_models, transactions, rec_models, symbol)
        serialized_metrics = json.loads(json.dumps(metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return handle_error(e, '/api/metrics')

# Эндпоинты для отчётов по ликвидности

@app.route('/api/liquidity', methods=['GET'])
def get_liquidity_report():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        liquidity_metrics = calculate_liquidity_metrics(transactions)
        serialized_metrics = json.loads(json.dumps(liquidity_metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return handle_error(e, '/api/liquidity')

@app.route('/api/liquidity/real-time', methods=['GET'])
def get_real_time_liquidity():
    current_time = datetime.now() + timedelta(hours=5)  # +05
    end = current_time.strftime('%Y-%m-%d')  # 2025-09-19
    start = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol, real_time=True)
        transactions = pd.DataFrame(data['transactions'])
        liquidity_metrics = calculate_liquidity_metrics(transactions)
        liquidity_metrics['timestamp'] = current_time.isoformat()
        serialized_metrics = json.loads(json.dumps(liquidity_metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return handle_error(e, '/api/liquidity/real-time')

# Эндпоинты для сценариев "что если"

@app.route('/api/what-if', methods=['GET'])
def get_what_if_scenarios():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    currency_growth = float(request.args.get('currency_growth', 0.1))
    currency_std = float(request.args.get('currency_std', 0.01))
    delay_factor = float(request.args.get('delay_factor', 0.05))
    purchase_shift_days = int(request.args.get('purchase_shift_days', 0))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        if transactions.empty:
            raise ValueError("No transaction data for simulation.")
        sim_results = monte_carlo_simulation(transactions, num_simulations, currency_growth, currency_std, 
                                             delay_factor, purchase_shift_days, scenario_type='all', symbol=symbol)
        summaries = {}
        for sym, results in sim_results.items():
            mean_cash_flow = round(results.mean(), 2)
            ci_low = round(results.quantile(0.025), 2)
            ci_high = round(results.quantile(0.975), 2)
            recommendation = "Scenario impacts are within acceptable range."
            if mean_cash_flow < 0:
                recommendation = f"Mitigate risks for {sym} to avoid negative cash flow."
            elif (ci_high - ci_low) / abs(mean_cash_flow) > 0.5 if mean_cash_flow != 0 else True:
                recommendation = f"High variability for {sym}; implement risk strategies."
            summaries[sym] = {'mean_cash_flow': mean_cash_flow, 'ci_low': ci_low, 'ci_high': ci_high, 'recommendation': recommendation}
        
        scenarios = {
            'scenarios': {sym: results.tolist() for sym, results in sim_results.items()},
            'parameters': {
                'currency_growth': currency_growth,
                'currency_std': currency_std,
                'delay_factor': delay_factor,
                'purchase_shift_days': purchase_shift_days
            },
            'summary': summaries
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return handle_error(e, '/api/what-if')

@app.route('/api/what-if/currency-growth', methods=['GET'])
def get_currency_growth_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    currency_growth = float(request.args.get('currency_growth', 0.1))
    currency_std = float(request.args.get('currency_std', 0.01))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        if transactions.empty:
            raise ValueError("No transaction data for simulation.")
        sim_results = monte_carlo_simulation(transactions, num_simulations, currency_growth, currency_std, 
                                             scenario_type='currency_growth', symbol=symbol)
        summaries = {}
        for sym, results in sim_results.items():
            mean_cash_flow = round(results.mean(), 2)
            ci_low = round(results.quantile(0.025), 2)
            ci_high = round(results.quantile(0.975), 2)
            recommendation = "Currency growth impact is manageable."
            if mean_cash_flow < 0:
                recommendation = f"Hedge against currency fluctuations for {sym}."
            elif (ci_high - ci_low) / abs(mean_cash_flow) > 0.5 if mean_cash_flow != 0 else True:
                recommendation = f"High uncertainty in {sym} currency growth; consider hedging."
            summaries[sym] = {'mean_cash_flow': mean_cash_flow, 'ci_low': ci_low, 'ci_high': ci_high, 'recommendation': recommendation}
        
        scenarios = {
            'scenarios': {sym: results.tolist() for sym, results in sim_results.items()},
            'parameters': {'currency_growth': currency_growth, 'currency_std': currency_std},
            'summary': summaries
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return handle_error(e, '/api/what-if/currency-growth')

@app.route('/api/what-if/payment-delay', methods=['GET'])
def get_payment_delay_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    delay_factor = float(request.args.get('delay_factor', 0.05))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        if transactions.empty:
            raise ValueError("No transaction data for simulation.")
        sim_results = monte_carlo_simulation(transactions, num_simulations, delay_factor=delay_factor, 
                                             scenario_type='payment_delay', symbol=symbol)
        summaries = {}
        for sym, results in sim_results.items():
            mean_cash_flow = round(results.mean(), 2)
            ci_low = round(results.quantile(0.025), 2)
            ci_high = round(results.quantile(0.975), 2)
            recommendation = "Payment delay impact is manageable."
            if mean_cash_flow < 0:
                recommendation = f"Improve payment collection for {sym}."
            elif (ci_high - ci_low) / abs(mean_cash_flow) > 0.5 if mean_cash_flow != 0 else True:
                recommendation = f"High variability in {sym} payment delays; tighten terms."
            summaries[sym] = {'mean_cash_flow': mean_cash_flow, 'ci_low': ci_low, 'ci_high': ci_high, 'recommendation': recommendation}
        
        scenarios = {
            'scenarios': {sym: results.tolist() for sym, results in sim_results.items()},
            'parameters': {'delay_factor': delay_factor},
            'summary': summaries
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return handle_error(e, '/api/what-if/payment-delay')

@app.route('/api/what-if/purchase-schedule', methods=['GET'])
def get_purchase_schedule_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    symbol = request.args.get('symbol')
    purchase_shift_days = int(request.args.get('purchase_shift_days', 0))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type, symbol)
        transactions = pd.DataFrame(data['transactions'])
        if transactions.empty:
            raise ValueError("No transaction data for simulation.")
        sim_results = monte_carlo_simulation(transactions, num_simulations, purchase_shift_days=purchase_shift_days, 
                                             scenario_type='purchase_schedule', symbol=symbol)
        summaries = {}
        for sym, results in sim_results.items():
            mean_cash_flow = round(results.mean(), 2)
            ci_low = round(results.quantile(0.025), 2)
            ci_high = round(results.quantile(0.975), 2)
            recommendation = "Purchase schedule changes are manageable."
            if mean_cash_flow < 0:
                recommendation = f"Optimize {sym} purchase schedule."
            elif (ci_high - ci_low) / abs(mean_cash_flow) > 0.5 if mean_cash_flow != 0 else True:
                recommendation = f"High variability in {sym} purchase shifts; stabilize planning."
            summaries[sym] = {'mean_cash_flow': mean_cash_flow, 'ci_low': ci_low, 'ci_high': ci_high, 'recommendation': recommendation}
        
        scenarios = {
            'scenarios': {sym: results.tolist() for sym, results in sim_results.items()},
            'parameters': {'purchase_shift_days': purchase_shift_days},
            'summary': summaries
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return handle_error(e, '/api/what-if/purchase-schedule')

# Health-check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "banks_supported": len(BANK_SYMBOLS)}), 200

if __name__ == "__main__":
    # Для локального запуска
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))