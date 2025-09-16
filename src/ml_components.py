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

# Добавляем src в sys.path, если запускаем из корня
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импорт функции из Datacollector.py
from Datacollector import collect_integrated_data, calculate_liquidity_metrics

# Путь к данным
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

app = Flask(__name__)

# Функция для обработки выбросов (IQR метод)
def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Удаляет выбросы из столбца с помощью IQR.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 1. Постройте модели: Прогноз потоков (Prophet с регрессорами)
def train_prophet_model(transactions: pd.DataFrame) -> Prophet:
    """
    Обучает Prophet на агрегированных данных Cash Flow с регрессорами (курсы валют).
    """
    df_prophet = transactions.groupby('Date').agg({'Cash Flow': 'sum', 'USD_Equivalent': 'mean'}).reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Cash Flow': 'y', 'USD_Equivalent': 'usd_rate'})
    df_prophet = remove_outliers(df_prophet, 'y')
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor('usd_rate')
    model.fit(df_prophet)
    return model

# 2. Сценарии "what-if" (Monte Carlo с уменьшенным шумом и расширенными параметрами)
def monte_carlo_simulation(transactions: pd.DataFrame, num_simulations: int = 2000, 
                           currency_growth: float = 0.1, currency_std: float = 0.01, 
                           delay_factor: float = 0.05, purchase_shift_days: int = 0,
                           scenario_type: str = 'all') -> pd.Series:
    """
    Симулирует сценарии "что если" с учётом роста курса, задержки платежей и сдвига графика закупок.
    scenario_type: 'all', 'currency_growth', 'payment_delay', 'purchase_schedule'
    """
    results = []
    usd_rate = transactions['USD_Equivalent'].mean()
    for _ in range(num_simulations):
        sim_transactions = transactions.copy()
        if scenario_type in ['all', 'currency_growth']:
            # Рост курса
            simulated_rate = usd_rate * (1 + np.random.normal(currency_growth, currency_std))
            sim_transactions['USD_Equivalent'] = sim_transactions['Transaction Amount'] / simulated_rate
            sim_transactions['Cash Flow'] /= simulated_rate  # Корректировка Cash Flow на новый курс
        if scenario_type in ['all', 'payment_delay']:
            # Задержка платежа (множитель + сдвиг дат)
            sim_transactions['Cash Flow'] *= np.random.uniform(1 - delay_factor, 1 + delay_factor)
            delay_days = np.random.randint(0, int(delay_factor * 30))  # Задержка до 30 дней
            sim_transactions['Date'] += pd.Timedelta(days=delay_days)
        if scenario_type in ['all', 'purchase_schedule']:
            # Сдвиг графика закупок
            purchase_mask = sim_transactions['Account Type'].str.contains('Purchase|Procurement', case=False, na=False)
            shift_days = np.random.randint(-purchase_shift_days, purchase_shift_days)
            sim_transactions.loc[purchase_mask, 'Date'] += pd.Timedelta(days=shift_days)
        # Агрегация по датам и суммирование Cash Flow
        sim_cash_flow = sim_transactions.groupby('Date')['Cash Flow'].sum().sum()
        results.append(sim_cash_flow)
    return pd.Series(results)

# 3. Рекомендации (Decision Tree)
def train_recommendation_model(transactions: pd.DataFrame) -> DecisionTreeClassifier:
    """
    Обучает Decision Tree для рекомендаций по кредитам/инвестициям.
    """
    X = transactions[['Debt-to-Equity Ratio', 'Profit Margin', 'Transaction Amount']].fillna(0)
    y = (transactions['Transaction Outcome'] == 1).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model

# 4. Измерьте метрики
def evaluate_models(prophet_model: Prophet, transactions: pd.DataFrame, rec_model: DecisionTreeClassifier) -> Dict[str, float]:
    """
    Оценивает точность моделей (MAE для Prophet, F1 для Decision Tree).
    """
    future = prophet_model.make_future_dataframe(periods=30)
    last_usd_rate = transactions.groupby('Date').agg({'USD_Equivalent': 'mean'}).iloc[-1]['USD_Equivalent']
    future['usd_rate'] = last_usd_rate
    forecast = prophet_model.predict(future)
    true_values = transactions.groupby('Date').agg({'Cash Flow': 'sum'}).reindex(future['ds']).fillna(0)['Cash Flow']
    mae = mean_absolute_error(true_values[-30:], forecast['yhat'][-30:])
    X_test = transactions[['Debt-to-Equity Ratio', 'Profit Margin', 'Transaction Amount']].fillna(0)
    y_pred = rec_model.predict(X_test)
    f1 = f1_score((transactions['Transaction Outcome'] == 1).astype(int), y_pred)
    return {'MAE (Prophet)': mae, 'F1-Score (Recommendations)': f1}

# Вспомогательная функция для сериализации
def convert_to_serializable(obj):
    """Преобразует NumPy-типы в стандартные Python-типы."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    else:
        return obj

# Эндпоинты API

# Эндпоинт для получения данных
@app.route('/api/data', methods=['GET'])
def get_data():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        # Преобразование всех значений в сериализуемые типы
        serialized_data = json.loads(json.dumps(data, default=convert_to_serializable))
        return jsonify(serialized_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Эндпоинт для получения метрик моделей
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        prophet_model = train_prophet_model(transactions)
        rec_model = train_recommendation_model(transactions)
        metrics = evaluate_models(prophet_model, transactions, rec_model)
        # Преобразование метрик в сериализуемые типы
        serialized_metrics = json.loads(json.dumps(metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Эндпоинты для отчётов по ликвидности

# Общий отчёт по ликвидности
@app.route('/api/liquidity', methods=['GET'])
def get_liquidity_report():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        liquidity_metrics = calculate_liquidity_metrics(transactions)
        serialized_metrics = json.loads(json.dumps(liquidity_metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Отчёт по ликвидности в реальном времени (последние 24 часа)
@app.route('/api/liquidity/real-time', methods=['GET'])
def get_real_time_liquidity():
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    account_type = request.args.get('account_type')
    
    try:
        data = collect_integrated_data(start, end, account_type, real_time=True)
        transactions = pd.DataFrame(data['transactions'])
        liquidity_metrics = calculate_liquidity_metrics(transactions)
        liquidity_metrics['timestamp'] = datetime.now().isoformat()
        serialized_metrics = json.loads(json.dumps(liquidity_metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Эндпоинты для сценариев "что если"

# Общий сценарий "что если"
@app.route('/api/what-if', methods=['GET'])
def get_what_if_scenarios():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    currency_growth = float(request.args.get('currency_growth', 0.1))
    currency_std = float(request.args.get('currency_std', 0.01))
    delay_factor = float(request.args.get('delay_factor', 0.05))
    purchase_shift_days = int(request.args.get('purchase_shift_days', 0))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        sim_results = monte_carlo_simulation(transactions, num_simulations, 
                                             currency_growth, currency_std, 
                                             delay_factor, purchase_shift_days, 
                                             scenario_type='all')
        scenarios = {
            'scenarios': sim_results.tolist(),
            'parameters': {
                'currency_growth': currency_growth,
                'currency_std': currency_std,
                'delay_factor': delay_factor,
                'purchase_shift_days': purchase_shift_days
            },
            'summary': {
                'mean_cash_flow': round(sim_results.mean(), 2),
                'ci_low': round(sim_results.quantile(0.025), 2),
                'ci_high': round(sim_results.quantile(0.975), 2)
            }
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Сценарий "рост курса валют"
@app.route('/api/what-if/currency-growth', methods=['GET'])
def get_currency_growth_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    currency_growth = float(request.args.get('currency_growth', 0.1))
    currency_std = float(request.args.get('currency_std', 0.01))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        sim_results = monte_carlo_simulation(transactions, num_simulations, 
                                             currency_growth, currency_std, 
                                             scenario_type='currency_growth')
        scenarios = {
            'scenarios': sim_results.tolist(),
            'parameters': {
                'currency_growth': currency_growth,
                'currency_std': currency_std
            },
            'summary': {
                'mean_cash_flow': round(sim_results.mean(), 2),
                'ci_low': round(sim_results.quantile(0.025), 2),
                'ci_high': round(sim_results.quantile(0.975), 2)
            }
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Сценарий "задержка платежа"
@app.route('/api/what-if/payment-delay', methods=['GET'])
def get_payment_delay_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    delay_factor = float(request.args.get('delay_factor', 0.05))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        sim_results = monte_carlo_simulation(transactions, num_simulations, 
                                             delay_factor=delay_factor, 
                                             scenario_type='payment_delay')
        scenarios = {
            'scenarios': sim_results.tolist(),
            'parameters': {
                'delay_factor': delay_factor
            },
            'summary': {
                'mean_cash_flow': round(sim_results.mean(), 2),
                'ci_low': round(sim_results.quantile(0.025), 2),
                'ci_high': round(sim_results.quantile(0.975), 2)
            }
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Сценарий "изменение графика закупок"
@app.route('/api/what-if/purchase-schedule', methods=['GET'])
def get_purchase_schedule_scenario():
    start = request.args.get('start')
    end = request.args.get('end')
    account_type = request.args.get('account_type')
    purchase_shift_days = int(request.args.get('purchase_shift_days', 0))
    num_simulations = int(request.args.get('num_simulations', 2000))
    
    if not start or not end:
        return jsonify({"error": "Parameters 'start' and 'end' are required"}), 400
    
    try:
        data = collect_integrated_data(start, end, account_type)
        transactions = pd.DataFrame(data['transactions'])
        sim_results = monte_carlo_simulation(transactions, num_simulations, 
                                             purchase_shift_days=purchase_shift_days, 
                                             scenario_type='purchase_schedule')
        scenarios = {
            'scenarios': sim_results.tolist(),
            'parameters': {
                'purchase_shift_days': purchase_shift_days
            },
            'summary': {
                'mean_cash_flow': round(sim_results.mean(), 2),
                'ci_low': round(sim_results.quantile(0.025), 2),
                'ci_high': round(sim_results.quantile(0.975), 2)
            }
        }
        serialized_scenarios = json.loads(json.dumps(scenarios, default=convert_to_serializable))
        return jsonify(serialized_scenarios)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)