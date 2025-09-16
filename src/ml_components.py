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

# Добавляем src в sys.path, если запускаем из корня
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импорт функции из Datacollector.py
from Datacollector import collect_integrated_data

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

# 2. Сценарии "what-if" (Monte Carlo с уменьшенным шумом)
def monte_carlo_simulation(transactions: pd.DataFrame, num_simulations: int = 2000) -> pd.Series:
    """
    Симулирует сценарии роста курсов и задержек платежей с уменьшенным шумом.
    """
    results = []
    usd_rate = transactions['USD_Equivalent'].mean()
    for _ in range(num_simulations):
        simulated_rate = usd_rate * (1 + np.random.normal(0.1, 0.01))
        delay_factor = np.random.uniform(0.95, 1.05)
        sim_cash_flow = transactions['Cash Flow'] * delay_factor / simulated_rate
        results.append(sim_cash_flow.sum())
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
        sim_results = monte_carlo_simulation(transactions)
        metrics = evaluate_models(prophet_model, transactions, rec_model)
        metrics['Monte_Carlo_Mean'] = sim_results.mean()
        metrics['Monte_Carlo_CI_Low'] = sim_results.quantile(0.025)
        metrics['Monte_Carlo_CI_High'] = sim_results.quantile(0.975)
        # Преобразование метрик в сериализуемые типы
        serialized_metrics = json.loads(json.dumps(metrics, default=convert_to_serializable))
        return jsonify(serialized_metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)