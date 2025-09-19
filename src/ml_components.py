import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS  # Import Flask-CORS
from typing import Dict
from datetime import datetime, timedelta
import traceback
import psutil
from sqlalchemy import create_engine, text

# Настройка базы данных
DB_URL = "postgresql://postgres.zolipjvrqejnhbendclq:fLXxkf42l6NtY@aws-1-eu-north-1.pooler.supabase.com:6543/postgres"
engine = create_engine(DB_URL)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Настройка sys.path для импорта Datacollector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт Datacollector
try:
    from Datacollector import collect_integrated_data, calculate_liquidity_metrics, BANK_SYMBOLS
except ImportError as e:
    logger.error(f"Import error: {e}. Ensure Datacollector.py is in the src directory.")
    def collect_integrated_data(start_date, end_date, account_type=None, symbol=None, real_time=False):
        raise ImportError("Datacollector not available.")
    def calculate_liquidity_metrics(transactions):
        return {'Liquidity Ratio': 0.0, 'status': 'Import error', 'recommendation': 'Fix Datacollector import.'}
    BANK_SYMBOLS = ['WFC', 'JPM', 'SBER']

app = Flask(__name__)

# Enable CORS for your frontend origin
CORS(app, resources={r"/api/*": {"origins": "https://financial-assistant-vite.vercel.app"}})

# Логирование памяти
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_usage:.2f} MB")

# Required columns for dataset validation
REQUIRED_COLUMNS = [
    'Transaction ID', 'Date', 'Account Type', 'Transaction Amount', 'Cash Flow',
    'Net Income', 'Revenue', 'Expenditure', 'Profit Margin', 'Debt-to-Equity Ratio',
    'Operating Expenses', 'Gross Profit', 'Transaction Volume', 'Processing Time (seconds)',
    'Accuracy Score', 'Missing Data Indicator', 'Normalized Transaction Amount',
    'Transaction Outcome', 'Symbol', 'Liquidity Ratio'
]

def validate_required_columns(transactions: pd.DataFrame) -> bool:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in transactions.columns]
    if missing_columns:
        return False, f'Missing required columns in dataset: {", ".join(missing_columns)}'
    return True, None

# 2. Monte Carlo
def monte_carlo_simulation(transactions: pd.DataFrame, num_simulations: int = 20,
                           currency_growth: float = 0.1, currency_std: float = 0.01, 
                           delay_factor: float = 0.05, purchase_shift_days: int = 0,
                           scenario_type: str = 'all', symbol: str = None) -> Dict[str, pd.Series]:
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
        logger.error(f"Error in Monte Carlo simulation: {str(e)} - Traceback: {traceback.format_exc()}")
        raise

def _run_monte_carlo(transactions: pd.DataFrame, num_simulations: int, currency_growth: float, currency_std: float, 
                     delay_factor: float, purchase_shift_days: int, scenario_type: str) -> pd.Series:
    sim_results = []
    usd_rate = transactions['USD_Equivalent'].mean()
    if np.isnan(usd_rate):
        usd_rate = 1.0
    for _ in range(num_simulations):
        sim_transactions = transactions.copy()
        if scenario_type in ['all', 'currency_growth']:
            simulated_rate = max(usd_rate * (1 + np.random.normal(currency_growth, currency_std)), 0.01)
            sim_transactions['USD_Equivalent'] = sim_transactions['Transaction Amount'] / simulated_rate
            sim_transactions['Cash Flow'] /= simulated_rate
            sim_transactions['Liquidity Ratio'] *= simulated_rate
        if scenario_type in ['all', 'payment_delay']:
            sim_transactions['Cash Flow'] *= np.random.uniform(1 - delay_factor, 1 + delay_factor)
            delay_days = np.random.randint(0, int(delay_factor * 30))
            sim_transactions['Date'] = pd.to_datetime(sim_transactions['Date']) + pd.Timedelta(days=delay_days)
        if scenario_type in ['all', 'purchase_schedule'] and purchase_shift_days != 0:
            purchase_mask = sim_transactions['Account Type'].str.contains('Purchase|Procurement', case=False, na=False)
            shift_days = np.random.randint(-purchase_shift_days, purchase_shift_days)
            sim_transactions.loc[purchase_mask, 'Date'] = pd.to_datetime(sim_transactions.loc[purchase_mask, 'Date']) + pd.Timedelta(days=shift_days)
        sim_cash_flow = sim_transactions.groupby('Date')['Cash Flow'].sum().sum()
        sim_results.append(sim_cash_flow)
    return pd.Series(sim_results)

# Сериализация
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): return float(obj) if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records')
    if isinstance(obj, dict) and 'model' in obj: return {'note': 'Model not serialized'}
    return obj

# Обработка ошибок
def handle_error(e, endpoint_name):
    logger.error(f"Error in {endpoint_name}: {str(e)} - {traceback.format_exc()}")
    return jsonify({'error': str(e), 'endpoint': endpoint_name, 'traceback': traceback.format_exc()[:500]}), 500

# Эндпоинты
@app.route('/api/banks', methods=['GET'])
def get_banks():
    try:
        log_memory_usage()
        return jsonify({'banks': BANK_SYMBOLS})
    except Exception as e:
        return handle_error(e, '/api/banks')

@app.route('/api/data', methods=['GET'])
def get_data():
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end: return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        return jsonify(json.loads(json.dumps(data, default=convert_to_serializable)))
    except Exception as e:
        return handle_error(e, '/api/data')

@app.route('/api/liquidity', methods=['GET'])
def get_liquidity_report():
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end: return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        return jsonify(json.loads(json.dumps(calculate_liquidity_metrics(transactions), default=convert_to_serializable)))
    except Exception as e:
        return handle_error(e, '/api/liquidity')

def get_what_if_common(scenario_type: str):
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end:
        return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        try:
            num_simulations = int(request.args.get('num_simulations', 20))
            currency_growth = float(request.args.get('currency_growth', 0.1))
            currency_std = float(request.args.get('currency_std', 0.01))
            delay_factor = float(request.args.get('delay_factor', 0.05))
            purchase_shift_days_raw = request.args.get('purchase_shift_days', '0')
            purchase_shift_days_clean = ''.join(c for c in purchase_shift_days_raw if c.isdigit() or c == '-')
            purchase_shift_days = int(purchase_shift_days_clean) if purchase_shift_days_clean else 0
        except ValueError as e:
            logger.error(f"Invalid query parameter: {str(e)}")
            return jsonify({"error": f"Invalid query parameter: {str(e)}"}), 400
        sim_results = monte_carlo_simulation(
            transactions,
            num_simulations,
            currency_growth,
            currency_std,
            delay_factor,
            purchase_shift_days,
            scenario_type,
            request.args.get('symbol')
        )
        summaries = {
            sym: {
                'mean': round(r.mean(), 2),
                'ci': [round(r.quantile(0.025), 2), round(r.quantile(0.975), 2)],
                'recommendation': "Within range." if r.mean() >= 0 else f"Mitigate risks for {sym}."
            }
            for sym, r in sim_results.items()
        }
        return jsonify({
            'scenarios': {sym: r.tolist() for sym, r in sim_results.items()},
            'summary': summaries
        })
    except Exception as e:
        return handle_error(e, f'/api/what-if/{scenario_type}')

@app.route('/api/what-if/all', methods=['GET'])
def get_what_if_all():
    return get_what_if_common('all')

@app.route('/api/what-if/currency-growth', methods=['GET'])
def get_what_if_currency_growth():
    return get_what_if_common('currency_growth')

@app.route('/api/what-if/payment-delay', methods=['GET'])
def get_what_if_payment_delay():
    return get_what_if_common('payment_delay')

@app.route('/api/what-if/purchase-schedule', methods=['GET'])
def get_what_if_purchase_schedule():
    return get_what_if_common('purchase_schedule')

@app.route('/health', methods=['GET'])
def health_check():
    try:
        log_memory_usage()
        return jsonify({"status": "OK", "banks_supported": len(BANK_SYMBOLS)})
    except Exception as e:
        return handle_error(e, '/health')

@app.route('/api/chart/cash-flow', methods=['GET'])
def get_cash_flow_chart():
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end:
        return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        if 'Date' not in transactions.columns or 'Cash Flow' not in transactions.columns:
            return jsonify({"error": "Required columns (Date, Cash Flow) not found"}), 400
            
        # Преобразование даты в строковый формат YYYY-MM-DD
        transactions['Date'] = pd.to_datetime(transactions['Date']).dt.strftime('%Y-%m-%d')
        chart_data = transactions.groupby('Date')['Cash Flow'].sum().reset_index()
        chart_data_json = chart_data.to_json(orient='records')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cash Flow Chart</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <canvas id="cashFlowChart" width="400" height="200"></canvas>
            <script>
                const ctx = document.getElementById('cashFlowChart').getContext('2d');
                const chartData = {chart_data_json};
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: chartData.map(row => row.Date),
                        datasets: [{{
                            label: 'Cash Flow',
                            data: chartData.map(row => row['Cash Flow']),
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Cash Flow Over Time'
                            }}
                        }},
                        scales: {{
                            y: {{ 
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Amount'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Date'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return render_template_string(html)
    except Exception as e:
        return handle_error(e, '/api/chart/cash-flow')

@app.route('/api/chart/liquidity', methods=['GET'])
def get_liquidity_chart():
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end:
        return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        if 'Date' not in transactions.columns or 'Liquidity Ratio' not in transactions.columns:
            return jsonify({"error": "Required columns (Date, Liquidity Ratio) not found"}), 400
            
        # Преобразование даты в строковый формат YYYY-MM-DD
        transactions['Date'] = pd.to_datetime(transactions['Date']).dt.strftime('%Y-%m-%d')
        chart_data = transactions.groupby('Date')['Liquidity Ratio'].mean().reset_index()
        chart_data_json = chart_data.to_json(orient='records')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Liquidity Chart</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <canvas id="liquidityChart" width="400" height="200"></canvas>
            <script>
                const ctx = document.getElementById('liquidityChart').getContext('2d');
                const chartData = {chart_data_json};
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: chartData.map(row => row.Date),
                        datasets: [{{
                            label: 'Liquidity Ratio',
                            data: chartData.map(row => row['Liquidity Ratio']),
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Liquidity Ratio Over Time'
                            }}
                        }},
                        scales: {{
                            y: {{ 
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Liquidity Ratio'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Date'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return render_template_string(html)
    except Exception as e:
        return handle_error(e, '/api/chart/liquidity')

@app.route('/api/chart/account-type-distribution', methods=['GET'])
def get_account_type_distribution_chart():
    start, end = request.args.get('start'), request.args.get('end')
    if not start or not end:
        return jsonify({"error": "Missing start/end"}), 400
    try:
        log_memory_usage()
        data = collect_integrated_data(start, end, request.args.get('account_type'), request.args.get('symbol'))
        transactions = pd.DataFrame(data['transactions'])
        is_valid, error_msg = validate_required_columns(transactions)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        if 'Account Type' not in transactions.columns or 'Transaction Amount' not in transactions.columns:
            return jsonify({"error": "Required columns (Account Type, Transaction Amount) not found"}), 400
            
        # Этот график не использует Date, поэтому преобразование не требуется
        chart_data = transactions.groupby('Account Type')['Transaction Amount'].sum().reset_index()
        chart_data_json = chart_data.to_json(orient='records')
        
        # Generate colors dynamically based on number of account types
        colors = [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)', 
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)',
            'rgba(255, 159, 64, 0.6)',
            'rgba(199, 199, 199, 0.6)',
            'rgba(83, 102, 255, 0.6)'
        ]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Account Type Distribution</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <canvas id="accountTypeChart" width="400" height="200"></canvas>
            <script>
                const ctx = document.getElementById('accountTypeChart').getContext('2d');
                const chartData = {chart_data_json};
                const colors = {colors};
                
                new Chart(ctx, {{
                    type: 'pie',
                    data: {{
                        labels: chartData.map(row => row['Account Type']),
                        datasets: [{{
                            label: 'Transaction Amount',
                            data: chartData.map(row => row['Transaction Amount']),
                            backgroundColor: colors.slice(0, chartData.length),
                            borderColor: colors.slice(0, chartData.length).map(color => color.replace('0.6', '1')),
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Transaction Amount by Account Type'
                            }},
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return render_template_string(html)
    except Exception as e:
        return handle_error(e, '/api/chart/account-type-distribution')

@app.route('/api/tables', methods=['GET'])
def get_tables():
    try:
        log_memory_usage()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """))
            tables = [row[0] for row in result.fetchall()]
        return jsonify({'tables': tables})
    except Exception as e:
        return handle_error(e, '/api/tables')

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    try:
        log_memory_usage()
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
            table_name = 'uploaded_csv_data'
            # Drop table if exists to overwrite
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()  # Ensure the DROP TABLE is committed
            df.to_sql(table_name, engine, index=False, if_exists='replace')
            return jsonify({'message': f'CSV uploaded and saved to table "{table_name}" successfully'})
        else:
            return jsonify({'error': 'Invalid file type. Only .csv files are allowed'}), 400
    except Exception as e:
        return handle_error(e, '/api/upload-csv')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))