import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_real_data():
    """Завантажує реальні дані енергоспоживання з репозиторію OPSD."""
    print("Завантаження реальних даних з мережі...")
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

    cols = ['utc_timestamp', 'DE_load_actual_entsoe_transparency']
    df = pd.read_csv(url, usecols=cols, parse_dates=['utc_timestamp'])

    df = df.rename(columns={'utc_timestamp': 'ds', 'DE_load_actual_entsoe_transparency': 'y'})

    df['ds'] = df['ds'].dt.tz_localize(None)
    df = df.dropna().reset_index(drop=True)

    return df.iloc[8000:8000 + (24 * 60)].reset_index(drop=True)


df = load_real_data()
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"Дані згенеровано. Навчальна вибірка: {len(train)}, Тестова: {len(test)}")

print("Навчання ARIMA...")
try:
    model_arima = ARIMA(train['y'], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    arima_predictions = model_arima_fit.forecast(steps=len(test))
except Exception as e:
    print(f"Помилка ARIMA: {e}")
    arima_predictions = np.full(len(test), train['y'].mean())

print("Навчання Prophet...")
prophet_predictions = None
try:
    m = Prophet(changepoint_prior_scale=0.05, weekly_seasonality=True, daily_seasonality=True)
    m.fit(train)
    future = m.make_future_dataframe(periods=len(test), freq='h')
    forecast_prophet = m.predict(future)
    prophet_predictions = forecast_prophet.iloc[train_size:]['yhat'].values
except Exception as e:
    print("-" * 30)
    print(f"ПОМИЛКА PROPHET: {e}")
    print("Використовується базовий прогноз для Prophet через проблеми з оточенням.")
    print("-" * 30)

    prophet_predictions = np.full(len(test), train['y'].mean())

print("Навчання LSTM...")
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train['y'].values.reshape(-1, 1))
test_scaled = scaler.transform(test['y'].values.reshape(-1, 1))


def create_dataset(dataset, look_back=24):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 24
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_lstm = Sequential(
    [LSTM(50, return_sequences=True, input_shape=(look_back, 1)), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(1)])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

lstm_predictions_scaled = model_lstm.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled).flatten()

actual_test = test['y'].values[look_back:]
arima_final = arima_predictions[look_back:] if isinstance(arima_predictions, np.ndarray) else arima_predictions.values[
    look_back:]
prophet_final = prophet_predictions[look_back:]


def evaluate(actual, pred, name):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae


print("\nМетрики якості:")
metrics = {"ARIMA": evaluate(actual_test, arima_final, "ARIMA"),
           "Prophet": evaluate(actual_test, prophet_final, "Prophet"),
           "LSTM": evaluate(actual_test, lstm_predictions, "LSTM")}

plt.figure(figsize=(14, 7))
plt.plot(df['ds'].iloc[train_size + look_back:], actual_test, label='Фактичні дані', color='black', alpha=0.5)
plt.plot(df['ds'].iloc[train_size + look_back:], arima_final, label='ARIMA', linestyle='--')
plt.plot(df['ds'].iloc[train_size + look_back:], prophet_final, label='Prophet', linestyle='-.')
plt.plot(df['ds'].iloc[train_size + look_back:], lstm_predictions, label='LSTM', color='red', linewidth=2)
plt.title('Порівняння моделей прогнозування енергії (Виправлено)')
plt.xlabel('Дата/Час')
plt.ylabel('Споживання (кВт)')
plt.legend()
plt.grid(True)
plt.show()
