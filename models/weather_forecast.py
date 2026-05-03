"""
models/weather_forecast.py - Time-series forecasting for weather & crop conditions
Uses Prophet for trend decomposition + XGBoost for feature-based prediction
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Using fallback forecasting.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config.settings import FORECAST_MODEL_PATH


class WeatherForecaster:
    """
    Multi-model time-series forecaster for weather variables.
    - Prophet for long-term trends (temperature, rainfall)
    - XGBoost for short-term anomaly detection
    - Fallback: statistical interpolation
    """

    def __init__(self, model_path: str = FORECAST_MODEL_PATH):
        self.models = {}
        self.scalers = {}
        self.model_path = model_path
        self.is_trained = False

        if Path(model_path).exists():
            self._load_models()

    def _load_models(self):
        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            self.models = saved.get("models", {})
            self.scalers = saved.get("scalers", {})
            self.is_trained = True
            logger.info("Loaded weather forecast models.")
        except Exception as e:
            logger.error(f"Failed to load forecast model: {e}")

    def _save_models(self):
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"models": self.models, "scalers": self.scalers}, f)

    def train(self, df: pd.DataFrame):
        """
        Train forecasting models.
        df must have columns: date, temperature, humidity, rainfall, soil_moisture
        """
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        target_cols = [c for c in ['temperature', 'humidity', 'rainfall', 'soil_moisture'] if c in df.columns]

        for col in target_cols:
            logger.info(f"Training forecast for: {col}")
            if PROPHET_AVAILABLE:
                prophet_df = df[['date', col]].rename(columns={'date': 'ds', col: 'y'})
                prophet_df = prophet_df.dropna()

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0
                )
                m.fit(prophet_df)
                self.models[f"prophet_{col}"] = m

            if XGB_AVAILABLE:
                features = self._create_features(df, col)
                y = df[col].values
                valid_mask = ~np.isnan(y)

                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                xgb_model.fit(features[valid_mask], y[valid_mask])
                self.models[f"xgb_{col}"] = xgb_model

        self.is_trained = True
        self._save_models()
        logger.info("Weather forecast models trained and saved.")

    def _create_features(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        """Create time-series features for XGBoost."""
        df = df.copy()
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek

        feature_cols = ['day_of_year', 'month', 'week', 'day_of_week']

        for lag in [1, 3, 7, 14]:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
            feature_cols.append(f'{target_col}_lag{lag}')

        df[f'{target_col}_rolling7'] = df[target_col].rolling(7).mean()
        df[f'{target_col}_rolling30'] = df[target_col].rolling(30).mean()
        feature_cols += [f'{target_col}_rolling7', f'{target_col}_rolling30']

        return df[feature_cols].fillna(df[feature_cols].mean()).values

    def forecast(self, days: int = 30) -> dict:
        """Generate forecast for next N days."""
        if not self.is_trained:
            logger.warning("Models not trained. Returning synthetic forecast.")
            return self._synthetic_forecast(days)

        results = {}
        future_dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        date_str = [d.strftime("%Y-%m-%d") for d in future_dates]

        target_cols = ['temperature', 'humidity', 'rainfall', 'soil_moisture']
        for col in target_cols:
            key = f"prophet_{col}"
            if key in self.models:
                future_df = pd.DataFrame({'ds': future_dates})
                forecast_df = self.models[key].predict(future_df)
                results[col] = {
                    "dates": date_str,
                    "values": forecast_df['yhat'].tolist(),
                    "lower": forecast_df['yhat_lower'].tolist(),
                    "upper": forecast_df['yhat_upper'].tolist()
                }
            else:
                results[col] = self._synthetic_forecast_col(col, days, date_str)

        return results

    def _synthetic_forecast(self, days: int) -> dict:
        future_dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        date_str = [d.strftime("%Y-%m-%d") for d in future_dates]

        return {
            "temperature": self._synthetic_forecast_col("temperature", days, date_str),
            "humidity": self._synthetic_forecast_col("humidity", days, date_str),
            "rainfall": self._synthetic_forecast_col("rainfall", days, date_str),
            "soil_moisture": self._synthetic_forecast_col("soil_moisture", days, date_str),
        }

    def _synthetic_forecast_col(self, col: str, days: int, date_str: list) -> dict:
        """Generate realistic synthetic forecast when model not trained."""
        base_values = {
            "temperature": 28.0, "humidity": 65.0, "rainfall": 5.0, "soil_moisture": 55.0
        }
        noise_scale = {
            "temperature": 3.0, "humidity": 8.0, "rainfall": 10.0, "soil_moisture": 5.0
        }

        base = base_values.get(col, 50.0)
        scale = noise_scale.get(col, 5.0)
        t = np.arange(days)
        values = base + scale * np.sin(2 * np.pi * t / 30) + np.random.normal(0, scale * 0.3, days)
        values = np.clip(values, 0, None).tolist()

        return {
            "dates": date_str,
            "values": values,
            "lower": [v - scale for v in values],
            "upper": [v + scale for v in values]
        }

    def detect_anomalies(self, current_data: dict) -> list:
        """Detect anomalies in current sensor readings."""
        anomalies = []
        thresholds = {
            "temperature": (10, 45),
            "humidity": (20, 95),
            "rainfall": (0, 250),
            "soil_moisture": (15, 85),
            "ph": (4.5, 8.5)
        }
        for key, (low, high) in thresholds.items():
            val = current_data.get(key)
            if val is not None:
                if val < low:
                    anomalies.append({
                        "parameter": key,
                        "value": val,
                        "issue": "below_threshold",
                        "threshold": low,
                        "severity": "high" if val < low * 0.8 else "medium"
                    })
                elif val > high:
                    anomalies.append({
                        "parameter": key,
                        "value": val,
                        "issue": "above_threshold",
                        "threshold": high,
                        "severity": "high" if val > high * 1.2 else "medium"
                    })
        return anomalies


def generate_sample_weather_data(days: int = 365) -> pd.DataFrame:
    """Generate realistic sample weather data for training/demo."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    t = np.arange(days)

    temperature = (
        30 + 8 * np.sin(2 * np.pi * t / 365) +
        3 * np.sin(2 * np.pi * t / 7) +
        np.random.normal(0, 1.5, days)
    )
    humidity = (
        65 + 15 * np.sin(2 * np.pi * (t + 90) / 365) +
        np.random.normal(0, 5, days)
    ).clip(20, 100)
    rainfall = np.maximum(0,
        5 * np.sin(2 * np.pi * t / 30) + np.random.exponential(3, days)
    )
    soil_moisture = (
        55 + 10 * np.sin(2 * np.pi * t / 30) + 0.3 * rainfall +
        np.random.normal(0, 3, days)
    ).clip(10, 100)

    return pd.DataFrame({
        "date": dates,
        "temperature": temperature.round(1),
        "humidity": humidity.round(1),
        "rainfall": rainfall.round(2),
        "soil_moisture": soil_moisture.round(1)
    })
